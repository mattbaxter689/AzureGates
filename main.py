import argparse
import json
import logging
import sys
import time
from enum import Enum
from pathlib import Path
import tempfile

from azure.ai.ml import MLClient, Input, command, Output
from azure.ai.ml.entities import Command
from azure.identity import DefaultAzureCredential
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("orchestrator")
console = Console()

ROOT = Path(__file__).parent
CONFIG: dict = json.loads((ROOT / "config.json").read_text())


# ------ Gating -------
class Gate(Enum):
    DATA = "data"
    DRIFT = "drift"
    # FIT = "fit"
    # PROMOTE = "promote"
    # DEPLOY = "deploy"


# More gates will go here, but will happen iteratively as things are built out
GATE_ORDER = [Gate.DATA, Gate.DRIFT]


# -------- Arg parsing -------
def parse_args() -> argparse.Namespace:

    # NOTE: again, we add logic for everything but comment and add iteratively
    p = argparse.ArgumentParser(description="AML Classification Pipeline Orchestrator")
    p.add_argument("--start-from", choices=[g.value for g in Gate], default="data")
    p.add_argument("--data-version", default=None)
    p.add_argument("--run-id", default=None)
    # p.add_argument("--n-trials", type=int, default=CONFIG.get("pipeline", {}).get("n_trials", 30))
    # p.add_argument("--max-epochs", type=int, default=100)
    # p.add_argument("--final-epochs", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    # p.add_argument("--force-promote", action="store_true")
    return p.parse_args()


# ----- AML client and environment -------
def get_client() -> MLClient:
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=CONFIG["workspace"]["subscription_id"],
        resource_group_name=CONFIG["workspace"]["resource_group"],
        workspace_name=CONFIG["workspace"]["workspace_name"],
    )


def get_aml_environment() -> str:
    cfg_env = CONFIG["environment"]
    return f"{cfg_env['name']}@latest"


# ------ Base Job Config ----------
def _base_job_kwargs(name: str, description: str) -> dict:
    compute = CONFIG["compute"]["training_cluster"]

    return dict(
        display_name=name,
        description=description,
        environment=get_aml_environment(),
        compute=compute,
        experiment_name=CONFIG["pipeline"]["experiment_name"],
        code="./src",
        environment_variables={"PYTHONPATH": "./"},
    )


# ------ Job Builders ---------
def build_data_job(ml_client: MLClient) -> Command:

    return command(
        **_base_job_kwargs(
            "data-versioning-gate", "Fetch, clean, split, and version data"
        ),
        command=(
            "python -m gates.data_versioning_gate "
            "--raw-data ${{inputs.raw_data}} "
            "--output-asset-name ${{inputs.asset_name}} "
            "--output-version-path ${{outputs.asset_version}}"  # AML mounts this path
        ),
        inputs={
            "raw_data": Input(type="uri_file", path=CONFIG["data"]["name"]),
            "asset_name": CONFIG["data"]["output_asset_name"],
        },
        outputs={"asset_version": Output(type="uri_file", mode="rw_mount")},
    )


def build_drift_job(data_asset_version: str) -> Command:
    return command(
        **_base_job_kwargs(
            "drift-detection-gate", "Statistical drift check against baseline"
        ),
        command=(
            "python -m gates.drift_detection_gate "
            "--new-data-version ${{inputs.training_asset}} "
            "--gold-data-version ${{inputs.gold_data}}"
        ),
        inputs={
            "gold_data": Input(type="uri_file", path=CONFIG["data"]["name"]),
            "training_asset": Input(
                type="uri_file",
                path=str(CONFIG["data"]["output_asset_name"])
                + f":{data_asset_version}",
            ),
        },
    )


# ------- Job Execution Helpers -------
def submit_and_wait(
    ml_client: MLClient,
    job: Command,
    gate_name: str,
    dry_run: bool = False,
) -> dict:
    """Submit a job, poll until terminal, return {'status', 'outputs'}."""
    if dry_run:
        console.print(f"[dim][DRY RUN] Would submit: {job.display_name}[/dim]")
        return {"status": "DryRun", "outputs": {}}

    console.print(f"\n[bold cyan]▶ Submitting gate:[/bold cyan] {gate_name}")
    returned_job = ml_client.jobs.create_or_update(job)
    job_url = returned_job.studio_url
    console.print(f"  Job ID  : [link={job_url}]{returned_job.name}[/link]")
    console.print(f"  Studio  : {job_url}")

    # Poll until terminal state
    terminal_states = {"Completed", "Failed", "Canceled"}
    poll_interval = 30  # seconds
    while True:
        job_name = returned_job.name
        assert job_name is not None
        current = ml_client.jobs.get(job_name)
        status = current.status
        if status in terminal_states:
            break
        console.print(
            f"  Status  : [yellow]{status}[/yellow] — polling in {poll_interval}s…"
        )
        time.sleep(poll_interval)

    if status != "Completed":
        console.print(
            f"[bold red]✘ Gate '{gate_name}' {status}. Pipeline halted.[/bold red]"
        )
        raise RuntimeError(f"Gate '{gate_name}' failed with status '{status}'")

    console.print(f"[bold green]✔ Gate '{gate_name}' Completed[/bold green]")
    outputs = getattr(current, "outputs", {})
    return {"status": status, "outputs": outputs, "job_name": current.name}


def read_output_string(
    ml_client: MLClient, job_name: str, job_outputs: dict, key: str
) -> str | None:
    """Download a uri_file output and return its text content."""
    try:
        output = job_outputs.get(key)
        if output is None:
            log.warning("Output '%s' not found in job outputs", key)
            return None

        with tempfile.TemporaryDirectory() as tmp:
            ml_client.jobs.download(name=job_name, output_name=key, download_path=tmp)
            # AML downloads to <tmp>/named-outputs/<key>/<filename>
            output_dir = Path(tmp) / "named-outputs" / key
            files = list(output_dir.iterdir())
            if not files:
                log.warning("No files found in output '%s'", key)
                return None
            return files[0].read_text().strip()

    except Exception as exc:
        log.warning("Could not read output '%s': %s", key, exc)
    return None


def main() -> None:
    args = parse_args()
    ml_client = get_client()

    # State accumulated across gates
    mlflow_run_id: str | None = args.run_id
    model_version: int | None = None

    start_gate = Gate(args.start_from)
    active_gates = GATE_ORDER[GATE_ORDER.index(start_gate) :]

    # ── Print plan ────────────────────────────────────────────────────────────
    table = Table(title="Pipeline Execution Plan", show_header=True)
    table.add_column("Gate", style="cyan")
    table.add_column("Status", style="white")
    for gate in GATE_ORDER:
        status = "ACTIVE" if gate in active_gates else "SKIPPED"
        style = "green" if status == "ACTIVE" else "dim"
        table.add_row(gate.value, f"[{style}]{status}[/{style}]")
    console.print(table)

    if args.dry_run:
        console.print(
            Panel("[yellow]DRY RUN — jobs will be built but not submitted[/yellow]")
        )

    try:
        # ── Gate 1: Data Versioning ───────────────────────────────────────────
        if Gate.DATA in active_gates:
            asset_name = CONFIG["data"]["output_asset_name"]
            if args.data_version:
                console.print(
                    f"[dim]⏭ Skipping data gate — asset '{asset_name}' already exists.[/dim]"
                )
                data_asset_version = args.data_version
            else:
                job = build_data_job(ml_client)
                result = submit_and_wait(
                    ml_client, job, "Data Versioning", args.dry_run
                )
                data_asset_version = (
                    read_output_string(
                        ml_client,
                        result["job_name"],
                        result["outputs"],
                        "asset_version",
                    )
                    or "latest"
                )

                console.print(
                    f"  Data asset version: [bold]{data_asset_version}[/bold]"
                )

        # --- Gate 2: Drift Detection ---------------
        if Gate.DRIFT in active_gates:
            job = build_drift_job(data_asset_version)
            submit_and_wait(ml_client, job, "Drift Detection", args.dry_run)

    except RuntimeError as exc:
        console.print(
            Panel(f"[bold red]Pipeline halted:\n{exc}[/bold red]", title="FAILED")
        )
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(
        Panel(
            "[bold green]Pipeline complete[/bold green]\n"
            # f"  Data Version   : {data_asset_version}\n"
            # f"  MLflow Run   : {mlflow_run_id}\n"
            # f"  Model Version: {model_version or 'not promoted'}",
            # title="✔ SUCCESS",
        )
    )


if __name__ == "__main__":
    main()
