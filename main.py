import argparse
import json
import logging
import sys
import time
from enum import Enum
from pathlib import Path
import tempfile
import os

from azure.ai.ml import MLClient, Input, command
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
    # DRIFT = "drift"
    # FIT = "fit"
    # PROMOTE = "promote"
    # DEPLOY = "deploy"


# More gates will go here, but will happen iteratively as things are built out
GATE_ORDER = [Gate.DATA]


# -------- Arg parsing -------
def parse_args() -> argparse.Namespace:

    # NOTE: again, we add logic for everything but comment and add iteratively
    p = argparse.ArgumentParser(description="AML Classification Pipeline Orchestrator")
    p.add_argument("--start-from", choices=[g.value for g in Gate], default="data")
    p.add_argument("--data-uuid", default=None)
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
    return f"{cfg_env['name']}:{cfg_env['version']}"


# ------ Base Job Config ----------
def _base_job_kwargs(name: str, description: str) -> dict:
    compute = CONFIG["compute"]["training_cluster"]

    return dict(
        display_name=name,
        description=description,
        environment=get_aml_environment(),
        compute=compute,
        experiment_name=CONFIG["pipeline"]["experiment_name"],
        # code=absolute_path,
        environment_variables={"PYTHONPATH": "./"},
    )


# ------ Job Builders ---------
def build_data_job(ml_client: MLClient) -> Command:
    # Fetch asset explicitly to avoid resolution errors
    data_asset = ml_client.data.get(
        name=CONFIG["data"]["name"], label=CONFIG["data"]["version"]
    )
    print(data_asset)

    return command(
        **_base_job_kwargs(
            "data-versioning-gate", "Fetch, clean, split, and version data"
        ),
        command=(
            "python -m gates.data_versioning_gate "
            "--raw-data ${{inputs.raw_data}} "
            "--output-base-uri ${{inputs.output_base_uri}} "
            "--output-uuid-file ${{outputs.uuid_file}}"
        ),
        inputs={
            "raw_data": Input(type="uri_file", path=data_asset.id),  # Pass asset ID
            "output_base_uri": CONFIG["data"]["output_base_uri"],
        },
        outputs={"uuid_file": {"type": "uri_file", "mode": "rw_mount"}},
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
    return {"status": status, "outputs": outputs}


def read_output_string(ml_client: MLClient, job_outputs: dict, key: str) -> str | None:
    """Download a uri_file output and return its text content."""
    try:
        output = job_outputs.get(key)
        if output and hasattr(output, "path"):
            # Download via SDK
            with tempfile.TemporaryDirectory() as tmp:
                local = os.path.join(tmp, "output.txt")
                ml_client.jobs.download(
                    name=output.name, output_name=key, download_path=tmp
                )
                return Path(local).read_text().strip()
    except Exception as exc:
        log.warning("Could not read output '%s': %s", key, exc)
    return None


def main() -> None:
    args = parse_args()
    ml_client = get_client()
    # data = ml_client.data.get(name="sleep_data", label="latest")
    # print(data)
    # print(data.path)
    # print(data.type)
    # State accumulated across gates
    data_uuid: str | None = args.data_uuid
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
        if Gate.DATA in active_gates and data_uuid is None:
            job = build_data_job(ml_client)
            result = submit_and_wait(ml_client, job, "Data Versioning", args.dry_run)
            if not args.dry_run:
                # The UUID is written to the output file by the gate script
                data_uuid = read_output_string(
                    ml_client, result["outputs"], "uuid_file"
                )
                if not data_uuid:
                    raise RuntimeError("Data versioning gate did not produce a UUID.")
            else:
                data_uuid = "dry-run-uuid"
            console.print(f"  [bold]Data UUID:[/bold] {data_uuid}")

        elif data_uuid:
            console.print(
                f"[dim]Skipping data gate — using provided UUID: {data_uuid}[/dim]"
            )

        assert data_uuid, "data_uuid must be set before drift gate"

    except RuntimeError as exc:
        console.print(
            Panel(f"[bold red]Pipeline halted:\n{exc}[/bold red]", title="FAILED")
        )
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(
        Panel(
            f"[bold green]Pipeline complete[/bold green]\n"
            f"  Data UUID    : {data_uuid}\n"
            f"  MLflow Run   : {mlflow_run_id}\n"
            f"  Model Version: {model_version or 'not promoted'}",
            title="✔ SUCCESS",
        )
    )


if __name__ == "__main__":
    main()
