import argparse
import json
import logging
import sys
from enum import Enum
from pathlib import Path

from azure.ai.ml import MLClient, Input, command, Output, dsl
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
    FIT = "fit"
    # PROMOTE = "promote"
    # DEPLOY = "deploy"


# More gates will go here, but will happen iteratively as things are built out
GATE_ORDER = [Gate.DATA, Gate.DRIFT, Gate.FIT]


# -------- Arg parsing -------
def parse_args() -> argparse.Namespace:

    # NOTE: again, we add logic for everything but comment and add iteratively
    p = argparse.ArgumentParser(description="AML Classification Pipeline Orchestrator")
    p.add_argument("--start-from", choices=[g.value for g in Gate], default="data")
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


def get_compute() -> str:
    return CONFIG["compute"]["training_cluster"]


# ------ Base Job Config ----------
def _base_job_kwargs(name: str, description: str) -> dict:

    return dict(
        display_name=name,
        description=description,
        environment=get_aml_environment(),
        compute=get_compute(),
        experiment_name=CONFIG["pipeline"]["experiment_name"],
        code="./src",
        environment_variables={"PYTHONPATH": "./"},
    )


# ------ Job Builders ---------
def data_versioning_component() -> Command:

    return command(
        **_base_job_kwargs(
            "data-versioning-gate", "Fetch, clean, split, and version data"
        ),
        command=(
            "python -m gates.data_versioning_gate "
            "--raw-data ${{inputs.raw_data}} "
            "--output-training-path ${{outputs.processed_data}}"
        ),
        inputs={
            "raw_data": Input(type="uri_file", path=CONFIG["data"]["name"]),
        },
        outputs={
            "processed_data": Output(
                type="uri_folder",
                mode="rw_mount",
                name=CONFIG["data"]["output_asset_name"],
            )
        },
    )


def drift_detection_component() -> Command:
    return command(
        **_base_job_kwargs(
            "drift-detection-gate", "Statistical drift check against baseline"
        ),
        command=(
            "python -m gates.drift_detection_gate "
            "--new-data-version ${{inputs.processed_data}} "
            "--gold-data-version ${{inputs.gold_data}} "
            "--drift-output-path ${{outputs.drift_output}}"
        ),
        inputs={
            "gold_data": Input(type="uri_file", path=CONFIG["data"]["name"]),
            "processed_data": Input(
                type="uri_folder",
            ),
        },
        outputs={"drift_output": Output(type="uri_file", mode="rw_mount")},
    )


def model_training_component() -> Command:
    return command(
        **_base_job_kwargs("model-training-gate", "Model fit gate to train new model"),
        command=(
            "python -m gates.model_training_gate "
            "--drift-detected ${{inputs.drift_detected}} "
            "--training-data ${{inputs.processed_data}}"
        ),
        inputs={
            "processed_data": Input(type="uri_folder"),
            "drift_detected": Input(type="uri_file"),
        },
    )


@dsl.pipeline(
    description="Gated classification pipeline",
    experiment_name=CONFIG["pipeline"]["experiment_name"],
    default_compute=get_compute(),
)
def build_pipeline(raw_data: Input, gold_data: Input):
    data_step = data_versioning_component()(
        raw_data=raw_data,
    )

    drift_step = drift_detection_component()(
        gold_data=gold_data,
        processed_data=data_step.outputs.processed_data,  # wired directly, AML mounts it
    )

    model_step = model_training_component()(
        processed_data=data_step.outputs.processed_data,
        drift_detected=drift_step.outputs.drift_output,
    )

    return {"drift_output": drift_step.outputs.drift_output}


def main() -> None:
    args = parse_args()
    ml_client = get_client()

    start_gate = Gate(args.start_from)
    active_gates = GATE_ORDER[GATE_ORDER.index(start_gate) :]

    table = Table(title="Pipeline Execution Plan", show_header=True)
    table.add_column("Gate", style="cyan")
    table.add_column("Status", style="white")
    for gate in GATE_ORDER:
        status = "ACTIVE" if gate in active_gates else "SKIPPED"
        style = "green" if status == "ACTIVE" else "dim"
        table.add_row(gate.value, f"[{style}]{status}[/{style}]")
    console.print(table)

    pipeline_job = build_pipeline(
        raw_data=Input(type="uri_file", path=CONFIG["data"]["name"]),
        gold_data=Input(type="uri_file", path=CONFIG["data"]["name"]),
    )

    try:
        returned_job = ml_client.jobs.create_or_update(pipeline_job)
        job_url = returned_job.studio_url
        console.print(f"\n[bold cyan]▶ Pipeline submitted[/bold cyan]")
        console.print(f"  Job ID : {returned_job.name}")
        console.print(f"  Studio : [link={job_url}]{job_url}[/link]")

        # Stream logs — blocks until pipeline completes
        ml_client.jobs.stream(returned_job.name)

        console.print(Panel("[bold green]Pipeline complete[/bold green]"))

    except RuntimeError as exc:
        console.print(
            Panel(f"[bold red]Pipeline halted:\n{exc}[/bold red]", title="FAILED")
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
