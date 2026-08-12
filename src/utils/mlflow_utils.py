import json
import logging
import os
from pathlib import Path

import mlflow
from mlflow.client import MlflowClient
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)


def get_baseline_stats() -> dict[str, float]:
    client = MlflowClient()

    run_id = os.environ["AZUREML_RUN_ID"]
    experiment_id = client.get_run(run_id).info.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="""
            attributes.status = 'FINISHED'
            and tags.gate = 'data-versioning'
            and tags.gate_status = 'PASSED'
        """,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No finished data-versioning-gate run found")

    logger.info("Run found")
    run_id = runs[0].info.run_id
    artifact_path = client.download_artifacts(run_id, "baseline/baseline.json")
    logger.info("Baseline artifacts downloaded")
    return json.loads(Path(artifact_path).read_text())


def tag_challenger(
    client: MlflowClient,
    run_id: str,
    model_name: str = "sleep_disorder_classifier",
    baseline_f1: float = 0.70,
) -> str | None:
    """
    Register the best fvinal model as a new model version and tag it as current challenger
    candidate, if it clears a baseline.
    """
    logger.info(f"Validating challenger candidate for run: {run_id}")
    run_info = client.get_run(run_id)
    val_f1 = run_info.data.metrics.get("best_val_f1")

    if val_f1 < baseline_f1:
        logger.info(
            f"Run {run_id} failed baseline gate: f1={val_f1:.4f} < {baseline_f1}"
        )
        return None

    model_uri = f"runs:/{run_id}/model"
    logger.info("Registering candidate model version")
    model_version_details = mlflow.register_model(model_uri, model_name)
    challenger_version = model_version_details.version

    client.transition_model_version_stage(
        name=model_name, version=challenger_version, stage="Staging"
    )
    client.set_model_version_tag(
        name=model_name,
        version=challenger_version,
        key="deployment_role",
        value="challenger",
    )

    return challenger_version


def resolve_version_by_role(
    client: MlflowClient, model_name: str, role: str
) -> ModelVersion | None:
    """
    Return the ModelVersion currently tagged deployment_role=<role> for
    the given model, or None if no version is available
    """
    versions = client.search_model_versions(
        f"name='{model_name}' and tags.deployment_role='{role}'"
    )

    if not versions:
        return None

    if len(versions) > 1:
        logger.warning(
            f"Multiple versions of '{model_name}' tagged deployment_role={role}: "
            f"{[v.version for v in versions]}. Using the highest version number. "
            f"This usually means a previous promotion/retire step didn't fully "
            f"clear the old tag -- worth checking the run logs from that step."
        )

    return max(versions, key=lambda v: int(v.version))
