import logging
import json
from pathlib import Path
from mlflow.client import MlflowClient
import os
import mlflow

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


def validate_and_promote_challenger(
    run_id: str, model_name: str = "sleep_disorder_classifier"
) -> str:
    """
    Perform quick validation of potential challenger model, and promote to challenger
    """

    client = MlflowClient()

    logger.info(f"Validation challenger promotion for run: {run_id}")

    run_info = client.get_run(run_id)
    run_metrics = run_info.data.metrics

    if run_metrics.get("best_val_f1", 0) < 0.70:
        raise ValueError(
            f"Validation F1 {run_metrics.get('best_val_f1', 0)} failed baseline gate"
        )

    logger.info("Archiving previous challenger models")
    archive_previous_challenger(client, model_name)

    model_uri = f"runs:/{run_id}/model"
    logger.info("Registering version run from registry path")

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


def archive_previous_challenger(client: MlflowClient, model_name: str) -> None:
    """
    Archive any old versions of staging models, as they are
    no longer needed
    """
    existing_versions = client.search_model_versions(f"name='{model_name}'")

    for mv in existing_versions:
        if mv.current_stage == "Staging":
            logger.info(f"Archiving old challenger: Version {mv.version}")

            # Transition it to Archived
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Archived"
            )

            # Update its tag
            client.set_model_version_tag(
                name=model_name,
                version=mv.version,
                key="deployment_role",
                value="archived_challenger",
            )
