import logging
import json
from pathlib import Path
from mlflow.client import MlflowClient
import os

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
