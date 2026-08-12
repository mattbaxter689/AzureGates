import logging
import os
import sys

import mlflow

from utils.asset_utils import get_ml_client

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a root logger with clean formatting for AML job logs"""

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Suppress noisy Azure SDK transport logs
    for noisy in ("azure.core", "azure.identity", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def configure_mlflow() -> str:
    """
    Set up MLflow to log to the AML workspace.

    In an AML Command Job the MLFLOW_TRACKING_URI environment variable is
    automatically injected. Locally, DefaultAzureCredential is used.

    Returns the active experiment ID.
    """
    try:
        ml_client = get_ml_client()
        tracking_uri = ml_client.workspaces.get(
            os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
        ).mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("Tracking URI for MLFlow set")
    except Exception as e:
        logger.warning(f"Could not resolve tracking URI: {e}")
        raise
