import logging
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import pandas as pd
import tempfile
from pathlib import Path
import os

logger = logging.getLogger(__name__)


def get_ml_client() -> MLClient:
    try:
        return MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
            resource_group_name=os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
            workspace_name=os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
        )
    except Exception as e:
        logger.error(
            "Error connecting to MlClient. Perhaps configure environment variables"
        )
        raise e


def get_next_version(ml_client: MLClient, name: str) -> str:
    """Return the next available version number for a data asset."""
    try:
        versions = ml_client.data.list(name=name)
        existing = [int(v.version) for v in versions if v.version.isdigit()]
        return str(max(existing) + 1) if existing else "1"
    except Exception:
        return "1"


def register_dataframes_as_asset(
    ml_client: MLClient,
    dataframes: dict[str, pd.DataFrame],
    name: str,
    description: str = "",
) -> Data:
    """
    Archive all previous versions, then upload dataframes as CSVs
    and register them as version 1 of a data asset.
    """
    version = get_next_version(ml_client, name)
    logger.info(f"Version: {version}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for file_name, df in dataframes.items():
            path = Path(tmp_dir) / file_name
            df.to_csv(path, index=False)
            logger.info(f"Saved {file_name} with {len(df)} rows")

        asset = Data(
            path=tmp_dir,
            type=AssetTypes.URI_FOLDER,
            description=description,
            name=name,
            version=version,
        )

        registered = ml_client.data.create_or_update(asset)
        logger.info(
            f"Registered data asset {name} version {registered.version} at {registered.path}"
        )
    return registered
