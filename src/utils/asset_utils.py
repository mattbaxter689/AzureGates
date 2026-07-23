import logging
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
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
