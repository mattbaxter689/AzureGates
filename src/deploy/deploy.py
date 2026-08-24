import argparse
import logging
from typing import TypedDict

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    DataCollector,
    DeploymentCollection,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)
from azure.core.exceptions import ResourceNotFoundError
from mlflow.tracking import MlflowClient

from utils.mlflow_utils import resolve_version_by_role

logger = logging.getLogger(__name__)

CHAMPION_DEPLOYMENT = "champion"
CHALLENGER_DEPLOYMENT = "challenger"


class Decision(TypedDict):
    decision: str  # "promote_first" | "shadow" | "reject"
    reason: str
    model_name: str
    challenger_version: str | None
    champion_version: str | None
    challenger_f1: float | None
    champion_f1: float | None
    challenger_latency_ms: float | None


def ensure_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    try:
        return ml_client.online_endpoints.get(endpoint_name)
    except ResourceNotFoundError:
        endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
        return ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def build_data_collector() -> DataCollector:
    """
    Build a data collector to collect API inputs and outputs. Will be used to audit the model performance
    over time and assess model performance. Can be used to build a model rollback system
    """
    return DataCollector(
        collections={
            "request": DeploymentCollection(enabled="True"),
            "response": DeploymentCollection(enabled="True"),
        }
    )


def ensure_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
    version: str,
    instance_type: str,
    environment_name: str,
) -> None:
    """
    Create or update a managed endpoint
    """
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=f"azureml:{model_name}:{version}",
        instance_type=instance_type,
        instance_count=1,
        environment=environment_name,
        data_collector=build_data_collector(),
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()


def retag(
    mlflow_client: MlflowClient,
    model_name: str,
    version: str,
    role: str,
    stage: str,
) -> None:
    mlflow_client.set_model_version_tag(model_name, version, "deployment_role", role)
    mlflow_client.transition_model_version_stage(
        name=model_name, version=version, stage=stage
    )


def handle_promote_first(
    ml_client: MLClient,
    mlflow_client: MlflowClient,
    decision: Decision,
    args: argparse.Namespace,
) -> None:
    """
    Safety net: If we do not have a current champion model live, create an endpoint of the
    current champion model to ensure we have a model serving endpoint
    """
    model_name = decision["model_name"]
    version = decision["challenger_version"]
    assert (
        version is not None
    ), "promote_first decision must include a challenger_version"

    logger.info(
        f"No champion exists — promoting challenger v{version} directly to champion."
    )
    retag(mlflow_client, model_name, version, "champion", "Production")

    ensure_endpoint(ml_client, args.endpoint_name)
    ensure_deployment(
        ml_client,
        args.endpoint_name,
        CHAMPION_DEPLOYMENT,
        model_name,
        version,
        args.instance_type,
    )

    endpoint = ml_client.online_endpoints.get(args.endpoint_name)
    endpoint.traffic = {CHAMPION_DEPLOYMENT: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info(f"Champion deployment live at 100% traffic (model v{version}).")


def handle_shadow(
    ml_client: MLClient,
    decision: Decision,
    args: argparse.Namespace,
) -> None:
    """
    If the challenger model meets criteria for shadow deployment, deploy model
    to mirror all traffic coming to the champion endpoint. This will allow direct
    model performance comparison to determine challenger promotion
    """
    model_name = decision["model_name"]
    challenger_version = decision["challenger_version"]
    champion_version = decision["champion_version"]
    assert (
        challenger_version is not None and champion_version is not None
    ), "shadow decision must include both challenger_version and champion_version"

    logger.info(
        f"Challenger v{challenger_version} beat champion v{champion_version} — "
        f"standing up shadow deployment."
    )
    ensure_endpoint(ml_client, args.endpoint_name)
    # Champion should already be live; ensure_deployment is a no-op if it's unchanged.
    ensure_deployment(
        ml_client,
        args.endpoint_name,
        CHAMPION_DEPLOYMENT,
        model_name,
        champion_version,
        args.instance_type,
    )
    ensure_deployment(
        ml_client,
        args.endpoint_name,
        CHALLENGER_DEPLOYMENT,
        model_name,
        challenger_version,
        args.instance_type,
    )

    endpoint = ml_client.online_endpoints.get(args.endpoint_name)
    endpoint.traffic = {CHAMPION_DEPLOYMENT: 100, CHALLENGER_DEPLOYMENT: 0}
    endpoint.mirror_traffic = {CHALLENGER_DEPLOYMENT: args.mirror_pct}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info(
        f"Challenger v{challenger_version} shadow-deployed, mirroring {args.mirror_pct}% of traffic."
    )


def handle_reject(
    ml_client: MLClient,
    mlflow_client: MlflowClient,
    decision: Decision,
) -> None:
    """
    If a challenger fails the required checks to be used in a shadow deployment
    then we gracefully archive the model to not bloat
    """
    model_name = decision["model_name"]
    version = decision["challenger_version"]

    if version is None:
        # Failed the baseline gate before a version was even registered — nothing to archive.
        logger.info(f"Challenger rejected before registration: {decision['reason']}")
        return

    logger.info(f"Challenger v{version} rejected: {decision['reason']}")
    retag(mlflow_client, model_name, version, "archived_challenger", "Archived")
    ml_client.models.archive(name=model_name, version=version)


def ensure_champion_live(
    ml_client: MLClient,
    mlflow_client: MlflowClient,
    model_name: str,
    endpoint_name: str,
    instance_type: str,
    environment_name: str,
) -> None:
    """
    Safety net: if a champion is tagged in the registry but isn't currently
    serving on the endpoint (endpoint deleted, deployment lost, or this is
    the first deploy run since champion tags were set some other way),
    stand it back up. Runs regardless of the challenger's decision, so a
    rejected or shadowed challenger never leaves the endpoint without
    champion coverage.
    """
    champion_mv = resolve_version_by_role(mlflow_client, model_name, "champion")
    if champion_mv is None:
        return

    try:
        ml_client.online_endpoints.get(endpoint_name)
        try:
            ml_client.online_deployments.get(
                name=CHAMPION_DEPLOYMENT, endpoint_name=endpoint_name
            )
            champion_live = True
        except ResourceNotFoundError:
            champion_live = False
    except ResourceNotFoundError:
        champion_live = False

    if champion_live:
        return

    logger.warning(
        f"Champion v{champion_mv.version} is tagged in the registry but not live "
        f"on endpoint '{endpoint_name}'. Deploying it now."
    )
    ensure_endpoint(ml_client, endpoint_name)
    ensure_deployment(
        ml_client,
        endpoint_name,
        CHAMPION_DEPLOYMENT,
        model_name,
        champion_mv.version,
        instance_type,
        environment_name,
    )

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    if endpoint.traffic.get(CHAMPION_DEPLOYMENT, 0) == 0:
        endpoint.traffic = {**(endpoint.traffic or {}), CHAMPION_DEPLOYMENT: 100}
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
