import argparse
import logging
from typing import TypedDict

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint
from azure.core.exceptions import ResourceNotFoundError
from mlflow.tracking import MlflowClient


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Execute the promotion / shadow-deployment decision"
    )
    p.add_argument(
        "--decision-input",
        type=str,
        required=True,
        help="Folder containing decision.json from the gate step",
    )
    p.add_argument("--endpoint-name", type=str, required=True)
    p.add_argument("--instance-type", type=str, default="Standard_DS3_v2")
    p.add_argument("--mirror-pct", type=int, default=10)
    return p.parse_args()


def ensure_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    try:
        return ml_client.online_endpoints.get(endpoint_name)
    except ResourceNotFoundError:
        endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
        return ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def ensure_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
    version: str,
    instance_type: str,
) -> None:
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=f"azureml:{model_name}:{version}",
        instance_type=instance_type,
        instance_count=1,
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
    mlflow_client: MlflowClient,
    decision: Decision,
    args: argparse.Namespace,
) -> None:
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
    model_name = decision["model_name"]
    version = decision["challenger_version"]

    if version is None:
        # Failed the baseline gate before a version was even registered — nothing to archive.
        logger.info(f"Challenger rejected before registration: {decision['reason']}")
        return

    logger.info(f"Challenger v{version} rejected: {decision['reason']}")
    retag(mlflow_client, model_name, version, "archived_challenger", "Archived")
    ml_client.models.archive(name=model_name, version=version)
