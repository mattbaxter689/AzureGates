import json

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import DefaultAzureCredential

with open("config.json") as f:
    cfg = json.load(f)

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=cfg["workspace"]["subscription_id"],
    resource_group_name=cfg["workspace"]["resource_group"],
    workspace_name=cfg["workspace"]["workspace_name"],
)

env_docker_context = Environment(
    build=BuildContext(path=".", dockerfile_path="infra/environment/Dockerfile"),
    name="gating-docker-context",
    description="Environment created from Docker context for azure gating project",
)

ml_client.environments.create_or_update(env_docker_context)
