from pydantic import BaseModel


class WorkspaceConfig(BaseModel):
    subscription_id: str
    workspace_name: str
    resource_group: str


class EnvironmentConfig(BaseModel):
    name: str
    version: str = "latest"

    @property
    def full_name(self) -> str:
        return f"{self.name}@{self.verison}"


class ComputeConfig(BaseModel):
    compute_cluster: str


class PipelineConfig(BaseModel):
    experiment_name: str


class DataConfig(BaseModel):
    name: str
    output_asset_name: str


class OrchestratorConfig(BaseModel):
    workspace: WorkspaceConfig
    environment: EnvironmentConfig
    compute: ComputeConfig
    pipeline: PipelineConfig
    data: DataConfig
