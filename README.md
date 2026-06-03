# Azure Gates

A production-like, gate based MLOps pipeline built on Azure ML SDK v2.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 main.py (Orchestrator)                                 │
│                                                                                        │
│  [Data Version Gate] ──Asset Version──► [Drift Gate] ──Drift Result──► [Model Fit Gate]│
│                                                                              │         │
│                                                                            Run ID      │
│                                                                              │         │
│                                                                      [Analysis & Promo]│
│                                                                              │         │
│                                                                     [Shadow Deployment]│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

## Quickstart

### 1. Provision Azure ML Resources

Provision the needed Azure ML resources via terraform and get your `config.json` from your workspace_name

```bash
terraform plan
terraform apply -auto-approve
```

Ensure to create a `terraform.tfvars` file to list the secrets for the project

### 2. Configure Workspace

```bash
cp config.json.example config.json
# Fill in your subscription_id, resource_group, workspace_name, and other azure details
```

### 3. Create Azure ML Execution Environment

We need an environment on Azure ML to execute our jobs. Our build environment is created based on a defined Docker image, as it allows us to install and us `uv` to build the environment image.

```bash
uv run infra/environment/create_environment.py 
```

This can take quite some time to run, so allow ample time to complete. To install the packages
in a local `uv` project, run

```bash
uv sync
```

To install the proper packages and versions

### 4. Submit an Azure ML DSL Pipeline Job

```bash
uv run main.py
```

This configures an Azure ML DSL pipeline build to execute the gates

## Data Asset Governance

For this project, we make use of a static csv file as our "raw" data. In reality, this data might come from an S3 bucket of Blob Storage, SQL tables, etc.
This raw data is stored in a `URI File` data asset. This is something that lives in your blob storage
associated with your Azure ML project workspace. With Azure and it's DSL pipelines, we can pass these
data assets to jobs as an `Input`. In this case, Azure automatically injects the asset into the job
runtime, and we can fetch the path to access these jobs (see `main.py` for how to pass `Input`'s,
and any of the gates for how to access the data).

For best practices and ensuring clear data versions between runs, we can register a data asset containing
our data after performing a train-test split. In a DSL pipeline, we define an `Output` that allows us
to create and register data assets from job runs simply by specifying the `name` field (again, see `main.py` for an example of how to use `Output`'s, and any gate script on how to send an `Output`). With DSL pipelines, we are able to directly any `Output` from a previous gate as an `Input` to another gate. This
is quite powerful and allows us to pass and share information between pipeline jobs easily and cleanly.

For this project, we mainly use `URI Files` and `URI Folders` for data assets, but other asset types exist
like model assets, compute assets, etc. The list of assets and their types can be found [in the official documentation](https://learn.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-v2?view=azureml-api-2&tabs=sdk#data)
