terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
  }
}

data "azurerm_client_config" "current" {}

# -------------------------------
# Resource Group
# -------------------------------
resource "azurerm_resource_group" "ml" {
  name     = var.resource_group_name
  location = var.location
}

# -------------------------------
# Azure Container Registry
# -------------------------------
resource "azurerm_container_registry" "acr" {
  name                          = var.acr_name
  resource_group_name           = azurerm_resource_group.ml.name
  location                      = azurerm_resource_group.ml.location
  sku                           = "Standard"
  admin_enabled                 = true # Needed for AML environment builds
  public_network_access_enabled = true
}

# -------------------------------
# Storage Account + Container
# -------------------------------
resource "azurerm_storage_account" "blob" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.ml.name
  location                 = azurerm_resource_group.ml.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_container" "ml_data" {
  name                  = "ml-data"
  storage_account_name  = azurerm_storage_account.blob.name
  container_access_type = "private"
}

# -------------------------------
# Key Vault
# -------------------------------
resource "azurerm_key_vault" "kv" {
  name                          = "${var.workspace_name}-kv"
  location                      = azurerm_resource_group.ml.location
  resource_group_name           = azurerm_resource_group.ml.name
  tenant_id                     = data.azurerm_client_config.current.tenant_id
  sku_name                      = "standard"
  purge_protection_enabled      = false
  public_network_access_enabled = true
}

# -------------------------------
# Application Insights
# -------------------------------
resource "azurerm_application_insights" "ai" {
  name                = "${var.workspace_name}-ai"
  location            = azurerm_resource_group.ml.location
  resource_group_name = azurerm_resource_group.ml.name
  application_type    = "web"

  workspace_id = null # Helps prevent those pesky managed resource group locking issues

  lifecycle {
    ignore_changes = [workspace_id]
  }
}

# -------------------------------
# Azure ML Workspace
# -------------------------------
resource "azurerm_machine_learning_workspace" "aml" {
  name                          = var.workspace_name
  location                      = azurerm_resource_group.ml.location
  resource_group_name           = azurerm_resource_group.ml.name
  public_network_access_enabled = true

  identity {
    type = "SystemAssigned"
  }

  container_registry_id   = azurerm_container_registry.acr.id
  storage_account_id      = azurerm_storage_account.blob.id
  key_vault_id            = azurerm_key_vault.kv.id
  application_insights_id = azurerm_application_insights.ai.id

  depends_on = [
    azurerm_container_registry.acr,
    azurerm_storage_account.blob,
    azurerm_key_vault.kv,
    azurerm_application_insights.ai,
  ]
}

# -------------------------------
# Compute Cluster
# -------------------------------

# CPU
resource "azurerm_machine_learning_compute_cluster" "cpu" {
  name                          = var.cpu_cluster_name
  location                      = azurerm_resource_group.ml.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.aml.id
  vm_size                       = "Standard_E4s_v3"
  vm_priority                   = "Dedicated"

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 2
    scale_down_nodes_after_idle_duration = "PT10M"
  }

  identity {
    type = "SystemAssigned"
  }

  depends_on = [azurerm_machine_learning_workspace.aml]
}

# GPU
resource "azurerm_machine_learning_compute_cluster" "gpu" {
  name                          = var.gpu_cluster_name
  location                      = azurerm_resource_group.ml.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.aml.id
  vm_size                       = "Standard_NC4as_T4_v3"
  vm_priority                   = "Dedicated"

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 1
    scale_down_nodes_after_idle_duration = "PT10M"
  }

  identity {
    type = "SystemAssigned"
  }

  depends_on = [azurerm_machine_learning_workspace.aml]
}


# -------------------------------
# Explicit Role Assignments
# (Only those NOT managed automatically by the Workspace)
# -------------------------------

# Compute — Storage Blob Data Reader (reading data assets at job runtime)
resource "azurerm_role_assignment" "compute_storage_access" {
  principal_id         = azurerm_machine_learning_compute_cluster.cpu.identity[0].principal_id
  role_definition_name = "Storage Blob Data Contributor"
  scope                = azurerm_storage_account.blob.id

  depends_on = [azurerm_machine_learning_compute_cluster.cpu]
}

resource "azurerm_role_assignment" "gpu_compute_storage_access" {
  principal_id         = azurerm_machine_learning_compute_cluster.gpu.identity[0].principal_id
  role_definition_name = "Storage Blob Data Contributor"
  scope                = azurerm_storage_account.blob.id

  depends_on = [azurerm_machine_learning_compute_cluster.gpu]
}

resource "azurerm_role_assignment" "gpu_endpoint_operator_access" {
  principal_id         = azurerm_machine_learning_compute_cluster.gpu.identity[0].principal_id
  role_definition_name = "AzureML Compute Operator"
  scope                = azurerm_machine_learning_workspace.aml.id

  depends_on = [azurerm_machine_learning_compute_cluster.gpu]
}

# Workspace — AzureML Data Scientist
resource "azurerm_role_assignment" "workspace_aml_data_scientist" {
  principal_id         = azurerm_machine_learning_workspace.aml.identity[0].principal_id
  role_definition_name = "AzureML Data Scientist"
  scope                = azurerm_machine_learning_workspace.aml.id

  depends_on = [azurerm_machine_learning_workspace.aml]
}
