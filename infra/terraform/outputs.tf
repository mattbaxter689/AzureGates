output "subscription_id" {
  value       = data.azurerm_client_config.current.subscription_id
  description = "Azure subscription ID"
}

output "resource_group_name" {
  value = azurerm_resource_group.ml.name
}

output "workspace_name" {
  value = azurerm_machine_learning_workspace.aml.name
}

# Storage outputs
output "storage_account_name" {
  value = azurerm_storage_account.blob.name
}

# ACR login server
output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}

# Managed Identity
output "managed_identity_client_id" {
  value = azurerm_user_assigned_identity.aml_identity.client_id
}

# Key Vault and App Insights IDs (optional, useful for Python SDK)
output "key_vault_id" {
  value = azurerm_key_vault.kv.id
}

output "application_insights_id" {
  value = azurerm_application_insights.ai.id
}
