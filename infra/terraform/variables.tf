variable "location" {
  description = "Azure region"
  type        = string
  default     = "canadacentral"
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
}

variable "workspace_name" {
  description = "Azure ML workspace name"
  type        = string
}

variable "acr_name" {
  description = "Azure Container Registry name (globally unique)"
  type        = string
}

variable "storage_account_name" {
  description = "Storage account name (globally unique)"
  type        = string
}
