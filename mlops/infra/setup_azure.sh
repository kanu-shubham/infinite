#!/usr/bin/env bash
# =============================================================================
# setup_azure.sh
# =============================================================================
# One-time script to provision every Azure resource the project needs.
# Run this ONCE from your local machine before any CI/CD runs.
#
# Prerequisites:
#   az login                          (Azure CLI authenticated)
#   az account set --subscription ID  (correct subscription selected)
#
# Usage:
#   chmod +x infra/setup_azure.sh
#   ./infra/setup_azure.sh
#
# After running, copy the printed values into GitHub repository secrets.
# =============================================================================
set -euo pipefail

# ── Configurable names ────────────────────────────────────────────────────────
RESOURCE_GROUP="mlops-hotel-rg"
LOCATION="eastus"
ACR_NAME="hotelpriceacr"                  # must be globally unique, lowercase
STORAGE_ACCOUNT="hotelmlopsstore"          # must be globally unique, lowercase
BLOB_CONTAINER="mlflow-artifacts"
CONTAINERAPPS_ENV="hotel-price-env"
CONTAINER_APP_NAME="hotel-price-api"
# =============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Provisioning Azure resources for Hotel Price MLOps"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── 1. Resource Group ─────────────────────────────────────────────────────────
echo "▶ Creating resource group: $RESOURCE_GROUP in $LOCATION"
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none
echo "  ✓ Resource group ready"

# ── 2. Azure Container Registry ───────────────────────────────────────────────
# ACR stores your Docker images — like DockerHub but private inside Azure.
echo "▶ Creating Container Registry: $ACR_NAME"
az acr create \
  --name "$ACR_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --sku Basic \
  --admin-enabled true \
  --output none
echo "  ✓ ACR ready"

# ── 3. Storage Account + Blob Container for MLflow ────────────────────────────
# MLflow saves model files and run artifacts here instead of your local disk.
# In production you never want experiment data sitting on a laptop.
echo "▶ Creating Storage Account: $STORAGE_ACCOUNT"
az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --output none

STORAGE_KEY=$(az storage account keys list \
  --account-name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[0].value" -o tsv)

az storage container create \
  --name "$BLOB_CONTAINER" \
  --account-name "$STORAGE_ACCOUNT" \
  --account-key "$STORAGE_KEY" \
  --output none
echo "  ✓ Blob storage ready (container: $BLOB_CONTAINER)"

# ── 4. Container Apps Environment ─────────────────────────────────────────────
# Container Apps is Azure's managed serverless container platform.
# You push a Docker image; Azure handles scaling, TLS, and load balancing.
echo "▶ Creating Container Apps Environment: $CONTAINERAPPS_ENV"
az containerapp env create \
  --name "$CONTAINERAPPS_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none
echo "  ✓ Container Apps environment ready"

# ── 5. Placeholder Container App (updated by CI/CD on every deploy) ───────────
echo "▶ Creating initial Container App: $CONTAINER_APP_NAME"
ACR_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)

az containerapp create \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$CONTAINERAPPS_ENV" \
  --image "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest" \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 0.5 \
  --memory 1.0Gi \
  --output none
echo "  ✓ Container App created (CI/CD will update the image on every push)"

# ── 6. Collect secrets for GitHub ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Add the following as GitHub Repository Secrets"
echo "  (Settings → Secrets → Actions → New repository secret)"
echo "═══════════════════════════════════════════════════════════"
echo ""

ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)
APP_URL=$(az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)

echo "  AZURE_RESOURCE_GROUP   = $RESOURCE_GROUP"
echo "  ACR_LOGIN_SERVER       = $ACR_SERVER"
echo "  ACR_USERNAME           = $ACR_USERNAME"
echo "  ACR_PASSWORD           = $ACR_PASSWORD"
echo "  AZURE_STORAGE_ACCOUNT  = $STORAGE_ACCOUNT"
echo "  AZURE_STORAGE_KEY      = $STORAGE_KEY"
echo "  AZURE_BLOB_CONTAINER   = $BLOB_CONTAINER"
echo "  CONTAINER_APP_NAME     = $CONTAINER_APP_NAME"
echo ""
echo "  Your API will be live at: https://$APP_URL"
echo ""
echo "  You also need:"
echo "  AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID,"
echo "  AZURE_SUBSCRIPTION_ID"
echo "  → Run: az ad sp create-for-rbac --name mlops-hotel-sp --role contributor"
echo "         --scopes /subscriptions/<your-subscription-id>"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Done. All resources created in resource group: $RESOURCE_GROUP"
echo "═══════════════════════════════════════════════════════════"
