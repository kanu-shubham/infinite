#!/usr/bin/env bash
# =============================================================================
# setup_azure.sh  (Recommendation System)
# =============================================================================
# Reuses the resource group and Container Apps environment from the price
# predictor project, and adds a new Container App for the recsys API.
# Run ONCE before any CI/CD.
#
# Prerequisites: az login, correct subscription selected
# Usage: chmod +x infra/setup_azure.sh && ./infra/setup_azure.sh
# =============================================================================
set -euo pipefail

RESOURCE_GROUP="mlops-hotel-rg"
LOCATION="eastus"
ACR_NAME="hotelrecsysacr"
STORAGE_ACCOUNT="hotelmlopsstore"    # same storage account, different container
BLOB_CONTAINER="recsys-artifacts"
CONTAINERAPPS_ENV="hotel-price-env"  # reuse existing environment
CONTAINER_APP_NAME="hotel-recsys-api"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Provisioning Azure resources for Hotel RecSys"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "▶ Creating Container Registry: $ACR_NAME"
az acr create \
  --name "$ACR_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --sku Basic \
  --admin-enabled true \
  --output none
echo "  ✓ ACR ready"

echo "▶ Creating Blob container for RecSys artifacts"
STORAGE_KEY=$(az storage account keys list \
  --account-name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[0].value" -o tsv)

az storage container create \
  --name "$BLOB_CONTAINER" \
  --account-name "$STORAGE_ACCOUNT" \
  --account-key "$STORAGE_KEY" \
  --output none
echo "  ✓ Blob container ready"

echo "▶ Creating Container App: $CONTAINER_APP_NAME"
ACR_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)

az containerapp create \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$CONTAINERAPPS_ENV" \
  --image "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest" \
  --target-port 8001 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --output none
echo "  ✓ Container App created"

ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)
APP_URL=$(az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Add these as GitHub Repository Secrets"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  RECSYS_ACR_LOGIN_SERVER  = $ACR_SERVER"
echo "  RECSYS_ACR_USERNAME      = $ACR_USERNAME"
echo "  RECSYS_ACR_PASSWORD      = $ACR_PASSWORD"
echo "  RECSYS_CONTAINER_APP     = $CONTAINER_APP_NAME"
echo "  AZURE_RESOURCE_GROUP     = $RESOURCE_GROUP"
echo ""
echo "  API live at: https://$APP_URL"
echo "═══════════════════════════════════════════════════════════"
