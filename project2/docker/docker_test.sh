#!/bin/bash

gcloud auth login
# ==============================================================================
# STEP 1: Environment Setup
# ==============================================================================
export PROJECT_ID="project-372174a7-7b3a-4058-a00"
export REGION="us-central1"
export REPO_NAME="my-docker-repo"  # add：Artifact Registry as the docker repo name
export SERVICE_NAME="shopping-assistant-api"

gcloud config set project $PROJECT_ID

# Source image provided by Alex
export SOURCE_IMAGE="gcr.io/gleaming-cove-490722-p5/shopping-assistant-api:latest"

# Target image path 
export MY_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:latest"

echo "Deploying to Project: $PROJECT_ID"

# ==============================================================================
# STEP 2: Prepare Artifact Registry & Image Migration
# ==============================================================================

echo "Step 2.1: Enabling Artifact Registry API..."
gcloud services enable artifactregistry.googleapis.com

echo "Step 2.2: Creating Artifact Registry repository (if not exists)..."
# Check if the repo exist, or create new one
gcloud artifacts repositories describe $REPO_NAME --location=$REGION > /dev/null 2>&1 || \
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for Shopping Assistant"

echo "Step 2.3: Authorizing Docker for Artifact Registry..."
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet

echo "Step 2.4: Pulling external image..."
docker pull $SOURCE_IMAGE

echo "Step 2.5: Re-tagging image for Artifact Registry..."
docker tag $SOURCE_IMAGE $MY_IMAGE

echo "Step 2.6: Pushing image to your private registry..."
docker push $MY_IMAGE


# ==============================================================================
# STEP 3: Enable Cloud Run API
# ==============================================================================
echo "Step 3: Enabling Cloud Run API..."
gcloud services enable run.googleapis.com

# ==============================================================================
# STEP 4: Deploy to Cloud Run
# ==============================================================================
echo "Step 4: Deploying service..."
gcloud run deploy $SERVICE_NAME \
  --image $MY_IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars PROJECT_ID=$PROJECT_ID,REGION=$REGION,ENDPOINT_ID=560716295549485056 \
  --memory 512Mi \
  --min-instances 0 \
  --max-instances 1

# ==============================================================================
# STEP 5: Output Results
# ==============================================================================
echo "--------------------------------------------------------"
echo "✅ SUCCESS: Deployment Finished!"
echo "Your Public Service URL is:"
gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --format="value(status.url)"
echo "--------------------------------------------------------"