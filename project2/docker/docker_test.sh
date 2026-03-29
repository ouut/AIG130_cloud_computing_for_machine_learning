#!/bin/bash

gcloud auth login
docker login
# ==============================================================================
# STEP 1: Environment Setup
# ==============================================================================
export PROJECT_ID="project-372174a7-7b3a-4058-a00"
export REGION="us-central1"
export SERVICE_NAME="shopping-assistant-api"

gcloud config set project $PROJECT_ID

# Source image provided by Alex
export SOURCE_IMAGE="gcr.io/gleaming-cove-490722-p5/shopping-assistant-api:latest"

# Target image path 
export MY_IMAGE="docker.io/chet2026/clouding:shopping-assistant-api"

echo "Deploying to Project: $PROJECT_ID"

# ==============================================================================
# STEP 2: Push Image to Docker Hub
# ==============================================================================
echo "Step 2: Pushing image to Docker Hub..."
docker tag $SOURCE_IMAGE $MY_IMAGE
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