"""
=============================================================
  AIG130 Project - Stage 3: Deploy Model + API to Cloud Run
=============================================================
  WHAT THIS DOES:
    Part A: Deploys your trained AutoML model to a Vertex AI Endpoint
            (makes the model callable via the internet)
    Part B: Deploys the FastAPI app (main.py) to Cloud Run
            (creates a public URL for your Shopping Assistant API)

  BEFORE RUNNING:
    1. Complete Stage 2 (model must be trained)
    2. Enable these APIs in GCP Console:
       → Cloud Run API
       → Container Registry API (or Artifact Registry API)
    3. Have your model_resource_name.txt from Stage 2

  HOW TO RUN:
    Run this script in Google Colab AFTER Stage 2 is complete.

  CHANGE THESE:
    YOUR_PROJECT_ID = "your-gcp-project-id"
=============================================================
"""

# ─── CHANGE THIS ────────────────────────────────────────────
YOUR_PROJECT_ID = "your-gcp-project-id"
YOUR_REGION     = "us-central1"
# ────────────────────────────────────────────────────────────

from google.cloud import aiplatform
from google.colab import auth
import subprocess
import os

auth.authenticate_user()
aiplatform.init(project=YOUR_PROJECT_ID, location=YOUR_REGION)

# ─────────────────────────────────────────────
# PART A: DEPLOY MODEL TO VERTEX AI ENDPOINT
# ─────────────────────────────────────────────
print("=" * 60)
print("  PART A: Deploying ML Model to Vertex AI Endpoint")
print("=" * 60)

# Read the model resource name saved by Stage 2
try:
    with open("model_resource_name.txt", "r") as f:
        model_resource_name = f.read().strip()
    print(f"  ✓ Found model: {model_resource_name}")
except FileNotFoundError:
    print("  ❌ ERROR: model_resource_name.txt not found!")
    print("     Make sure you ran Stage 2 first and the file was saved.")
    raise

# Load the model object
model = aiplatform.Model(model_resource_name)
print(f"  ✓ Model loaded: {model.display_name}")

# Create a Vertex AI Endpoint
print("\n  → Creating Vertex AI Endpoint (2-3 minutes)...")
endpoint = aiplatform.Endpoint.create(
    display_name="shopping_assistant_endpoint",
    description="AIG130 Project - Shopping Recommendation Endpoint"
)
print(f"  ✓ Endpoint created: {endpoint.resource_name}")

# Deploy model to endpoint
print("\n  → Deploying model to endpoint (5-10 minutes)...")
print("  → Using the smallest machine type to stay within free tier")

deployed_model = endpoint.deploy(
    model=model,
    deployed_model_display_name="shopping_model_deployed",
    machine_type="n1-standard-2",   # Smallest available - free tier friendly
    min_replica_count=1,
    max_replica_count=2,            # Auto-scales up to 2 if busy
    traffic_percentage=100,
)

# Get the endpoint ID (needed for main.py)
endpoint_id = endpoint.name
print(f"\n  ✅ Model deployed successfully!")
print(f"  📌 ENDPOINT ID: {endpoint_id}")
print(f"     (Save this! You need it in main.py → ENDPOINT_ID variable)")

# Save endpoint ID to file
with open("endpoint_id.txt", "w") as f:
    f.write(endpoint_id)
print(f"  ✓ Saved endpoint ID to: endpoint_id.txt")

# ─────────────────────────────────────────────
# PART B: DEPLOY FASTAPI TO CLOUD RUN
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PART B: Deploying API to Cloud Run")
print("=" * 60)

print("""
  IMPORTANT: Before running the next steps, you need to:

  1. Open Google Cloud Shell (click the terminal icon at top of GCP Console)
     → https://console.cloud.google.com

  2. Upload your 3 API files to Cloud Shell:
     - main.py
     - requirements.txt
     - Dockerfile
     (Click the 3-dot menu → Upload files)

  3. Run these commands IN Cloud Shell (copy-paste each line):
""")

SERVICE_NAME = "shopping-assistant-api"
IMAGE_NAME   = f"gcr.io/{YOUR_PROJECT_ID}/{SERVICE_NAME}"

cloud_shell_commands = f"""
# ─── RUN THESE IN GOOGLE CLOUD SHELL ───────────────────────

# Step 1: Set your project
gcloud config set project {YOUR_PROJECT_ID}

# Step 2: Enable required APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com

# Step 3: Build and push the Docker container
gcloud builds submit --tag {IMAGE_NAME}

# Step 4: Deploy to Cloud Run
gcloud run deploy {SERVICE_NAME} \\
  --image {IMAGE_NAME} \\
  --platform managed \\
  --region {YOUR_REGION} \\
  --allow-unauthenticated \\
  --set-env-vars PROJECT_ID={YOUR_PROJECT_ID},REGION={YOUR_REGION},ENDPOINT_ID={endpoint_id} \\
  --memory 512Mi \\
  --min-instances 0 \\
  --max-instances 3

# Step 5: Get your public URL
gcloud run services describe {SERVICE_NAME} \\
  --region {YOUR_REGION} \\
  --format="value(status.url)"

# ────────────────────────────────────────────────────────────
# After running, you will get a URL like:
# https://shopping-assistant-api-xxxxxxx-uc.a.run.app
# That is your live Shopping Assistant API!
"""

print(cloud_shell_commands)

# Save commands to a file for easy reference
with open("cloud_shell_commands.sh", "w") as f:
    f.write(cloud_shell_commands)
print("  ✓ Commands saved to: cloud_shell_commands.sh")

print("=" * 60)
print("  ✅ STAGE 3 SETUP COMPLETE!")
print("=" * 60)
print(f"""
  Summary:
  ✓ Vertex AI Endpoint ID: {endpoint_id}
  ✓ Cloud Run commands ready in: cloud_shell_commands.sh

  After running Cloud Shell commands, your API will be live at:
  https://{SERVICE_NAME}-[hash]-uc.a.run.app

  Test it by going to:
  https://{SERVICE_NAME}-[hash]-uc.a.run.app/docs

  Next step: Run stage4_monitoring.py to set up monitoring.
""")
