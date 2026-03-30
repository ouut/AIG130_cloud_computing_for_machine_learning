"""
=============================================================
  AIG130 Project - Stage 2: Train ML Model with AutoML Tables
=============================================================
  WHAT THIS DOES:
    - Takes your BigQuery data and trains an ML model automatically
    - AutoML figures out the best algorithm FOR YOU (no ML expertise needed!)
    - Creates a trained model you can use to make predictions

  BEFORE RUNNING:
    1. Complete Stage 1 first (data must be in BigQuery)
    2. Enable Vertex AI API:
       → Go to: https://console.cloud.google.com/apis/library
       → Search "Vertex AI API" → Click Enable
    3. Enable Cloud Storage API (same steps, search "Cloud Storage")

  COST: Training uses ~1 compute hour = approx $0 on free trial credits
        ($300 free credits covers this many times over)

  TIME: Training takes 30-60 minutes. You can close Colab and come back!

  CHANGE THESE:
    YOUR_PROJECT_ID = "gleaming-cove-490722-p5"
    YOUR_REGION     = "us-central1"   ← Keep this unless you're outside North America
=============================================================
"""

# ─── CHANGE THESE LINES ─────────────────────────────────────
YOUR_PROJECT_ID = "your-gcp-project-id"   # ← Your GCP project ID
YOUR_REGION     = "us-central1"            # ← GCP region (keep as-is for free tier)
# ────────────────────────────────────────────────────────────

import time
from google.cloud import aiplatform
from google.colab import auth

# ─────────────────────────────────────────────
# AUTHENTICATE & INITIALIZE
# ─────────────────────────────────────────────
print("=" * 60)
print("  Step 1: Setting up Vertex AI...")
print("=" * 60)

auth.authenticate_user()

aiplatform.init(
    project=YOUR_PROJECT_ID,
    location=YOUR_REGION,
)
print(f"  ✓ Vertex AI initialized")
print(f"  ✓ Project: {YOUR_PROJECT_ID}")
print(f"  ✓ Region:  {YOUR_REGION}\n")

# ─────────────────────────────────────────────
# STEP 1: CREATE A DATASET IN VERTEX AI
# (Points Vertex AI to your BigQuery table)
# ─────────────────────────────────────────────
print("=" * 60)
print("  Step 2: Registering BigQuery data with Vertex AI...")
print("=" * 60)

DATASET_NAME = "shopping_interactions_dataset"
BQ_SOURCE = f"bq://{YOUR_PROJECT_ID}.shopping_assistant.user_interactions"

print(f"  → Connecting to BigQuery table: {BQ_SOURCE}")
print("  → This creates a Vertex AI Dataset (1-2 minutes)...")

dataset = aiplatform.TabularDataset.create(
    display_name=DATASET_NAME,
    bq_source=BQ_SOURCE,
)
dataset.wait()

print(f"  ✓ Dataset registered!")
print(f"  ✓ Dataset resource name: {dataset.resource_name}\n")

# ─────────────────────────────────────────────
# STEP 2: LAUNCH AUTOML TRAINING
# AutoML will automatically:
#   - Try multiple ML algorithms
#   - Tune hyperparameters
#   - Select the best model
#   - All without you needing to write any ML code!
# ─────────────────────────────────────────────
print("=" * 60)
print("  Step 3: Launching AutoML Tables training job...")
print("=" * 60)

print("""
  AutoML Training Configuration:
  ─────────────────────────────────
  Target column : purchased   (what we want to predict)
  Task type     : Binary classification (bought=1 or not=0)
  Budget        : 1 compute hour (free tier friendly)
  Split         : 80% train, 10% validation, 10% test
  ─────────────────────────────────
  AutoML will try: Gradient Boosted Trees, Neural Networks,
                   Ensembles, and more - picking the best!
""")

# Columns to use as input features for prediction
FEATURE_COLUMNS = [
    "category",
    "price_tier",
    "price",
    "avg_rating",
    "review_count",
    "session_duration_seconds",
    "time_of_day",
    "day_of_week",
    "is_weekend",
    "page_views_in_session",
    "items_in_cart",
    "user_total_purchases",
]

# Column the model should PREDICT
TARGET_COLUMN = "purchased"

column_transformations = []
categorical_cols = ["category", "price_tier"]
numeric_cols = [c for c in FEATURE_COLUMNS if c not in categorical_cols]

for col in categorical_cols:
    column_transformations.append({"categorical": {"column_name": col}})
for col in numeric_cols:
    column_transformations.append({"numeric": {"column_name": col}})

job = aiplatform.AutoMLTabularTrainingJob(
    display_name="shopping_assistant_model_v1",
    optimization_prediction_type="classification",
    optimization_objective="maximize-au-roc",    # Maximize Area Under ROC curve
    column_transformations=column_transformations,
)

print("  → Training job submitted! This will take 30-60 minutes.")
print("  → You can monitor progress at:")
print(f"     https://console.cloud.google.com/vertex-ai/training?project={YOUR_PROJECT_ID}")
print("\n  ⏳ Waiting for training to complete (do not close this tab)...\n")

model = job.run(
    dataset=dataset,
    target_column=TARGET_COLUMN,
    budget_milli_node_hours=1000,   # 1 compute hour = 1000 milli-hours
    model_display_name="shopping_recommendation_model",
    disable_early_stopping=False,   # AutoML will stop early if good enough
)

# ─────────────────────────────────────────────
# STEP 3: VIEW TRAINING RESULTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 4: Training complete! Viewing results...")
print("=" * 60)

# Get model evaluation
model_evaluations = model.list_model_evaluations()
for evaluation in model_evaluations:
    metrics = evaluation.metrics

    print("\n  📊 MODEL PERFORMANCE METRICS:")
    print("  ─────────────────────────────────────────")

    if "auRoc" in metrics:
        auc = metrics["auRoc"]
        grade = "Excellent ✅" if auc > 0.85 else ("Good ✅" if auc > 0.75 else "Fair ⚠️")
        print(f"  AUC-ROC Score:     {auc:.4f}   ({grade})")
        print(f"  → Interpretation:  {auc:.0%} probability of ranking a")
        print(f"                     purchased item above a non-purchased one")

    if "logLoss" in metrics:
        print(f"  Log Loss:          {metrics['logLoss']:.4f}  (lower is better)")

    if "confidenceMetrics" in metrics:
        conf_metrics = metrics["confidenceMetrics"]
        if conf_metrics:
            best = max(conf_metrics, key=lambda x: x.get("f1Score", 0))
            print(f"  Best F1 Score:     {best.get('f1Score', 'N/A'):.4f}")
            print(f"  Precision:         {best.get('precision', 'N/A'):.4f}")
            print(f"  Recall:            {best.get('recall', 'N/A'):.4f}")

print("\n  📌 Your model resource name (save this!):")
print(f"  {model.resource_name}")

# Save model name to file for use in Stage 3
with open("model_resource_name.txt", "w") as f:
    f.write(model.resource_name)
print("\n  ✓ Saved model name to: model_resource_name.txt")

print("\n" + "=" * 60)
print("  ✅ STAGE 2 COMPLETE! Your ML model is trained!")
print("=" * 60)
print(f"""
  View your model at:
  https://console.cloud.google.com/vertex-ai/models?project={YOUR_PROJECT_ID}

  Next step: Run stage3_deploy_api.py to deploy the model as an API.
""")
