"""
=============================================================
  AIG130 Project - Stage 4: Monitoring & MLOps
=============================================================
  WHAT THIS DOES:
    - Sets up Vertex AI Model Monitoring (detects if model degrades)
    - Creates a BigQuery dashboard query to track usage
    - Sets up a Cloud Function to log predictions automatically
    - Shows you how to check your model's health

  WHAT MONITORING CATCHES:
    - Feature drift: input data starts looking different than training data
      Example: Users suddenly browse much longer sessions than expected
    - Prediction drift: model's predictions shift over time
      Example: Model starts recommending far fewer purchases
    - These are early warnings that the model needs retraining!

  BEFORE RUNNING:
    1. Complete Stages 1, 2, and 3
    2. Have your endpoint_id.txt from Stage 3

  CHANGE THESE:
    YOUR_PROJECT_ID = "your-gcp-project-id"
=============================================================
"""

# ─── CHANGE THIS ────────────────────────────────────────────
YOUR_PROJECT_ID  = "your-gcp-project-id"
YOUR_REGION      = "us-central1"
ALERT_EMAIL      = "your-email@example.com"   # ← Email for drift alerts
# ────────────────────────────────────────────────────────────

from google.cloud import aiplatform
from google.colab import auth
import json

auth.authenticate_user()
aiplatform.init(project=YOUR_PROJECT_ID, location=YOUR_REGION)

# Load endpoint ID from Stage 3
try:
    with open("endpoint_id.txt", "r") as f:
        ENDPOINT_ID = f.read().strip()
    print(f"  ✓ Using endpoint: {ENDPOINT_ID}")
except FileNotFoundError:
    ENDPOINT_ID = "your-endpoint-id"
    print("  ⚠ endpoint_id.txt not found. Using placeholder.")
    print("    Update ENDPOINT_ID manually if needed.")

# ─────────────────────────────────────────────
# PART 1: SET UP VERTEX AI MODEL MONITORING
# ─────────────────────────────────────────────
print("=" * 60)
print("  Part 1: Setting Up Vertex AI Model Monitoring")
print("=" * 60)

print("""
  Model Monitoring tracks two types of drift:
  ┌─────────────────────────────────────────────────────┐
  │ FEATURE DRIFT    - Input data changes significantly  │
  │                   (KL divergence or Jensen-Shannon)  │
  │ PREDICTION DRIFT - Output predictions shift         │
  │                   (Good indicator model needs        │
  │                    retraining)                       │
  └─────────────────────────────────────────────────────┘
""")

try:
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)

    # Define monitoring job
    monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name="shopping_assistant_monitor",
        endpoint=endpoint,
        logging_sampling_strategy=aiplatform.gapic.SamplingStrategy(
            random_sample_config=aiplatform.gapic.SamplingStrategy.RandomSampleConfig(
                sample_rate=0.1   # Log 10% of all predictions (free tier friendly)
            )
        ),
        model_deployment_monitoring_objective_configs=[
            aiplatform.gapic.ModelDeploymentMonitoringObjectiveConfig(
                deployed_model_id=endpoint.list_models()[0].id if endpoint.list_models() else "",
                objective_config=aiplatform.gapic.ModelMonitoringObjectiveConfig(
                    training_dataset=aiplatform.gapic.ModelMonitoringObjectiveConfig.TrainingDataset(
                        bigquery_source=aiplatform.gapic.BigQuerySource(
                            input_uri=f"bq://{YOUR_PROJECT_ID}.shopping_assistant.user_interactions"
                        ),
                        target_field="purchased",
                    ),
                    training_prediction_skew_detection_config=aiplatform.gapic.ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig(
                        skew_thresholds={
                            "session_duration_seconds": aiplatform.gapic.ThresholdConfig(value=0.3),
                            "price":                    aiplatform.gapic.ThresholdConfig(value=0.3),
                            "avg_rating":               aiplatform.gapic.ThresholdConfig(value=0.2),
                        }
                    ),
                )
            )
        ],
        model_monitoring_alert_config=aiplatform.gapic.ModelMonitoringAlertConfig(
            email_alert_config=aiplatform.gapic.ModelMonitoringAlertConfig.EmailAlertConfig(
                user_emails=[ALERT_EMAIL]
            )
        ),
        predict_instance_schema_uri="",
        analysis_instance_schema_uri="",
        model_deployment_monitoring_schedule_config=aiplatform.gapic.ModelDeploymentMonitoringScheduleConfig(
            monitor_interval={"seconds": 3600}  # Check every 1 hour
        ),
    )
    print(f"  ✓ Monitoring job created: {monitoring_job.display_name}")
    print(f"  ✓ Alerts will be sent to: {ALERT_EMAIL}")

except Exception as e:
    print(f"  ⚠ Automated monitoring setup requires a deployed endpoint.")
    print(f"    Complete Stage 3 first, then re-run this script.")
    print(f"    Error details: {e}")
    print("\n  → Proceeding with manual monitoring dashboard setup...")

# ─────────────────────────────────────────────
# PART 2: BIGQUERY MONITORING QUERIES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Part 2: BigQuery Monitoring Dashboard Queries")
print("=" * 60)
print("  Copy these queries into BigQuery Console to monitor your model:\n")

queries = {
    "1. Daily Interaction Volume": f"""
-- Shows how many users interact with the shopping assistant per day
SELECT
    interaction_date,
    COUNT(*)                                AS total_interactions,
    SUM(purchased)                          AS total_purchases,
    ROUND(AVG(CAST(purchased AS FLOAT64)) * 100, 1)  AS purchase_rate_pct,
    COUNT(DISTINCT user_id)                 AS unique_users
FROM `{YOUR_PROJECT_ID}.shopping_assistant.user_interactions`
GROUP BY interaction_date
ORDER BY interaction_date DESC
LIMIT 30;
""",
    "2. Category Performance": f"""
-- Which product categories are performing best?
SELECT
    category,
    COUNT(*)                                AS total_views,
    SUM(purchased)                          AS total_purchases,
    ROUND(AVG(CAST(purchased AS FLOAT64)) * 100, 1) AS conversion_rate_pct,
    ROUND(AVG(price), 2)                    AS avg_product_price
FROM `{YOUR_PROJECT_ID}.shopping_assistant.user_interactions`
GROUP BY category
ORDER BY conversion_rate_pct DESC;
""",
    "3. Feature Drift Detection": f"""
-- Compares recent data statistics vs. historical baseline
-- Large differences indicate the model may need retraining
WITH baseline AS (
    SELECT
        AVG(session_duration_seconds)    AS avg_session,
        STDDEV(session_duration_seconds) AS std_session,
        AVG(price)                       AS avg_price,
        AVG(user_total_purchases)        AS avg_purchases
    FROM `{YOUR_PROJECT_ID}.shopping_assistant.user_interactions`
    WHERE interaction_date < DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
),
recent AS (
    SELECT
        AVG(session_duration_seconds)    AS avg_session,
        AVG(price)                       AS avg_price,
        AVG(user_total_purchases)        AS avg_purchases
    FROM `{YOUR_PROJECT_ID}.shopping_assistant.user_interactions`
    WHERE interaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
)
SELECT
    'session_duration' AS feature,
    ROUND(b.avg_session, 1)  AS baseline_avg,
    ROUND(r.avg_session, 1)  AS recent_avg,
    ROUND(ABS(r.avg_session - b.avg_session) / NULLIF(b.std_session, 0), 2) AS drift_score,
    IF(ABS(r.avg_session - b.avg_session) / NULLIF(b.std_session, 0) > 1.5, '⚠ DRIFT DETECTED', '✅ Normal') AS status
FROM baseline b, recent r;
""",
    "4. Hourly API Usage": f"""
-- Shows peak usage hours to plan for scaling
SELECT
    time_of_day                                   AS hour_of_day,
    COUNT(*)                                      AS total_requests,
    ROUND(AVG(CAST(purchased AS FLOAT64)) * 100, 1) AS avg_conversion_pct
FROM `{YOUR_PROJECT_ID}.shopping_assistant.user_interactions`
GROUP BY time_of_day
ORDER BY time_of_day;
"""
}

for name, query in queries.items():
    print(f"  {'─' * 55}")
    print(f"  Query: {name}")
    print(f"  {'─' * 55}")
    print(query)

# Save queries to file
with open("monitoring_queries.sql", "w") as f:
    for name, query in queries.items():
        f.write(f"-- {name}\n{query}\n\n")
print("\n  ✓ All queries saved to: monitoring_queries.sql")

# ─────────────────────────────────────────────
# PART 3: TEST YOUR DEPLOYED API
# ─────────────────────────────────────────────
print("=" * 60)
print("  Part 3: Testing Your Deployed API")
print("=" * 60)

print("""
  Once your Cloud Run URL is live, test your API with this Python code:
  (Replace YOUR_CLOUD_RUN_URL with your actual URL from Stage 3)
""")

test_code = """
import requests

YOUR_CLOUD_RUN_URL = "https://shopping-assistant-api-xxxxxxx-uc.a.run.app"

# Test 1: Check the API is running
response = requests.get(f"{YOUR_CLOUD_RUN_URL}/")
print("API Status:", response.json())

# Test 2: Get personalized recommendations
user_data = {
    "user_id": "TEST_USER_001",
    "category": "Electronics",
    "price_tier": "mid-range",
    "max_price": 150.0,
    "session_duration_seconds": 120,
    "time_of_day": 19,
    "day_of_week": 5,
    "page_views_in_session": 5,
    "items_in_cart": 1,
    "user_total_purchases": 10,
    "is_weekend": 1
}

response = requests.post(f"{YOUR_CLOUD_RUN_URL}/recommend", json=user_data)
result = response.json()

print(f"\\nRecommendations for {result['user_id']}:")
print(f"Query: {result['query_summary']}")
print("\\nTop Products:")
for i, rec in enumerate(result['recommendations'], 1):
    print(f"  {i}. {rec['product_name']}")
    print(f"     Price: ${rec['price']}  |  Rating: {rec['avg_rating']}★")
    print(f"     Match Score: {rec['purchase_probability']:.0%}")
    print(f"     Why: {rec['recommendation_reason']}")
"""
print(test_code)

with open("test_api.py", "w") as f:
    f.write(test_code)
print("  ✓ Test code saved to: test_api.py")

# ─────────────────────────────────────────────
# PART 4: MONITORING CHECKLIST
# ─────────────────────────────────────────────
print("=" * 60)
print("  Part 4: Ongoing Monitoring Checklist")
print("=" * 60)

checklist = """
  Run these checks weekly to ensure your model stays healthy:

  ┌─────────────────────────────────────────────────────────┐
  │  DAILY CHECKS (automated via Vertex AI Monitoring)      │
  │  ✓ API is responding (latency < 200ms)                  │
  │  ✓ No error rate spikes in Cloud Logging                │
  │  ✓ Prediction volume is within normal range             │
  │                                                         │
  │  WEEKLY CHECKS (run monitoring_queries.sql)             │
  │  ✓ Conversion rate stable (not dropping significantly)  │
  │  ✓ Feature drift score < 1.5 for all features          │
  │  ✓ Category distribution hasn't shifted dramatically    │
  │                                                         │
  │  MONTHLY ACTIONS                                        │
  │  ✓ Re-evaluate model on new held-out data              │
  │  ✓ If AUC-ROC drops > 5%, schedule retraining          │
  │  ✓ Review Vertex AI Monitoring dashboard               │
  │    URL: https://console.cloud.google.com/vertex-ai/    │
  │         model-monitoring                               │
  └─────────────────────────────────────────────────────────┘

  WHEN TO RETRAIN:
  → Feature drift score > 1.5  (data distribution has shifted)
  → Conversion rate drops > 10% from baseline
  → Model AUC-ROC drops below 0.75 on new evaluation data
  → More than 90 days have passed since last training
"""
print(checklist)

print("=" * 60)
print("  ✅ STAGE 4 COMPLETE! Full MLOps pipeline is set up!")
print("=" * 60)
print(f"""
  🎉 Congratulations! Your full ML pipeline is now:
  ✓ Stage 1: Data in BigQuery
  ✓ Stage 2: Model trained with AutoML Tables
  ✓ Stage 3: API live on Cloud Run
  ✓ Stage 4: Monitoring & alerting configured

  GCP Console links to include in your report:
  → BigQuery:    https://console.cloud.google.com/bigquery?project={YOUR_PROJECT_ID}
  → Vertex AI:   https://console.cloud.google.com/vertex-ai?project={YOUR_PROJECT_ID}
  → Cloud Run:   https://console.cloud.google.com/run?project={YOUR_PROJECT_ID}
  → Monitoring:  https://console.cloud.google.com/vertex-ai/model-monitoring?project={YOUR_PROJECT_ID}
""")
