"""
=============================================================
  AIG130 Project - Stage 1, Step 2: Upload Data to BigQuery
=============================================================
  WHAT THIS DOES:
    - Connects to your Google Cloud project
    - Creates a BigQuery dataset and table
    - Uploads the cleaned CSV file to BigQuery

  BEFORE RUNNING:
    1. Create a FREE GCP account: https://cloud.google.com/free
       (You get $300 free credits - NO charges for this project)
    2. Create a new project in GCP Console
    3. Enable BigQuery API:
       → Go to: https://console.cloud.google.com/apis/library
       → Search "BigQuery API" → Click Enable
    4. In Google Colab, run this first:
         from google.colab import auth
         auth.authenticate_user()

  CHANGE THIS:
    YOUR_PROJECT_ID = "your-gcp-project-id"   ← Replace with your actual project ID
                                                  (found in GCP Console top bar)
=============================================================
"""

# ─── CHANGE THIS LINE ───────────────────────────────────────
YOUR_PROJECT_ID = "your-gcp-project-id"   # ← PUT YOUR GCP PROJECT ID HERE
# ────────────────────────────────────────────────────────────

import pandas as pd
from google.cloud import bigquery
from google.colab import auth

# ─────────────────────────────────────────────
# AUTHENTICATE WITH GCP
# (A popup will ask you to sign in with your Google account)
# ─────────────────────────────────────────────
print("=" * 60)
print("  Step 1: Authenticating with Google Cloud...")
print("=" * 60)
print("  → A popup will appear. Sign in with your Google account.")
print("  → Make sure it's the same account you used to create GCP.\n")

auth.authenticate_user()
print("  ✓ Authentication successful!\n")

# ─────────────────────────────────────────────
# SET UP BIGQUERY CLIENT
# ─────────────────────────────────────────────
print("=" * 60)
print("  Step 2: Connecting to BigQuery...")
print("=" * 60)

client = bigquery.Client(project=YOUR_PROJECT_ID)
print(f"  ✓ Connected to project: {YOUR_PROJECT_ID}\n")

# ─────────────────────────────────────────────
# CREATE DATASET
# ─────────────────────────────────────────────
DATASET_ID = "shopping_assistant"
TABLE_ID    = "user_interactions"

print("=" * 60)
print("  Step 3: Creating BigQuery dataset...")
print("=" * 60)

dataset_ref = bigquery.Dataset(f"{YOUR_PROJECT_ID}.{DATASET_ID}")
dataset_ref.location = "US"
dataset_ref.description = "AIG130 Project - Shopping Assistant ML Data"

try:
    dataset = client.create_dataset(dataset_ref, exists_ok=True)
    print(f"  ✓ Dataset created: {DATASET_ID}\n")
except Exception as e:
    print(f"  ! Warning: {e}")
    print("  → This is OK if the dataset already exists\n")

# ─────────────────────────────────────────────
# DEFINE TABLE SCHEMA
# (Tells BigQuery what columns and data types to expect)
# ─────────────────────────────────────────────
print("=" * 60)
print("  Step 4: Defining table schema...")
print("=" * 60)

schema = [
    bigquery.SchemaField("user_id",                   "STRING",  description="Unique user ID"),
    bigquery.SchemaField("product_id",                "STRING",  description="Unique product ID"),
    bigquery.SchemaField("category",                  "STRING",  description="Product category"),
    bigquery.SchemaField("price_tier",                "STRING",  description="budget/mid-range/premium/luxury"),
    bigquery.SchemaField("price",                     "FLOAT64", description="Product price USD"),
    bigquery.SchemaField("avg_rating",                "FLOAT64", description="Average product rating"),
    bigquery.SchemaField("review_count",              "INTEGER", description="Number of reviews"),
    bigquery.SchemaField("session_duration_seconds",  "INTEGER", description="Browse duration"),
    bigquery.SchemaField("time_of_day",               "INTEGER", description="Hour 0-23"),
    bigquery.SchemaField("day_of_week",               "INTEGER", description="0=Mon to 6=Sun"),
    bigquery.SchemaField("is_weekend",                "INTEGER", description="1 if weekend"),
    bigquery.SchemaField("page_views_in_session",     "INTEGER", description="Pages viewed"),
    bigquery.SchemaField("items_in_cart",             "INTEGER", description="Cart size"),
    bigquery.SchemaField("user_total_purchases",      "INTEGER", description="User purchase history"),
    bigquery.SchemaField("purchased",                 "INTEGER", description="TARGET: 1=bought, 0=browsed"),
    bigquery.SchemaField("interaction_date",          "STRING",  description="Date of interaction"),
]

table_ref = f"{YOUR_PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
table = bigquery.Table(table_ref, schema=schema)

try:
    table = client.create_table(table, exists_ok=True)
    print(f"  ✓ Table created: {TABLE_ID}")
    print(f"  ✓ Full path: {table_ref}\n")
except Exception as e:
    print(f"  ! Error creating table: {e}")

# ─────────────────────────────────────────────
# UPLOAD CSV TO BIGQUERY
# ─────────────────────────────────────────────
print("=" * 60)
print("  Step 5: Uploading data to BigQuery...")
print("  (This may take 1-2 minutes)")
print("=" * 60)

df = pd.read_csv("shopping_data_clean.csv")
print(f"  → Uploading {len(df):,} rows...")

job_config = bigquery.LoadJobConfig(
    schema=schema,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Overwrite if exists
    skip_leading_rows=1,  # Skip CSV header row
    source_format=bigquery.SourceFormat.CSV,
)

job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
job.result()  # Wait for upload to complete

# Verify upload
table_info = client.get_table(table_ref)
print(f"  ✓ Upload complete!")
print(f"  ✓ Rows in BigQuery: {table_info.num_rows:,}")
print(f"  ✓ Table size: {table_info.num_bytes / 1024:.1f} KB\n")

# ─────────────────────────────────────────────
# TEST: RUN A QUERY TO VERIFY DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  Step 6: Testing data with a BigQuery SQL query...")
print("=" * 60)

test_query = f"""
    SELECT
        category,
        COUNT(*) AS total_interactions,
        SUM(purchased) AS total_purchases,
        ROUND(AVG(price), 2) AS avg_price,
        ROUND(AVG(session_duration_seconds), 0) AS avg_session_secs
    FROM `{table_ref}`
    GROUP BY category
    ORDER BY total_interactions DESC
"""

results = client.query(test_query).to_dataframe()
print("  ✓ Query results:")
print(results.to_string(index=False))

print("\n" + "=" * 60)
print("  ✅ STAGE 1 COMPLETE!")
print("=" * 60)
print(f"""
  Your data is now live in BigQuery!
  You can view it at:
  https://console.cloud.google.com/bigquery?project={YOUR_PROJECT_ID}

  Next step: Run stage2_train_automl.py to train your ML model.
""")
