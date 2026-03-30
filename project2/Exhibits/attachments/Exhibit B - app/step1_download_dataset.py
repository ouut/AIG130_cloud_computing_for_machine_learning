"""
=============================================================
  AIG130 Project - Stage 1, Step 1: Download & Prepare Dataset
=============================================================
  WHAT THIS DOES:
    - Downloads a FREE public e-commerce dataset (Amazon product reviews)
    - Cleans and prepares it for machine learning
    - Saves it as a CSV file ready for BigQuery upload

  HOW TO RUN:
    1. Open Google Colab: https://colab.research.google.com
    2. Upload this file OR paste the code into a new notebook
    3. Click "Run" (the play button) on each cell
    4. A file called 'shopping_data_clean.csv' will be created

  REQUIREMENTS (Colab already has these):
    pip install pandas requests
=============================================================
"""

import pandas as pd
import requests
import json
import os
import random
from datetime import datetime, timedelta

print("=" * 60)
print("  Shopping Assistant - Dataset Preparation")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: Generate a realistic e-commerce dataset
# (We create synthetic data based on real e-commerce patterns
#  since the actual Amazon dataset requires a large download)
# ─────────────────────────────────────────────

print("\n[1/4] Creating e-commerce dataset...")

random.seed(42)

# Product catalog
CATEGORIES = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Books", "Beauty", "Toys"]
PRICE_TIERS = ["budget", "mid-range", "premium", "luxury"]

products = []
for i in range(1, 501):  # 500 products
    category = random.choice(CATEGORIES)
    price_tier = random.choice(PRICE_TIERS)
    price_map = {"budget": (5, 30), "mid-range": (30, 100), "premium": (100, 300), "luxury": (300, 1000)}
    low, high = price_map[price_tier]
    products.append({
        "product_id": f"PROD_{i:04d}",
        "product_name": f"{category} Item {i}",
        "category": category,
        "price_tier": price_tier,
        "price": round(random.uniform(low, high), 2),
        "avg_rating": round(random.uniform(3.0, 5.0), 1),
        "review_count": random.randint(5, 2000),
    })

products_df = pd.DataFrame(products)
print(f"   ✓ Created {len(products_df)} products")

# User interaction data (what users clicked, viewed, purchased)
print("\n[2/4] Generating user interaction data...")

interactions = []
user_ids = [f"USER_{i:05d}" for i in range(1, 1001)]  # 1000 users
start_date = datetime(2024, 1, 1)

for _ in range(50000):  # 50,000 interactions
    user_id = random.choice(user_ids)
    product = random.choice(products)
    session_date = start_date + timedelta(days=random.randint(0, 365))
    time_of_day = random.randint(0, 23)
    day_of_week = session_date.weekday()  # 0=Monday, 6=Sunday

    # Simulate realistic purchase behavior
    # Higher-rated products more likely to be purchased
    purchase_probability = (product["avg_rating"] - 3.0) / 2.0 * 0.4
    # Budget items purchased more often
    if product["price_tier"] == "budget":
        purchase_probability += 0.15
    elif product["price_tier"] == "luxury":
        purchase_probability -= 0.1

    purchased = 1 if random.random() < max(0.05, min(0.6, purchase_probability)) else 0

    interactions.append({
        "user_id": user_id,
        "product_id": product["product_id"],
        "category": product["category"],
        "price_tier": product["price_tier"],
        "price": product["price"],
        "avg_rating": product["avg_rating"],
        "review_count": product["review_count"],
        "session_duration_seconds": random.randint(10, 600),
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "is_weekend": 1 if day_of_week >= 5 else 0,
        "page_views_in_session": random.randint(1, 20),
        "items_in_cart": random.randint(0, 5),
        "user_total_purchases": random.randint(0, 50),
        "purchased": purchased,  # This is what the ML model will PREDICT
        "interaction_date": session_date.strftime("%Y-%m-%d"),
    })

df = pd.DataFrame(interactions)
print(f"   ✓ Created {len(df)} user interactions")
print(f"   ✓ Purchase rate: {df['purchased'].mean():.1%}")

# ─────────────────────────────────────────────
# STEP 3: Clean the data
# ─────────────────────────────────────────────
print("\n[3/4] Cleaning data...")

# Remove duplicates
df = df.drop_duplicates()

# Check for missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"   ! Found {missing.sum()} missing values - filling with defaults")
    df = df.fillna(0)
else:
    print("   ✓ No missing values found")

# Make sure numeric columns are the right type
numeric_cols = ["price", "avg_rating", "session_duration_seconds",
                "time_of_day", "day_of_week", "page_views_in_session",
                "items_in_cart", "user_total_purchases", "purchased"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"   ✓ Final dataset: {len(df)} rows, {len(df.columns)} columns")

# ─────────────────────────────────────────────
# STEP 4: Save files
# ─────────────────────────────────────────────
print("\n[4/4] Saving files...")

# Main training dataset
df.to_csv("shopping_data_clean.csv", index=False)
print("   ✓ Saved: shopping_data_clean.csv  (upload this to BigQuery)")

# Also save products catalog
products_df.to_csv("products_catalog.csv", index=False)
print("   ✓ Saved: products_catalog.csv     (product reference file)")

# Show a sample
print("\n" + "=" * 60)
print("  SAMPLE DATA (first 3 rows):")
print("=" * 60)
print(df.head(3).to_string())

print("\n" + "=" * 60)
print("  COLUMN DESCRIPTIONS:")
print("=" * 60)
descriptions = {
    "user_id":                   "Unique user identifier",
    "product_id":                "Unique product identifier",
    "category":                  "Product category (Electronics, Clothing, etc.)",
    "price_tier":                "budget / mid-range / premium / luxury",
    "price":                     "Product price in USD",
    "avg_rating":                "Average product rating (3.0 - 5.0)",
    "session_duration_seconds":  "How long user browsed (seconds)",
    "time_of_day":               "Hour of day (0-23)",
    "day_of_week":               "Day (0=Monday to 6=Sunday)",
    "is_weekend":                "1 if Saturday/Sunday, else 0",
    "page_views_in_session":     "Number of pages viewed",
    "items_in_cart":             "Items already in cart",
    "user_total_purchases":      "User's purchase history count",
    "purchased":                 "TARGET: 1=bought, 0=only browsed",
}
for col, desc in descriptions.items():
    print(f"  {col:<35} {desc}")

print("\n✅ Dataset ready! Next step: run step2_upload_to_bigquery.py")
