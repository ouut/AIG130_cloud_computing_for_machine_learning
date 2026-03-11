"""
=============================================================
  AIG130 Project - Stage 3: Shopping Assistant API (main.py)
=============================================================
  WHAT THIS IS:
    This is the backend API for your Shopping Assistant.
    It receives user data and returns product recommendations
    by calling your trained AutoML model.

  This file is the main application code - it runs on Cloud Run
  (Google's serverless platform) so it's accessible from the internet.

  HOW IT WORKS:
    1. User sends their info (category interest, budget, etc.)
    2. This API calls your Vertex AI model
    3. Model predicts purchase probability for each product
    4. API returns the top recommended products

  FILES NEEDED (all 3 must be in the same folder):
    ✓ main.py          ← This file
    ✓ requirements.txt ← Python packages
    ✓ Dockerfile       ← Tells Cloud Run how to run the app
=============================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from google.cloud import aiplatform
import json
import os
import uvicorn

# ─── CHANGE THIS ────────────────────────────────────────────
PROJECT_ID   = os.environ.get("PROJECT_ID", "your-gcp-project-id")
REGION       = os.environ.get("REGION", "us-central1")
ENDPOINT_ID  = os.environ.get("ENDPOINT_ID", "your-endpoint-id")  # From Stage 3 deploy step
# ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Shopping Assistant API",
    description="AI-Powered Hyper-Personalized Shopping Recommendations",
    version="1.0.0"
)

# Allow web browsers to call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# DATA MODELS
# (Defines what the API accepts and returns)
# ─────────────────────────────────────────────

class UserContext(BaseModel):
    """What the user sends to get recommendations"""
    user_id: str
    category: str                          # "Electronics", "Clothing", etc.
    price_tier: str                        # "budget", "mid-range", "premium", "luxury"
    max_price: Optional[float] = 500.0
    session_duration_seconds: int = 60
    time_of_day: int = 12                  # 0-23
    day_of_week: int = 1                   # 0=Monday, 6=Sunday
    page_views_in_session: int = 3
    items_in_cart: int = 0
    user_total_purchases: int = 0
    is_weekend: int = 0

class ProductRecommendation(BaseModel):
    """A single product recommendation"""
    product_id: str
    product_name: str
    category: str
    price: float
    avg_rating: float
    purchase_probability: float            # How likely user is to buy (0-1)
    recommendation_reason: str

class RecommendationResponse(BaseModel):
    """What the API returns"""
    user_id: str
    query_summary: str
    recommendations: List[ProductRecommendation]
    model_version: str = "AutoML-v1"

# ─────────────────────────────────────────────
# PRODUCT CATALOG
# (In production this would be a database query)
# ─────────────────────────────────────────────

SAMPLE_PRODUCTS = [
    {"product_id": "PROD_0001", "name": "Wireless Bluetooth Headphones", "category": "Electronics", "price": 79.99, "rating": 4.5, "price_tier": "mid-range"},
    {"product_id": "PROD_0002", "name": "USB-C Fast Charger", "category": "Electronics", "price": 24.99, "rating": 4.3, "price_tier": "budget"},
    {"product_id": "PROD_0003", "name": "4K Smart TV 55-inch", "category": "Electronics", "price": 449.99, "rating": 4.7, "price_tier": "premium"},
    {"product_id": "PROD_0004", "name": "Running Shoes Pro", "category": "Sports", "price": 89.99, "rating": 4.4, "price_tier": "mid-range"},
    {"product_id": "PROD_0005", "name": "Yoga Mat Premium", "category": "Sports", "price": 34.99, "rating": 4.6, "price_tier": "budget"},
    {"product_id": "PROD_0006", "name": "Winter Jacket", "category": "Clothing", "price": 129.99, "rating": 4.2, "price_tier": "mid-range"},
    {"product_id": "PROD_0007", "name": "Coffee Maker Deluxe", "category": "Home & Kitchen", "price": 59.99, "rating": 4.5, "price_tier": "mid-range"},
    {"product_id": "PROD_0008", "name": "Air Fryer XL", "category": "Home & Kitchen", "price": 99.99, "rating": 4.8, "price_tier": "mid-range"},
    {"product_id": "PROD_0009", "name": "Skincare Set Luxury", "category": "Beauty", "price": 189.99, "rating": 4.6, "price_tier": "premium"},
    {"product_id": "PROD_0010", "name": "Python Programming Book", "category": "Books", "price": 39.99, "rating": 4.9, "price_tier": "budget"},
]

# ─────────────────────────────────────────────
# VERTEX AI PREDICTION FUNCTION
# ─────────────────────────────────────────────

def get_purchase_prediction(user_context: UserContext, product: dict) -> float:
    """
    Calls the Vertex AI AutoML model to predict purchase probability.
    Returns a float between 0 and 1.
    """
    try:
        # Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=REGION)
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)

        # Prepare the input features (must match training data columns exactly)
        instance = {
            "category":                 product["category"],
            "price_tier":               product["price_tier"],
            "price":                    product["price"],
            "avg_rating":               product["rating"],
            "review_count":             100,  # Default value
            "session_duration_seconds": user_context.session_duration_seconds,
            "time_of_day":              user_context.time_of_day,
            "day_of_week":              user_context.day_of_week,
            "is_weekend":               user_context.is_weekend,
            "page_views_in_session":    user_context.page_views_in_session,
            "items_in_cart":            user_context.items_in_cart,
            "user_total_purchases":     user_context.user_total_purchases,
        }

        # Call the model
        prediction = endpoint.predict(instances=[instance])
        probabilities = prediction.predictions[0]

        # AutoML returns probabilities for each class [class_0_prob, class_1_prob]
        # We want class 1 (purchased=1) probability
        if isinstance(probabilities, dict):
            scores = probabilities.get("scores", [0.5, 0.5])
            classes = probabilities.get("classes", ["0", "1"])
            if "1" in classes:
                idx = classes.index("1")
                return float(scores[idx])
        return 0.5

    except Exception as e:
        # If model call fails, use a simple rule-based fallback
        print(f"Model prediction failed: {e}. Using fallback.")
        return _fallback_prediction(user_context, product)


def _fallback_prediction(user_context: UserContext, product: dict) -> float:
    """Simple rule-based prediction when model is unavailable"""
    score = 0.3  # Base score
    if product["category"] == user_context.category:
        score += 0.3
    if product["price_tier"] == user_context.price_tier:
        score += 0.2
    if product["price"] <= user_context.max_price:
        score += 0.1
    score += (product["rating"] - 3.0) / 2.0 * 0.1
    return min(0.95, max(0.05, score))


def get_recommendation_reason(probability: float, product: dict, user: UserContext) -> str:
    """Generate a human-readable explanation for the recommendation"""
    reasons = []
    if product["category"] == user.category:
        reasons.append(f"matches your interest in {user.category}")
    if product["price_tier"] == user.price_tier:
        reasons.append("fits your budget preference")
    if product["rating"] >= 4.5:
        reasons.append(f"highly rated at {product['rating']}★")
    if probability > 0.7:
        reasons.append("strong match for your browsing pattern")
    if not reasons:
        reasons.append("popular in your selected category")
    return "Recommended because it " + " and ".join(reasons[:2])


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "✅ Shopping Assistant API is running!",
        "version": "1.0.0",
        "endpoints": {
            "recommendations": "POST /recommend",
            "health":          "GET /health",
            "docs":            "GET /docs"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "project": PROJECT_ID}


@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(user_context: UserContext):
    """
    Main recommendation endpoint.
    Send user context, receive personalized product recommendations.

    Example request body:
    {
        "user_id": "USER_00001",
        "category": "Electronics",
        "price_tier": "mid-range",
        "max_price": 150.0,
        "session_duration_seconds": 120,
        "time_of_day": 19,
        "day_of_week": 5,
        "items_in_cart": 1,
        "user_total_purchases": 5,
        "is_weekend": 1
    }
    """
    print(f"📥 Request from user: {user_context.user_id}")

    # Filter products by category and price
    candidate_products = [
        p for p in SAMPLE_PRODUCTS
        if p["category"] == user_context.category
        and p["price"] <= user_context.max_price
    ]

    # If no exact category match, use all products
    if not candidate_products:
        candidate_products = [p for p in SAMPLE_PRODUCTS if p["price"] <= user_context.max_price]

    if not candidate_products:
        candidate_products = SAMPLE_PRODUCTS[:5]

    # Get purchase predictions for each product
    scored_products = []
    for product in candidate_products:
        probability = get_purchase_prediction(user_context, product)
        reason = get_recommendation_reason(probability, product, user_context)
        scored_products.append({
            "product": product,
            "probability": probability,
            "reason": reason
        })

    # Sort by purchase probability (highest first)
    scored_products.sort(key=lambda x: x["probability"], reverse=True)

    # Take top 5
    recommendations = []
    for item in scored_products[:5]:
        p = item["product"]
        recommendations.append(ProductRecommendation(
            product_id=p["product_id"],
            product_name=p["name"],
            category=p["category"],
            price=p["price"],
            avg_rating=p["rating"],
            purchase_probability=round(item["probability"], 3),
            recommendation_reason=item["reason"]
        ))

    return RecommendationResponse(
        user_id=user_context.user_id,
        query_summary=f"{user_context.category} items in {user_context.price_tier} range",
        recommendations=recommendations
    )


@app.get("/products")
def list_products():
    """Returns the product catalog"""
    return {"products": SAMPLE_PRODUCTS, "count": len(SAMPLE_PRODUCTS)}


# ─────────────────────────────────────────────
# RUN LOCALLY (for testing before Cloud Run)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting Shopping Assistant API locally...")
    print("   Open: http://localhost:8080/docs  (interactive API explorer)")
    uvicorn.run(app, host="0.0.0.0", port=8080)
