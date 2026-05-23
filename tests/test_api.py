import pytest
from fastapi.testclient import TestClient
from app.main import app

# Initialize test client
client = TestClient(app)

def test_health_endpoint():
    """Verify that the health check endpoint returns 200 and indicates service readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Healthy"
    assert data["model_loaded"] is True
    assert "features_expected" in data

def test_predict_low_risk_customer():
    """Verify that a customer with strong brand indicators (recent purchases, gold tier, high web activity)

    is correctly predicted as low risk (class 0) with matching explanation.
    """
    payload = {
        "city_tier": "Tier 1",
        "age_group": "25-34",
        "acquisition_channel": "Organic",
        "loyalty_tier": "Gold",
        "preferred_category": "Skin Care",
        "skin_type": "Combination",
        "marketing_consent": "Yes",
        "recency_days": 4.0,
        "frequency_180d": 6,
        "monetary_180d": 4200.0,
        "return_rate_180d": 0.0,
        "avg_discount_pct_180d": 0.05,
        "avg_rating_180d": 4.8,
        "category_diversity_180d": 3,
        "ticket_count_90d": 0,
        "negative_ticket_rate_90d": 0.0,
        "avg_resolution_hours_90d": 0.0,
        "days_since_signup": 365,
        "sessions_30d": 20,
        "product_views_30d": 80,
        "cart_adds_30d": 6,
        "wishlist_adds_30d": 3,
        "abandoned_carts_30d": 0,
        "email_opens_30d": 12,
        "campaign_clicks_30d": 4,
        "last_visit_days_ago": 1
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "churn_probability" in data
    assert "predicted_class" in data
    assert "risk_explanation" in data
    assert data["churn_probability"] < 0.20
    assert data["predicted_class"] == 0
    assert "LOW RISK" in data["risk_explanation"]

def test_predict_high_risk_customer():
    """Verify that a customer with high-risk behaviors (prolonged inactivity, support friction, cart abandonments)

    is correctly predicted as high risk (class 1, probability >= 0.20) with matching explanation.
    """
    payload = {
        "city_tier": "Tier 3",
        "age_group": "35-44",
        "acquisition_channel": "Google Search",
        "loyalty_tier": "None",
        "preferred_category": "Makeup",
        "skin_type": "Dry",
        "marketing_consent": "No",
        "recency_days": 140.0,
        "frequency_180d": 1,
        "monetary_180d": 350.0,
        "return_rate_180d": 0.0,
        "avg_discount_pct_180d": 0.35,
        "avg_rating_180d": 2.0,
        "category_diversity_180d": 1,
        "ticket_count_90d": 3,
        "negative_ticket_rate_90d": 1.0,
        "avg_resolution_hours_90d": 48.0,
        "days_since_signup": 150,
        "sessions_30d": 1,
        "product_views_30d": 2,
        "cart_adds_30d": 0,
        "wishlist_adds_30d": 0,
        "abandoned_carts_30d": 4,
        "email_opens_30d": 0,
        "campaign_clicks_30d": 0,
        "last_visit_days_ago": 25
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "churn_probability" in data
    assert "predicted_class" in data
    assert "risk_explanation" in data
    assert data["churn_probability"] >= 0.20
    assert data["predicted_class"] == 1
    assert "HIGH RISK" in data["risk_explanation"]

def test_predict_invalid_payload_error():
    """Verify that posting a malformed or invalid payload triggers a 422 Unprocessable Entity error."""
    # Send numeric field with an invalid string
    payload = {
        "recency_days": "one hundred days",
        "frequency_180d": "not a number"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_batch_predict_endpoint():
    """Verify that the batch scoring endpoint returns bulk predictions with matching payload structures."""
    payload = [
        # Customer 1: Low Risk
        {
            "recency_days": 5.0,
            "frequency_180d": 4,
            "monetary_180d": 2000.0,
            "loyalty_tier": "Silver"
        },
        # Customer 2: High Risk
        {
            "recency_days": 120.0,
            "frequency_180d": 1,
            "monetary_180d": 300.0,
            "loyalty_tier": "None",
            "ticket_count_90d": 2,
            "negative_ticket_rate_90d": 1.0
        }
    ]
    
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    
    # Assert Customer 1 matches Low Risk
    assert data["predictions"][0]["churn_probability"] < 0.20
    assert data["predictions"][0]["predicted_class"] == 0
    
    # Assert Customer 2 matches High Risk
    assert data["predictions"][1]["churn_probability"] >= 0.20
    assert data["predictions"][1]["predicted_class"] == 1
