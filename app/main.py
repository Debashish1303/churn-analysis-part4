import json
import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Feature columns aligned with Part 3 training pipeline (see DATA_DICTIONARY.md)
CAT_COLS = [
    "city_tier",
    "age_group",
    "acquisition_channel",
    "loyalty_tier",
    "preferred_category",
    "skin_type",
    "marketing_consent",
]
NUM_COLS = [
    "recency_days",
    "frequency_180d",
    "monetary_180d",
    "return_rate_180d",
    "avg_discount_pct_180d",
    "avg_rating_180d",
    "category_diversity_180d",
    "ticket_count_90d",
    "negative_ticket_rate_90d",
    "avg_resolution_hours_90d",
    "days_since_signup",
    "sessions_30d",
    "product_views_30d",
    "cart_adds_30d",
    "wishlist_adds_30d",
    "abandoned_carts_30d",
    "email_opens_30d",
    "campaign_clicks_30d",
    "last_visit_days_ago",
]
FEATURE_COLS = NUM_COLS + CAT_COLS

PART4_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PART4_ROOT / "model" / "model.pkl"
METRICS_PATH = PART4_ROOT / "model" / "metrics.json"
DEFAULT_DECISION_THRESHOLD = 0.20

model_pipeline = None
decision_threshold = DEFAULT_DECISION_THRESHOLD


class CustomerFeatures(BaseModel):
    city_tier: Optional[str] = Field(default="Unknown", description="City tier (Tier 1, Tier 2, Tier 3)")
    age_group: Optional[str] = Field(default="Unknown", description="Age bracket (18-24, 25-34, 35-44, 45+)")
    acquisition_channel: Optional[str] = Field(
        default="Unknown",
        description="Acquisition channel (e.g. Instagram, Organic)",
    )
    loyalty_tier: Optional[str] = Field(
        default="None",
        description="Loyalty tier (None, Silver, Gold, Platinum)",
    )
    preferred_category: Optional[str] = Field(default="Unknown", description="Preferred product category")
    skin_type: Optional[str] = Field(default="Unknown", description="Skin type (Normal, Dry, Oily, etc.)")
    marketing_consent: Optional[str] = Field(default="Unknown", description="Marketing opt-in (Yes or No)")

    recency_days: Optional[float] = Field(default=90.0, ge=0, description="Days since last purchase")
    frequency_180d: Optional[int] = Field(default=0, ge=0, description="Orders in last 180 days")
    monetary_180d: Optional[float] = Field(default=0.0, ge=0, description="Spend (INR) in last 180 days")
    return_rate_180d: Optional[float] = Field(default=0.0, ge=0, le=1, description="Return rate (0–1)")
    avg_discount_pct_180d: Optional[float] = Field(default=0.0, ge=0, le=1, description="Avg discount fraction")
    avg_rating_180d: Optional[float] = Field(default=None, ge=1, le=5, description="Avg order rating (1–5)")
    category_diversity_180d: Optional[int] = Field(default=0, ge=0, description="Distinct categories purchased")

    ticket_count_90d: Optional[int] = Field(default=0, ge=0, description="Support tickets in last 90 days")
    negative_ticket_rate_90d: Optional[float] = Field(
        default=0.0, ge=0, le=1, description="Share of negative-sentiment tickets"
    )
    avg_resolution_hours_90d: Optional[float] = Field(
        default=0.0, ge=0, description="Avg ticket resolution hours"
    )
    days_since_signup: Optional[int] = Field(default=30, ge=0, description="Days since signup")

    sessions_30d: Optional[int] = Field(default=0, ge=0, description="Web/app sessions (30d)")
    product_views_30d: Optional[int] = Field(default=0, ge=0, description="Product page views (30d)")
    cart_adds_30d: Optional[int] = Field(default=0, ge=0, description="Cart additions (30d)")
    wishlist_adds_30d: Optional[int] = Field(default=0, ge=0, description="Wishlist additions (30d)")
    abandoned_carts_30d: Optional[int] = Field(default=0, ge=0, description="Abandoned carts (30d)")
    email_opens_30d: Optional[int] = Field(default=0, ge=0, description="Marketing emails opened (30d)")
    campaign_clicks_30d: Optional[int] = Field(default=0, ge=0, description="Campaign link clicks (30d)")
    last_visit_days_ago: Optional[int] = Field(default=30, ge=0, description="Days since last site visit")

    model_config = {
        "json_schema_extra": {
            "example": {
                "city_tier": "Tier 1",
                "age_group": "25-34",
                "acquisition_channel": "Instagram",
                "loyalty_tier": "Silver",
                "preferred_category": "Skin Care",
                "skin_type": "Combination",
                "marketing_consent": "Yes",
                "recency_days": 42.0,
                "frequency_180d": 3,
                "monetary_180d": 1850.50,
                "return_rate_180d": 0.0,
                "avg_discount_pct_180d": 0.15,
                "avg_rating_180d": 4.5,
                "category_diversity_180d": 2,
                "ticket_count_90d": 1,
                "negative_ticket_rate_90d": 0.0,
                "avg_resolution_hours_90d": 2.5,
                "days_since_signup": 120,
                "sessions_30d": 12,
                "product_views_30d": 45,
                "cart_adds_30d": 3,
                "wishlist_adds_30d": 1,
                "abandoned_carts_30d": 0,
                "email_opens_30d": 4,
                "campaign_clicks_30d": 1,
                "last_visit_days_ago": 2,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(description="Predicted churn probability (0.0–1.0)")
    predicted_class: int = Field(description="1 = churn risk, 0 = active (threshold from Part 3)")
    risk_explanation: str = Field(description="CRM-oriented explanation and recommended action")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


def load_decision_threshold() -> float:
    if not METRICS_PATH.exists():
        return DEFAULT_DECISION_THRESHOLD
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    return float(metrics.get("selected_decision_threshold", DEFAULT_DECISION_THRESHOLD))


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Copy model.pkl from Part 3 into part-4/model/ before starting the API."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def features_to_dataframe(features: CustomerFeatures) -> pd.DataFrame:
    row = features.model_dump()
    return pd.DataFrame([row])[FEATURE_COLS]


def generate_risk_explanation(prob: float, features: CustomerFeatures, threshold: float) -> str:
    if prob >= threshold:
        reasons = []
        if features.recency_days is not None and features.recency_days >= 60:
            reasons.append(f"prolonged purchasing silence of {int(features.recency_days)} days")
        if (features.ticket_count_90d or 0) >= 2 or (features.negative_ticket_rate_90d or 0) > 0.5:
            reasons.append(
                "recent customer support friction (unresolved complaints or negative sentiment)"
            )
        if (features.abandoned_carts_30d or 0) >= 2:
            reasons.append(
                f"significant cart abandonments ({features.abandoned_carts_30d}) in the last month"
            )
        if (features.last_visit_days_ago or 0) >= 20:
            reasons.append(
                f"digital disengagement (last visit {features.last_visit_days_ago} days ago)"
            )
        if features.frequency_180d == 1 and (features.recency_days or 0) >= 40:
            reasons.append("single purchase without repeat buying habit")

        reason_str = ", ".join(reasons) if reasons else "declining engagement and transactional inactivity"

        if features.loyalty_tier in ("Gold", "Silver", "Platinum"):
            action = (
                "VIP advocacy: prioritize support outreach and loyalty rewards; "
                "avoid blanket discounting."
            )
        else:
            action = "Standard retention: trigger the ₹30 welcome-back offer via approved channels."

        return (
            f"HIGH RISK (Score: {prob:.1%}). Primary drivers: {reason_str}. "
            f"Recommended CRM action: {action}"
        )

    safeguards = []
    if features.loyalty_tier in ("Gold", "Silver", "Platinum"):
        safeguards.append(f"active {features.loyalty_tier} loyalty member")
    if features.recency_days is not None and features.recency_days <= 15:
        safeguards.append(f"recent purchase ({int(features.recency_days)} days ago)")
    if (features.sessions_30d or 0) >= 8:
        safeguards.append("strong recent web/app activity")

    safeguard_str = " and ".join(safeguards) if safeguards else "stable recent activity signals"
    return (
        f"LOW RISK (Score: {prob:.1%}). Supported by {safeguard_str}. "
        "Recommended CRM action: standard engagement; suppress aggressive discounting."
    )


def predict_one(features: CustomerFeatures) -> PredictionResponse:
    input_df = features_to_dataframe(features)
    prob = float(model_pipeline.predict_proba(input_df)[0, 1])
    predicted_class = 1 if prob >= decision_threshold else 0
    explanation = generate_risk_explanation(prob, features, decision_threshold)
    return PredictionResponse(
        churn_probability=prob,
        predicted_class=predicted_class,
        risk_explanation=explanation,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, decision_threshold
    decision_threshold = load_decision_threshold()
    model_pipeline = load_model()
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Decision threshold: {decision_threshold}")
    yield
    print("Shutting down API...")


app = FastAPI(
    title="D2C Customer Churn Scoring API",
    description="Churn scoring service for internal CRM retention workflows (Part 4 capstone).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Churn model pipeline is not loaded.")
    return {
        "status": "Healthy",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
        "decision_threshold": decision_threshold,
        "features_expected": len(FEATURE_COLS),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        return predict_one(features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}") from e


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(payload: List[CustomerFeatures]):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if not payload:
        return BatchPredictionResponse(predictions=[])

    try:
        rows = [f.model_dump() for f in payload]
        input_df = pd.DataFrame(rows)[FEATURE_COLS]
        probs = model_pipeline.predict_proba(input_df)[:, 1]
        predictions = []
        for i, prob in enumerate(probs):
            p_val = float(prob)
            predictions.append(
                PredictionResponse(
                    churn_probability=p_val,
                    predicted_class=1 if p_val >= decision_threshold else 0,
                    risk_explanation=generate_risk_explanation(
                        p_val, payload[i], decision_threshold
                    ),
                )
            )
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch inference failed: {e}") from e
