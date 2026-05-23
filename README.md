# Part 4 — FastAPI Churn Scoring Service

Production-style API that serves the Part 3 XGBoost churn classifier for internal CRM workflows. Predictions use the saved pipeline at `model/model.pkl` and the business decision threshold from `model/metrics.json` (default **0.20**).

---

## Project structure

```text
part-4/
├── app/
│   └── main.py           # FastAPI app (Pydantic validation, risk explanations)
├── model/
│   ├── model.pkl         # Champion pipeline from Part 3 (required)
│   └── metrics.json      # Validation metrics and decision threshold
├── tests/
│   └── test_api.py       # API tests (health, predict, batch, validation errors)
├── Dockerfile            # Optional container build
├── requirements.txt
├── main.py               # Optional uvicorn entry script
├── monitoring_plan.md    # Post-deployment monitoring and responsible use
└── README.md
```

**Model artifact:** If `model/model.pkl` is missing, copy it from Part 3 (e.g. `part-3/model.pkl` or the notebook output) into `part-4/model/model.pkl`. Optionally copy `metrics.json` alongside it.

---

## Run locally with uvicorn

All commands below assume your shell is in the **`part-4`** directory.

### 1. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
cd c:\Users\agraw\OneDrive\Desktop\masai_capstone\part-4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
cd path/to/masai_capstone/part-4
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Confirm the model file is present

```powershell
# PowerShell
Test-Path .\model\model.pkl
```

```bash
# bash
test -f model/model.pkl && echo "model.pkl OK"
```

If this fails, place `model.pkl` from Part 3 into `part-4/model/` before starting the server.

### 4. Start the API with uvicorn

From `part-4/` (recommended):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Alternative (uses `main.py` wrapper):

```bash
python main.py
```

On startup you should see logs indicating the model loaded from `part-4/model/model.pkl` and the decision threshold (e.g. `0.2`).

### 5. Verify the service

| Step | Action |
|------|--------|
| Health | Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) — expect `"status": "Healthy"` and `"model_loaded": true` |
| Docs | Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI |
| Single predict | `POST /predict` with a JSON body matching `CustomerFeatures` (see sample below) |
| Batch | `POST /batch_predict` with a JSON array of customer payloads |

**Example health response:**

```json
{
  "status": "Healthy",
  "model_loaded": true,
  "model_path": ".../part-4/model/model.pkl",
  "decision_threshold": 0.2,
  "features_expected": 26
}
```

### 6. Run automated tests

With the venv active and cwd = `part-4/`:

```bash
pytest -v
```

Tests spin up the app via `TestClient` and expect the real `model/model.pkl` to load at startup.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service and model load status |
| `POST` | `/predict` | Score one customer (26 features; optional fields use defaults) |
| `POST` | `/batch_predict` | Score a list of customers |

**Decision rule:** `predicted_class = 1` when `churn_probability >= 0.20` (from Part 3 economics: ₹30 offer vs expected retention value).

### Sample `POST /predict` request

```json
{
  "city_tier": "Tier 1",
  "age_group": "25-34",
  "acquisition_channel": "Instagram",
  "loyalty_tier": "Silver",
  "preferred_category": "Skin Care",
  "skin_type": "Combination",
  "marketing_consent": "Yes",
  "recency_days": 110.0,
  "frequency_180d": 1,
  "monetary_180d": 350.0,
  "return_rate_180d": 0.0,
  "avg_discount_pct_180d": 0.25,
  "avg_rating_180d": 3.0,
  "category_diversity_180d": 1,
  "ticket_count_90d": 2,
  "negative_ticket_rate_90d": 1.0,
  "avg_resolution_hours_90d": 24.5,
  "days_since_signup": 120,
  "sessions_30d": 2,
  "product_views_30d": 5,
  "cart_adds_30d": 0,
  "wishlist_adds_30d": 0,
  "abandoned_carts_30d": 3,
  "email_opens_30d": 1,
  "campaign_clicks_30d": 0,
  "last_visit_days_ago": 15
}
```

### Sample response

```json
{
  "churn_probability": 0.42,
  "predicted_class": 1,
  "risk_explanation": "HIGH RISK (Score: 42.0%). Primary drivers: ..."
}
```

Invalid types (e.g. string for `recency_days`) return **422** from Pydantic validation.

### Sample `POST /batch_predict`

```json
[
  {
    "loyalty_tier": "Gold",
    "recency_days": 4.0,
    "frequency_180d": 4,
    "monetary_180d": 3200.0,
    "sessions_30d": 15
  },
  {
    "loyalty_tier": "None",
    "recency_days": 85.0,
    "frequency_180d": 1,
    "monetary_180d": 150.0,
    "ticket_count_90d": 2,
    "negative_ticket_rate_90d": 1.0
  }
]
```

---

## Docker (optional)

From `part-4/`:

```bash
docker build -t churn-scoring-service .
docker run -p 8000:8000 churn-scoring-service
```

The image includes `model/model.pkl` baked in at build time; rebuild after updating the artifact.

---

## Feature inputs (snapshot 2025-09-30)

See `DATA_DICTIONARY.md` in the capstone root. The API expects the same 26 features as Part 3: RFM and support metrics, 30-day web activity, and customer profile fields. Do not send post-snapshot order data as features.

---

## Responsible use

See `monitoring_plan.md` for drift monitoring, retraining triggers, and CRM guardrails (what to do and what not to do with scores).
