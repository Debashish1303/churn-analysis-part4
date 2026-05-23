# Part 4 — FastAPI Churn Scoring Service

Production-style API that serves the Part 3 XGBoost churn classifier for internal CRM workflows. Predictions use the saved pipeline at `model/model.pkl` and the business decision threshold from `model/metrics.json` (default **0.20**).

---

## Project Structure

```text
part-4/
├── app/
│   └── main.py           # FastAPI app (Pydantic validation, risk explanations)
├── model/
│   ├── model.pkl         # Champion pipeline from Part 3 (required)
│   └── metrics.json      # Validation metrics and decision threshold
├── tests/
│   └── test_api.py       # API tests (health, predict, batch, validation errors)
├── Dockerfile            # Container build specification
├── requirements.txt
├── main.py               # Optional uvicorn entry script
├── monitoring_plan.md    # Post-deployment monitoring and responsible use
└── README.md

```

---

## Run Locally with Uvicorn

All commands below assume your shell is in the **`part-4`** root directory.

### 1. Create and Activate a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Confirm Model Presence

```powershell
# PowerShell
Test-Path .\model\model.pkl

```

```bash
# Bash
test -f model/model.pkl && echo "model.pkl OK"

```

### 4. Start the API Service

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```

On startup, logs will indicate that the model has successfully loaded along with its active decision threshold (e.g., `0.2`).

---

## Service Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service and model load status |
| `POST` | `/predict` | Score one customer (26 Pydantic-validated features) |
| `POST` | `/batch_predict` | Score an array of multiple customer payloads |

### Verification Steps

1. **Health Check:** Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health). Expect `"status": "Healthy"` and `"model_loaded": true`.
2. **Interactive Docs:** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to view the auto-generated Swagger UI.
3. **Run Suite:** Execute `pytest -v` to spin up automated internal tests.

---

## Feature Input Specifications (Snapshot 2025-09-30)

The API strictly adheres to the definitions in `DATA_DICTIONARY.md`. It processes exactly **26 features** mapped directly to the modeling snapshot:

* **Categorical (7 fields):** `city_tier`, `age_group`, `acquisition_channel`, `loyalty_tier`, `preferred_category`, `skin_type`, `marketing_consent`.
* **Numeric RFM & Support (11 fields):** `recency_days`, `frequency_180d`, `monetary_180d`, `return_rate_180d`, `avg_discount_pct_180d`, `avg_rating_180d`, `category_diversity_180d`, `ticket_count_90d`, `negative_ticket_rate_90d`, `avg_resolution_hours_90d`, `days_since_signup`.
* **Web Traffic Metrics (8 fields):** `sessions_30d`, `product_views_30d`, `cart_adds_30d`, `wishlist_adds_30d`, `abandoned_carts_30d`, `email_opens_30d`, `campaign_clicks_30d`, `last_visit_days_ago`.

⚠️ **Critical Data Leakage Rule:** Production feature streams must *only* pass activity timestamps up to the snapshot date (`2025-09-30`). Post-snapshot transactional activity (`order_date > 2025-09-30`) must **never** be supplied to the prediction payload.

---

## API Request/Response Examples

### Sample Single Prediction Payload (`POST /predict`)

```json
{
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
  "last_visit_days_ago": 2
}

```

### Expected Response Format

```json
{
  "churn_probability": 0.42,
  "predicted_class": 1,
  "risk_explanation": "HIGH RISK (Score: 42.0%). Primary drivers: digital disengagement. Recommended CRM action: Standard retention: trigger the ₹30 welcome-back offer via approved channels."
}

```

*Note: Passing incorrect data types or out-of-range constraints (e.g., negative frequency counts) triggers a native `422 Unprocessable Entity` response validation block.*

---

## Docker Integration

To build and serve the application within a reproducible, isolated container:

```bash
# Build the image from root
docker build -t churn-scoring-service .

# Run the container
docker run -p 8000:8000 churn-scoring-service

```

The Docker image bakes in `model/model.pkl` at build time. Be sure to rebuild your container whenever a fresh, retrained model artifact is exported.

---

## Post-Deployment Monitoring & Responsible Use

Before exposing the service endpoints to live CRM triggers, please read **`monitoring_plan.md`**. It covers crucial guardrails for production including:

* **Drift Alerts:** Procedures for tracking Population Stability Index (PSI) on key drivers like `recency_days` or `ticket_count_90d`.
* **Retraining Triggers:** Hard thresholds for automated scheduling (90-day intervals) or performance flags (when rolling recall drops below 0.60).
* **Ethical Guardrails:** Prohibitions against dynamic pricing, demographic discrimination (such as altering treatments by `age_group` or `city_tier`), and strict enforcement of `marketing_consent == "No"` filters.
