# Usage Guide — mp1-ticket-sentiment (Flavor C)

End-to-end **support ticket / review sentiment** classifier.  
Works on **local machine (CPU)** and **Google Colab (CPU is enough; T4 GPU not required** unless you later fine-tune a transformer).

---

## 1. Local usage

### 1.1 Setup (once)

```powershell
cd "D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
cd mp1-ticket-sentiment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the **repo root**.

### Execution options

| Path | Best for | Commands |
|------|----------|----------|
| **Option A** | Quick full run | `python scripts/run_m2_pipeline.py` then `python scripts/run_train.py` |
| **Option B** | Learn / debug each stage | prepare → validate → features → train → serve → drift |

### 1.1b Data source: amazon / yelp / sentiment140 / support_tickets / all

```powershell
python data/prepare_dataset.py --source amazon
python data/prepare_dataset.py --source yelp
python data/prepare_dataset.py --source sentiment140
python data/prepare_dataset.py --source support_tickets
python data/prepare_dataset.py --source all
```

Or set `configs/data_source.yaml` (default: `support_tickets`). Drop CSVs under `data/external/kaggle/{amazon,yelp,sentiment140,support_tickets}/` — see that folder’s README. Option A’s `run_m2_pipeline.py` calls `prepare_dataset.py`. `--source synthetic` aliases to `support_tickets`.

### 1.2 Option A — easiest (recommended)

```powershell
python tools/verify_setup.py
python scripts/run_m2_pipeline.py
python scripts/run_train.py
```

(`scripts/run_pipeline.py` is an alias for `scripts/run_train.py`.)

**Terminal 1 — API**

```powershell
uvicorn serving.api:app --reload --port 8000
```

**Terminal 2 — UI (optional, Lab4)**

```powershell
streamlit run ui/app.py
```

**Terminal 3 — predict + drift**

```powershell
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"Support resolved my issue quickly, thank you!\",\"channel\":\"chat\"}"
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

- Swagger: http://127.0.0.1:8000/docs  
- MLflow UI (optional): `mlflow ui` → http://127.0.0.1:5000  
- Prediction logs: `monitoring/predictions.db` and/or `monitoring/predictions.jsonl` (see § Prediction log backends)

### 1.3 Option B — step by step

```powershell
python data/prepare_dataset.py --source support_tickets
python validation/validate_data.py
python features/build_features.py
python training/train.py
uvicorn serving.api:app --reload --port 8000
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

| Step | Entry file |
|------|------------|
| Prepare tickets | `data/prepare_dataset.py` (`--source amazon\|yelp\|sentiment140\|support_tickets\|all`) |
| Validate | `validation/validate_data.py` (shim: `data/validate.py`) |
| Features | `features/build_features.py` → `feature_store/feature_store.db` |
| Train | `training/train.py` |
| Serve | `serving/api.py` via uvicorn |
| UI | `streamlit run ui/app.py` |
| Drift simulate | `monitoring/simulate_concept_drift.py` |
| Drift check | `monitoring/check_drift.py` (mean-shift + PSI) |

---

## 2. Google Colab usage

### 2.1 Runtime

1. Open [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **CPU** (T4 only if you add transformer fine-tuning later)

### 2.2 Option A — easiest

```python
!git clone https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment.git
%cd mp1-ticket-sentiment
!pip install -q -r requirements.txt

!python scripts/run_m2_pipeline.py
!python scripts/run_train.py
```

Predict with TestClient (recommended on Colab):

```python
from fastapi.testclient import TestClient
from serving.api import app

client = TestClient(app)
print(client.get("/health").json())
print(client.post("/predict", json={
    "text": "Support resolved my issue quickly, thank you!",
    "channel": "chat",
}).json())
print(client.post("/predict", json={
    "text": "Terrible experience, refund still pending after one week.",
    "channel": "email",
}).json())
```

Drift:

```python
!python monitoring/simulate_concept_drift.py
!python monitoring/check_drift.py
```

### 2.3 Option B — step by step on Colab

```python
!python data/prepare_dataset.py --source support_tickets
!python validation/validate_data.py
!python features/build_features.py
!python training/train.py
```

Then use the TestClient cell for `/predict`.

### 2.4 Optional: uvicorn on Colab

Possible with background + ngrok; **not required**. Prefer TestClient for demos.

---

## 3. What each option does

| Stage | Option A | Option B |
|-------|----------|----------|
| Setup check | `tools/verify_setup.py` | same |
| M2 data | `scripts/run_m2_pipeline.py` | prepare → `validation/validate_data.py` → features |
| M3 train | `scripts/run_train.py` (includes M2) | `training/train.py` |
| M4 serve | uvicorn / TestClient / Streamlit UI | same |
| M5 drift | simulate + check_drift (mean-shift + PSI) | same |

---

## Docker

Requires a trained `model_store/sentiment_model.joblib` (run Option A first).

```bash
docker build -f docker/Dockerfile -t mp1-ticket-sentiment .
docker run --rm -p 8000:8000 mp1-ticket-sentiment
```

VaayuGrid-style multi-service (mlflow | trainer | api | monitor):

```bash
docker compose up -d mlflow
docker compose run --rm trainer
docker compose up -d api
docker compose run --rm monitor
```

Then call `/health` and `/predict` as in [reports/api_examples.md](reports/api_examples.md).

### Streamlit UI (Teams Lab4)

With the API running:

```bash
streamlit run ui/app.py
```

### Prediction log backends (SQLite / JSONL)

Set in `configs/config.yaml`:

```yaml
monitoring:
  log_backend: both          # sqlite | jsonl | both
  drift_read_backend: auto   # auto | sqlite | jsonl
  predictions_db: monitoring/predictions.db
  predictions_jsonl: monitoring/predictions.jsonl
```

Or override at runtime:

```powershell
$env:PREDICTION_LOG_BACKEND="jsonl"   # or sqlite / both
uvicorn serving.api:app --port 8000
```

`monitoring/check_drift.py` reads the configured backend(s). When both have data, `auto` prefers SQLite so the same event is not double-counted.

---

## 5. Optional DistilBERT comparison

```bash
pip install -r requirements-transformer.txt
python training/train_transformer.py
```

Does not change the served classical model. Metrics: `model_store/transformer_decision.json`.
