# mp1-ticket-sentiment

**BITS Pilani WILP · PGAIML · PCAM ZC412 Machine Learning Engineering**  
**Mini-Project-1 · Flavor C — Support Ticket / Review Sentiment Classifier**

Taxila-aligned end-to-end NLP pipeline (QuickBite + M2 ELS patterns adapted for text):

raw tickets → Pandera validation → shared features + SQLite feature store → MLflow experiments → FastAPI → prediction logs / concept-drift checks.

## Architecture (Taxila-style)

```text
data/raw/tickets.csv          (immutable raw)
        |
        v
data/validate.py              (Pandera schema + statistical checks)
        |
        v
features/build_features.py    (shared clean_text + channel flags)
        |
        +--> data/feature_store.db      (offline feature store)
        +--> data/feature_schema.json   (feature contract)
        |
        v
training/train.py             (MLflow: LogReg C=1/C=10 + LinearSVC)
        |
        +--> model_store/sentiment_model.joblib
        |
        v
serving/api.py                (FastAPI + Pydantic + prediction logging)
        |
        v
monitoring/check_drift.py     (shift checks + retrain trigger notes)
```

## Repository layout

```text
mp1-ticket-sentiment/
├── data/
│   ├── raw/                 # immutable raw tickets
│   ├── generate_data.py
│   ├── validate.py          # Pandera
│   ├── feature_store.db     # generated
│   └── feature_schema.json  # generated
├── features/
│   ├── text_utils.py        # shared train/serve cleaning
│   └── build_features.py
├── training/train.py
├── serving/api.py
├── monitoring/
│   ├── logger.py
│   ├── check_drift.py
│   └── simulate_concept_drift.py
├── model_store/
├── reports/
├── configs/config.yaml
└── scripts/run_pipeline.py
```

## Setup

```bash
cd mp1-ticket-sentiment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run (aligned with Taxila tutorial flow)

```bash
python scripts/run_pipeline.py
```

Or step by step:

```bash
python data/generate_data.py
python data/validate.py
python features/build_features.py
python training/train.py
uvicorn serving.api:app --reload --port 8000
```

Sample predict:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Support resolved my issue quickly, thank you!\",\"channel\":\"chat\"}"
```

Drift simulation:

```bash
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

MLflow UI:

```bash
mlflow ui
```

## Design decisions (for report)

1. **Shared feature logic** in `features/text_utils.py` used by feature build and API (avoids training-serving skew).
2. **TF-IDF fit on train split only** inside `training/train.py` (leakage-safe, M2 classroom lesson).
3. **Offline SQLite feature store** for cleaned text + numeric/channel features (Taxila M2 pattern).
4. **Pandera** for schema + statistical validation before features.
5. **Three MLflow runs** compared; best macro-F1 promoted to `model_store/sentiment_model.joblib`.

## Process updates

See [`reports/PROCESS_UPDATES.md`](reports/PROCESS_UPDATES.md) for:

- progress completed till now
- M1 foundation checklist
- pending items to close **Week 1 / M2** (and M1 polish)

## Team

- Org: https://github.com/bits-pgaiml-mle
- Repo: https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment
