# mp1-ticket-sentiment

**BITS Pilani WILP · PGAIML · PCAM ZC412 Machine Learning Engineering**  
**Mini-Project-1 · Flavor C — Support Ticket / Review Sentiment Classifier**

Taxila-aligned end-to-end NLP pipeline (QuickBite + M2 ELS patterns adapted for text):

raw tickets → Pandera validation → shared features + SQLite feature store → MLflow experiments → FastAPI → prediction logs / concept-drift checks.

## Process status

See **[reports/PROCESS_UPDATES.md](reports/PROCESS_UPDATES.md)** for progress. M2–M5 pipeline, DVC, DistilBERT comparison, API/drift evidence, and submission reports are complete; remaining team actions are Taxila group upload and demo recording.

## Architecture (Taxila / Teams-style)

```text
data/prepare_dataset.py
        |
        v
data/raw/tickets.csv
        |
        v
validation/validate_data.py   (Pandera; Teams validation/ layout)
        |
        v
features/build_features.py    (shared build_feature_row for train + serve)
        |
        +--> feature_store/feature_store.db
        +--> model_store/feature_columns.json
        |
        v
training/train.py             (MLflow multi-run)
        |
        +--> model_store/sentiment_model.joblib
        |
        v
serving/                      (Lab4: api + model_loader + inference_schema)
        |
        +--> ui/app.py        (Streamlit calls /predict only)
        |
        v
monitoring/check_drift.py     (mean-shift + PSI)
```

## Repository layout

```text
mp1-ticket-sentiment/
├── data/                    # prepare + immutable raw
├── validation/              # Pandera (Teams layout)
├── features/                # shared train/serve feature logic
├── feature_store/           # SQLite offline store
├── training/
├── serving/                 # api.py, model_loader.py, inference_schema.py
├── monitoring/
├── model_store/             # joblib + feature_columns.json
├── ui/app.py                # Streamlit (Lab4 pattern)
├── tools/verify_setup.py
├── scripts/                 # Option A runners
├── docker/Dockerfile
├── docker-compose.yml       # mlflow | trainer | api | monitor
└── reports/
```

## Usage (local + Colab)

Full instructions (Option A / Option B, local and Google Colab): **[USAGE.md](USAGE.md)**

There are **two execution paths**:

| Path | When to use | Entry |
|------|-------------|--------|
| **Option A** | Fastest end-to-end run | `scripts/run_m2_pipeline.py` then `scripts/run_train.py` |
| **Option B** | Step-by-step / debugging | prepare → validate → features → train → serve |

### Local — Option A (quick, recommended)

```bash
python tools/verify_setup.py
python scripts/run_m2_pipeline.py
python scripts/run_train.py
uvicorn serving.api:app --reload --port 8000
# optional UI (separate terminal): streamlit run ui/app.py
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

### Local — Option B (step by step)

```bash
python data/prepare_dataset.py --source support_tickets
python validation/validate_data.py
python features/build_features.py
python training/train.py
uvicorn serving.api:app --reload --port 8000
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

Swagger: http://127.0.0.1:8000/docs

### Data source switch (before Option A or B)

```bash
python data/prepare_dataset.py --source amazon
python data/prepare_dataset.py --source yelp
python data/prepare_dataset.py --source sentiment140
python data/prepare_dataset.py --source support_tickets   # default
python data/prepare_dataset.py --source all
```

(`--source synthetic` is an alias for `support_tickets`.) Drop-in layout: `data/external/kaggle/README.md`.

### Colab — Option A (easiest)

```python
!git clone https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment.git
%cd mp1-ticket-sentiment
!pip install -q -r requirements.txt
!python scripts/run_m2_pipeline.py
!python scripts/run_train.py
```

### Colab — Option B (step by step)

```python
!python data/prepare_dataset.py --source support_tickets
!python data/validate.py
!python features/build_features.py
!python training/train.py
```

Use `fastapi.testclient.TestClient` for `/predict` on Colab (details in USAGE.md). T4 GPU is not required for the current classical ML baseline.

## Dataset versioning (DVC)

All modes (`amazon`, `yelp`, `sentiment140`, `support_tickets`, `all`) are snapshotted under `data/versions/` and tracked with DVC. Details: **[docs/DVC.md](docs/DVC.md)**.

```bash
dvc repro
dvc push
git tag week1-data-v1
```

## Docker (M4 packaging)

```bash
docker build -f docker/Dockerfile -t mp1-ticket-sentiment .
docker run --rm -p 8000:8000 mp1-ticket-sentiment
```

API docs: http://127.0.0.1:8000/docs — sample curls in **[reports/api_examples.md](reports/api_examples.md)**.

## Transformer comparison (optional M3)

Classical training is the default served path. For the brief’s classical-vs-transformer comparison:

```bash
pip install -r requirements-transformer.txt
python training/train_transformer.py
```

DistilBERT metrics land in `model_store/transformer_decision.json` and do **not** replace `sentiment_model.joblib`. See **[reports/model_comparison.md](reports/model_comparison.md)**.

## Design decisions (for report)

1. **Shared feature logic** in `features/text_utils.py` used by feature build and API (avoids training-serving skew).
2. **TF-IDF fit on train split only** inside `training/train.py` (leakage-safe, M2 classroom lesson).
3. **Offline SQLite feature store** for cleaned text + numeric/channel features (Taxila M2 pattern).
4. **Pandera** for schema + statistical validation before features.
5. **MLflow runs** compare LogReg / LinearSVC / DistilBERT; best classical macro-F1 is promoted to `model_store/sentiment_model.joblib`.
6. **Multi-source ingest**: Amazon, Yelp, Sentiment140, support tickets, or all — via `configs/data_source.yaml`.

## Submission package

- Final report: [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md)
- Demo outline: [reports/DEMO_SCRIPT.md](reports/DEMO_SCRIPT.md)
- Process tracker: [reports/PROCESS_UPDATES.md](reports/PROCESS_UPDATES.md)

## Team

- Org: https://github.com/bits-pgaiml-mle
- Repo: https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment
- Members / roles: TBD (fill before Taxila upload)
