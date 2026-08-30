# mp1-ticket-sentiment

**BITS Pilani WILP · PGAIML · PCAM ZC412 Machine Learning Engineering**  
**Mini-Project-1 · Flavor C — Support Ticket / Review Sentiment Classifier**

Teams / Taxila-aligned end-to-end NLP pipeline:

raw tickets → Pandera validation → shared features + SQLite feature store → MLflow experiments → FastAPI (+ Streamlit UI) → SQLite/JSONL prediction logs → mean-shift + PSI drift checks.

## Process status

See **[reports/PROCESS_UPDATES.md](reports/PROCESS_UPDATES.md)**. Code/evidence for M1–M5 is complete. Remaining team actions: make repo public, record 10–15 min demo to Drive, fill Drive link in [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md), one member uploads to Taxila (≤10 MB).

## Architecture (Taxila / Teams-style)

Full write-up: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** (context, batch pipeline, Lab4 serve, drift, Compose, artifacts).

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
training/train.py             (MLflow: LogReg / LinearSVC)
training/train_transformer.py (optional DistilBERT comparison)
        |
        +--> model_store/sentiment_model.joblib   (served = best classical)
        |
        v
serving/                      (Lab4: api + model_loader + inference_schema)
        |
        +--> ui/app.py        (Streamlit calls /predict only)
        +--> monitoring logs  (SQLite and/or JSONL)
        |
        v
monitoring/check_drift.py     (mean-shift + PSI; exit 1 on PSI alert)
```

## Repository layout

```text
mp1-ticket-sentiment/
├── data/                    # prepare + immutable raw
├── validation/              # Pandera (Teams layout)
├── features/                # shared train/serve feature logic
├── feature_store/           # SQLite offline store
├── training/                # classical + optional transformer
├── serving/                 # api.py, model_loader.py, inference_schema.py
├── monitoring/              # logger (sqlite|jsonl|both), drift, simulate
├── model_store/             # joblib + feature_columns.json + decision JSONs
├── ui/app.py                # Streamlit (Lab4 pattern)
├── tools/verify_setup.py
├── scripts/                 # Option A runners + DVC snapshot
├── docker/Dockerfile
├── docker-compose.yml       # mlflow | trainer | api | monitor
└── reports/                 # FINAL_REPORT, DEMO_SCRIPT, evidence logs
```

## Usage (local + Colab)

Full instructions: **[USAGE.md](USAGE.md)**

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
# optional UI: streamlit run ui/app.py
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

### Data source switch

```bash
python data/prepare_dataset.py --source amazon
python data/prepare_dataset.py --source yelp
python data/prepare_dataset.py --source sentiment140
python data/prepare_dataset.py --source support_tickets   # default
python data/prepare_dataset.py --source all
```

(`--source synthetic` aliases to `support_tickets`.) Drop-in layout: `data/external/kaggle/README.md`.

Per-source train/validate profiles: [`reports/datasets/COMPARISON.md`](reports/datasets/COMPARISON.md). Regenerate:

```bash
python scripts/expand_demo_datasets.py   # local amazon/yelp/sentiment140 extracts
python scripts/generate_dataset_reports.py
```

### Colab — Option A

```python
!git clone https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment.git
%cd mp1-ticket-sentiment
!pip install -q -r requirements.txt
!python scripts/run_m2_pipeline.py
!python scripts/run_train.py
```

### Colab — Option B

```python
!python data/prepare_dataset.py --source support_tickets
!python validation/validate_data.py
!python features/build_features.py
!python training/train.py
```

Use `fastapi.testclient.TestClient` for `/predict` on Colab (details in USAGE.md).

## Dataset versioning (DVC)

Details: **[docs/DVC.md](docs/DVC.md)**.

```bash
dvc repro
dvc push
git tag week1-data-v1
```

## Docker + Compose

```bash
docker build -f docker/Dockerfile -t mp1-ticket-sentiment .
docker run --rm -p 8000:8000 mp1-ticket-sentiment
```

VaayuGrid-style stack:

```bash
docker compose up -d mlflow
docker compose run --rm trainer
docker compose up -d api
docker compose run --rm monitor
```

Sample curls: **[reports/api_examples.md](reports/api_examples.md)**.

## Prediction logs (SQLite / JSONL)

Default `monitoring.log_backend: both` in `configs/config.yaml` (`sqlite` | `jsonl` | `both`). Override with `PREDICTION_LOG_BACKEND`. Drift prefers one source when both exist (`drift_read_backend: auto`).

## Transformer comparison (optional M3)

```bash
pip install -r requirements-transformer.txt
python training/train_transformer.py
```

Does **not** replace the served classical model. See **[reports/model_comparison.md](reports/model_comparison.md)**.

## Design decisions (for report)

1. Shared `build_feature_row` / `text_utils` for train and serve (anti train–serve skew).
2. TF-IDF fit on train split only (leakage-safe).
3. Offline SQLite feature store + `model_store/feature_columns.json` (Teams pattern).
4. Pandera schema + statistical validation.
5. MLflow classical multi-run; DistilBERT comparison-only; serve LinearSVC.
6. Dual prediction logs (SQLite + JSONL) and PSI + mean-shift drift (Taxila M5/M6).
7. Lab4 serving split (`model_loader`, `inference_schema`) + optional Streamlit UI.

## Submission package

- Evaluator links (fill Drive URL): [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md)
- Demo outline (10–15 min voiceover): [reports/DEMO_SCRIPT.md](reports/DEMO_SCRIPT.md)
- Process tracker: [reports/PROCESS_UPDATES.md](reports/PROCESS_UPDATES.md)

## Team

- Org: https://github.com/bits-pgaiml-mle
- Repo: https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment (**make public** before Taxila upload)
- Members / roles: TBD (fill before Taxila upload)
