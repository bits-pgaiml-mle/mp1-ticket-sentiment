# Architecture — Flavor C Ticket / Review Sentiment

**Project:** `mp1-ticket-sentiment`  
**Course:** BITS Pilani WILP · PCAM ZC412 · Mini-Project-1 Flavor C  
**Goal:** End-to-end NLP ML engineering pipeline (data → validate → features → train → serve → monitor), not a notebook-only classifier.

Related docs: [README](../README.md) · [USAGE](../USAGE.md) · [DVC](DVC.md) · [FINAL_REPORT](../reports/FINAL_REPORT.md) · [dataset comparison](../reports/datasets/COMPARISON.md)

---

## 1. System context

```text
                     ┌──────────────────────┐
  Analyst / Demo UI  │  Streamlit ui/app.py │
                     └──────────┬───────────┘
                                │ HTTP POST /predict
                                v
  Client / curl / Swagger ──► FastAPI serving/api.py
                                │
                                ├─► model_store/*.joblib (+ feature_columns.json)
                                ├─► shared features.build_feature_row
                                └─► monitoring logs (SQLite and/or JSONL)
                                              │
                                              v
                                    monitoring/check_drift.py
                                    (mean-shift + PSI)
```

**Classes:** `negative` | `neutral` | `positive`  
**Default active source:** `support_tickets` (`configs/data_source.yaml`)

---

## 2. Batch ML pipeline (M2–M3)

```text
configs/data_source.yaml          configs/config.yaml
        │                                  │
        v                                  │
data/prepare_dataset.py ──► data/raw/tickets.csv
        │                                  │
        v                                  │
validation/validate_data.py (Pandera)      │
        │                                  │
        v                                  │
features/build_features.py ◄───────────────┘
        │
        ├──► feature_store/feature_store.db   (table ticket_features)
        └──► model_store/feature_columns.json
                │
                v
        training/train.py  ── MLflow experiment: ticket_sentiment_prediction
                │              LogReg (C=1, C=10), LinearSVC
                │
                ├──► model_store/sentiment_model.joblib   ◄── served artifact
                ├──► model_store/best_model_decision.json
                └──► (optional) training/train_transformer.py → DistilBERT
                       comparison only; not promoted to serve
```

### Data sources

| `--source` | Role | Notes |
|------------|------|--------|
| `support_tickets` | Default | Synthetic generator if CSV missing (~1500 rows) |
| `amazon` / `yelp` / `sentiment140` | Alternate review/tweet extracts | Local CSVs under `data/external/kaggle/` |
| `all` | Mix | Caps per source from `data_source.yaml` |
| `synthetic` | Alias | Maps to `support_tickets` |

Prepare normalizes text + label + channel into a single schema before validation.

### Features (train = serve)

Shared builder: `features.build_features.build_feature_row` / `features.text_utils`

| Feature | Description |
|---------|-------------|
| `text_clean` | Normalized ticket/review text |
| `text_len`, `word_count` | Length signals |
| `channel_email`, `channel_chat`, `channel_app` | One-hot channel flags |

Training wraps text in a sklearn `Pipeline` (TF-IDF → classifier). Online path builds the same tabular row, then feeds the saved pipeline.

### Model selection

- Classical runs logged to MLflow; best by macro-F1 written to `model_store/best_model_decision.json`.
- DistilBERT is an optional accuracy comparison (`requirements-transformer.txt`); production serve stays classical for latency/size.

---

## 3. Online serving (M4 / Lab4)

| Module | Responsibility |
|--------|----------------|
| `serving/inference_schema.py` | Pydantic request/response (`text`, `channel` → `label`, `confidence`, `model_version`) |
| `serving/model_loader.py` | Load joblib bundle + feature column schema + version |
| `serving/api.py` | FastAPI: `GET /health`, `POST /predict` |
| `ui/app.py` | Streamlit client; calls `/predict` only (no local model) |

**Request contract**

- `text`: non-empty string  
- `channel`: one of `email` | `chat` | `app` (invalid → HTTP 422)

**Response**

- Predicted label, confidence (proba or softmax of decision scores), model version string from config.

---

## 4. Monitoring & drift (M5)

```text
/predict
   │
   v
monitoring/logger.py
   ├── sqlite → monitoring/predictions.db
   └── jsonl  → monitoring/predictions.jsonl
         (config: monitoring.log_backend = sqlite | jsonl | both;
          override: PREDICTION_LOG_BACKEND)

monitoring/simulate_concept_drift.py  → inject shifted traffic into logs
monitoring/check_drift.py
   ├── mean-shift on feature aggregates vs training reference
   └── PSI (Population Stability Index); exit code 1 on PSI alert
```

Thresholds live in `configs/config.yaml` (`drift_shift_threshold`, `psi_alert`). Drift read backend can prefer SQLite, JSONL, or auto when both exist.

---

## 5. Runtime & deployment views

### Local (developer)

```text
Option A:  scripts/run_m2_pipeline.py → scripts/run_train.py → uvicorn
Option B:  prepare → validate → features → train → serve → drift
Verify:    tools/verify_setup.py
```

### Docker Compose

| Service | Role | Ports / notes |
|---------|------|----------------|
| `mlflow` | Tracking server | `5000` |
| `trainer` | `scripts/run_train.py` | Shares `model_store` / `feature_store` volumes |
| `api` | FastAPI | `8000`; dual prediction logs |
| `monitor` | One-shot `check_drift.py` | Depends on API / shared monitoring volume |

Image: `docker/Dockerfile` → `mp1-ticket-sentiment:1.0.0`.

### Data versioning

DVC stage `snapshot_datasets` (`dvc.yaml`) materializes `data/versions/` and active `data/raw/tickets.csv`. Details: [docs/DVC.md](DVC.md).

---

## 6. Configuration & artifact map

| Path | Purpose |
|------|---------|
| `configs/config.yaml` | Paths, TF-IDF/train params, serve version, monitoring backends/thresholds |
| `configs/data_source.yaml` | Active source + per-source column maps / row caps |
| `feature_store/feature_store.db` | Offline feature table |
| `model_store/sentiment_model.joblib` | Served classical bundle |
| `model_store/feature_columns.json` | Schema contract for train/serve |
| `model_store/best_model_decision.json` | Promotion record |
| `monitoring/predictions.db` / `.jsonl` | Online prediction telemetry |
| `reports/` | Submission evidence, drift, API examples, per-source dataset reports |

---

## 7. Design principles

1. **Same features online and offline** — one `build_feature_row` used by feature store build and `/predict`.
2. **Validate before train** — Pandera gate on canonical `tickets.csv`.
3. **Track then promote** — MLflow for experiments; only classical winner is served.
4. **Observable serve** — every prediction is logged for drift; dual backends for lab flexibility.
5. **Composable sources** — switch datasets without changing train/serve code; reports under `reports/datasets/`.
6. **Lab / Taxila alignment** — `validation/`, Lab4 serving split, compose + PSI mirror course demos.

---

## 8. Out of scope / non-goals

- DistilBERT is not the production serving path.
- Real Kaggle dumps are optional drop-ins; repo ships generators/demo extracts for reproducibility.
- Multi-tenant auth, autoscaling, and managed feature stores are outside Flavor C scope.

---

## 9. Reproduce architecture evidence

```bash
python tools/verify_setup.py
python scripts/run_m2_pipeline.py
python scripts/run_train.py
uvicorn serving.api:app --port 8000
# optional: streamlit run ui/app.py
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
python scripts/generate_dataset_reports.py   # all --source modes
```
