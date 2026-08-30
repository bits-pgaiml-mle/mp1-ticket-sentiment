# Final Report — Flavor C Support Ticket / Review Sentiment Classifier

**Course:** PCAM ZC412 Machine Learning Engineering  
**Group org:** [bits-pgaiml-mle](https://github.com/bits-pgaiml-mle)  
**Repository:** [mp1-ticket-sentiment](https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment)  
**Flavor:** C — NLP / text sentiment (support tickets + optional Amazon/Yelp/Sentiment140)

## 1. Problem

A customer-support / e-commerce platform needs to classify incoming ticket text as **negative / neutral / positive** so agents can prioritize urgent unhappy customers. The system must cover the full ML lifecycle: ingest → validate → features → experiments → REST serve → monitor drift → retrain design.

## 2. Architecture

```text
prepare_dataset → raw tickets → Pandera validate → shared features + SQLite store
        → MLflow (LogReg / LinearSVC / DistilBERT) → promote classical joblib
        → FastAPI /predict + prediction DB → drift check + retrain triggers
```

Taxila patterns used: immutable raw data, feature schema contract, offline feature store, MLflow multi-run comparison, FastAPI packaging, prediction logging, and drift detective style checks (M2–M5 ELS / Teams churn labs adapted to text).

## 3. M1 — Foundations (gaps addressed)

Notebook-style fragility addressed in this repo:

| Gap | Fix |
|-----|-----|
| Hidden preprocessing in notebooks | Shared `features/text_utils.py` for train and serve |
| Unversioned data | DVC snapshots under `data/versions/` + tag `week1-data-v1` |
| No input validation | Pandera schema + statistical checks |
| No experiment tracking | MLflow experiment `ticket_sentiment_prediction` |
| Ad-hoc serving | FastAPI with Pydantic edge-case handling |
| No monitoring | SQLite prediction logs + drift script |

## 4. M2 — Data engineering & versioning

- Sources: `support_tickets` (default synthetic), plus `amazon` / `yelp` / `sentiment140` / `all` via `configs/data_source.yaml`.
- Synthetic generator expanded with paraphrases, ambiguity, typos, and light label noise so metrics are not trivially 1.0 (~1463 unique texts / 1500 rows).
- Validation evidence: `reports/validation_log.txt`.
- Features: cleaned text, `text_len`, `word_count`, channel one-hots → SQLite `ticket_features`.
- Versioning: `dvc.yaml` / `dvc.lock`, local remote `dvc-storage/`, docs in `docs/DVC.md`.

## 5. M3 — Experimentation

| Run | Macro-F1 | Role |
|-----|----------|------|
| logreg_C1 | 0.8251 | baseline |
| logreg_C10 | 0.8257 | classical |
| **linear_svc** | **0.8423** | **served** |
| distilbert_finetune | 0.8683 | comparison only |

TF-IDF is fit on the training split only (leakage lesson from Taxila M2). DistilBERT is slightly more accurate but heavier; production stays classical. Details: `reports/model_comparison.md`.

## 6. M4 — Packaging & deployment

- Artifact: `model_store/sentiment_model.joblib` (+ label encoder / decision JSON).
- API: `GET /health`, `POST /predict` with empty-text and bad-channel 400s.
- Evidence: `reports/api_examples.md`, `reports/api_smoke_log.txt`.
- Docker: `docker/Dockerfile` (slim classical deps only).

## 7. M5 — Monitoring, drift, retraining

- Simulation posts slang/short tickets; length/word shifts flagged **DRIFTED** (shift ≈ 2.0 > 0.8 threshold).
- Retrain if rolling accuracy < 0.80 with ≥200 labels, or feature shift exceeds threshold for 3 consecutive batches.
- Evidence: `reports/drift_report.md`, `reports/drift_*_log.txt`.

## 8. How to reproduce

```bash
pip install -r requirements.txt
python scripts/run_m2_pipeline.py
python scripts/run_train.py
uvicorn serving.api:app --port 8000
# optional: pip install -r requirements-transformer.txt && python training/train_transformer.py
```

Full local/Colab/Docker steps: `USAGE.md`.

## 9. Citations

- Course brief: *ML Engineering Mini-Project Assignment Brief* (Flavor C).
- Suggested datasets: Amazon/Yelp reviews, Twitter Sentiment140, support-ticket style text.
- Tools: scikit-learn, Pandera, MLflow, DVC, FastAPI, Hugging Face DistilBERT (comparison).
- Course references: Taxila M1–M5 ELS materials; Teams churn week labs for feature-store / serving patterns.
- Texts: Crowe et al., *Machine Learning Production Systems*; Burkov, *Machine Learning Engineering*; McMahon, *MLE with Python*.
