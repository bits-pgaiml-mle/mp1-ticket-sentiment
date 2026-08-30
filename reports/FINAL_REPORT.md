# Final Report — Flavor C Support Ticket / Review Sentiment Classifier

## Submission links (for evaluators — read first)

| Item | Link / note |
|------|-------------|
| **GitHub repository (public)** | https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment |
| **Demo recording (Drive)** | `TODO: paste Google Drive / OneDrive share link here` |
| **Demo script used** | [DEMO_INSTRUCTIONS.md](DEMO_INSTRUCTIONS.md) (practice runbook) + [DEMO_SCRIPT.md](DEMO_SCRIPT.md) (target **10–15 min** with voiceover) |
| **Quick run** | See repo [USAGE.md](../USAGE.md) — Option A |
| **Taxila upload tip** | Upload this report (or a short PDF/zip &lt; 10 MB). Keep large video + artifacts on Drive; put links here. |

**Access checklist before submit**

1. Repo is **public** (or evaluators can clone without requesting access).
2. Drive folder is shared so **anyone with the link** can view.
3. One group member uploads to Taxila (group-level grading).
4. Links above are filled and tested in an incognito window.

---

**Course:** PCAM ZC412 Machine Learning Engineering  
**Group org:** [bits-pgaiml-mle](https://github.com/bits-pgaiml-mle)  
**Repository:** [mp1-ticket-sentiment](https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment)  
**Flavor:** C — NLP / text sentiment (support tickets + optional Amazon/Yelp/Sentiment140)

## 1. Problem

A customer-support / e-commerce platform needs to classify incoming ticket text as **negative / neutral / positive** so agents can prioritize urgent unhappy customers. The system must cover the full ML lifecycle: ingest → validate → features → experiments → REST serve → monitor drift → retrain design.

## 2. Architecture

Canonical document: **[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)**.

```text
prepare_dataset → raw tickets → validation/validate_data.py (Pandera)
        → features/build_features.py (shared build_feature_row)
        → feature_store/feature_store.db + model_store/feature_columns.json
        → MLflow (LogReg / LinearSVC / DistilBERT) → promote classical joblib
        → serving/ (Lab4: api + model_loader + inference_schema) + ui/app.py
        → prediction logs (SQLite and/or JSONL) → mean-shift + PSI drift
```

Aligned with Teams churn labs (M2–M4), Taxila ELS patterns, and VaayuGrid Docker/PSI ideas (M5/M6). Instructor guidance: end-to-end pipeline matters more than any single tool; Docker is optional polish.

## 3. M1 — Foundations (gaps addressed)

| Gap | Fix |
|-----|-----|
| Hidden preprocessing in notebooks | Shared `build_feature_row` / `text_utils` for train and serve |
| Unversioned data | DVC snapshots under `data/versions/` + tag `week1-data-v1` |
| No input validation | Pandera in `validation/validate_data.py` |
| No experiment tracking | MLflow experiment `ticket_sentiment_prediction` |
| Ad-hoc serving | FastAPI Lab4 split + Pydantic schema |
| No monitoring | SQLite/JSONL prediction logs + drift script |

## 4. M2 — Data engineering & versioning

- Sources: `support_tickets` (default synthetic), plus `amazon` / `yelp` / `sentiment140` / `all`.
- Synthetic generator uses paraphrases, ambiguity, typos, and light label noise (~1463 unique texts / 1500 rows) so metrics are not trivially 1.0.
- Validation evidence: `reports/validation_log.txt`.
- Features: cleaned text, `text_len`, `word_count`, channel one-hots → `feature_store/feature_store.db`.
- Versioning: `dvc.yaml` / `dvc.lock`, docs in `docs/DVC.md`.

## 5. M3 — Experimentation

| Run | Macro-F1 | Role |
|-----|----------|------|
| logreg_C1 | 0.8251 | baseline |
| logreg_C10 | 0.8257 | classical |
| **linear_svc** | **0.8423** | **served** |
| distilbert_finetune | 0.8683 | comparison only |

TF-IDF is fit on the training split only. DistilBERT is slightly more accurate but heavier; production stays classical. Details: `reports/model_comparison.md`.

## 6. M4 — Packaging & deployment

- Artifact: `model_store/sentiment_model.joblib`.
- API: `GET /health`, `POST /predict` (empty text → 400; invalid channel → 422).
- Lab4 modules: `serving/inference_schema.py`, `serving/model_loader.py`.
- Optional UI: `streamlit run ui/app.py`.
- Evidence: `reports/api_examples.md`, `reports/api_smoke_log.txt`.
- Docker: `docker/Dockerfile`; optional Compose: `docker-compose.yml` (mlflow / trainer / api / monitor).

## 7. M5 — Monitoring, drift, retraining

- Logs: SQLite `monitoring/predictions.db` and/or JSONL `monitoring/predictions.jsonl` (`log_backend: both`).
- Simulation posts slang/short tickets; mean-shift and **PSI** flag `text_len` / `word_count` (PSI &gt; 0.25 → exit 1).
- Retrain if rolling accuracy &lt; 0.80 with ≥200 labels, shift exceeds threshold for 3 batches, or PSI alert fires.
- Evidence: `reports/drift_report.md`, `reports/drift_*_log.txt`.

## 8. How to reproduce

```bash
pip install -r requirements.txt
python tools/verify_setup.py
python scripts/run_m2_pipeline.py
python scripts/run_train.py
uvicorn serving.api:app --port 8000
# optional: streamlit run ui/app.py
# optional: pip install -r requirements-transformer.txt && python training/train_transformer.py
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

Full local/Colab/Docker steps: `USAGE.md`.

## 9. Citations

- Course brief: *ML Engineering Mini-Project Assignment Brief* (Flavor C).
- Suggested datasets: Amazon/Yelp reviews, Twitter Sentiment140, support-ticket style text.
- Tools: scikit-learn, Pandera, MLflow, DVC, FastAPI, Streamlit, Hugging Face DistilBERT (comparison).
- Course references: Taxila M1–M6 ELS / Docker demo; Teams churn week labs (feature store, MLflow, Lab4 serving).
- Lecture guidance (submission): group upload, public Git, Drive demo link, report ≤10 MB.
- Texts: Crowe et al., *Machine Learning Production Systems*; Burkov, *Machine Learning Engineering*; McMahon, *MLE with Python*.
