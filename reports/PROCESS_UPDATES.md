# Process Updates — mp1-ticket-sentiment (Flavor C)

**Course:** PCAM ZC412 Machine Learning Engineering  
**Group org:** [bits-pgaiml-mle](https://github.com/bits-pgaiml-mle)  
**Repo:** [mp1-ticket-sentiment](https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment)  
**Last updated:** 30 Aug 2026

## Module mapping (course vs mini-project weeks)

| Mini-project week | Course module | Focus |
|-------------------|---------------|--------|
| Day 1–2 / foundations | **M1** | Repo, environment, system layout, basic service thinking |
| Week 1 | **M2** | Ingest, validate, feature pipeline, dataset versioning |
| Week 2 | **M3** | Experiments, MLflow, model comparison |
| Week 3 | **M4** | Package + REST API (+ optional Docker / UI) |
| Week 4 | **M5** | Monitoring, drift, retraining |

---

## Progress (30 Aug 2026) — submission-ready code

### Setup / process

- [x] Group org + Flavor C repo: `bits-pgaiml-mle/mp1-ticket-sentiment`
- [x] Taxila / Teams lab alignment (validation/, feature_store/, Lab4 serving, Streamlit, compose)
- [x] Dual prediction logs (SQLite + JSONL) + PSI drift
- [x] Submission report package with evaluator links block

### M1 — foundations

| Item | Status | Evidence |
|------|--------|----------|
| Isolated project layout | Done | `data/`, `validation/`, `features/`, `feature_store/`, `training/`, `serving/`, `monitoring/`, `model_store/`, `ui/` |
| Dependencies | Done | `requirements.txt`, `requirements-transformer.txt` |
| Model persistence | Done | `model_store/sentiment_model.joblib` |
| Feature schema contract | Done | `model_store/feature_columns.json` |
| REST service | Done | Lab4 split: `serving/api.py`, `model_loader.py`, `inference_schema.py` |
| M1 gaps write-up | Done | `reports/FINAL_REPORT.md` |

### M2 / Week 1 — data pipeline

| Item | Status | Evidence |
|------|--------|----------|
| Multi-source prepare | Done | amazon / yelp / sentiment140 / support_tickets / all |
| Diversified synthetic tickets | Done | `data/generate_data.py` (~1463 unique / 1500) |
| Pandera validation | Done | `validation/validate_data.py`, `reports/validation_log.txt` |
| Feature store + shared features | Done | `feature_store/feature_store.db` + `build_feature_row` |
| DVC snapshots | Done | `dvc.lock`, `docs/DVC.md` |
| Week-1 design write-up | Done | `reports/FINAL_REPORT.md` §4 |

### M3 — experiments

| Item | Status | Evidence |
|------|--------|----------|
| Classical MLflow runs | Done | LogReg C=1/C=10 + LinearSVC |
| DistilBERT comparison | Done | `training/train_transformer.py`, `transformer_decision.json` |
| Model comparison report | Done | `reports/model_comparison.md` (served = **linear_svc**, F1 ≈ 0.84) |

### M4 — packaging & API

| Item | Status | Evidence |
|------|--------|----------|
| FastAPI + edge cases | Done | empty text → 400; bad channel → 422 (Pydantic Literal) |
| Lab4 module split | Done | `inference_schema.py`, `model_loader.py` |
| Streamlit UI | Done | `ui/app.py` (calls `/predict` only) |
| Sample requests | Done | `reports/api_examples.md`, `api_smoke_log.txt` |
| Docker + Compose | Done | `docker/Dockerfile`, `docker-compose.yml` |

### M5 — monitoring

| Item | Status | Evidence |
|------|--------|----------|
| Prediction logging | Done | SQLite and/or JSONL (`log_backend: both`) |
| Drift simulation | Done | slang tickets via `simulate_concept_drift.py` |
| Mean-shift + PSI | Done | `check_drift.py` (PSI alert exits non-zero) |
| Drift report | Done | `reports/drift_report.md` |
| Retrain trigger design | Done | accuracy + shift batches + PSI |

### Submission artifacts

- [x] `reports/FINAL_REPORT.md` (evaluator links at top)
- [x] `reports/DEMO_SCRIPT.md` (10–15 min voiceover target)
- [ ] Recorded demo video → Drive → paste URL in FINAL_REPORT
- [ ] GitHub repo set **public**
- [ ] Taxila group membership / one-member upload (≤10 MB)
- [ ] Team names/roles in README

---

## Outside-repo blockers (from lecture transcripts)

1. **One group member** submits on Taxila (group grading).
2. **Demo with voiceover** (~10–15 min); upload to Drive; link in report (Taxila file limit ~10 MB).
3. **GitHub public** + Drive share tested in incognito.
4. Instructor indicated hard stop around **31 Aug 2026** for evaluation timeline — submit ASAP.
5. Fill teammate names in README before upload.
