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
| Week 3 | **M4** | Package + REST API |
| Week 4 | **M5** | Monitoring, drift, retraining |

---

## Progress (30 Aug 2026) — submission package

### Setup / process

- [x] Group formed; Teams coordination started
- [x] GitHub org created: `bits-pgaiml-mle`
- [x] Flavor C repo created and cloned locally
- [x] Scaffold, Taxila align, multi-source data, DVC, docs sync
- [x] Submission package branch: reports, DistilBERT comparison, API/drift evidence

### M1 — foundations

| Item | Status | Evidence |
|------|--------|----------|
| Isolated project layout | Done | `data/`, `features/`, `training/`, `serving/`, `monitoring/`, `model_store/` |
| Dependency list | Done | `requirements.txt`, optional `requirements-transformer.txt` |
| Model persistence | Done | `model_store/sentiment_model.joblib` |
| Feature schema / contract | Done | `data/feature_schema.json` |
| REST service | Done | `serving/api.py` |
| M1 gaps write-up | Done | `reports/FINAL_REPORT.md` §3 |

### M2 / Week 1 — data pipeline

| Item | Status | Evidence |
|------|--------|----------|
| Multi-source prepare | Done | amazon / yelp / sentiment140 / support_tickets / all |
| Diversified synthetic tickets | Done | `data/generate_data.py` (~1463 unique / 1500) |
| Pandera validation | Done | `reports/validation_log.txt` |
| Feature store + shared clean_text | Done | `features/` + SQLite |
| DVC snapshots | Done | `dvc.lock` refreshed; `data/versions/` |
| Week-1 design write-up | Done | `reports/FINAL_REPORT.md` §4 |

### M3 — experiments

| Item | Status | Evidence |
|------|--------|----------|
| Classical MLflow runs | Done | LogReg C=1/C=10 + LinearSVC |
| DistilBERT comparison | Done | `training/train_transformer.py`, `transformer_decision.json` |
| Model comparison report | Done | `reports/model_comparison.md` (served = linear_svc) |

### M4 — packaging & API

| Item | Status | Evidence |
|------|--------|----------|
| FastAPI + edge cases | Done | empty text / bad channel → 400 |
| Sample requests | Done | `reports/api_examples.md`, `api_smoke_log.txt` |
| Docker | Done | `docker/Dockerfile` + README/USAGE |

### M5 — monitoring

| Item | Status | Evidence |
|------|--------|----------|
| Drift simulation | Done | slang tickets logged |
| Drift report with numbers | Done | `reports/drift_report.md` (text_len/word_count DRIFTED) |
| Retrain trigger design | Done | accuracy + consecutive shift batches |

### Submission artifacts

- [x] `reports/FINAL_REPORT.md`
- [x] `reports/DEMO_SCRIPT.md` (5–7 min outline)
- [ ] Recorded demo video (team action)
- [ ] Taxila group membership / upload (instructor must add group)
- [ ] Team names/roles in README (fill TBD)

---

## Outside-repo blockers

1. Taxila: user must be added to a submission group (see Mini-Project-1 `notes.txt`).
2. Confirm late/extension if needed (brief due 24 Aug 2026).
3. Record demo from `DEMO_SCRIPT.md`.
