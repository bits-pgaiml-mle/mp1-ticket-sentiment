# Process Updates — mp1-ticket-sentiment (Flavor C)

**Course:** PCAM ZC412 Machine Learning Engineering  
**Group org:** [bits-pgaiml-mle](https://github.com/bits-pgaiml-mle)  
**Repo:** [mp1-ticket-sentiment](https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment)  
**Last updated:** 02 Aug 2026

## Module mapping (course vs mini-project weeks)

| Mini-project week | Course module | Focus |
|-------------------|---------------|--------|
| Day 1–2 / foundations | **M1** | Repo, environment, system layout, basic service thinking |
| Week 1 | **M2** | Ingest, validate, feature pipeline, dataset versioning |
| Week 2 | **M3** | Experiments, MLflow, model comparison |
| Week 3 | **M4** | Package + REST API |
| Week 4 | **M5** | Monitoring, drift, retraining |

> Note: Taxila “ELS Week 1” is **M1** (fragile churn service). Mini-project “Week 1” in the brief is **M2** (data pipeline). Both are tracked below.

---

## Progress till now (02 Aug 2026)

### Setup / process

- [x] Group formed; Teams coordination started
- [x] GitHub org created: `bits-pgaiml-mle`
- [x] Flavor C repo created and cloned locally
- [x] Local Git identity set to WILP email for commits
- [x] Scaffold, Taxila align, multi-source data, and DVC PRs merged to `main`

### M1 — foundations (aligned with Taxila ELS Week 1 ideas)

| Item | Status | Evidence |
|------|--------|----------|
| Isolated project layout | Done | `data/`, `features/`, `training/`, `serving/`, `monitoring/`, `model_store/` |
| Dependency list / reproducible install | Done | `requirements.txt`, README + USAGE setup |
| Model persistence concept | Done | `model_store/sentiment_model.joblib` |
| Feature schema / contract | Done | `data/feature_schema.json` |
| Basic REST service shape | Done | `serving/api.py` (`/health`, `/predict`) |
| Document known fragility / engineering gaps | Pending | Add short M1 “gaps & fixes” note in report |

### M2 / Mini-project Week 1 — data pipeline

| Item | Status | Evidence |
|------|--------|----------|
| Multi-source prepare | Done | `amazon` / `yelp` / `sentiment140` / `support_tickets` / `all` |
| Raw text ingest | Done | `data/prepare_dataset.py` (+ `generate_data.py` for ticket templates) |
| Schema + statistical validation | Done | `data/validate.py` (Pandera) |
| Clean / tokenize-style text prep | Done | `features/text_utils.py` (`clean_text`) |
| Feature pipeline + offline store | Done | `features/build_features.py` + SQLite `data/feature_store.db` |
| Shared train/serve feature logic | Done | same `text_utils` used by features + API |
| Option A / Option B runners | Done | `scripts/run_m2_pipeline.py`, `scripts/run_train.py` (`run_pipeline.py` alias) |
| Dataset versioning (DVC / tagged data snapshot) | **Done** | `dvc.yaml` snapshots all sources; tag `week1-data-v1` |
| Week-1 notes in formal report | **Pending** | Decisions, schema, feature list |

### Ahead of Week 1 (already started — OK per instructor “progress as modules unfold”)

| Item | Status | Module |
|------|--------|--------|
| MLflow multi-run training + best-model selection | Done (early) | M3 |
| FastAPI predict + prediction logging | Done (early) | M4/M5 |
| Concept-drift simulation + drift check script | Done (early) | M5 |
| Formal Taxila upload report | Pending | Submission |
| Demo video with voiceover | Pending | Submission |

---

## Pending for Week 1 (process / report — code done)

1. **Week-1 validation evidence**
   - Capture sample `python data/validate.py` output (screenshot or log) into `reports/`
2. **Week-1 design write-up** (short section in formal report)
   - Why support tickets + optional Amazon/Yelp/Sentiment140 sources
   - Schema/statistical checks chosen
   - Feature list and SQLite store rationale
   - Shared `clean_text` to avoid train–serve skew
   - DVC multi-source snapshots (`docs/DVC.md`)
3. **Confirm Flavor C** on group registration spreadsheet if C is the submission choice
4. **Remaining M1 polish**
   - Short “M1 gaps addressed” note in the formal report
   - Peer run-through: clone → venv → Option A (`scripts/run_m2_pipeline.py`)

### Explicitly not required to close Week 1 / M1

- Full formal final report (needed by submission day)
- Demo video (Week 4 / final)
- Transformer fine-tuning (optional later M3 enhancement)
- Cloud deployment (instructor: local demo is enough)

---

## Suggested teammate split (Week 1 close-out)

| Owner | Task |
|-------|------|
| TBD | Validation log / screenshots into `reports/` |
| TBD | Week-1 + M1 write-up draft |
| TBD | Peer run-through of setup on a clean machine |
| TBD | README Team names/roles |

---

## Next checkpoint

Week-1 code (ingest/validate/features/DVC) is frozen via `week1-data-v1`. Treat MLflow comparison evidence as **Week 2 / M3** official deliverable (already runnable via `python training/train.py` or Option A `scripts/run_train.py`).
