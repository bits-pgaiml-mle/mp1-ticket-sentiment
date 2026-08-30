# Demo Script (5–7 minutes)

**Project:** mp1-ticket-sentiment (Flavor C)  
**Goal:** Walk graders through M2→M5 with live commands. Record with voiceover (instructor guidance: **~10–15 minutes** ideal; keep under ~15).  
**After recording:** upload to Google Drive / OneDrive, share with link access, paste the URL into the **Submission links** table at the top of [FINAL_REPORT.md](FINAL_REPORT.md).

## 0:00–0:40 — Hook & architecture

- State Flavor C: support-ticket sentiment (negative / neutral / positive).
- Show README architecture diagram (prepare → validate → features → MLflow → FastAPI → drift).
- Mention Taxila alignment: feature store, shared preprocess, MLflow, monitoring.

## 0:40–2:00 — M2 data pipeline (Option A)

```bash
python scripts/run_m2_pipeline.py
```

- Point at Pandera PASS in the terminal / `reports/validation_log.txt`.
- Open `data/feature_schema.json` and mention SQLite feature store.
- Briefly show `dvc.yaml` / `data/versions/` and tag `week1-data-v1`.

## 2:00–3:30 — M3 experiments

```bash
python training/train.py
# optional clip: python training/train_transformer.py
```

- Show three classical metrics (~0.83–0.84) and DistilBERT (~0.87) from `reports/model_comparison.md`.
- Justify serving **LinearSVC**: latency/size vs small F1 gain from DistilBERT.
- Open `model_store/best_model_decision.json`.

## 3:30–5:00 — M4 API

```bash
uvicorn serving.api:app --port 8000
```

- Hit Swagger or curl from `reports/api_examples.md` (positive / negative / empty text).
- Mention Docker one-liner from README.

## 5:00–6:30 — M5 drift & retrain

```bash
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

- Call out DRIFTED on `text_len` / `word_count`.
- State retrain triggers from `reports/drift_report.md`.

## 6:30–7:00 — Close

- Repo link + commit history reflects weekly progress.
- Point to `reports/FINAL_REPORT.md` for design justifications.
- Thank graders / Q&A.
