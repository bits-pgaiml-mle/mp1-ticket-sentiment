# Demo Script (~10–15 minutes with voiceover)

**Project:** mp1-ticket-sentiment (Flavor C)  
**Goal:** Walk graders through M2→M5 with live commands and narration.  
**Instructor guidance:** about **10–15 minutes** with voiceover (not 30–40).  
**Practice first:** use the full runbook **[DEMO_INSTRUCTIONS.md](DEMO_INSTRUCTIONS.md)** (Windows commands, talking points, checklist), then record with this timeline.  
**After recording:** upload to Google Drive / OneDrive → share with link access → paste URL into the **Submission links** table at the top of [FINAL_REPORT.md](FINAL_REPORT.md).

## 0:00–1:00 — Hook & architecture

- State Flavor C: support-ticket sentiment (negative / neutral / positive).
- Show README architecture (prepare → `validation/` → features → MLflow → FastAPI → drift).
- Mention Teams/Taxila alignment: feature store, shared `build_feature_row`, Lab4 serving split, dual logs, PSI.

## 1:00–3:00 — M2 data pipeline (Option A)

```bash
python tools/verify_setup.py
python scripts/run_m2_pipeline.py
```

- Point at Pandera PASS / `reports/validation_log.txt`.
- Show `feature_store/feature_store.db` and `model_store/feature_columns.json`.
- Briefly show `dvc.yaml` / `data/versions/` and tag `week1-data-v1`.

## 3:00–5:30 — M3 experiments

```bash
python training/train.py
# optional short clip: python training/train_transformer.py
```

- Classical metrics (~0.83–0.84) vs DistilBERT (~0.87) from `reports/model_comparison.md`.
- Justify serving **LinearSVC**: latency/size vs small F1 gain.
- Open `model_store/best_model_decision.json`.

## 5:30–8:00 — M4 API (+ optional UI)

```bash
uvicorn serving.api:app --port 8000
```

- Curl or Swagger from `reports/api_examples.md` (positive / negative / empty text / bad channel).
- Optional: `streamlit run ui/app.py` — UI has no model knowledge (Lab4 teaching point).
- Mention Docker / Compose one-liners from README.

## 8:00–12:00 — M5 drift & retrain

```bash
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

- Call out mean-shift **DRIFTED** and **PSI SHIFTED** / exit code 1.
- Mention SQLite + JSONL logs (`log_backend: both`).
- State retrain triggers from `reports/drift_report.md`.

## 12:00–15:00 — Close

- Repo is public; commit history shows weekly progress.
- Point to `reports/FINAL_REPORT.md` submission links + design justifications.
- Thank graders / Q&A.
