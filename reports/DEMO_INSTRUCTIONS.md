# Demo instructions — practice once, then record

**Project:** `mp1-ticket-sentiment` (Flavor C)  
**Target length:** **10–15 minutes** with voiceover  
**Voiceover timeline:** [DEMO_SCRIPT.md](DEMO_SCRIPT.md)  
**Architecture backup:** [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)

Use this file as your **rehearsal runbook**. Run it once without recording. When smooth, record with the same steps and the short talking points below.

---

## Before you start (5–10 min, no recording)

### Checklist

- [ ] Repo root open in terminal:
  `D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment`
- [ ] Virtualenv active: `.\.venv\Scripts\activate` (or your usual env)
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] GitHub repo is **public** (or you can show it in browser)
- [ ] Screen ready: IDE + browser + 2–3 terminals
- [ ] Close Slack / email popups; hide unrelated tabs
- [ ] Optional: enlarge font in terminal / VS Code for readability on video

### Windows layout (recommended)

| Window | Use for |
|--------|---------|
| **Terminal 1** | Pipeline, train, drift commands |
| **Terminal 2** | Keep API running (`uvicorn`) |
| **Terminal 3** (optional) | Streamlit UI |
| **Browser** | Swagger `http://127.0.0.1:8000/docs`, GitHub, docs |
| **Editor** | Open `README.md`, `docs/ARCHITECTURE.md`, `reports/model_comparison.md`, `model_store/best_model_decision.json` |

### One-time warm-up (so train/API are not cold on camera)

```powershell
cd "D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment"
.\.venv\Scripts\activate
python tools/verify_setup.py
python scripts/run_m2_pipeline.py
python scripts/run_train.py
```

If this fails, fix it **before** practice or recording. You can skip re-training during the demo if artifacts already exist and you only re-show logs/metrics — but graders like seeing at least one live train or a clear MLflow/decision file.

---

## Practice run — exact sequence

Say the **Talk** lines out loud while you click/run. Time yourself; aim for **≤15 min**.

### Segment A — Hook & architecture (~1 min)

**Do**

1. Open GitHub repo in browser (or local README).
2. Scroll to Architecture (or open `docs/ARCHITECTURE.md`).

**Talk**

> This is Flavor C for PCAM ZC412 — a support-ticket and review sentiment classifier with three classes: negative, neutral, and positive.  
> The point of the mini-project is the ML engineering pipeline, not just accuracy. Flow is: prepare data, Pandera validation, shared features and a SQLite feature store, MLflow experiments, FastAPI serving with a Lab4 split, prediction logs, then mean-shift and PSI drift checks.  
> Layout matches Teams labs and Taxila-style demos.

---

### Segment B — M2 data pipeline (~2 min)

**Do** (Terminal 1)

```powershell
python tools/verify_setup.py
python scripts/run_m2_pipeline.py
```

**Show briefly**

- Terminal: validation **PASS**
- File: `reports/validation_log.txt` (if present)
- Paths: `feature_store/feature_store.db`, `model_store/feature_columns.json`
- Optional 10 sec: `dvc.yaml` and `data/versions/` (data versioning)

**Talk**

> Option A runs prepare, Pandera validation, and feature build. Validation fails the pipeline on bad schema or empty text. Features use the same `build_feature_row` helper that serving will use later, so train and serve stay aligned. We also version datasets with DVC snapshots.

**If slow:** you may skip re-running prepare on the recording take and only open the validation log + feature store — say you already ran Option A successfully.

---

### Segment C — M3 experiments (~2.5 min)

**Do**

```powershell
python training/train.py
```

Or, if already trained and you are short on time:

```powershell
python scripts/run_train.py
```

**Show**

- Terminal metrics for LogReg / LinearSVC
- `model_store/best_model_decision.json`
- `reports/model_comparison.md` (classical ~0.84 vs DistilBERT ~0.87)

**Do not** run DistilBERT live unless you have GPU time and ≥3 spare minutes. Point at the comparison report instead.

**Talk**

> We track classical models in MLflow. LinearSVC is our served model — about 0.84 macro-F1. DistilBERT is a few points higher but larger and slower, so we keep it as a comparison only. The decision JSON records why we promote the classical joblib artifact.

---

### Segment D — M4 API (+ optional UI) (~2.5 min)

**Do** (Terminal 2 — leave running)

```powershell
uvicorn serving.api:app --port 8000
```

**Browser:** open http://127.0.0.1:8000/docs  

**Calls** (Terminal 1 or Swagger “Try it out”)

Health:

```powershell
curl -s http://127.0.0.1:8000/health
```

Positive:

```powershell
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"Support resolved my issue quickly, thank you!\",\"channel\":\"chat\"}"
```

Negative:

```powershell
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"Terrible experience, refund still pending after one week.\",\"channel\":\"email\"}"
```

Edge — empty text → **400**:

```powershell
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"   \",\"channel\":\"chat\"}"
```

Edge — bad channel → **422**:

```powershell
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"hello\",\"channel\":\"sms\"}"
```

**Optional UI** (Terminal 3):

```powershell
streamlit run ui/app.py
```

Submit one ticket in the UI; emphasize the UI only calls `/predict`.

**Talk**

> Serving follows Lab4: schema, model loader, and thin API. Shared feature builder, joblib model, health and predict endpoints. Invalid channel is rejected by the contract. Streamlit has no local model — it only calls the API. Docker Compose is available for MLflow, trainer, API, and monitor if graders want deployment polish.

Full curl catalog: [api_examples.md](api_examples.md).

---

### Segment E — M5 drift & retrain (~3–4 min)

**Do** (API still running helps if simulate hits the live log; simulate/check mainly use log files)

```powershell
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

**Show**

- Script output: mean-shift **DRIFTED** and/or **PSI SHIFTED**
- Exit code 1 on PSI alert is intentional
- Mention `monitoring/predictions.db` and `monitoring/predictions.jsonl` (`log_backend: both`)
- Open `reports/drift_report.md` for retrain triggers

**Talk**

> Every prediction is logged to SQLite and JSONL. We simulate concept drift, then check mean-shift and PSI against the training reference. When PSI crosses the threshold the checker exits non-zero — that is our alert signal. Retrain triggers are documented in the drift report: sustained PSI alert, accuracy drop on a holdout sample, or a material schema/feature change.

---

### Segment F — Close (~1–2 min)

**Do**

- Show `reports/FINAL_REPORT.md` (submission links table)
- Show recent commits / public GitHub briefly
- Optional: `reports/datasets/COMPARISON.md` (multi-source support) — **10 seconds only**

**Talk**

> Code and evidence for M1 through M5 are in the public repo. The final report has the Drive demo link and design justifications. Thank you — happy to take questions.

---

## Timing cheat sheet

| Time | Segment | Must show |
|------|---------|-----------|
| 0:00–1:00 | Hook | Architecture |
| 1:00–3:00 | M2 | Option A / validation PASS / feature store |
| 3:00–5:30 | M3 | Train metrics + best_model_decision + comparison doc |
| 5:30–8:00 | M4 | uvicorn + 2 predicts + one edge case |
| 8:00–12:00 | M5 | simulate + check_drift + dual logs |
| 12:00–15:00 | Close | FINAL_REPORT + public repo |

If you are over time: skip Streamlit, skip live DistilBERT, skip Compose live start, skip dataset comparison.

---

## Practice scoring (do this after your dry run)

Mark each item. Re-practice only the fails.

- [ ] Finished in **≤15 minutes**
- [ ] Said “Flavor C” and three class labels
- [ ] Named prepare → validate → features → MLflow → FastAPI → drift
- [ ] Validation PASS visible
- [ ] Explained shared `build_feature_row`
- [ ] Named served model (**LinearSVC**) and why not DistilBERT in prod
- [ ] Live `/predict` positive and negative
- [ ] Showed 400 or 422 edge case
- [ ] Drift alert / PSI visible
- [ ] Mentioned dual logs (SQLite + JSONL)
- [ ] Pointed at FINAL_REPORT / public Git

---

## Recording day tips

1. Use the **same** path as practice; do not improvise new scripts on camera.
2. Start recording **after** venv is active and warm-up succeeded (or start with verify_setup if you want “from cold”).
3. Narrate **while** commands run so silence does not waste minutes.
4. If a command fails: pause recording, fix, re-take that segment — or cut in an editor.
5. Prefer **1080p**, clear mic; avoid music.
6. Title suggestion: `PCAM-ZC412-FlavorC-mp1-ticket-sentiment-demo`

### After recording

1. Upload video to **Google Drive** or OneDrive.
2. Share: **Anyone with the link can view**.
3. Paste the URL into the **Submission links** table at the top of [FINAL_REPORT.md](FINAL_REPORT.md).
4. Commit/push that link update (or include the updated report in the Taxila zip ≤10 MB).
5. Keep the large video **off** Taxila — only the link belongs in the report.

---

## Troubleshooting (quick)

| Problem | Fix |
|---------|-----|
| `Model not loaded` / 503 | Run `python training/train.py` then restart uvicorn |
| Port 8000 in use | `netstat -ano \| findstr :8000` then stop that PID, or use `--port 8001` and update curls |
| Pandera / validation fail | Re-run `python data/prepare_dataset.py --source support_tickets` then validate |
| Drift script finds no logs | Hit `/predict` a few times, or run `simulate_concept_drift.py` first |
| curl JSON quoting pain on PowerShell | Use Swagger Try it out instead |
| Streamlit cannot reach API | Confirm uvicorn is on `127.0.0.1:8000` |
| Transformer too slow | Do **not** run live; open `reports/model_comparison.md` |

---

## Command block — copy for Terminal 1 (practice)

```powershell
cd "D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment"
.\.venv\Scripts\activate

python tools/verify_setup.py
python scripts/run_m2_pipeline.py
python training/train.py

# After Terminal 2 has uvicorn running:
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"Support resolved my issue quickly, thank you!\",\"channel\":\"chat\"}"
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"Terrible experience, refund still pending after one week.\",\"channel\":\"email\"}"
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"   \",\"channel\":\"chat\"}"
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"hello\",\"channel\":\"sms\"}"

python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

**Terminal 2**

```powershell
cd "D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment"
.\.venv\Scripts\activate
uvicorn serving.api:app --port 8000
```

**Terminal 3 (optional)**

```powershell
cd "D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment"
.\.venv\Scripts\activate
streamlit run ui/app.py
```

---

When practice checklist is all green, record using [DEMO_SCRIPT.md](DEMO_SCRIPT.md) timing and this runbook’s commands.
