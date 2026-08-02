# Usage Guide — mp1-ticket-sentiment (Flavor C)

End-to-end **support ticket / review sentiment** classifier.  
Works on **local machine (CPU)** and **Google Colab (CPU is enough; T4 GPU not required** unless you later fine-tune a transformer).

---

## 1. Local usage

### 1.1 Setup (once)

```powershell
cd "D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
cd mp1-ticket-sentiment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the **repo root**.

### 1.2 Option A — easiest (recommended)

```powershell
python scripts/run_m2_pipeline.py
python scripts/run_train.py
```

(`scripts/run_pipeline.py` is an alias for `scripts/run_train.py`.)

**Terminal 1 — API**

```powershell
uvicorn serving.api:app --reload --port 8000
```

**Terminal 2 — predict + drift**

```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"Support resolved my issue quickly, thank you!\",\"channel\":\"chat\"}"
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

- Swagger: http://127.0.0.1:8000/docs  
- MLflow UI (optional): `mlflow ui` → http://127.0.0.1:5000  

### 1.3 Option B — step by step

```powershell
python data/generate_data.py
python data/validate.py
python features/build_features.py
python training/train.py
uvicorn serving.api:app --reload --port 8000
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```

| Step | Entry file |
|------|------------|
| Generate tickets | `data/generate_data.py` |
| Validate | `data/validate.py` |
| Features | `features/build_features.py` |
| Train | `training/train.py` |
| Serve | `serving/api.py` via uvicorn |
| Drift simulate | `monitoring/simulate_concept_drift.py` |
| Drift check | `monitoring/check_drift.py` |

---

## 2. Google Colab usage

### 2.1 Runtime

1. Open [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **CPU** (T4 only if you add transformer fine-tuning later)

### 2.2 Option A — easiest

```python
!git clone https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment.git
%cd mp1-ticket-sentiment
!pip install -q -r requirements.txt

!python scripts/run_m2_pipeline.py
!python scripts/run_train.py
```

Predict with TestClient (recommended on Colab):

```python
from fastapi.testclient import TestClient
from serving.api import app

client = TestClient(app)
print(client.get("/health").json())
print(client.post("/predict", json={
    "text": "Support resolved my issue quickly, thank you!",
    "channel": "chat",
}).json())
print(client.post("/predict", json={
    "text": "Terrible experience, refund still pending after one week.",
    "channel": "email",
}).json())
```

Drift:

```python
!python monitoring/simulate_concept_drift.py
!python monitoring/check_drift.py
```

### 2.3 Option B — step by step on Colab

```python
!python data/generate_data.py
!python data/validate.py
!python features/build_features.py
!python training/train.py
```

Then use the TestClient cell for `/predict`.

### 2.4 Optional: uvicorn on Colab

Possible with background + ngrok; **not required**. Prefer TestClient for demos.

---

## 3. What each option does

| Stage | Option A | Option B |
|-------|----------|----------|
| M2 data | `scripts/run_m2_pipeline.py` | generate → validate → features |
| M3 train | `scripts/run_train.py` (includes M2) | `training/train.py` |
| M4 serve | uvicorn / TestClient | same |
| M5 drift | simulate + check_drift | same |
