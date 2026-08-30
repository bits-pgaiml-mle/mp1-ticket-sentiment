# Drift Simulation & Retraining Design

**Project:** mp1-ticket-sentiment (Flavor C)  
**Module:** M5 — Monitoring, Drift & Retraining  
**Evidence logs:** `reports/drift_sim_log.txt`, `reports/drift_check_log.txt`  
**Aligned with:** Taxila M5 ELS (monitoring / observability / retraining)

## Monitoring signals

- Prediction DB: `monitoring/predictions.db`
- Compared against training feature store: `data/feature_store.db`
- Numeric shifts: `text_len`, `word_count` (z-style shift vs train std)
- Channel mix and predicted label distribution
- Drift threshold from `configs/config.yaml`: `monitoring.drift_shift_threshold = 0.8`

## Drift scenario

`monitoring/simulate_concept_drift.py` posts slang/topic-shifted tickets (`bruh`, `fr`, `no cap`, short chat phrasing) to the live API.

Captured run (8 production predictions):

| Feature | Train mean | Prod mean | Shift | Flag |
|---------|------------|-----------|-------|------|
| text_len | 104.25 | 38.12 | 2.01 | DRIFTED |
| word_count | 17.56 | 7.12 | 1.90 | DRIFTED |

Production channel mix: email 37.5%, chat 37.5%, app 25.0%.  
Predicted labels under drift: neutral 50.0%, negative 37.5%, positive 12.5%.

Interpretation: shorter slang tickets shift length/word features well beyond threshold, and confidence scores drop (often ~0.4–0.6), which is a useful early warning even before labeled accuracy is available.

## Retraining trigger

Retrain when **either**:

1. Rolling labeled accuracy < 0.80 with at least 200 newly labeled tickets, or  
2. `text_len` / `word_count` shift score > threshold for **3 consecutive batches**.

Operational notes (Taxila M5 pattern):

- Log every prediction with features used at serve time (shared `clean_text` / channel flags).
- Prefer feature/prediction-distribution monitors when labels lag.
- After retrain, promote a new joblib only if held-out macro-F1 improves and smoke `/predict` passes.
- Keep DistilBERT as an offline experiment unless latency budget changes.

Reproduce:

```bash
uvicorn serving.api:app --port 8000
python monitoring/simulate_concept_drift.py
python monitoring/check_drift.py
```
