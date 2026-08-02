# Drift Simulation & Retraining Design

**Project:** mp1-ticket-sentiment (Flavor C)  
**Module:** M5 — Monitoring, Drift & Retraining

## Monitoring signals

- Prediction DB: `monitoring/predictions.db`
- Compared against training feature store: `data/feature_store.db`
- Numeric shifts: `text_len`, `word_count`
- Channel mix and predicted label distribution

## Drift scenario

`monitoring/simulate_concept_drift.py` sends slang/topic-shifted tickets (`bruh`, `fr`, `no cap`, etc.) to the live API.

## Retraining trigger

Retrain when:
1. Rolling labeled accuracy < 0.80 with at least 200 new labels, or
2. `text_len` / `word_count` shift score > threshold (config) for 3 consecutive batches.
