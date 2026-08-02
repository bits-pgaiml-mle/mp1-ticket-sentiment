# Drift Simulation & Retraining Design

**Project:** mp1-ticket-sentiment (Flavor C)  
**Module:** M5 — Monitoring, Drift & Retraining  
**Status:** Placeholder — fill after Week 4

## Drift scenarios to simulate

1. New slang / abbreviations in tickets
2. Topic shift (billing issues surge)
3. Channel mix change (more chat than email)

## Retraining trigger (draft)

Retrain when rolling accuracy falls below `configs/config.yaml -> monitoring.accuracy_retrain_threshold` with at least N=200 labeled tickets.
