# Model Comparison Report

**Project:** mp1-ticket-sentiment (Flavor C)  
**Module:** M3 — Experimentation & Reproducibility  
**Pattern:** Taxila QuickBite MLflow multi-run comparison

## Experiments

| Run | Model | Key params | Accuracy | Macro-F1 | Notes |
|-----|-------|------------|----------|----------|-------|
| logreg_C1 | LogisticRegression | C=1.0 | fill after train | fill | baseline |
| logreg_C10 | LogisticRegression | C=10.0 | fill after train | fill | stronger fit |
| linear_svc | LinearSVC | C=1.0 | fill after train | fill | margin classifier |

## Decision

Best model: see `model_store/best_model_decision.json` after `python training/train.py`.

Justification principles:
- Prefer highest macro-F1 on held-out stratified split.
- If metrics are nearly tied, prefer simpler/default `C=1.0` LogisticRegression (Taxila engineering judgment pattern).
