# Model Comparison Report

**Project:** mp1-ticket-sentiment (Flavor C)  
**Module:** M3 — Experimentation & Reproducibility  
**Pattern:** Taxila QuickBite MLflow multi-run comparison  
**Evidence logs:** `reports/train_classical_log.txt`, `reports/train_transformer_log.txt`

## Experiments (MLflow experiment: `ticket_sentiment_prediction`)

| Run | Model | Key params | Accuracy | Macro-F1 | Notes |
|-----|-------|------------|----------|----------|-------|
| logreg_C1 | LogisticRegression | C=1.0, TF-IDF+numeric | 0.8267 | 0.8251 | classical baseline |
| logreg_C10 | LogisticRegression | C=10.0, TF-IDF+numeric | 0.8267 | 0.8257 | stronger regularization inverse |
| linear_svc | LinearSVC | C=1.0, TF-IDF+numeric | 0.8433 | 0.8423 | best classical |
| distilbert_finetune | DistilBERT | 1 epoch, max_len=96, batch=8 | 0.8688 | 0.8683 | comparison-only; ~28 ms/text CPU |

Classical training used the full feature store (1500 tickets). DistilBERT used a stratified sample of 800 rows for CPU-friendly fine-tuning (see `training/train_transformer.py`).

## Decision — production model

**Promoted artifact:** `model_store/sentiment_model.joblib` ← **linear_svc**  
**Decision file:** `model_store/best_model_decision.json`  
**Transformer comparison file:** `model_store/transformer_decision.json` (not served)

### Justification

1. **Highest classical macro-F1** among LogReg C=1 / C=10 / LinearSVC on a leakage-safe TF-IDF fit (vectorizer fit on train split only).
2. **DistilBERT is slightly stronger** (+~2.6 pp macro-F1) but costs ~2 minutes of CPU fine-tune and ~28 ms/request plus a large PyTorch/transformers stack.
3. **Serving choice:** keep LinearSVC for Docker/FastAPI — small joblib artifact, no GPU, low latency, slim `requirements.txt`. DistilBERT remains an M3 experiment for the brief’s “classical vs transformer” requirement.

Reproduce:

```bash
python scripts/run_m2_pipeline.py
python training/train.py
pip install -r requirements-transformer.txt
python training/train_transformer.py
mlflow ui
```
