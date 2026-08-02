# mp1-ticket-sentiment

**BITS Pilani WILP · PGAIML · PCAM ZC412 Machine Learning Engineering**  
**Mini-Project-1 · Flavor C — Support Ticket / Review Sentiment Classifier**

End-to-end NLP ML pipeline: raw text ingest → cleaning & features → tracked experiments → REST API → monitoring / concept drift / retraining design.

## Architecture

```text
raw tickets.csv
      |
      v
  validate  ----->  clean + features  ----->  tickets_features.csv (DVC)
                         |
                         v
                   train + MLflow  ----->  models/best_model.joblib
                         |
                         v
                      FastAPI  ----->  POST /predict (text)
                         |
                         v
                    monitoring  ----->  logs + drift report + retrain trigger
```

## Repository layout

```text
mp1-ticket-sentiment/
├── configs/config.yaml
├── data/raw|processed/
├── src/data|features|training|serving|monitoring/
├── models/  monitoring/logs/  notebooks/  reports/
├── docker/Dockerfile
├── scripts/run_week1.py
└── requirements.txt
```

## Setup

```bash
cd mp1-ticket-sentiment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Week 1 — data pipeline (ready)

```bash
python scripts/run_week1.py
```

## Later weeks (stubs)

| Week | Module | Focus |
|------|--------|-------|
| 2 | M3 | Classical ML (+ optional transformer) + MLflow |
| 3 | M4 | FastAPI text inference + empty/malformed handling |
| 4 | M5 | Concept drift (new slang/topics) + retrain trigger |

## Dataset

Starter generates **synthetic support tickets** labeled `negative` / `neutral` / `positive`.  
Optional upgrade: Amazon/Yelp reviews or Sentiment140.

## Team

- Org: [bits-pgaiml-mle](https://github.com/bits-pgaiml-mle)
- Repo: [mp1-ticket-sentiment](https://github.com/bits-pgaiml-mle/mp1-ticket-sentiment)
