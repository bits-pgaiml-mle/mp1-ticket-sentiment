# API Examples — Ticket Sentiment Service

**Module:** M4 — Packaging & Deployment  
**Service:** FastAPI Lab4 layout — `serving/api.py` + `model_loader.py` + `inference_schema.py`  
**Smoke log:** `reports/api_smoke_log.txt`

Base URL (local): `http://127.0.0.1:8000`

## Start the API

```bash
uvicorn serving.api:app --reload --port 8000
```

Optional UI (separate terminal): `streamlit run ui/app.py`  
Swagger UI: http://127.0.0.1:8000/docs

## Health

```bash
curl -s http://127.0.0.1:8000/health
```

Expected (shape):

```json
{
  "status": "ok",
  "service": "ticket-sentiment",
  "model_loaded": true,
  "model_version": "ticket-sentiment-v1",
  "feature_columns": ["text_clean", "text_len", "word_count", "channel_email", "channel_chat", "channel_app"]
}
```

## Predict — positive

```bash
curl -s -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Support resolved my issue quickly, thank you!\",\"channel\":\"chat\"}"
```

Example response:

```json
{"label":"positive","confidence":0.8430912034673379,"model_version":"ticket-sentiment-v1"}
```

## Predict — negative

```bash
curl -s -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Terrible experience, refund still pending after one week.\",\"channel\":\"email\"}"
```

## Predict — neutral

```bash
curl -s -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Can you confirm the warranty period for this product?\",\"channel\":\"app\"}"
```

## Edge cases

Empty / whitespace text → HTTP **400**:

```bash
curl -s -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"   \",\"channel\":\"chat\"}"
```

```json
{"detail":"text must not be empty"}
```

Invalid channel → HTTP **422** (Pydantic `Literal["email","chat","app"]`, Lab4-style contract):

```bash
curl -s -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"hello\",\"channel\":\"fax\"}"
```

Expect a validation error body listing allowed channel values.

## Prediction logging

Each successful `/predict` appends to the configured backend(s) (`configs/config.yaml` → `monitoring.log_backend`):

- SQLite: `monitoring/predictions.db`
- JSONL: `monitoring/predictions.jsonl`

## Docker

```bash
docker build -f docker/Dockerfile -t mp1-ticket-sentiment .
docker run --rm -p 8000:8000 mp1-ticket-sentiment
```

Or Compose API service: `docker compose up -d api` (after trainer has written `model_store`).
