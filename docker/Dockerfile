FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY data ./data
COPY validation ./validation
COPY features ./features
COPY feature_store ./feature_store
COPY training ./training
COPY serving ./serving
COPY monitoring ./monitoring
COPY scripts ./scripts
COPY model_store ./model_store
COPY ui ./ui

EXPOSE 8000 5000

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
