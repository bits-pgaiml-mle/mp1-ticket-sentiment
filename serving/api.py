import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.build_features import build_feature_row
from monitoring.logger import init, log
from serving.inference_schema import TextRequest, TextResponse
from serving.model_loader import load_model_and_schema

app = FastAPI(
    title="Ticket Sentiment API",
    description="PCAM ZC412 Mini-Project-1 Flavor C — Taxila/Teams-aligned text classifier",
    version="1.0.0",
)

BUNDLE, FEATURE_COLUMNS, VERSION = load_model_and_schema()
init()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "ticket-sentiment",
        "model_loaded": BUNDLE is not None,
        "model_version": VERSION,
        "feature_columns": FEATURE_COLUMNS,
    }


@app.post("/predict", response_model=TextResponse)
def predict(payload: TextRequest) -> TextResponse:
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training/train.py first.")
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    channel = payload.channel.lower().strip()
    X, features = build_feature_row(payload.text, channel)
    pipe = BUNDLE["pipeline"]
    label_encoder = BUNDLE["label_encoder"]

    pred_id = pipe.predict(X)[0]
    label = str(label_encoder.inverse_transform([pred_id])[0])

    confidence = 0.0
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        confidence = float(pipe.predict_proba(X).max())
    elif hasattr(clf, "decision_function"):
        scores = np.asarray(pipe.decision_function(X), dtype=float).reshape(-1)
        exp = np.exp(scores - np.max(scores))
        confidence = float((exp / exp.sum()).max())

    result = {"label": label, "confidence": confidence, "model_version": VERSION}
    log({"text": payload.text, "channel": channel}, result, features)
    return TextResponse(**result)
