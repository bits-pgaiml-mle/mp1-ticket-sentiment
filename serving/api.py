import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.text_utils import channel_flags, clean_text
from monitoring.logger import init, log

CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()
VERSION = CFG["serving"]["model_version"]
MODEL_PATH = ROOT / CFG["training"]["model_path"]

app = FastAPI(
    title="Ticket Sentiment API",
    description="PCAM ZC412 Mini-Project-1 Flavor C — Taxila-aligned text classifier",
    version="1.0.0",
)

BUNDLE = None
if MODEL_PATH.exists():
    BUNDLE = joblib.load(MODEL_PATH)
init()


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    channel: str = Field(default="app")


class TextResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


def build_feature_row(text: str, channel: str) -> tuple[pd.DataFrame, dict]:
    text_clean = clean_text(text)
    flags = channel_flags(channel)
    features = {
        "text_clean": text_clean,
        "text_len": len(text_clean),
        "word_count": len(text_clean.split()) if text_clean else 0,
        **flags,
    }
    return pd.DataFrame([features]), features


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "ticket-sentiment",
        "model_loaded": BUNDLE is not None,
        "model_version": VERSION,
    }


@app.post("/predict", response_model=TextResponse)
def predict(payload: TextRequest) -> TextResponse:
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training/train.py first.")
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    channel = payload.channel.lower().strip()
    if channel not in {"email", "chat", "app"}:
        raise HTTPException(status_code=400, detail="channel must be one of: email, chat, app")

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
