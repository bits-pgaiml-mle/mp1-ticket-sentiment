import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_backends(cfg: dict | None = None) -> set[str]:
    cfg = cfg or load_config()
    mon = cfg.get("monitoring", {})
    env = os.getenv("PREDICTION_LOG_BACKEND", "").strip().lower()
    raw = env or str(mon.get("log_backend", "sqlite")).strip().lower()
    if raw in {"both", "all"}:
        return {"sqlite", "jsonl"}
    if raw in {"sqlite", "db"}:
        return {"sqlite"}
    if raw in {"jsonl", "json", "jsonlines"}:
        return {"jsonl"}
    raise ValueError(f"Unsupported prediction log backend: {raw}. Use sqlite, jsonl, or both.")


def sqlite_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    env = os.getenv("PREDICTIONS_DB")
    if env:
        return Path(env)
    return ROOT / cfg["monitoring"]["predictions_db"]


def jsonl_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    env = os.getenv("PREDICTION_LOG")
    if env:
        return Path(env)
    return ROOT / cfg["monitoring"].get("predictions_jsonl", "monitoring/predictions.jsonl")


def _record(raw: dict, result: dict, features: dict) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": raw.get("text", ""),
        "channel": raw.get("channel", ""),
        "text_len": int(features.get("text_len", 0)),
        "word_count": int(features.get("word_count", 0)),
        "channel_email": int(features.get("channel_email", 0)),
        "channel_chat": int(features.get("channel_chat", 0)),
        "channel_app": int(features.get("channel_app", 0)),
        "label": result["label"],
        "confidence": float(result["confidence"]),
        "model_version": result["model_version"],
    }


def init() -> None:
    cfg = load_config()
    backends = get_backends(cfg)
    if "sqlite" in backends:
        path = sqlite_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                text TEXT,
                channel TEXT,
                text_len INTEGER,
                word_count INTEGER,
                channel_email INTEGER,
                channel_chat INTEGER,
                channel_app INTEGER,
                label TEXT,
                confidence REAL,
                model_version TEXT
            )
            """
        )
        conn.commit()
        conn.close()
    if "jsonl" in backends:
        path = jsonl_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


def log(raw: dict, result: dict, features: dict) -> None:
    cfg = load_config()
    backends = get_backends(cfg)
    row = _record(raw, result, features)

    if "sqlite" in backends:
        path = sqlite_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute(
            """
            INSERT INTO predictions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row["timestamp"],
                row["text"],
                row["channel"],
                row["text_len"],
                row["word_count"],
                row["channel_email"],
                row["channel_chat"],
                row["channel_app"],
                row["label"],
                row["confidence"],
                row["model_version"],
            ),
        )
        conn.commit()
        conn.close()

    if "jsonl" in backends:
        path = jsonl_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
