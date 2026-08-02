import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "monitoring" / "predictions.db"


def init() -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
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


def log(raw: dict, result: dict, features: dict) -> None:
    conn = sqlite3.connect(DB)
    conn.execute(
        """
        INSERT INTO predictions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            raw.get("text", ""),
            raw.get("channel", ""),
            int(features.get("text_len", 0)),
            int(features.get("word_count", 0)),
            int(features.get("channel_email", 0)),
            int(features.get("channel_chat", 0)),
            int(features.get("channel_app", 0)),
            result["label"],
            float(result["confidence"]),
            result["model_version"],
        ),
    )
    conn.commit()
    conn.close()
