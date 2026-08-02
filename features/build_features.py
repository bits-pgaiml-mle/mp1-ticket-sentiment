import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.text_utils import channel_flags, clean_text

CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def transform_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        text_clean = clean_text(row["text"])
        flags = channel_flags(row["channel"])
        item = {
            "ticket_id": row["ticket_id"],
            "text_clean": text_clean,
            "text_len": len(text_clean),
            "word_count": len(text_clean.split()) if text_clean else 0,
            **flags,
            "label": row["label"],
        }
        rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    raw_path = ROOT / cfg["data"]["raw_path"]
    store_path = ROOT / cfg["data"]["feature_store"]
    schema_path = ROOT / cfg["data"]["feature_schema"]
    table = cfg["data"]["feature_table"]

    df = pd.read_csv(raw_path)
    feature_df = transform_frame(df)

    store_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(store_path)
    feature_df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()

    schema = [
        "text_clean",
        "text_len",
        "word_count",
        "channel_email",
        "channel_chat",
        "channel_app",
    ]
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"Feature store: {len(feature_df)} rows -> {store_path} [{table}]")
    print(f"Feature schema contract -> {schema_path}")
    print(f"Features: {schema}")


if __name__ == "__main__":
    main()
