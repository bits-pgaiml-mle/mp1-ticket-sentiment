import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text_clean"] = out["text"].map(clean_text)
    out["text_len"] = out["text_clean"].str.len()
    out["word_count"] = out["text_clean"].str.split().str.len()
    out["channel_email"] = (out["channel"] == "email").astype(int)
    out["channel_chat"] = (out["channel"] == "chat").astype(int)
    out["channel_app"] = (out["channel"] == "app").astype(int)
    return out


def main() -> None:
    cfg = load_config()
    raw_path = ROOT / cfg["data"]["raw_path"]
    processed_path = ROOT / cfg["data"]["processed_path"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    featured = build_features(df)
    featured.to_csv(processed_path, index=False)
    print(f"Wrote features -> {processed_path} ({featured.shape[0]} rows, {featured.shape[1]} cols)")


if __name__ == "__main__":
    main()
