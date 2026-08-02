import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
ALLOWED_LABELS = {"negative", "neutral", "positive"}
ALLOWED_CHANNELS = {"email", "chat", "app"}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = ["ticket_id", "text", "channel", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors

    if df["ticket_id"].duplicated().any():
        errors.append("ticket_id has duplicates")

    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        errors.append("Empty or null text rows found")

    bad_labels = set(df["label"].unique()) - ALLOWED_LABELS
    if bad_labels:
        errors.append(f"Unexpected labels: {bad_labels}")

    bad_channels = set(df["channel"].unique()) - ALLOWED_CHANNELS
    if bad_channels:
        errors.append(f"Unexpected channels: {bad_channels}")

    if df["text"].astype(str).str.len().mean() < 5:
        errors.append("Average text length is suspiciously short")

    return errors


def main() -> None:
    cfg = load_config()
    path = ROOT / cfg["data"]["raw_path"]
    if not path.exists():
        print(f"Raw data not found: {path}")
        print("Run: python -m src.data.generate")
        sys.exit(1)

    df = pd.read_csv(path)
    errors = validate(df)
    if errors:
        print("Validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"Validation PASSED: {len(df)} rows")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
