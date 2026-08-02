import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

SCHEMA = pa.DataFrameSchema(
    {
        "ticket_id": pa.Column(str, nullable=False, unique=True),
        "text": pa.Column(
            str,
            checks=[
                pa.Check(lambda s: s.str.strip().str.len() > 0, error="text must be non-empty"),
                pa.Check(lambda s: s.str.len().mean() >= 5, error="average text length too short"),
            ],
            nullable=False,
        ),
        "channel": pa.Column(str, checks=pa.Check.isin(["email", "chat", "app"]), nullable=False),
        "label": pa.Column(
            str,
            checks=pa.Check.isin(["negative", "neutral", "positive"]),
            nullable=False,
        ),
        "data_source": pa.Column(
            str,
            checks=pa.Check.isin(
                ["amazon", "yelp", "sentiment140", "support_tickets", "synthetic", "kaggle"]
            ),
            required=False,
            nullable=True,
        ),
    },
    coerce=True,
    strict=False,
)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def statistical_checks(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    label_share = df["label"].value_counts(normalize=True)
    if label_share.min() < 0.15:
        errors.append(f"Class imbalance too high: {label_share.to_dict()}")
    if df["text"].str.len().median() < 10:
        errors.append("Median text length < 10 characters")
    if df["channel"].nunique() < 2:
        errors.append("Expected multiple channels in raw data")
    return errors


def main() -> None:
    cfg = load_config()
    path = ROOT / cfg["data"]["raw_path"]
    if not path.exists():
        print(f"Raw data not found: {path}")
        print("Run: python data/prepare_dataset.py [--source amazon|yelp|sentiment140|support_tickets|all]")
        sys.exit(1)

    df = pd.read_csv(path)
    try:
        validated = SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        print("Schema validation FAILED:")
        print(exc.failure_cases.head(20).to_string(index=False))
        sys.exit(1)

    stats_errors = statistical_checks(validated)
    if stats_errors:
        print("Statistical validation FAILED:")
        for e in stats_errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"PASS: {len(validated)} tickets validated — schema + statistical checks passed")
    print(validated["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
