import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from generate_data import generate_tickets

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "data_source.yaml"
RAW_PATH = ROOT / "data" / "raw" / "tickets.csv"
SYN_EXT = ROOT / "data" / "external" / "synthetic" / "tickets.csv"
KAG_EXT = ROOT / "data" / "external" / "kaggle" / "tickets_from_kaggle.csv"
DEFAULT_KAGGLE_CSV = ROOT / "data" / "external" / "kaggle" / "reviews_sample.csv"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def map_label(value) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"negative", "neutral", "positive"}:
            return v
        if v in {"neg", "0", "1"}:
            return "negative" if v != "1" else "positive"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if score <= 2:
        return "negative"
    if score >= 4:
        return "positive"
    return "neutral"


def ensure_demo_reviews(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    demo = pd.DataFrame(
        {
            "text": [
                "Amazing product, fast delivery and great support.",
                "Package arrived damaged and nobody replied.",
                "Where can I find the warranty information?",
                "Refund was processed within a day, thank you.",
                "App crashes every time I open checkout.",
                "Need to update my billing address please.",
            ],
            "label": [5, 1, 3, 5, 1, 3],
        }
    )
    demo.to_csv(path, index=False)
    print(f"Created demo reviews CSV at {path} (replace with Amazon/Yelp/Sentiment140 extract)")
    return demo


def adapt_kaggle(df: pd.DataFrame, text_col: str, label_col: str, max_rows: int, seed: int) -> pd.DataFrame:
    if text_col not in df.columns:
        for c in ["text", "reviewText", "review", "content", "sentence"]:
            if c in df.columns:
                text_col = c
                break
    if label_col not in df.columns:
        for c in ["label", "overall", "sentiment", "stars", "rating"]:
            if c in df.columns:
                label_col = c
                break
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Could not find text/label columns in {list(df.columns)}")

    df = df.dropna(subset=[text_col, label_col]).copy()
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)

    rng = np.random.default_rng(seed)
    out = pd.DataFrame(
        {
            "ticket_id": [f"K{i:06d}" for i in range(len(df))],
            "text": df[text_col].astype(str).str.strip(),
            "channel": rng.choice(["email", "chat", "app"], len(df)),
            "label": df[label_col].map(map_label),
            "data_source": "kaggle",
        }
    )
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def prepare(source: str | None = None) -> Path:
    cfg = load_config()
    source = (source or cfg.get("data_source") or "synthetic").lower().strip()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    if source == "synthetic":
        n = int(cfg.get("synthetic", {}).get("n_rows", 1500))
        seed = int(cfg.get("synthetic", {}).get("seed", 42))
        df = generate_tickets(n, seed)
        df["data_source"] = "synthetic"
        SYN_EXT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SYN_EXT, index=False)
        shutil.copy(SYN_EXT, RAW_PATH)

    elif source == "kaggle":
        kcfg = cfg.get("kaggle", {})
        raw = ensure_demo_reviews(ROOT / kcfg.get("local_csv", DEFAULT_KAGGLE_CSV))
        df = adapt_kaggle(
            raw,
            kcfg.get("text_col", "text"),
            kcfg.get("label_col", "label"),
            int(kcfg.get("max_rows", 5000)),
            int(cfg.get("synthetic", {}).get("seed", 42)),
        )
        KAG_EXT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(KAG_EXT, index=False)
        df.to_csv(RAW_PATH, index=False)

    elif source == "both":
        syn_n = int(cfg.get("both", {}).get("synthetic_rows", 800))
        kag_n = int(cfg.get("both", {}).get("kaggle_rows", 2000))
        seed = int(cfg.get("synthetic", {}).get("seed", 42))
        syn = generate_tickets(syn_n, seed)
        syn["data_source"] = "synthetic"
        kcfg = cfg.get("kaggle", {})
        raw = ensure_demo_reviews(ROOT / kcfg.get("local_csv", DEFAULT_KAGGLE_CSV))
        kag = adapt_kaggle(
            raw,
            kcfg.get("text_col", "text"),
            kcfg.get("label_col", "label"),
            kag_n,
            seed,
        )
        df = pd.concat([syn, kag], ignore_index=True)
        df["ticket_id"] = [f"M{i:06d}" for i in range(len(df))]
        df.to_csv(RAW_PATH, index=False)
    else:
        raise ValueError("data_source must be one of: synthetic, kaggle, both")

    print(f"Prepared source={source} -> {RAW_PATH} ({len(df)} rows)")
    print(df["data_source"].value_counts().to_string())
    return RAW_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "kaggle", "both"], default=None)
    args = parser.parse_args()
    prepare(args.source)


if __name__ == "__main__":
    main()
