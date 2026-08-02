import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from generate_data import generate_tickets

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "data_source.yaml"
RAW_PATH = ROOT / "data" / "raw" / "tickets.csv"

SOURCE_ALIASES = {
    "synthetic": "support_tickets",
    "kaggle": "amazon",
    "both": "all",
    "twitter": "sentiment140",
    "sentiment_140": "sentiment140",
    "support": "support_tickets",
    "tickets": "support_tickets",
}
VALID_SOURCES = ("amazon", "yelp", "sentiment140", "support_tickets", "all")


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_source(source: str) -> str:
    source = source.lower().strip().replace("-", "_")
    return SOURCE_ALIASES.get(source, source)


def map_label(value) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"negative", "neutral", "positive"}:
            return v
        if v in {"neg", "0"}:
            return "negative"
        if v in {"pos", "4", "1"} and v != "0":
            if v == "1":
                return "positive"
            return "positive" if v in {"pos", "4"} else "negative"
        if v in {"2", "neutral"}:
            return "neutral"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if score in (0, 4) and score == int(score):
        return "negative" if int(score) == 0 else "positive"
    if score <= 2:
        return "negative"
    if score >= 4:
        return "positive"
    return "neutral"


def map_sentiment140_target(value) -> str:
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return map_label(value)
    if v == 0:
        return "negative"
    if v == 2:
        return "neutral"
    if v == 4:
        return "positive"
    return map_label(value)


DEMO_TEXT = {
    "amazon": [
        ("Great battery life and build quality, highly recommend.", 5),
        ("Stopped working after two days, waste of money.", 1),
        ("Average product, nothing special but does the job.", 3),
        ("Excellent packaging and fast Amazon delivery.", 5),
        ("Screen cracked in transit, support ignored me.", 1),
        ("Okay for the price, a few missing features.", 3),
    ],
    "yelp": [
        ("Best pasta in town, friendly staff and cozy vibe.", 5),
        ("Rude service and cold food, never going back.", 1),
        ("Decent brunch, wait time was long though.", 3),
        ("Amazing cocktails and live music on Friday.", 5),
        ("Dirty tables and overpriced drinks.", 1),
        ("Solid neighborhood spot for a quick bite.", 3),
    ],
    "sentiment140": [
        ("loving this sunny weather today", 4),
        ("stuck in traffic again this sucks", 0),
        ("just another monday", 2),
        ("congrats on the launch everyone", 4),
        ("my phone died mid call", 0),
        ("watching the game later", 2),
    ],
    "support_tickets": [
        ("Please refund my duplicate charge from yesterday.", "negative"),
        ("Thanks, the agent resolved my login issue quickly.", "positive"),
        ("How do I update the shipping address on my order?", "neutral"),
        ("App keeps crashing when I open payment settings.", "negative"),
        ("Great support chat experience, problem fixed.", "positive"),
        ("Need documentation for API rate limits please.", "neutral"),
    ],
}


def ensure_demo_csv(source: str, path: Path, section: dict) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = DEMO_TEXT[source]
    text_col = section.get("text_col", "text")
    label_col = section.get("label_col", "label")

    if source == "sentiment140" and not section.get("has_header", True):
        demo = pd.DataFrame(
            [
                {
                    "target": lab,
                    "id": i,
                    "date": "Mon Jun 01 00:00:00 PDT 2009",
                    "flag": "NO_QUERY",
                    "user": f"user{i}",
                    "text": txt,
                }
                for i, (txt, lab) in enumerate(rows)
            ]
        )
        demo.to_csv(path, index=False, header=False)
    else:
        demo = pd.DataFrame([{text_col: txt, label_col: lab} for txt, lab in rows])
        demo.to_csv(path, index=False)
    print(f"Created demo {source} CSV at {path} (replace with real dataset extract)")
    return path


def read_source_csv(path: Path, section: dict) -> pd.DataFrame:
    if section.get("has_header") is False:
        names = section.get("column_names") or ["target", "id", "date", "flag", "user", "text"]
        return pd.read_csv(path, header=None, names=names, encoding="latin-1", on_bad_lines="skip")
    return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")


def adapt_dataframe(
    df: pd.DataFrame,
    source: str,
    text_col: str,
    label_col: str,
    max_rows: int,
    seed: int,
    channel: str,
) -> pd.DataFrame:
    if text_col not in df.columns:
        for c in ["text", "reviewText", "review", "content", "sentence", "tweet"]:
            if c in df.columns:
                text_col = c
                break
    if label_col not in df.columns:
        for c in ["label", "overall", "sentiment", "stars", "rating", "target", "score"]:
            if c in df.columns:
                label_col = c
                break
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"{source}: could not find text/label columns in {list(df.columns)}")

    df = df.dropna(subset=[text_col, label_col]).copy()
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)

    label_mapper = map_sentiment140_target if source == "sentiment140" else map_label
    prefix = {"amazon": "A", "yelp": "Y", "sentiment140": "T", "support_tickets": "S"}.get(source, "X")
    rng = np.random.default_rng(seed)
    channels = rng.choice(
        ["email", "chat", "app"],
        size=len(df),
        p=[0.45, 0.35, 0.20] if channel == "email" else ([0.20, 0.45, 0.35] if channel == "chat" else [0.20, 0.30, 0.50]),
    )
    out = pd.DataFrame(
        {
            "ticket_id": [f"{prefix}{i:06d}" for i in range(len(df))],
            "text": df[text_col].astype(str).str.strip(),
            "channel": channels,
            "label": df[label_col].map(label_mapper),
            "data_source": source,
        }
    )
    return out[out["text"].str.len() > 0].reset_index(drop=True)


def load_named_source(cfg: dict, source: str, max_rows: int | None = None) -> pd.DataFrame:
    section = cfg.get(source, {})
    seed = int(section.get("seed", cfg.get("support_tickets", {}).get("seed", 42)))
    path = ROOT / section.get("local_csv", f"data/external/kaggle/{source}/data.csv")
    n = int(max_rows if max_rows is not None else section.get("max_rows", 3000))

    if source == "support_tickets" and (not path.exists()) and section.get("use_synthetic_if_missing", True):
        syn_n = int(section.get("synthetic_rows", n))
        df = generate_tickets(syn_n, seed)
        df["data_source"] = "support_tickets"
        return df.head(n).reset_index(drop=True)

    ensure_demo_csv(source, path, section)
    raw = read_source_csv(path, section)
    return adapt_dataframe(
        raw,
        source,
        section.get("text_col", "text"),
        section.get("label_col", "label"),
        n,
        seed,
        section.get("channel", "email"),
    )


def prepare(source: str | None = None) -> Path:
    cfg = load_config()
    source = normalize_source(source or cfg.get("data_source") or "support_tickets")
    if source not in VALID_SOURCES:
        raise ValueError(f"data_source must be one of: {', '.join(VALID_SOURCES)}")

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    if source == "all":
        all_cfg = cfg.get("all", {})
        parts = [
            load_named_source(cfg, "amazon", int(all_cfg.get("amazon_rows", 800))),
            load_named_source(cfg, "yelp", int(all_cfg.get("yelp_rows", 800))),
            load_named_source(cfg, "sentiment140", int(all_cfg.get("sentiment140_rows", 800))),
            load_named_source(cfg, "support_tickets", int(all_cfg.get("support_ticket_rows", 800))),
        ]
        df = pd.concat(parts, ignore_index=True)
        df["ticket_id"] = [f"M{i:06d}" for i in range(len(df))]
    else:
        df = load_named_source(cfg, source)

    df.to_csv(RAW_PATH, index=False)
    print(f"Prepared source={source} -> {RAW_PATH} ({len(df)} rows)")
    print(df["data_source"].value_counts().to_string())
    return RAW_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Flavor C data: amazon | yelp | sentiment140 | support_tickets | all"
    )
    parser.add_argument(
        "--source",
        choices=[*VALID_SOURCES, "synthetic", "kaggle", "both", "twitter", "support", "tickets"],
        default=None,
        help="Override configs/data_source.yaml",
    )
    args = parser.parse_args()
    prepare(args.source)


if __name__ == "__main__":
    main()
