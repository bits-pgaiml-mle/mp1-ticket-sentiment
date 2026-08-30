"""Expand Flavor C demo CSVs so amazon/yelp/sentiment140 can run full M2+M3 reports."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(42)
N = 1500

AMAZON_POS = [
    "Great battery life and build quality, highly recommend.",
    "Excellent packaging and fast Amazon delivery.",
    "Works better than expected for the price.",
    "Crystal clear sound and solid construction.",
    "Setup was easy and performance is excellent.",
]
AMAZON_NEG = [
    "Stopped working after two days, waste of money.",
    "Screen cracked in transit, support ignored me.",
    "Terribly slow and overheats constantly.",
    "False advertising, product does not match listing.",
    "Charging port failed within a week.",
]
AMAZON_NEU = [
    "Average product, nothing special but does the job.",
    "Okay for the price, a few missing features.",
    "Decent quality, shipping took longer than stated.",
    "Fine as a backup device, not for daily heavy use.",
    "Acceptable build, instructions could be clearer.",
]

YELP_POS = [
    "Best pasta in town, friendly staff and cozy vibe.",
    "Amazing cocktails and live music on Friday.",
    "Fantastic brunch and attentive service.",
    "Loved the desserts, will definitely come back.",
    "Clean place, generous portions, fair prices.",
]
YELP_NEG = [
    "Rude service and cold food, never going back.",
    "Dirty tables and overpriced drinks.",
    "Waited an hour and the order was wrong.",
    "Food was bland and the place was noisy.",
    "Manager refused to fix a clearly wrong bill.",
]
YELP_NEU = [
    "Decent brunch, wait time was long though.",
    "Solid neighborhood spot for a quick bite.",
    "Average coffee, parking is limited nearby.",
    "Okay pizza, nothing memorable either way.",
    "Service was fine, ambience is just average.",
]

S140_POS = [
    "loving this sunny weather today",
    "congrats on the launch everyone",
    "just got promoted feeling great",
    "weekend vibes with good friends",
    "this playlist is making my day",
]
S140_NEG = [
    "stuck in traffic again this sucks",
    "my phone died mid call",
    "feeling sick and exhausted today",
    "another delay on my train ride",
    "worst customer service ever",
]
S140_NEU = [
    "just another monday",
    "watching the game later",
    "heading to the store now",
    "need to finish this report",
    "coffee then emails then meetings",
]

SUFFIX = ["", "!", "!!", " asap", " thanks", " fyi", " please help"]


def _pick(pool: list[str]) -> str:
    return pool[int(RNG.integers(0, len(pool)))]


def _noise(text: str) -> str:
    return f"{text}{_pick(SUFFIX)}".strip()


def write_amazon(path: Path, n: int = N) -> None:
    rows = []
    for _ in range(n):
        bucket = int(RNG.integers(0, 3))
        if bucket == 0:
            text, stars = _noise(_pick(AMAZON_NEG)), int(RNG.choice([1, 2]))
        elif bucket == 1:
            text, stars = _noise(_pick(AMAZON_NEU)), 3
        else:
            text, stars = _noise(_pick(AMAZON_POS)), int(RNG.choice([4, 5]))
        rows.append({"reviewText": text, "overall": stars})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {n} -> {path}")


def write_yelp(path: Path, n: int = N) -> None:
    rows = []
    for _ in range(n):
        bucket = int(RNG.integers(0, 3))
        if bucket == 0:
            text, stars = _noise(_pick(YELP_NEG)), int(RNG.choice([1, 2]))
        elif bucket == 1:
            text, stars = _noise(_pick(YELP_NEU)), 3
        else:
            text, stars = _noise(_pick(YELP_POS)), int(RNG.choice([4, 5]))
        rows.append({"text": text, "stars": stars})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {n} -> {path}")


def write_sentiment140(path: Path, n: int = N) -> None:
    rows = []
    for i in range(n):
        bucket = int(RNG.integers(0, 3))
        if bucket == 0:
            text, target = _noise(_pick(S140_NEG)), 0
        elif bucket == 1:
            text, target = _noise(_pick(S140_NEU)), 2
        else:
            text, target = _noise(_pick(S140_POS)), 4
        rows.append(
            {
                "target": target,
                "id": i,
                "date": "Mon Jun 01 00:00:00 PDT 2009",
                "flag": "NO_QUERY",
                "user": f"user{i}",
                "text": text,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, header=False)
    print(f"Wrote {n} -> {path}")


def main() -> None:
    write_amazon(ROOT / "data/external/kaggle/amazon/reviews.csv")
    write_yelp(ROOT / "data/external/kaggle/yelp/reviews.csv")
    write_sentiment140(ROOT / "data/external/kaggle/sentiment140/tweets.csv")


if __name__ == "__main__":
    main()
