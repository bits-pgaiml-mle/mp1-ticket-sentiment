from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

TEMPLATES = {
    "negative": [
        "My order arrived damaged and support has not replied.",
        "The app keeps crashing whenever I try to checkout.",
        "Terrible experience, refund still pending after one week.",
        "Agent was rude and the ticket was closed without resolution.",
        "Delivery is late again and tracking is completely wrong.",
        "I am extremely unhappy with the product quality.",
        "Charged twice for the same invoice, please fix this.",
    ],
    "neutral": [
        "Can you confirm the warranty period for this product?",
        "I need help updating my billing address.",
        "Please share the status of ticket related to invoice copy.",
        "Where can I find documentation for API rate limits?",
        "I want to change the delivery slot for tomorrow.",
        "Looking for details about return policy window.",
        "Need clarification on subscription renewal date.",
    ],
    "positive": [
        "Support resolved my issue quickly, thank you!",
        "Great product quality and packaging was perfect.",
        "The new chat assistant answered my question instantly.",
        "Happy with the replacement, excellent service.",
        "Fast delivery and the item matches the description.",
        "Very satisfied with the quick refund process.",
        "Awesome experience, will recommend to friends.",
    ],
}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_tickets(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = list(TEMPLATES.keys())
    rows = []
    for i in range(n):
        label = labels[int(rng.integers(0, len(labels)))]
        text = TEMPLATES[label][int(rng.integers(0, len(TEMPLATES[label])))]
        noise = rng.choice(
            ["", " please help", " asap", " thanks", "!!", " FYI"],
            p=[0.35, 0.2, 0.15, 0.15, 0.1, 0.05],
        )
        rows.append(
            {
                "ticket_id": f"TKT{i:06d}",
                "text": f"{text}{noise}".strip(),
                "channel": rng.choice(["email", "chat", "app"]),
                "label": label,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    out_path = ROOT / cfg["data"]["raw_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_tickets(int(cfg["data"]["sample_size"]), int(cfg["data"]["random_seed"]))
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} tickets -> {out_path}")
    print("Raw data is immutable after this step. Do not edit in place.")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
