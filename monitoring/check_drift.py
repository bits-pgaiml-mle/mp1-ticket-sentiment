import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    train = pd.read_sql(
        f"SELECT * FROM {cfg['data']['feature_table']}",
        sqlite3.connect(ROOT / cfg["data"]["feature_store"]),
    )
    pred_db = ROOT / cfg["monitoring"]["predictions_db"]
    if not pred_db.exists():
        print(f"No predictions DB at {pred_db}. Call /predict first.")
        sys.exit(1)

    prod = pd.read_sql("SELECT * FROM predictions", sqlite3.connect(pred_db))
    threshold = float(cfg["monitoring"]["drift_shift_threshold"])
    print(f"Training: {len(train)} tickets | Production logs: {len(prod)} predictions")
    print()

    for feat in ["text_len", "word_count"]:
        t_mean, t_std = train[feat].mean(), train[feat].std()
        p_mean = prod[feat].mean()
        shift = abs(p_mean - t_mean) / (t_std + 1e-9)
        flag = "DRIFTED" if shift > threshold else "OK"
        print(f"{feat:22s} train={t_mean:6.2f} prod={p_mean:6.2f} shift={shift:.2f} {flag}")

    print()
    print("Channel mix (production predictions):")
    for ch in ["channel_email", "channel_chat", "channel_app"]:
        print(f"  {ch:18s} {prod[ch].mean()*100:5.1f}%")

    print()
    print("Predicted label breakdown:")
    dist = prod["label"].value_counts(normalize=True).mul(100)
    for lbl, pct in dist.items():
        print(f"  {lbl:10s} {pct:5.1f}%")

    print()
    print("Retrain trigger (draft):")
    print(
        "Retrain when rolling accuracy < 0.80 with >=200 labeled tickets, "
        "or when text_len/word_count shift > threshold for 3 consecutive batches."
    )


if __name__ == "__main__":
    main()
