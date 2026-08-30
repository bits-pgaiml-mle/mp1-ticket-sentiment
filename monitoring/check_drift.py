import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitoring.logger import get_backends, jsonl_path, load_config, sqlite_path

NUMERIC_FEATURES = ["text_len", "word_count"]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def psi(expected, actual, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    eps = 1e-6
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def load_predictions(cfg: dict) -> tuple[pd.DataFrame, str]:
    backends = get_backends(cfg)
    prefer = str(cfg.get("monitoring", {}).get("drift_read_backend", "auto")).strip().lower()
    frames: dict[str, pd.DataFrame] = {}

    if "sqlite" in backends:
        db = sqlite_path(cfg)
        if db.exists():
            try:
                df = pd.read_sql("SELECT * FROM predictions", sqlite3.connect(db))
                if len(df):
                    frames["sqlite"] = df
            except Exception as exc:
                print(f"Warning: could not read SQLite predictions ({exc})")

    if "jsonl" in backends:
        path = jsonl_path(cfg)
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_json(path, lines=True)
            if len(df):
                frames["jsonl"] = df

    if not frames:
        return pd.DataFrame(), ""

    if prefer in frames:
        return frames[prefer], f"{prefer}:{sqlite_path(cfg) if prefer == 'sqlite' else jsonl_path(cfg)}"

    if len(frames) == 1:
        name, df = next(iter(frames.items()))
        path = sqlite_path(cfg) if name == "sqlite" else jsonl_path(cfg)
        return df, f"{name}:{path}"

    # both backends enabled and both have rows: prefer sqlite for drift to avoid double-count
    return frames["sqlite"], f"sqlite:{sqlite_path(cfg)} (jsonl also present at {jsonl_path(cfg)})"


def main() -> None:
    cfg = load_config()
    train = pd.read_sql(
        f"SELECT * FROM {cfg['data']['feature_table']}",
        sqlite3.connect(ROOT / cfg["data"]["feature_store"]),
    )
    prod, source = load_predictions(cfg)
    if prod.empty:
        backends = ", ".join(sorted(get_backends(cfg)))
        print(f"No prediction logs found for backend(s): {backends}. Call /predict first.")
        sys.exit(1)

    shift_threshold = float(cfg["monitoring"]["drift_shift_threshold"])
    psi_alert = float(cfg["monitoring"].get("psi_alert", 0.25))
    print(f"Training: {len(train)} tickets | Production logs: {len(prod)} predictions")
    print(f"Log source: {source}")
    print()

    print("Mean-shift checks (Taxila-style z-shift):")
    for feat in NUMERIC_FEATURES:
        t_mean, t_std = train[feat].mean(), train[feat].std()
        p_mean = prod[feat].mean()
        shift = abs(p_mean - t_mean) / (t_std + 1e-9)
        flag = "DRIFTED" if shift > shift_threshold else "OK"
        print(f"  {feat:22s} train={t_mean:6.2f} prod={p_mean:6.2f} shift={shift:.2f} {flag}")

    print()
    print("Population Stability Index (VaayuGrid / Taxila M6):")
    shifted = []
    for feat in NUMERIC_FEATURES:
        score = psi(train[feat], prod[feat])
        status = "STABLE" if score < 0.10 else ("MODERATE" if score < 0.25 else "SHIFTED")
        print(f"  {feat:22s} psi={score:.4f} {status}")
        if score > psi_alert:
            shifted.append(feat)

    print()
    print("Channel mix (production predictions):")
    for ch in ["channel_email", "channel_chat", "channel_app"]:
        if ch in prod.columns:
            print(f"  {ch:18s} {prod[ch].mean()*100:5.1f}%")

    print()
    print("Predicted label breakdown:")
    dist = prod["label"].value_counts(normalize=True).mul(100)
    for lbl, pct in dist.items():
        print(f"  {lbl:10s} {pct:5.1f}%")

    print()
    print("Retrain trigger:")
    print(
        "Retrain when rolling accuracy < 0.80 with >=200 labeled tickets, "
        "or when text_len/word_count mean-shift > threshold for 3 consecutive batches, "
        f"or when PSI > {psi_alert} on any numeric feature."
    )

    if shifted:
        print(f"\nRETRAINING TRIGGER FIRED: PSI alert on {', '.join(shifted)}")
        sys.exit(1)
    print(f"\nNo PSI drift above {psi_alert}.")


if __name__ == "__main__":
    main()
