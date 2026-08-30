import json
import sqlite3
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "config.yaml"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 96
EPOCHS = 1
BATCH_SIZE = 8
LR = 5e-5
MAX_TRAIN_ROWS = 800


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit(
            "Transformer extras missing. Install with:\n"
            "  pip install -r requirements-transformer.txt\n"
            f"Details: {exc}"
        ) from exc

    cfg = load_config()
    store_path = ROOT / cfg["data"]["feature_store"]
    table = cfg["data"]["feature_table"]
    model_dir = ROOT / "model_store"
    model_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(store_path)
    df = pd.read_sql(f"SELECT text_clean, label FROM {table}", conn)
    conn.close()

    if len(df) > MAX_TRAIN_ROWS:
        df = df.sample(n=MAX_TRAIN_ROWS, random_state=int(cfg["data"]["random_seed"])).reset_index(drop=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    texts = df["text_clean"].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=float(cfg["training"]["test_size"]),
        random_state=int(cfg["data"]["random_seed"]),
        stratify=y,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TextDataset(Dataset):
        def __init__(self, texts_, labels_):
            self.enc = tokenizer(
                list(texts_),
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels_, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.enc.items()}
            item["labels"] = self.labels[idx]
            return item

    train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TextDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = max(1, len(train_loader) * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    mlflow.set_experiment(cfg["training"]["mlflow_experiment"])
    t0 = time.perf_counter()

    with mlflow.start_run(run_name="distilbert_finetune"):
        mlflow.log_param("model_type", "distilbert")
        mlflow.log_param("base_model", MODEL_NAME)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("lr", LR)
        mlflow.log_param("max_train_rows", MAX_TRAIN_ROWS)
        mlflow.log_param("device", str(device))

        model.train()
        for epoch in range(EPOCHS):
            running = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                out = model(**batch)
                out.loss.backward()
                optimizer.step()
                scheduler.step()
                running += float(out.loss.item())
            print(f"epoch {epoch + 1}/{EPOCHS} mean_loss={running / max(1, len(train_loader)):.4f}")

        model.eval()
        preds = []
        gold = []
        with torch.no_grad():
            for batch in test_loader:
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                gold.extend(labels.numpy().tolist())

        metrics = {
            "accuracy": float(accuracy_score(gold, preds)),
            "f1_macro": float(f1_score(gold, preds, average="macro")),
        }
        train_seconds = time.perf_counter() - t0
        mlflow.log_metrics(metrics)
        mlflow.log_metric("train_seconds", train_seconds)

        sample = tokenizer(
            list(X_test[:8]),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        t_inf0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**sample)
        infer_ms = (time.perf_counter() - t_inf0) * 1000 / max(1, len(X_test[:8]))
        mlflow.log_metric("approx_infer_ms_per_text", infer_ms)

        decision = {
            "run_name": "distilbert_finetune",
            "base_model": MODEL_NAME,
            "metrics": metrics,
            "train_seconds": train_seconds,
            "approx_infer_ms_per_text": infer_ms,
            "device": str(device),
            "note": (
                "Comparison-only run. Production artifact remains the classical "
                "pipeline in model_store/sentiment_model.joblib for latency and size."
            ),
            "label_classes": list(label_encoder.classes_),
        }
        out_path = model_dir / "transformer_decision.json"
        out_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out_path))
        print(json.dumps(decision, indent=2))
        print(f"Wrote {out_path} (did not overwrite served classical model)")


if __name__ == "__main__":
    main()
