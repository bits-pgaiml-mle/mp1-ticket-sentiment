import json
import sqlite3
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.text_utils import select_numeric, select_text

CONFIG_PATH = ROOT / "configs" / "config.yaml"

RUNS = [
    {
        "run_name": "logreg_C1",
        "model_type": "logistic_regression",
        "params": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs", "random_state": 42},
    },
    {
        "run_name": "logreg_C10",
        "model_type": "logistic_regression",
        "params": {"C": 10.0, "max_iter": 2000, "solver": "lbfgs", "random_state": 42},
    },
    {
        "run_name": "linear_svc",
        "model_type": "linear_svc",
        "params": {"C": 1.0, "max_iter": 5000, "random_state": 42},
    },
]


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_estimator(model_type: str, params: dict):
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    if model_type == "linear_svc":
        return LinearSVC(**params)
    raise ValueError(f"Unsupported model_type: {model_type}")


def main() -> None:
    cfg = load_config()
    store_path = ROOT / cfg["data"]["feature_store"]
    table = cfg["data"]["feature_table"]
    model_dir = ROOT / "model_store"
    model_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(store_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()

    X = df.drop(columns=["ticket_id", "label"])
    y_raw = df["label"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg["training"]["test_size"]),
        random_state=int(cfg["data"]["random_seed"]),
        stratify=y,
    )

    mlflow.set_experiment(cfg["training"]["mlflow_experiment"])
    best = {"f1_macro": -1.0, "run_name": None, "artifact": None}

    for run in RUNS:
        features = FeatureUnion(
            [
                (
                    "tfidf",
                    Pipeline(
                        [
                            ("text", FunctionTransformer(select_text, validate=False)),
                            (
                                "vect",
                                TfidfVectorizer(
                                    max_features=int(cfg["features"]["max_features"]),
                                    ngram_range=tuple(cfg["features"]["ngram_range"]),
                                    min_df=int(cfg["features"]["min_df"]),
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "numeric",
                    Pipeline(
                        [
                            ("num", FunctionTransformer(select_numeric, validate=False)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                ),
            ]
        )
        clf = build_estimator(run["model_type"], run["params"])
        pipe = Pipeline([("features", features), ("clf", clf)])

        with mlflow.start_run(run_name=run["run_name"]):
            mlflow.log_param("model_type", run["model_type"])
            mlflow.log_params({f"clf__{k}": v for k, v in run["params"].items()})
            mlflow.log_param("max_features", cfg["features"]["max_features"])
            mlflow.log_param("ngram_range", str(cfg["features"]["ngram_range"]))

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            }
            mlflow.log_metrics(metrics)

            artifact = model_dir / f"{run['run_name']}.joblib"
            joblib.dump(
                {
                    "pipeline": pipe,
                    "label_encoder": label_encoder,
                    "model_type": run["model_type"],
                    "params": run["params"],
                },
                artifact,
            )
            mlflow.log_artifact(str(artifact))
            schema_path = ROOT / cfg["data"]["feature_schema"]
            if schema_path.exists():
                mlflow.log_artifact(str(schema_path))

            print(f"Run {run['run_name']}: accuracy={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f}")
            if metrics["f1_macro"] > best["f1_macro"]:
                best = {
                    "f1_macro": metrics["f1_macro"],
                    "run_name": run["run_name"],
                    "artifact": artifact,
                    "metrics": metrics,
                }

    best_path = ROOT / cfg["training"]["model_path"]
    joblib.dump(joblib.load(best["artifact"]), best_path)
    joblib.dump(label_encoder, ROOT / cfg["training"]["label_encoder_path"])

    decision = {
        "best_run": best["run_name"],
        "metrics": best["metrics"],
        "justification": (
            "Selected highest macro-F1 among LogisticRegression (C=1/C=10) and LinearSVC. "
            "TF-IDF is fit only on the training split to avoid leakage."
        ),
    }
    (model_dir / "best_model_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"Best model: {best['run_name']} -> {best_path}")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
