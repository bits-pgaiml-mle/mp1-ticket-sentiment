from pathlib import Path

import joblib
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_schema():
    cfg = load_config()
    model_path = ROOT / cfg["training"]["model_path"]
    schema_path = ROOT / cfg["data"]["feature_schema"]
    version_tag = cfg["serving"]["model_version"]

    if not model_path.exists():
        return None, None, version_tag

    bundle = joblib.load(model_path)
    feature_columns = None
    if schema_path.exists():
        import json

        feature_columns = json.loads(schema_path.read_text(encoding="utf-8"))
    return bundle, feature_columns, version_tag
