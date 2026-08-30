from pathlib import Path

REQUIRED = [
    "data/raw",
    "validation/validate_data.py",
    "features/build_features.py",
    "feature_store",
    "training/train.py",
    "serving/api.py",
    "serving/model_loader.py",
    "serving/inference_schema.py",
    "monitoring/check_drift.py",
    "model_store",
    "ui/app.py",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [p for p in REQUIRED if not (root / p).exists()]
    if missing:
        print("Setup incomplete. Missing:")
        for m in missing:
            print(f"  - {m}")
        raise SystemExit(1)
    print("OK: Teams/Taxila layout packages are present.")


if __name__ == "__main__":
    main()
