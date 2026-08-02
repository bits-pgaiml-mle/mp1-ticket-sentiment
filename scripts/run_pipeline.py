import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n=== {' '.join(cmd)} ===")
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    py = sys.executable
    run([py, "data/generate_data.py"])
    run([py, "data/validate.py"])
    run([py, "features/build_features.py"])
    run([py, "training/train.py"])
    print("\nPipeline complete (M2+M3). Serve with:")
    print("  uvicorn serving.api:app --reload --port 8000")
    print("Then:")
    print("  python monitoring/simulate_concept_drift.py")
    print("  python monitoring/check_drift.py")


if __name__ == "__main__":
    main()
