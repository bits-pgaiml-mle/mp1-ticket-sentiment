import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print(f"\n=== {' '.join(args)} ===")
    result = subprocess.run([sys.executable, *args], cwd=ROOT, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    run(["scripts/run_m2_pipeline.py"])
    run(["training/train.py"])
    print("\nM2+M3 complete. Start API with:")
    print("  uvicorn serving.api:app --reload --port 8000")
    print("Then:")
    print("  python monitoring/simulate_concept_drift.py")
    print("  python monitoring/check_drift.py")


if __name__ == "__main__":
    main()
