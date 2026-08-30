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
    run(["data/prepare_dataset.py"])
    run(["validation/validate_data.py"])
    run(["features/build_features.py"])
    print("\nM2 pipeline complete (prepare -> validate -> feature store).")


if __name__ == "__main__":
    main()
