import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(module: str) -> None:
    print(f"\n=== {module} ===")
    result = subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    run("src.data.generate")
    run("src.data.validate")
    run("src.features.build_features")
    print("\nWeek 1 pipeline complete.")


if __name__ == "__main__":
    main()
