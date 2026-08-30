import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "validation" / "validate_data.py"), run_name="__main__")
