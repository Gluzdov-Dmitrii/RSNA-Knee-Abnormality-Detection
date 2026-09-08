"""Build the public notebook from verified probe and Dataset receipts."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "recheck" / "build_public.py"), run_name="__main__")
