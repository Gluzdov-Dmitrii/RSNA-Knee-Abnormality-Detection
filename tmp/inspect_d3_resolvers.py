import json
import re
from pathlib import Path

root = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\open_kernels_d3")
needles = [
    "def _asset_root",
    "def _comp_root",
    "def _competition_root",
    "def locate_competition_root",
    "def find_test_root",
    "ASSET = Path",
]


def cell_src(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src or ""


for name in [
    "prvsiyan__head-and-shoulders-knees-and-toes",
    "junaid512__rsna-base-enc",
    "maverickss26__rsna-knee-restructured-version-2",
    "nishantkharga__rsna-knee-full-4-arm-ensemble-v55",
    "hyakumanben2025__rsna-knee-dinosaur-v5-repro",
]:
    d = root / name
    nb = json.loads(next(d.glob("*.ipynb")).read_text(encoding="utf-8"))
    src = "\n".join(cell_src(c) for c in nb.get("cells", []))
    print("=" * 60, name)
    for needle in needles:
        i = src.find(needle)
        if i < 0:
            continue
        print(src[i : i + 900])
        print("---")
