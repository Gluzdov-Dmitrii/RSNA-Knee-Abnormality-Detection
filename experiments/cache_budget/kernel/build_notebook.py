from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KERNEL = Path(__file__).resolve().parent
FOLDS_SRC = ROOT / "ops" / "assets" / "FOLDS_V1" / "folds.csv"
EXPECTED = "3086df3341333f44adb883292da386857c3230eaa2d501514ddf827a2da11b1a"
CODE_FILE = "rsna-cache-budget-probe.ipynb"


def main() -> None:
    raw = FOLDS_SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != EXPECTED:
        raise SystemExit(f"FOLDS_V1 hash mismatch: {sha}")
    (KERNEL / "folds.csv").write_bytes(raw)
    b64 = base64.b64encode(gzip.compress(raw)).decode("ascii")
    probe = (KERNEL / "probe.py").read_text(encoding="utf-8")
    if not probe.endswith("\n"):
        probe += "\n"

    prelude = f'''from pathlib import Path
import base64, gzip, hashlib

EXPECTED = "{EXPECTED}"
raw = gzip.decompress(base64.b64decode("""{b64}"""))
sha = hashlib.sha256(raw).hexdigest()
if sha != EXPECTED:
    raise SystemExit(f"embedded FOLDS_V1 hash mismatch: {{sha}}")
Path("/kaggle/working/folds.csv").write_bytes(raw)
print("wrote folds.csv", sha[:12], "n=", raw.count(b"\\n") - 1)
'''

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": (
                    "# RSNA Knee cache-budget probe\n\n"
                    "Private CPU measurement of **quality vs full-corpus GiB**, citing "
                    "[Steven Lee](https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache) "
                    "(Apache 2.0). Geometry knobs only; we do **not** vendor his report lexicon.\n\n"
                    "- Labels: `pilkwang/rsna-knee-llm-labels` `report_labels_v2.csv` (`LABEL_PILKWANG_V1`)\n"
                    "- Folds: embedded `FOLDS_V1` (seed 2026)\n"
                    "- Subset: 32 studies/fold, seed 2026\n"
                    "- Not a leaderboard run. No dataset publish. No GPU.\n"
                ),
            },
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": prelude},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": probe},
        ],
    }
    (KERNEL / CODE_FILE).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": "dmitriigluzdov/rsna-cache-budget-probe",
        "title": "rsna-cache-budget-probe",
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": ["pilkwang/rsna-knee-llm-labels"],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
    }
    (KERNEL / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote", KERNEL / CODE_FILE, "bytes", (KERNEL / CODE_FILE).stat().st_size)


if __name__ == "__main__":
    main()
