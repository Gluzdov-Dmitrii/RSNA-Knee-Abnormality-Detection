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
CODE_FILE = "rsna-knee-on-a-storage-budget.ipynb"
PROBE = KERNEL / "probe_v2.py"


def cell(kind: str, source: str, cid: str) -> dict:
    body = source.strip("\n") + "\n"
    if kind == "markdown":
        return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": body}
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": body,
    }


def split_probe() -> dict[str, str]:
    text = PROBE.read_text(encoding="utf-8")
    parts: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in text.splitlines(True):
        if line.startswith("# === CELL:"):
            if current is not None:
                parts[current] = "".join(buf)
            current = line.split("CELL:")[1].strip().strip("= ").strip()
            buf = []
        else:
            if current is None:
                continue
            buf.append(line)
    if current is not None:
        parts[current] = "".join(buf)
    return parts


def plots_only() -> str:
    parts = split_probe()
    text = parts["plots"]
    marker = "if __name__"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    return text


def main() -> None:
    raw = FOLDS_SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != EXPECTED:
        raise SystemExit(f"FOLDS_V1 hash mismatch: {sha}")
    (KERNEL / "folds.csv").write_bytes(raw)
    b64 = base64.b64encode(gzip.compress(raw)).decode("ascii")
    parts = split_probe()

    folds_cell = f'''from pathlib import Path
import base64, gzip, hashlib

EXPECTED = "{EXPECTED}"
raw = gzip.decompress(base64.b64decode("""{b64}"""))
sha = hashlib.sha256(raw).hexdigest()
if sha != EXPECTED:
    raise SystemExit(f"embedded FOLDS_V1 hash mismatch: {{sha}}")
Path("/kaggle/working/folds.csv").write_bytes(raw)
print("FOLDS_V1", sha[:12], "studies", raw.count(b"\\n") - 1)
'''

    cells = [
        cell(
            "markdown",
            """# RSNA Knee on a storage budget

The competition train set is on the order of **half a terabyte**. You do not need that much resident disk to train a 2.5D model.

[Steven Lee](https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache) published an Apache 2.0 CPU pipeline that writes a **11.1 GiB** uint8 cache: six slots × nine physically ordered slices × 224², 1st–99th percentile, centre crop in millimetres. This notebook asks the next question:

> If 11 GiB is a *design*, where is the knee of the quality-vs-size curve?

We do **not** materialise an 11 GiB file here, we do **not** publish a Dataset of derived MRI, and we do **not** score the leaderboard. The metric is locked-fold weak-label OOF macro AUC of a tiny GBDT on slot summary statistics, plus SSIM to the densest crop-130 cache in the sweep. That is a probe of *how much signal survives the cache*, not a DINOv2 ranking.

**Runtime.** CPU, internet off, competition data + Pilkwang public labels. About 200 studies (40 per `FOLDS_V1` fold). Expect well under an hour.
""",
            "md-intro",
        ),
        cell(
            "markdown",
            """## Credits and licence

Geometry and the 11 GiB recipe follow Steven Lee, who in turn credits Pilkwang's slot scheme, Karnakbayev's 2.5D grouping, and Will's physical millimetres. This notebook reimplements a compatible decode (public `train_series` plane × `Fluid_Sensitive`, not his recovered FS/T1 mapping) and **does not copy his report lexicon**.

| Piece | Source |
| --- | --- |
| Pixel cache geometry | `stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache` (Apache 2.0) |
| Weak labels | `pilkwang/rsna-knee-llm-labels` / `report_labels_v2.csv` |
| Study folds | study-level 5-fold iterative stratification, seed 2026 (`FOLDS_V1`) |

Use of the MRI is governed by the competition rules and [MIRA](http://rsna.org/mira-license). A notebook that reads the mount is fine. A public Dataset of derived pixels needs a separate permission check; this kernel does not write one.
""",
            "md-credits",
        ),
        cell("code", folds_cell, "code-folds"),
        cell(
            "markdown",
            """## Size model

A uint8 cache is linear in studies, slots and slices, and quadratic in `img`:

`GiB = 4407 × 6 × n_slices × img² / 1024³`

Steven's published shape is `4407 × 6 × 9 × 224 × 224` → **11.12 GiB**. Crop millimetres do not appear in that formula. Changing crop at fixed `img` and `n_slices` is a *geometry* change, not a storage change. The plots keep that distinction: the left panel is size, the right panel is crop at 11 GiB.
""",
            "md-size",
        ),
        cell(
            "code",
            "from __future__ import annotations\n\n" + parts["setup"] + "\n" + parts["variants"],
            "code-setup",
        ),
        cell(
            "code",
            """plan = planned_variants()
print(f"Steven 224²×9 = {STEVEN_GIB:.3f} GiB")
try:
    display(plan)
except Exception:
    print(plan.to_string(index=False))
""",
            "code-plan",
        ),
        cell(
            "markdown",
            """## Decode

Filenames in this corpus are SOP UIDs, so lexicographic order is almost random (~5% agreement with anatomy in Steven's measurement). We order by `ImagePositionPatient` projected on the orientation normal, then `SliceLocation`, then `InstanceNumber`.

The **resolution** family is decoded once at 336² × 9, crop 130 mm, then bilinearly downsampled. That holds the slices and the crop fixed so the x-axis is really resolution. Slice-count and crop families are native decodes at 224².
""",
            "md-decode",
        ),
        cell("code", parts["io"] + "\n" + parts["decode"], "code-decode"),
        cell(
            "markdown",
            """## Probe

Same model on every cache:

- targets: Pilkwang scores binarized at 0.5
- split: `FOLDS_V1` fold ids on a fixed 200-study subset (seed 2026)
- features: 8 numbers per slot (mean, std, p10/p90, centre-slice mean, adjacent-slice MAD, in-plane gradient, missing flag)
- model: `HistGradientBoostingClassifier`, depth 3, 80 iterations
- uncertainty: 800 study-level bootstrap draws of the OOF predictions

A 6-epoch toy CNN was tried on the first five-point sweep and sat at ~0.49. It is not repeated here. This GBDT is also **not** a substitute for a frozen DINOv2 head; it is a cheap, identical probe across geometries.
""",
            "md-probe",
        ),
        cell("code", parts["metrics"] + "\n" + plots_only() + "\n" + parts["run"] + "\nmain()\n", "code-run"),
        cell(
            "markdown",
            """## How to read the curve

- **Left.** Crop 130 mm only. Resolution (circles), extra slice counts at 224² (squares), and the 3-slice tiny corner (triangles). The dotted vertical line is Steven's 11.1 GiB design. Error bars are 95% bootstrap intervals on this 200-study subset; they are *not* a public-LB CI.
- **Right.** Same 224² × 9 cache size, three crops. If 160 mm wins here, remember Steven's median-FOV trap: 160 mm sits on the median field of view, so the physical crop often does not apply. Intensity statistics like a larger FOV; a DINO token still has a millimetre pitch of `14 × crop_mm / img`.
- **SSIM plot.** Fidelity to 336² × 9 crop 130, after resampling to 64², not PSNR to raw DICOM. 224 vs 336 at the same crop should look almost lossless at that scale.

Materialise a **full-corpus** cache only after you pick a point. Do not upload derived MRI as a Dataset without a separate licence check.
""",
            "md-read",
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    (KERNEL / CODE_FILE).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": "dmitriigluzdov/rsna-knee-on-a-storage-budget",
        "title": "RSNA Knee on a storage budget",
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["eda"],
        "dataset_sources": ["pilkwang/rsna-knee-llm-labels"],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
    }
    (KERNEL / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote", KERNEL / CODE_FILE, "bytes", (KERNEL / CODE_FILE).stat().st_size)
    print("cells", [c["id"] for c in cells])


if __name__ == "__main__":
    main()
