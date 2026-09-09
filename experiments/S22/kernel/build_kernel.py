"""Copy the A0 batch notebook, pin SELECTED_SUBMISSION_SID=S01, append S22 overlay."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_NB = ROOT / "experiments" / "S01" / "kernel" / "rsna-week1-a0-s01-s05.ipynb"
CELL = Path(__file__).resolve().parent / "s22_overlay_cell.py"
OUT_DIR = ROOT / "tmp" / "kernel_s22"
CODE_FILE = "rsna-s22-a0-rank-blend.ipynb"
DATASET = "dmitriigluzdov/rsna-s22-resnet18-25d-folds"

SELECTED = """# SELECTED_SUBMISSION_SID
from pathlib import Path
import hashlib
SID = "S01"
src = Path(f"/kaggle/working/submission_{SID}.csv")
dst = Path("/kaggle/working/submission.csv")
if not src.is_file():
    raise FileNotFoundError(src)
payload = src.read_bytes()
dst.write_bytes(payload)
print(f"SELECTED {SID} sha256={hashlib.sha256(payload).hexdigest()} bytes={len(payload)}", flush=True)
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    overlay = CELL.read_text(encoding="utf-8")
    if not overlay.startswith("# S22_A0_TRANSFORMER_MIX_OVERLAY"):
        raise SystemExit("overlay cell marker missing")
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    found = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if src.startswith("# SELECTED_SUBMISSION_SID"):
            cell["source"] = [line + "\n" for line in SELECTED.splitlines()]
            found = True
            break
    if not found:
        raise SystemExit("SELECTED_SUBMISSION_SID cell missing")
    last = "".join(nb["cells"][-1].get("source", []))
    if last.startswith("# S22_A0_TRANSFORMER_MIX_OVERLAY"):
        nb["cells"][-1]["source"] = [line + "\n" for line in overlay.splitlines()]
    else:
        nb["cells"].append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [line + "\n" for line in overlay.splitlines()],
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest_nb = OUT_DIR / CODE_FILE
    dest_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {
        "id": "dmitriigluzdov/rsna-s22-a0-rank-blend",
        "title": "rsna-s22-a0-rank-blend",
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dreaddevelopment/raptor-knee-maxspan",
            "dreaddevelopment/raptor-knee-native384",
            "dreaddevelopment/raptor-knee-native384dense",
            "tonylica/rsna-knee-bend-dinov3-0917-repro-assets",
            "renta0426/rsna-knee-public0033-meniscus-bag-v1",
            DATASET,
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaTeslaT4",
    }
    meta_path = OUT_DIR / "kernel-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_notebook_sha256": sha256_file(SRC_NB),
        "overlay_cell_sha256": hashlib.sha256(overlay.encode("utf-8")).hexdigest(),
        "notebook_sha256": sha256_file(dest_nb),
        "metadata_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "kernel": metadata["id"],
        "dataset": DATASET,
        "selected_submission_sid": "S01",
        "n_cells": len(nb["cells"]),
    }
    (OUT_DIR / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(CELL, OUT_DIR / "s22_overlay_cell.py")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
