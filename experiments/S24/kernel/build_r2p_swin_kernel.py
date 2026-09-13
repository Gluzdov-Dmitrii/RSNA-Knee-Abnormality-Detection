"""Build r2plus1d + Swin3D-T rank-mean inference notebook."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CELL = Path(__file__).resolve().parent / "s24_r2p_swin_cell.py"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
KERNEL = "dmitriigluzdov/rsna-s24-r2p-swin-rankmean"
TITLE = "rsna-s24-r2p-swin-rankmean"
CODE_FILE = "rsna-s24-r2p-swin-rankmean.ipynb"
DATASETS = [
    "dmitriigluzdov/rsna-s24-r2plus1d-9slice-folds",
    "dmitriigluzdov/rsna-s24-swin3d-t-9slice-folds",
]
OUT_DIR = ROOT / "tmp" / "kernel_s24_r2p_swin"
PUSH_DIR = ROOT / "tmp" / "kernel_s24_r2p_swin_push"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = CELL.read_text(encoding="utf-8")
    if not source.startswith("# S24_RANKMEAN_R2P_SWIN"):
        raise SystemExit("r2p+swin cell marker missing")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "python3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# S24 r2plus1d_18 + Swin3D-T rank-mean\n",
                    "\n",
                    "Own PIXEL_CACHE_V1 decoder once, then two complementary 5-fold 3D bags. No A0/Raptor mix.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [line + "\n" for line in source.splitlines()],
            },
        ],
    }
    metadata = {
        "id": KERNEL,
        "title": TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": DATASETS,
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "docker_image": DOCKER,
        "machine_shape": "NvidiaTeslaT4",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PUSH_DIR.mkdir(parents=True, exist_ok=True)
    dest_nb = OUT_DIR / CODE_FILE
    dest_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta_text = json.dumps(metadata, indent=2) + "\n"
    (OUT_DIR / "kernel-metadata.json").write_text(meta_text, encoding="utf-8")
    (PUSH_DIR / CODE_FILE).write_bytes(dest_nb.read_bytes())
    (PUSH_DIR / "kernel-metadata.json").write_text(meta_text, encoding="utf-8")
    receipt = {
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kernel": KERNEL,
        "notebook_sha256": sha256_file(dest_nb),
        "metadata_sha256": hashlib.sha256(meta_text.encode("utf-8")).hexdigest(),
        "cell_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "datasets": DATASETS,
    }
    (OUT_DIR / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
