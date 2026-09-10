"""Build a one-cell GPU notebook: live PIXEL_CACHE_V1 decode + S22 ResNet-18 bag."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CELL = Path(__file__).resolve().parent / "s22_standalone_cell.py"
OUT_DIR = ROOT / "tmp" / "kernel_s22_local"
PUSH_DIR = ROOT / "tmp" / "kernel_s22_local_push"
CODE_FILE = "rsna-s22-local-cnn.ipynb"
KERNEL = "dmitriigluzdov/rsna-s22-local-cnn"
DATASET = "dmitriigluzdov/rsna-s22-resnet18-25d-folds"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = CELL.read_text(encoding="utf-8")
    if not source.startswith("# S22_LOCAL_CNN_BAG"):
        raise SystemExit("standalone cell marker missing")
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
                    "# S22 local CNN (ResNet-18 2.5D bag)\n",
                    "\n",
                    "Own PIXEL_CACHE_V1 decoder + 5-fold bag. No A0/Raptor mix.\n",
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
        "title": "rsna-s22-local-cnn",
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [DATASET],
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
    push_nb = PUSH_DIR / CODE_FILE
    push_nb.write_bytes(dest_nb.read_bytes())
    (PUSH_DIR / "kernel-metadata.json").write_text(meta_text, encoding="utf-8")
    receipt = {
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kernel": KERNEL,
        "notebook_sha256": sha256_file(dest_nb),
        "metadata_sha256": hashlib.sha256(meta_text.encode("utf-8")).hexdigest(),
        "cell_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "dataset": DATASET,
    }
    (OUT_DIR / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
