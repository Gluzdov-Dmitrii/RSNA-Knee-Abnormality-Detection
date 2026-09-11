"""Build a one-cell GPU notebook: live PIXEL_CACHE_V1 decode + S24 3D bag."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CELL = Path(__file__).resolve().parent / "s24_standalone_cell.py"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
VARIANTS = {
    "r3d": {
        "variant": "r3d_18",
        "kernel": "dmitriigluzdov/rsna-s24-r3d18-9slice",
        "title": "rsna-s24-r3d18-9slice",
        "code_file": "rsna-s24-r3d18-9slice.ipynb",
        "dataset": "dmitriigluzdov/rsna-s24-r3d18-9slice-folds",
        "out_dir": ROOT / "tmp" / "kernel_s24_r3d",
        "push_dir": ROOT / "tmp" / "kernel_s24_r3d_push",
        "heading": "S24 r3d_18 9-slice bag",
    },
    "r2p": {
        "variant": "r2plus1d_18",
        "kernel": "dmitriigluzdov/rsna-s24-r2plus1d-9slice",
        "title": "rsna-s24-r2plus1d-9slice",
        "code_file": "rsna-s24-r2plus1d-9slice.ipynb",
        "dataset": "dmitriigluzdov/rsna-s24-r2plus1d-9slice-folds",
        "out_dir": ROOT / "tmp" / "kernel_s24_r2p",
        "push_dir": ROOT / "tmp" / "kernel_s24_r2p_push",
        "heading": "S24 r2plus1d_18 9-slice bag",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    args = parser.parse_args()
    spec = VARIANTS[args.variant]
    source = CELL.read_text(encoding="utf-8")
    if not source.startswith("# S24_STANDALONE_BAG"):
        raise SystemExit("standalone cell marker missing")
    if 'VARIANT = "r3d_18"' not in source:
        raise SystemExit("VARIANT assignment missing")
    source = source.replace('VARIANT = "r3d_18"', f'VARIANT = "{spec["variant"]}"', 1)
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
                    f"# {spec['heading']}\n",
                    "\n",
                    "Own PIXEL_CACHE_V1 decoder + 5-fold 3D bag. No A0/Raptor mix.\n",
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
        "id": spec["kernel"],
        "title": spec["title"],
        "code_file": spec["code_file"],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [spec["dataset"]],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "docker_image": DOCKER,
        "machine_shape": "NvidiaTeslaT4",
    }
    spec["out_dir"].mkdir(parents=True, exist_ok=True)
    spec["push_dir"].mkdir(parents=True, exist_ok=True)
    dest_nb = spec["out_dir"] / spec["code_file"]
    dest_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta_text = json.dumps(metadata, indent=2) + "\n"
    (spec["out_dir"] / "kernel-metadata.json").write_text(meta_text, encoding="utf-8")
    push_nb = spec["push_dir"] / spec["code_file"]
    push_nb.write_bytes(dest_nb.read_bytes())
    (spec["push_dir"] / "kernel-metadata.json").write_text(meta_text, encoding="utf-8")
    receipt = {
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kernel": spec["kernel"],
        "variant": spec["variant"],
        "notebook_sha256": sha256_file(dest_nb),
        "metadata_sha256": hashlib.sha256(meta_text.encode("utf-8")).hexdigest(),
        "cell_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "dataset": spec["dataset"],
    }
    (spec["out_dir"] / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
