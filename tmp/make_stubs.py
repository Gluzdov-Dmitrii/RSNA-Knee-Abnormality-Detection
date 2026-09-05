import json
from pathlib import Path

root = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\forks")
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": [
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": 'print("stub")\n',
        }
    ],
}
for slug in ["rsna-open-sol-v6", "rsna-open-sol-fusion", "rsna-open-sol-raptor"]:
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "stub.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = {
        "id": f"dmitriigluzdov/{slug}",
        "title": slug,
        "code_file": "stub.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("prepared", slug)
