import json
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection")
OPEN = ROOT / "tmp" / "open_kernels_d4"
OUT = ROOT / "tmp" / "ready_d4"

DIAG = """from pathlib import Path
import os, sys
print('python', sys.version)
print('cuda_visible', os.environ.get('CUDA_VISIBLE_DEVICES'))
try:
    import torch
    print('cuda_is_available', torch.cuda.is_available(), 'count', torch.cuda.device_count())
except Exception as e:
    print('torch_err', e)
inp = Path('/kaggle/input')
print('inputs', sorted(p.name for p in inp.iterdir()) if inp.exists() else None)
found = list(inp.rglob('rsna-knee-weights/manifest.json')) if inp.exists() else []
print('manifests', [str(p) for p in found])
"""

RESOLVER = """def _resolve_asset():
    root = Path('/kaggle/input')
    candidates = [
        root / 'rsna-knee-bend-dinov3-0917-repro-assets',
        root / 'datasets' / 'tonylica' / 'rsna-knee-bend-dinov3-0917-repro-assets',
    ]
    if root.exists():
        candidates.extend(sorted(root.iterdir()))
    for cand in candidates:
        if (cand / 'rsna-knee-weights' / 'manifest.json').is_file():
            return cand
    found = list(root.rglob('rsna-knee-weights/manifest.json')) if root.exists() else []
    if found:
        return found[0].parent.parent
    raise FileNotFoundError('rsna-knee-weights/manifest.json not found under /kaggle/input')
ASSET = _resolve_asset()
print('ASSET', ASSET)
"""

COMP_ROOT = """def _resolve_comp():
    for cand in (
        Path('/kaggle/input/rsna-knee-abnormality-detection'),
        Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
    ):
        if (cand / 'test.csv').is_file() or (cand / 'train.csv').is_file():
            return cand
    raise FileNotFoundError('competition data not found under /kaggle/input')
ROOT = _resolve_comp()
print('ROOT', ROOT)
"""

OLD_ASSETS = [
    "ASSET = Path('/kaggle/input/rsna-knee-bend-dinov3-0917-repro-assets')",
    'ASSET = Path("/kaggle/input/rsna-knee-bend-dinov3-0917-repro-assets")',
    "ASSET = Path('/kaggle/input/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets')",
    'ASSET = Path("/kaggle/input/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets")',
]

OLD_ROOTS = [
    'ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")',
    "ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')",
    "ROOT = Path('/kaggle/input/rsna-knee-abnormality-detection')",
    'ROOT = Path("/kaggle/input/rsna-knee-abnormality-detection")',
]

JOBS = [
    {
        "origin": "hyakumanben2025__rsna-knee-dinosaur-v5-repro",
        "kernel": "dmitriigluzdov/rsna-knee-open-rsna-base-0936",
        "title": "RSNA Knee Open rsna-base 0936",
        "folder": "d4-dino-v5",
        "id_no": 132790146,
        "origin_ref": "hyakumanben2025/rsna-knee-dinosaur-v5-repro",
        "score": "0.936",
    },
    {
        "origin": "nishantkharga__rsna-knee-4-arm-v16-slothead-nk",
        "kernel": "dmitriigluzdov/rsna-knee-take-care-of-your-knee",
        "title": "RSNA Knee: Take Care Of Your Knee",
        "folder": "d4-nishant-v16",
        "origin_ref": "nishantkharga/rsna-knee-4-arm-v16-slothead-nk",
        "score": "0.935",
    },
    {
        "origin": "fanch123__rsna-knee-e0-v16-crossfit-gated",
        "kernel": "dmitriigluzdov/rsna-open-sol-v6",
        "title": "rsna-open-sol-v6",
        "folder": "d4-fanch",
        "origin_ref": "fanch123/rsna-knee-e0-v16-crossfit-gated",
        "score": "0.935",
    },
    {
        "origin": "anthonytherrien__rsna-knee-dino-radimagenet-dual-coatnet",
        "kernel": "dmitriigluzdov/rsna-open-sol-fusion",
        "title": "rsna-open-sol-fusion",
        "folder": "d4-anthony",
        "origin_ref": "anthonytherrien/rsna-knee-dino-radimagenet-dual-coatnet",
        "score": "0.935",
        "wrap_script": True,
    },
    {
        "origin": "sankurero__rsna-knee-coatnet-raptor-only",
        "kernel": "dmitriigluzdov/rsna-open-sol-raptor",
        "title": "rsna-open-sol-raptor",
        "folder": "d4-sankurero",
        "origin_ref": "sankurero/rsna-knee-coatnet-raptor-only",
        "score": "0.935",
    },
    {
        "origin": "hdhsjdjd__rsna-knee-raptor-quad-w080",
        "kernel": "dmitriigluzdov/rsna-knee-open-rsna-base-0936",
        "title": "RSNA Knee Open rsna-base 0936",
        "folder": "d4-raptor-w080",
        "id_no": 132790146,
        "origin_ref": "hdhsjdjd/rsna-knee-raptor-quad-w080",
        "score": "0.935",
    },
    {
        "origin": "pranjalverma08__rsna-knee-climbing-up-the-hill-the-ensemble-way",
        "kernel": "dmitriigluzdov/rsna-open-sol-v6",
        "title": "rsna-open-sol-v6",
        "folder": "d4-pranjal",
        "origin_ref": "pranjalverma08/rsna-knee-climbing-up-the-hill-the-ensemble-way",
        "score": "0.935",
    },
    {
        "origin": "pranjalverma08__rsna-knee-climbing-up-the-hill-the-ensemble-way",
        "kernel": "dmitriigluzdov/rsna-open-sol-fusion",
        "title": "rsna-open-sol-fusion",
        "folder": "d4-pranjal-fusion",
        "origin_ref": "pranjalverma08/rsna-knee-climbing-up-the-hill-the-ensemble-way",
        "score": "0.935",
    },
]


def cell_source(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src or ""


def script_to_notebook(py_src: str) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": py_src,
            }
        ],
    }


def patch_source(src: str) -> tuple[str, int]:
    replaced = 0
    for old in OLD_ASSETS:
        if old in src:
            src = src.replace(old, RESOLVER, 1)
            replaced += 1
    for old in OLD_ROOTS:
        if old in src:
            src = src.replace(old, COMP_ROOT, 1)
            replaced += 1
    return src, replaced


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for job in JOBS:
        src_dir = OPEN / job["origin"]
        meta = json.loads((src_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        code_name = meta["code_file"]
        if job.get("wrap_script") or code_name.endswith(".py"):
            py_src = (src_dir / code_name).read_text(encoding="utf-8")
            nb = script_to_notebook(py_src)
            code_name = Path(code_name).with_suffix(".ipynb").name
        else:
            nb = json.loads((src_dir / code_name).read_text(encoding="utf-8"))
        diag = {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": DIAG,
        }
        nb["cells"] = [diag] + nb.get("cells", [])
        total = 0
        for cell in nb["cells"]:
            src = cell_source(cell)
            new_src, n = patch_source(src)
            total += n
            if new_src != src:
                cell["source"] = new_src
        print(job["folder"], "patches", total)
        dst = OUT / job["folder"]
        dst.mkdir()
        (dst / code_name).write_text(json.dumps(nb), encoding="utf-8")
        (dst / "ORIGIN.txt").write_text(job["origin_ref"] + " " + job["score"] + "\n", encoding="utf-8")
        new_meta = {
            "id": job["kernel"],
            "title": job["title"],
            "code_file": code_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["gpu"],
            "dataset_sources": [x for x in meta.get("dataset_sources", []) if x],
            "kernel_sources": [x for x in meta.get("kernel_sources", []) if x],
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "model_sources": [x for x in meta.get("model_sources", []) if x],
            "machine_shape": "NvidiaTeslaT4",
        }
        if job.get("id_no"):
            new_meta["id_no"] = job["id_no"]
        (dst / "kernel-metadata.json").write_text(json.dumps(new_meta, indent=2), encoding="utf-8")
        print("ready", job["kernel"], "from", job["origin_ref"])


if __name__ == "__main__":
    main()
