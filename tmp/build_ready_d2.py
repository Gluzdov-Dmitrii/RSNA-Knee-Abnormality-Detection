import json
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection")
OPEN = ROOT / "tmp" / "open_kernels_d2"
OUT = ROOT / "tmp" / "ready_d2"

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

OLD_ASSETS = [
    "ASSET = Path('/kaggle/input/rsna-knee-bend-dinov3-0917-repro-assets')",
    "ASSET = Path('/kaggle/input/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets')",
]

JOBS = [
    {
        "origin": "kunaldesale2408__rsna-knee-abnormality-detectionv1",
        "kernel": "dmitriigluzdov/rsna-knee-open-rsna-base-0936",
        "title": "RSNA Knee Open rsna-base 0936",
        "folder": "d2-kunal",
        "id_no": 132790146,
        "origin_ref": "kunaldesale2408/rsna-knee-abnormality-detectionv1",
        "score": "0.936",
    },
    {
        "origin": "romantamrazov__rsna-knee-dinosaur-v4",
        "kernel": "dmitriigluzdov/rsna-knee-take-care-of-your-knee",
        "title": "RSNA Knee: Take Care Of Your Knee",
        "folder": "d2-dinosaur-v4",
        "origin_ref": "romantamrazov/rsna-knee-dinosaur-v4",
        "score": "0.936",
    },
    {
        "origin": "evgendvorkin__rsna-baseline",
        "kernel": "dmitriigluzdov/rsna-open-sol-v6",
        "title": "rsna-open-sol-v6",
        "folder": "d2-evgen",
        "origin_ref": "evgendvorkin/rsna-baseline",
        "score": "0.936",
    },
    {
        "origin": "shashwat1729__rsna-knee-infer-gpu-hsv44fixc-calw",
        "kernel": "dmitriigluzdov/rsna-open-sol-fusion",
        "title": "rsna-open-sol-fusion",
        "folder": "d2-shashwat",
        "origin_ref": "shashwat1729/rsna-knee-infer-gpu-hsv44fixc-calw",
        "score": "0.936",
    },
    {
        "origin": "nishantkharga__bend-the-knee-to-dinov3-ensembled-nk",
        "kernel": "dmitriigluzdov/rsna-open-sol-raptor",
        "title": "rsna-open-sol-raptor",
        "folder": "d2-nishant",
        "origin_ref": "nishantkharga/bend-the-knee-to-dinov3-ensembled-nk",
        "score": "0.935",
    },
]


def cell_source(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src or ""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for job in JOBS:
        src_dir = OPEN / job["origin"]
        meta = json.loads((src_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        code_name = meta["code_file"]
        nb = json.loads((src_dir / code_name).read_text(encoding="utf-8"))
        diag = {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": DIAG,
        }
        nb["cells"] = [diag] + nb.get("cells", [])
        replaced = 0
        for cell in nb["cells"]:
            src = cell_source(cell)
            for old in OLD_ASSETS:
                if old in src:
                    cell["source"] = src.replace(old, RESOLVER, 1)
                    replaced += 1
                    src = cell["source"]
        print(job["folder"], "asset_patches", replaced)
        dst = OUT / job["folder"]
        dst.mkdir()
        (dst / code_name).write_text(json.dumps(nb), encoding="utf-8")
        (dst / "ORIGIN.txt").write_text(job["origin_ref"] + " " + job["score"] + "\n", encoding="utf-8")
        datasets = [x for x in meta.get("dataset_sources", []) if x]
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
            "dataset_sources": datasets,
            "kernel_sources": meta.get("kernel_sources", []),
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "model_sources": meta.get("model_sources", []),
            "machine_shape": "NvidiaTeslaT4",
        }
        if job.get("id_no"):
            new_meta["id_no"] = job["id_no"]
        (dst / "kernel-metadata.json").write_text(json.dumps(new_meta, indent=2), encoding="utf-8")
        print("ready", job["kernel"], "from", job["origin_ref"])


if __name__ == "__main__":
    main()
