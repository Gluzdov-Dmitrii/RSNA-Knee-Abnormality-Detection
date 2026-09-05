import json
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection")
OPEN = ROOT / "tmp" / "open_kernels"
OUT = ROOT / "tmp" / "ready"

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
    for p in candidates:
        if (p / 'rsna-knee-weights' / 'manifest.json').is_file():
            return p
    found = list(root.rglob('rsna-knee-weights/manifest.json')) if root.exists() else []
    if found:
        return found[0].parent.parent
    raise FileNotFoundError('rsna-knee-weights/manifest.json not found under /kaggle/input')
ASSET = _resolve_asset()
print('ASSET', ASSET)
"""

OLD_ASSET = "ASSET = Path('/kaggle/input/rsna-knee-bend-dinov3-0917-repro-assets')"

JOBS = [
    {
        "origin": "anvithpothula__rsna-base",
        "kernel": "dmitriigluzdov/rsna-knee-open-rsna-base-0936",
        "title": "RSNA Knee Open rsna-base 0936",
        "folder": "ready-base",
        "patch_asset": True,
        "id_no": 132790146,
    },
    {
        "origin": "anhadmahajan06__rsna-knee-take-care-of-your-knee",
        "kernel": "dmitriigluzdov/rsna-knee-take-care-of-your-knee",
        "title": "RSNA Knee: Take Care Of Your Knee",
        "folder": "ready-takecare",
        "patch_asset": True,
    },
    {
        "origin": "hyakumanben2025__rsna-knee-dinosaur-v6-repro",
        "kernel": "dmitriigluzdov/rsna-open-sol-v6",
        "title": "rsna-open-sol-v6",
        "folder": "ready-v6",
        "patch_asset": True,
    },
    {
        "origin": "llccqq624__rsna-knee-dino-protocol-fusion",
        "kernel": "dmitriigluzdov/rsna-open-sol-fusion",
        "title": "rsna-open-sol-fusion",
        "folder": "ready-fusion",
        "patch_asset": False,
    },
    {
        "origin": "hdhsjdjd__rsna-knee-raptor-quad-w065",
        "kernel": "dmitriigluzdov/rsna-open-sol-raptor",
        "title": "rsna-open-sol-raptor",
        "folder": "ready-raptor",
        "patch_asset": True,
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
        if job["patch_asset"]:
            replaced = 0
            for cell in nb["cells"]:
                src = cell_source(cell)
                if OLD_ASSET in src:
                    cell["source"] = src.replace(OLD_ASSET, RESOLVER, 1)
                    replaced += 1
            print(job["folder"], "asset_patches", replaced)
        dst = OUT / job["folder"]
        dst.mkdir()
        (dst / code_name).write_text(json.dumps(nb), encoding="utf-8")
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
            "dataset_sources": meta.get("dataset_sources", []),
            "kernel_sources": meta.get("kernel_sources", []),
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "model_sources": meta.get("model_sources", []),
            "machine_shape": "NvidiaTeslaT4",
        }
        if job.get("id_no"):
            new_meta["id_no"] = job["id_no"]
        (dst / "kernel-metadata.json").write_text(json.dumps(new_meta, indent=2), encoding="utf-8")
        print("ready", job["kernel"])


if __name__ == "__main__":
    main()
