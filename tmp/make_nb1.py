import json
import shutil
from pathlib import Path

src_nb = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\open_kernels\anvithpothula__rsna-base\rsna-base.ipynb")
dst = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\forks\rsna-open-nb1")
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True)

nb = json.loads(src_nb.read_text(encoding="utf-8"))
diag = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": """from pathlib import Path
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
""",
}
nb["cells"] = [diag] + nb.get("cells", [])
old = "ASSET = Path('/kaggle/input/rsna-knee-bend-dinov3-0917-repro-assets')"
new = """def _resolve_asset():
    root = Path('/kaggle/input')
    candidates = [root / 'rsna-knee-bend-dinov3-0917-repro-assets', root / 'datasets' / 'tonylica' / 'rsna-knee-bend-dinov3-0917-repro-assets']
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
print('ASSET', ASSET)"""
replaced = 0
for cell in nb["cells"]:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    if old in src:
        cell["source"] = src.replace(old, new, 1)
        replaced += 1
if replaced != 1:
    raise SystemExit(f"ASSET assignment replacements={replaced}")
(dst / "rsna-base.ipynb").write_text(json.dumps(nb), encoding="utf-8")
meta = {
    "id": "dmitriigluzdov/rsna-open-nb1",
    "title": "rsna-open-nb1",
    "code_file": "rsna-base.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "dreaddevelopment/raptor-knee-maxspan",
        "dreaddevelopment/raptor-knee-widedense",
        "tonylica/rsna-knee-bend-dinov3-0917-repro-assets",
    ],
    "kernel_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
    "machine_shape": "NvidiaTeslaT4",
}
(dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("prepared", dst)
