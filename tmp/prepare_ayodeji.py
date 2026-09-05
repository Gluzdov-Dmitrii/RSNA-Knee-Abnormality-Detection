import json
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection")
SRC = ROOT / "tmp" / "open_kernels_d2" / "ayodejiibrahimlateef__rsna-knee-c01-dinosaur-v4-dual-fusion"
DST = ROOT / "tmp" / "ready_d2" / "d2-ayodeji"

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


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    code = meta["code_file"]
    nb = json.loads((SRC / code).read_text(encoding="utf-8"))
    diag = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": DIAG,
    }
    nb["cells"] = [diag] + nb.get("cells", [])
    (DST / code).write_text(json.dumps(nb), encoding="utf-8")
    (DST / "ORIGIN.txt").write_text(
        "ayodejiibrahimlateef/rsna-knee-c01-dinosaur-v4-dual-fusion 0.936\n",
        encoding="utf-8",
    )
    new_meta = {
        "id": "dmitriigluzdov/rsna-open-sol-v6",
        "title": "rsna-open-sol-v6",
        "code_file": code,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [x for x in meta.get("dataset_sources", []) if x],
        "kernel_sources": meta.get("kernel_sources", []),
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": meta.get("model_sources", []),
        "machine_shape": "NvidiaTeslaT4",
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(new_meta, indent=2), encoding="utf-8")
    print("ready", DST)


if __name__ == "__main__":
    main()
