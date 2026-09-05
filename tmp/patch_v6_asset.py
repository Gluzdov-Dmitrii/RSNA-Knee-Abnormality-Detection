import json
from pathlib import Path

p = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\ready\ready-v6")
meta = json.loads((p / "kernel-metadata.json").read_text(encoding="utf-8"))
nb = json.loads((p / meta["code_file"]).read_text(encoding="utf-8"))
old = "ASSET = Path('/kaggle/input/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets')"
new = """def _resolve_asset():
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
    raise FileNotFoundError('rsna-knee-weights/manifest.json not found')
ASSET = _resolve_asset()
print('ASSET', ASSET)"""
n = 0
for cell in nb["cells"]:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    if old in src:
        cell["source"] = src.replace(old, new, 1)
        n += 1
print("patches", n)
if n != 1:
    raise SystemExit("patch failed")
(p / meta["code_file"]).write_text(json.dumps(nb), encoding="utf-8")
print("updated", p)
