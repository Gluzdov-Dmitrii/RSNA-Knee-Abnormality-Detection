import json
from pathlib import Path

p = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\ready_d2\d2-evgen")
meta = json.loads((p / "kernel-metadata.json").read_text(encoding="utf-8"))
nb = json.loads((p / meta["code_file"]).read_text(encoding="utf-8"))
old = "ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')"
new = """def _resolve_root():
    candidates = [
        Path('/kaggle/input/rsna-knee-abnormality-detection'),
        Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
    ]
    for cand in candidates:
        if (cand / 'test.csv').is_file() or (cand / 'train.csv').is_file():
            return cand
    raise FileNotFoundError('competition data not found under /kaggle/input')
ROOT = _resolve_root()
print('ROOT', ROOT)"""
n = 0
for cell in nb["cells"]:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    if old in src:
        cell["source"] = src.replace(old, new, 1)
        n += 1
print("root_patches", n)
if n != 1:
    raise SystemExit("root patch failed")
(p / meta["code_file"]).write_text(json.dumps(nb), encoding="utf-8")
print("updated", p)
