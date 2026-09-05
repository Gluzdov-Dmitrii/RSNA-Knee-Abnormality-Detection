import json
import re
from pathlib import Path

p = Path(
    r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\open_kernels_d2"
    r"\ayodejiibrahimlateef__rsna-knee-c01-dinosaur-v4-dual-fusion"
)
nb = json.loads(next(p.glob("*.ipynb")).read_text(encoding="utf-8"))
src = "\n".join(
    ("".join(c.get("source") or []) if isinstance(c.get("source"), list) else (c.get("source") or ""))
    for c in nb["cells"]
)
for pat in [r"ASSET = .{0,220}", r"ROOT = .{0,220}", r"Path\('/kaggle/input[^']+'\)"]:
    print("PAT", pat)
    for i, m in enumerate(re.finditer(pat, src)):
        print(" ", m.group(0)[:240])
        if i >= 12:
            break
