import json
import re
from pathlib import Path

root = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\open_kernels")
needles = [
    "cuda:1",
    "device_count",
    "T4",
    "submission.csv",
    "sample_submission",
    "test.csv",
    "test_series",
    "StudyInstanceUID",
]


def cell_src(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src or ""


for name in [
    "anvithpothula__rsna-base",
    "anhadmahajan06__rsna-knee-take-care-of-your-knee",
    "hdhsjdjd__rsna-knee-raptor-quad-w065",
    "llccqq624__rsna-knee-dino-protocol-fusion",
    "shashwat1729__rsna-knee-infer-gpu-hsv44fixc-calw",
    "hyakumanben2025__rsna-knee-dinosaur-v6-repro",
]:
    d = root / name
    ipynb = next(d.glob("*.ipynb"))
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    src = "\n".join(cell_src(c) for c in nb.get("cells", []))
    print("=" * 80)
    print(name, "len", len(src))
    kmeta = nb.get("metadata", {}).get("kaggle", {})
    print("kaggle keys", sorted(kmeta.keys()))
    for k in ["accelerator", "isGpuEnabled", "isInternetEnabled", "isTpuEnabled", "enableGpu", "gpuCount"]:
        if k in kmeta:
            print(f"  {k}={kmeta[k]}")
    for needle in needles:
        print(f"  count[{needle}]={src.count(needle)}")
    # print GPU-related snippets
    for m in re.finditer(r".{0,80}(cuda:1|device_count|torch.device).{0,80}", src):
        line = " ".join(m.group(0).split())
        print("  GPU:", line[:180])
    for m in re.finditer(r".{0,60}(to_csv\(|submission\.csv|sample_submission).{0,80}", src):
        line = " ".join(m.group(0).split())
        print("  OUT:", line[:200])
    print()
