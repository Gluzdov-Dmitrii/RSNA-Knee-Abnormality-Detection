import hashlib
import json
import re
from pathlib import Path

root = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\open_kernels_d2")


def cell_src(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src or ""


for d in sorted(p for p in root.iterdir() if p.is_dir()):
    ipynb = next(d.glob("*.ipynb"))
    meta = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    src = "\n".join(cell_src(c) for c in cells)
    h = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    print(d.name)
    print(f"  bytes={ipynb.stat().st_size} cells={len(cells)} src={h}")
    print(
        "  gpu={gpu} net={net} ds={ds} ks={ks} ms={ms}".format(
            gpu=meta.get("enable_gpu"),
            net=meta.get("enable_internet"),
            ds=len(meta.get("dataset_sources", [])),
            ks=len(meta.get("kernel_sources", [])),
            ms=len(meta.get("model_sources", [])),
        )
    )
    print("  datasets", meta.get("dataset_sources"))
    print("  kernels", meta.get("kernel_sources"))
    print("  models", meta.get("model_sources"))
    m = re.search(r"ASSET = .{0,180}", src)
    print("  ASSET", (m.group(0).replace("\n", " | ")[:200] if m else "NONE"))
    print(
        "  flags",
        {
            "submission.csv": "submission.csv" in src,
            "test.csv": "test.csv" in src,
            "to_csv": "to_csv" in src,
            "internet": nb.get("metadata", {}).get("kaggle", {}).get("isInternetEnabled"),
        },
    )
    print()
