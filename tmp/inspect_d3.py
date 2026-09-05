import json
import re
from pathlib import Path

root = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\open_kernels_d3")


def cell_src(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src or ""


for d in sorted(p for p in root.iterdir() if p.is_dir()):
    meta = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))
    ipynb = next(d.glob("*.ipynb"))
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    src = "\n".join(cell_src(c) for c in nb.get("cells", []))
    print("=" * 60)
    print(d.name, "bytes", ipynb.stat().st_size, "cells", len(nb.get("cells", [])))
    print("gpu", meta.get("enable_gpu"), "net", meta.get("enable_internet"), "shape", meta.get("machine_shape"))
    print("docker", bool(meta.get("docker_image")))
    print("ds", [x for x in meta.get("dataset_sources", [])])
    print("ks", meta.get("kernel_sources"))
    print("ms", meta.get("model_sources"))
    flags = {
        "submission.csv": "submission.csv" in src,
        "to_csv": "to_csv" in src,
        "test.csv": "test.csv" in src,
        "ASSET =": "ASSET =" in src,
        "ROOT =": "ROOT =" in src,
        "find_root": "find_root" in src,
        "_comp_root": "_comp_root" in src,
        "_resolve": "_resolve" in src,
    }
    print("flags", flags)
    for pat in [r"ASSET = .{0,180}", r"ROOT = .{0,180}", r"COMP = .{0,180}"]:
        m = re.search(pat, src)
        if m:
            print(pat, "->", m.group(0).replace("\n", " | ")[:220])
    hard = sorted(set(re.findall(r"/kaggle/input/[A-Za-z0-9_./-]+", src)))
    print("hardcoded", hard[:25], "count", len(hard))
    print()
