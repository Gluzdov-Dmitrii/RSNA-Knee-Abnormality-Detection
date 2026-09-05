import json
import re
from pathlib import Path

root = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\open_kernels_d4")


def cell_src(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src or ""


for d in sorted(p for p in root.iterdir() if p.is_dir()):
    meta = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebooks = list(d.glob("*.ipynb"))
    scripts = list(d.glob("*.py"))
    if notebooks:
        ipynb = notebooks[0]
        nb = json.loads(ipynb.read_text(encoding="utf-8"))
        src = "\n".join(cell_src(c) for c in nb.get("cells", []))
        n_cells = len(nb.get("cells", []))
        size = ipynb.stat().st_size
    elif scripts:
        src = scripts[0].read_text(encoding="utf-8")
        n_cells = 0
        size = scripts[0].stat().st_size
    else:
        print("NO CODE", d.name)
        continue
    print("=" * 60)
    print(d.name, "bytes", size, "cells", n_cells, "type", meta.get("kernel_type"))
    print("gpu", meta.get("enable_gpu"), "net", meta.get("enable_internet"), "shape", meta.get("machine_shape"))
    print("docker", bool(meta.get("docker_image")))
    print("ds", [x for x in meta.get("dataset_sources", [])])
    print("ks", meta.get("kernel_sources"))
    print("ms", meta.get("model_sources"))
    print(
        "flags",
        {
            "submission.csv": "submission.csv" in src,
            "to_csv": "to_csv" in src,
            "ASSET =": "ASSET =" in src,
            "ROOT =": "ROOT =" in src,
            "_comp_root": "_comp_root" in src or "_competition_root" in src,
            "_asset_root": "_asset_root" in src or "_resolve_asset" in src,
            "find_test_root": "find_test_root" in src,
        },
    )
    for pat in [r"ASSET = .{0,160}", r"ROOT = .{0,160}"]:
        m = re.search(pat, src)
        if m:
            print(pat, "->", m.group(0).replace("\n", " | ")[:200])
    print()
