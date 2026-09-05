import hashlib
import json
import re
from pathlib import Path

root = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\open_kernels")


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
    file_h = hashlib.sha256(ipynb.read_bytes()).hexdigest()[:16]
    kmeta = nb.get("metadata", {}).get("kaggle", {})
    repro = nb.get("metadata", {}).get("rsna_one_dataset_reproduction", {})
    flags = {
        "submission.csv": "submission.csv" in src,
        "test.csv": "test.csv" in src,
        "sample_submission": "sample_submission" in src,
        "to_csv": "to_csv" in src,
        "cuda:1": "cuda:1" in src,
        "device_count": "device_count" in src,
        "hard_3": bool(re.search(r"(==\s*3|len\([^\)]*\)\s*==\s*3|head\(3\)|iloc\[:3\])", src)),
        "internet": kmeta.get("isInternetEnabled"),
        "acc": kmeta.get("accelerator"),
        "src_nb": repro.get("source_notebook"),
        "src_sha": (repro.get("source_cells_sha256") or "")[:16],
    }
    print(d.name)
    print(f"  bytes={ipynb.stat().st_size} cells={len(cells)} src={h} file={file_h}")
    print(
        "  gpu={gpu} net={net} shape={shape} ds={ds} ks={ks} ms={ms}".format(
            gpu=meta.get("enable_gpu"),
            net=meta.get("enable_internet"),
            shape=meta.get("machine_shape"),
            ds=len(meta.get("dataset_sources", [])),
            ks=len(meta.get("kernel_sources", [])),
            ms=len(meta.get("model_sources", [])),
        )
    )
    print(f"  flags={flags}")
    print()
