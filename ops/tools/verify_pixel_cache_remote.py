#!/usr/bin/env python3
"""Size + sample-hash check for PIXEL_CACHE_V1. Stdlib only."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(
    "/home/scientists/gluz_d_s/kaggle/projects/"
    "rsna-knee-abnormality-detection/data/rsna-knee-uint8-224-9-c130"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    spec = json.loads((root / "SPEC.json").read_text(encoding="utf-8"))
    npy = sorted(root.glob("pixels-*.npy"))
    if len(npy) != 35:
        raise SystemExit(f"shard count {len(npy)}")
    for shard in spec["shards"]:
        path = root / shard["file"]
        size = path.stat().st_size
        if size != int(shard["bytes"]):
            raise SystemExit(f"size {path.name}: {size} != {shard['bytes']}")
    names = ["studies.csv", "slot_mask.npy", "AUDIT.json", "pixels-000.npy", "pixels-034.npy"]
    checked = []
    for name in names:
        got = sha256_file(root / name)
        if got != spec["sha256"][name]:
            raise SystemExit(f"hash mismatch {name}")
        checked.append(name)
    n_csv = sum(1 for _ in (root / "studies.csv").read_text(encoding="utf-8").splitlines() if _) - 1
    print(json.dumps({"root": str(root), "n_shards": len(npy), "n_csv": n_csv, "checked": checked}, indent=2))


if __name__ == "__main__":
    main()
