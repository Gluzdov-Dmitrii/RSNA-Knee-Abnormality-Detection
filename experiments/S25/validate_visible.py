"""Validate visible S25 r3d+r2p rank-mean kernel output before submit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

HEADER = [
    "StudyInstanceUID",
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(output_dir: Path) -> dict:
    import numpy as np

    sub_path = output_dir / "submission.csv"
    receipt_path = output_dir / "s25_local_receipt.json"
    if not sub_path.is_file():
        raise FileNotFoundError(sub_path)
    frame = pd.read_csv(sub_path, dtype={"StudyInstanceUID": str})
    if frame.columns.tolist() != HEADER:
        raise RuntimeError(f"column order drift {frame.columns.tolist()}")
    uids = frame["StudyInstanceUID"].tolist()
    if len(set(uids)) != len(uids):
        raise RuntimeError("duplicate UID")
    values = frame[HEADER[1:]].to_numpy(float)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError("predictions outside finite [0,1]")
    if len(uids) >= 3:
        for label in HEADER[1:]:
            if pd.unique(frame[label]).size < 2:
                raise RuntimeError(f"{label} is constant")
    receipt = {}
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "S25_RANKMEAN_PASS":
            raise RuntimeError(f"status {receipt.get('status')}")
        if receipt.get("blend") != "rank_mean_r3d_r2p":
            raise RuntimeError("unexpected blend")
        if receipt.get("members") != ["r3d_18", "r2plus1d_18"]:
            raise RuntimeError(f"members drift {receipt.get('members')}")
        if int(receipt.get("present_studies") or 0) < 1:
            raise RuntimeError("zero present studies")
    hidden = None
    if receipt.get("seconds") is not None:
        hidden = round(float(receipt["seconds"]) * (1322 / max(len(uids), 1)), 1)
    if hidden is not None and hidden > 8.5 * 3600:
        raise RuntimeError(f"hidden estimate {hidden}s exceeds 8.5h engineering gate")
    result = {
        "rows": int(len(frame)),
        "sha256": sha256_file(sub_path),
        "min": float(values.min()),
        "max": float(values.max()),
        "receipt_status": receipt.get("status"),
        "members": receipt.get("members"),
        "present_studies": receipt.get("present_studies"),
        "seconds": receipt.get("seconds"),
        "hidden_estimate_seconds": hidden,
        "device": receipt.get("device"),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]))
