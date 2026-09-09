"""Validate visible S22 kernel output before the one authorized submit."""
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
VISIBLE_A0_SHA = "11bb66f2bc7ca21d4282de3696f5ea5531f90c6c2958164b6ad6931811af2213"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(output_dir: Path) -> dict:
    sample_path = output_dir / "sample_submission.csv"
    sub_path = output_dir / "submission.csv"
    receipt_path = output_dir / "s22_overlay_receipt.json"
    if not sub_path.is_file():
        raise FileNotFoundError(sub_path)
    frame = pd.read_csv(sub_path, dtype={"StudyInstanceUID": str})
    if frame.columns.tolist() != HEADER:
        raise RuntimeError(f"column order drift {frame.columns.tolist()}")
    uids = frame["StudyInstanceUID"].tolist()
    if len(set(uids)) != len(uids):
        raise RuntimeError("duplicate UID")
    values = frame[HEADER[1:]].to_numpy(float)
    if not (values.shape[1] == 12 and values.shape[0] == len(uids)):
        raise RuntimeError(f"shape {values.shape}")
    if not (np_isfinite(values) and values.min() >= 0 and values.max() <= 1):
        raise RuntimeError("predictions outside finite [0,1]")
    if len(uids) >= 3:
        for label in HEADER[1:]:
            if pd.unique(frame[label]).size < 2:
                raise RuntimeError(f"{label} is constant")
    receipt = {}
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "S22_OVERLAY_PASS":
            raise RuntimeError(f"overlay status {receipt.get('status')}")
        if receipt.get("parent_a0_sha256") != VISIBLE_A0_SHA and len(uids) == 3:
            raise RuntimeError(f"visible parent SHA drift {receipt.get('parent_a0_sha256')}")
    result = {
        "rows": int(len(frame)),
        "sha256": sha256_file(sub_path),
        "min": float(values.min()),
        "max": float(values.max()),
        "receipt_status": receipt.get("status"),
        "present_studies": receipt.get("present_studies"),
        "seconds": receipt.get("seconds"),
        "sample_path_present": sample_path.is_file(),
    }
    print(json.dumps(result, indent=2))
    return result


def np_isfinite(values) -> bool:
    import numpy as np

    return bool(np.isfinite(values).all())


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]))
