"""S25: rank-mean of S22 and S24 locked OOF (same-seed 2026).

CPU only. Writes oof.csv + metrics.json. Does not train.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT = Path("/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection")
TARGETS = [
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
DEFAULT_S22 = PROJECT / "runs/20260908T1635Z-s22-resnet18-25d/checkpoints/oof.csv"
DEFAULT_S24 = PROJECT / "runs/20260910T0545Z-s24-r3d18-9slice/checkpoints/oof.csv"
DEFAULT_LABELS = PROJECT / "data/labels/pilkwang/report_labels_v2.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def per_target_auc(y: np.ndarray, p: np.ndarray, labeled: np.ndarray) -> list[float | None]:
    scores: list[float | None] = []
    for col in range(y.shape[1]):
        mask = labeled[:, col] > 0.5
        yt = (y[mask, col] >= 0.5).astype(np.int32)
        if mask.sum() < 8 or len(np.unique(yt)) < 2:
            scores.append(None)
            continue
        scores.append(float(roc_auc_score(yt, p[mask, col])))
    return scores


def rank_mean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0][["StudyInstanceUID"]].copy()
    for name in TARGETS:
        ranks = np.stack(
            [df[name].rank(method="average", pct=True).to_numpy(dtype=np.float64) for df in frames],
            axis=0,
        )
        base[name] = np.mean(ranks, axis=0)
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s22", type=str, default=str(DEFAULT_S22))
    parser.add_argument("--s24", type=str, default=str(DEFAULT_S24))
    parser.add_argument("--oof", action="append", nargs=2, metavar=("NAME", "PATH"), default=None)
    parser.add_argument("--key", type=str, default="")
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS))
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)
    labels["StudyInstanceUID"] = labels["StudyInstanceUID"].astype(str)
    if args.oof:
        named_paths = [(name, path) for name, path in args.oof]
    else:
        named_paths = [("S22_RESNET18_25D", args.s22), ("S24_R3D18_9SLICE", args.s24)]
    members = []
    member_names = []
    for name, path in named_paths:
        df = pd.read_csv(path)
        df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
        members.append(df.sort_values("StudyInstanceUID").reset_index(drop=True))
        member_names.append(name)
    uids = list(members[0]["StudyInstanceUID"])
    if any(list(frame["StudyInstanceUID"]) != uids for frame in members[1:]):
        raise SystemExit("OOF UID order mismatch after sort")
    blend = rank_mean(members)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    oof_path = out / "oof.csv"
    blend.to_csv(oof_path, index=False)
    merged = labels.merge(blend, on="StudyInstanceUID", suffixes=("_y", "_p"), how="inner")
    y = merged[[f"{name}_y" for name in TARGETS]].to_numpy(dtype=np.float32)
    p = merged[[f"{name}_p" for name in TARGETS]].to_numpy(dtype=np.float32)
    labeled = np.isfinite(y).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0)
    scores = per_target_auc(y, p, labeled)
    macro = float(np.mean([s for s in scores if s is not None]))
    key = args.key or ("S25_RANKMEAN_" + "_".join(member_names))
    summary = {
        "key": key,
        "checked_at_utc": utc_now(),
        "members": member_names,
        "n": int(len(merged)),
        "oof_macro_auc": macro,
        "oof_per_target_auc": {name: scores[i] for i, name in enumerate(TARGETS)},
        "oof_csv": str(oof_path),
        "oof_paths": {name: path for name, path in named_paths},
        "seed_note": "same-seed 2026 rank-mean",
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
