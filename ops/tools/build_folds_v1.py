"""Build locked FOLDS_V1: study-level 5-fold iterative stratification.

Reads official train StudyInstanceUID values and Pilkwang V1 target scores.
Does not write reports or raw label tables.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CANDIDATES = [
    ROOT / "data" / "official" / "train.csv",
    ROOT / "tmp" / "schema" / "train.csv",
]
PILKWANG_CSV = ROOT / "tmp" / "labels" / "pilkwang" / "report_labels_v2.csv"
OUT_DIR = ROOT / "ops" / "assets" / "FOLDS_V1"
N_SPLITS = 5
SEED = 2026
EXPECTED_STUDIES = 4407
EXPECTED_PILKWANG_SHA = (
    "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920"
)
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
POSITIVE_THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iterative_stratification(y: np.ndarray, n_splits: int, rng: np.random.Generator) -> np.ndarray:
    """Sechidis et al. 2011 iterative stratification for k-fold assignment.

    y is a binary matrix of shape (n_samples, n_labels).
    Returns fold ids in 0 .. n_splits-1.
    """
    n_samples = y.shape[0]
    remaining = set(range(n_samples))
    folds = np.full(n_samples, -1, dtype=np.int32)
    desired_pos = y.sum(axis=0, dtype=np.float64) / n_splits
    remaining_pos = np.tile(desired_pos, (n_splits, 1))
    remaining_size = np.full(n_splits, n_samples / n_splits, dtype=np.float64)

    def choose_fold_by_size() -> int:
        tied = np.flatnonzero(np.isclose(remaining_size, remaining_size.max()))
        return int(rng.choice(tied))

    while remaining:
        rem = np.fromiter(remaining, dtype=np.int32)
        rem_pos = y[rem].sum(axis=0)
        active = np.flatnonzero(rem_pos > 0)
        if active.size == 0:
            leftover = rem.copy()
            rng.shuffle(leftover)
            for idx in leftover:
                fold = choose_fold_by_size()
                folds[idx] = fold
                remaining.remove(int(idx))
                remaining_size[fold] -= 1
            break

        label = int(active[np.argmin(rem_pos[active])])
        candidates = rem[y[rem, label] == 1]
        rng.shuffle(candidates)
        for idx in candidates:
            idx = int(idx)
            if idx not in remaining:
                continue
            demand = remaining_pos[:, label]
            # Tie-break: most remaining size, then random.
            max_demand = demand.max()
            fold_candidates = np.flatnonzero(np.isclose(demand, max_demand))
            sizes = remaining_size[fold_candidates]
            fold_candidates = fold_candidates[np.isclose(sizes, sizes.max())]
            fold = int(rng.choice(fold_candidates))
            folds[idx] = fold
            remaining.remove(idx)
            remaining_pos[fold] -= y[idx]
            remaining_size[fold] -= 1

    if (folds < 0).any():
        raise RuntimeError("fold assignment left unassigned studies")
    return folds


def main() -> None:
    train_csv = next((path for path in TRAIN_CANDIDATES if path.is_file()), None)
    if train_csv is None:
        raise SystemExit("official train.csv not found under data/official or tmp/schema")

    pilkwang_sha = sha256_file(PILKWANG_CSV)
    if pilkwang_sha != EXPECTED_PILKWANG_SHA:
        raise SystemExit(f"Pilkwang SHA mismatch: {pilkwang_sha}")

    train = pd.read_csv(train_csv, usecols=["StudyInstanceUID"])
    labels = pd.read_csv(PILKWANG_CSV, usecols=["StudyInstanceUID"] + TARGETS)
    if len(train) != EXPECTED_STUDIES:
        raise SystemExit(f"unexpected train size {len(train)}")
    if train["StudyInstanceUID"].nunique() != EXPECTED_STUDIES:
        raise SystemExit("train StudyInstanceUID is not unique")
    if labels["StudyInstanceUID"].duplicated().any():
        raise SystemExit("Pilkwang StudyInstanceUID duplicates")

    extra = int((~labels["StudyInstanceUID"].isin(train["StudyInstanceUID"])).sum())
    if extra:
        raise SystemExit(f"Pilkwang has {extra} UIDs outside official train")
    merged = train.merge(labels, on="StudyInstanceUID", how="left")
    unlabeled = merged.loc[merged[TARGETS[0]].isna(), "StudyInstanceUID"].tolist()
    merged[TARGETS] = merged[TARGETS].fillna(0.0)
    merged = merged.sort_values("StudyInstanceUID").reset_index(drop=True)

    y = (merged[TARGETS].to_numpy(dtype=np.float64) >= POSITIVE_THRESHOLD).astype(np.int32)
    rng = np.random.default_rng(SEED)
    folds = iterative_stratification(y, N_SPLITS, rng)

    out = pd.DataFrame(
        {
            "StudyInstanceUID": merged["StudyInstanceUID"],
            "fold": folds,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds_path = OUT_DIR / "folds.csv"
    out.to_csv(folds_path, index=False)
    folds_sha = sha256_file(folds_path)

    fold_sizes = out["fold"].value_counts().sort_index().to_dict()
    pos_by_fold = {}
    for fold_id in range(N_SPLITS):
        mask = folds == fold_id
        pos_by_fold[str(fold_id)] = {
            target: int(y[mask, col].sum()) for col, target in enumerate(TARGETS)
        }

    spec = {
        "key": "FOLDS_V1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_splits": N_SPLITS,
        "seed": SEED,
        "unit": "StudyInstanceUID",
        "n_studies": int(len(out)),
        "algorithm": "iterative_stratification_sechidis_2011",
        "positive_threshold": POSITIVE_THRESHOLD,
        "stratify_source": "LABEL_PILKWANG_V1",
        "stratify_file": "pilkwang/rsna-knee-llm-labels@report_labels_v2.csv",
        "stratify_sha256": pilkwang_sha,
        "train_uid_source": "kaggle competitions download -f train.csv",
        "train_csv_path_kind": "local_gitignored_official_or_tmp_schema",
        "train_n_studies": EXPECTED_STUDIES,
        "pilkwang_unlabeled_train_uids": unlabeled,
        "folds_csv_sha256": folds_sha,
        "fold_sizes": {str(k): int(v) for k, v in fold_sizes.items()},
        "positives_by_fold": pos_by_fold,
        "notes": [
            "Study-level folds over the official 4407 train studies.",
            "Stratified on Pilkwang V1 12-target scores binarized at 0.5 (YES vs NO/UNK).",
            "One official train study is absent from Pilkwang V1; it is kept in FOLDS_V1 with an all-zero stratify row.",
            "train.csv Report column was not written into this artifact.",
        ],
    }
    spec_path = OUT_DIR / "SPEC.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "folds_csv": str(folds_path),
        "folds_sha256": folds_sha,
        "fold_sizes": spec["fold_sizes"],
        "n_studies": spec["n_studies"],
    }, indent=2))


if __name__ == "__main__":
    main()
