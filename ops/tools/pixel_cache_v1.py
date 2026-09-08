from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SLOT_NAMES = [
    "SAG_FLUID",
    "COR_FLUID",
    "AX_FLUID",
    "SAG_STRUCT",
    "COR_STRUCT",
    "AX_STRUCT",
]
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
SHAPE = (6, 9, 224, 224)
N_STUDIES = 4407
N_SHARDS = 35
EXPECTED_PRESENT_SLOTS = 21334
EXPECTED_ABSENT_SLOTS = 5108
FOLDS_SHA256 = "3086df3341333f44adb883292da386857c3230eaa2d501514ddf827a2da11b1a"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_folds_csv() -> Path:
    local = repo_root() / "ops" / "assets" / "FOLDS_V1" / "folds.csv"
    if local.is_file():
        return local
    nsu = repo_root() / "data" / "FOLDS_V1" / "folds.csv"
    if nsu.is_file():
        return nsu
    raise FileNotFoundError("FOLDS_V1 folds.csv not found")


def default_pilkwang_csv() -> Path:
    local = repo_root() / "tmp" / "labels" / "pilkwang" / "report_labels_v2.csv"
    if local.is_file():
        return local
    nsu = repo_root() / "data" / "labels" / "pilkwang" / "report_labels_v2.csv"
    if nsu.is_file():
        return nsu
    raise FileNotFoundError("LABEL_PILKWANG_V1 CSV not found")


def default_cache_root() -> Path:
    local = repo_root() / "data" / "rsna-knee-uint8-224-9-c130"
    if (local / "studies.csv").is_file():
        return local
    nsu = Path(
        "/home/scientists/gluz_d_s/kaggle/projects/"
        "rsna-knee-abnormality-detection/data/rsna-knee-uint8-224-9-c130"
    )
    if (nsu / "studies.csv").is_file():
        return nsu
    raise FileNotFoundError(
        "PIXEL_CACHE_V1 not found; download dmitriigluzdov/rsna-knee-uint8-224-9-c130"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PixelCacheV1:
    """Memory-map PIXEL_CACHE_V1 shards. One study is ~2.58 MiB uint8."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else default_cache_root()
        self.spec = json.loads((self.root / "SPEC.json").read_text(encoding="utf-8"))
        self.index = pd.read_csv(self.root / "studies.csv")
        if len(self.index) != N_STUDIES:
            raise ValueError(f"studies.csv has {len(self.index)} rows, expected {N_STUDIES}")
        missing = {"StudyInstanceUID", "shard", "row"} - set(self.index.columns)
        if missing:
            raise ValueError(f"studies.csv missing {missing}")
        if self.index["StudyInstanceUID"].duplicated().any():
            raise ValueError("studies.csv has duplicate StudyInstanceUID")
        self.mask = np.load(self.root / "slot_mask.npy", mmap_mode="r", allow_pickle=False)
        if self.mask.shape != (N_STUDIES, 6):
            raise ValueError(f"slot_mask shape {self.mask.shape}")
        self._by_uid = {
            uid: i for i, uid in enumerate(self.index["StudyInstanceUID"].astype(str))
        }
        self._shards: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _shard(self, name: str) -> np.ndarray:
        array = self._shards.get(name)
        if array is None:
            array = np.load(self.root / name, mmap_mode="r", allow_pickle=False)
            self._shards[name] = array
        return array

    def study_index(self, uid: str) -> int:
        return self._by_uid[uid]

    def get(self, uid: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.study_index(uid)
        row = self.index.iloc[i]
        pixels = self._shard(str(row["shard"]))[int(row["row"])]
        if pixels.shape != SHAPE or pixels.dtype != np.uint8:
            raise ValueError(f"{uid} has shape/dtype {pixels.shape} {pixels.dtype}")
        return pixels, np.array(self.mask[i], copy=True)

    def join_folds(self, folds_csv: Path) -> pd.DataFrame:
        folds = pd.read_csv(folds_csv)
        out = self.index.merge(folds, on="StudyInstanceUID", how="left", validate="one_to_one")
        if out["fold"].isna().any():
            raise ValueError("FOLDS_V1 missing UIDs that exist in the pixel cache")
        return out

    def join_labels(self, labels_csv: Path, targets: list[str] | None = None) -> pd.DataFrame:
        targets = targets or TARGETS
        labels = pd.read_csv(labels_csv)
        keep = ["StudyInstanceUID"] + [name for name in targets if name in labels.columns]
        missing_targets = [name for name in targets if name not in labels.columns]
        if missing_targets:
            raise ValueError(f"{labels_csv} missing targets {missing_targets}")
        labels = labels[keep]
        if labels["StudyInstanceUID"].duplicated().any():
            raise ValueError(f"{labels_csv} has duplicate StudyInstanceUID")
        return self.index.merge(labels, on="StudyInstanceUID", how="left", validate="many_to_one")


def _spec_hashes(spec: dict) -> dict[str, str]:
    hashes = spec.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("SPEC.json missing sha256 dict")
    return {str(name): str(value) for name, value in hashes.items()}


def verify_spec(
    root: Path | None = None,
    max_shards: int | None = 1,
    check_present_slots: int = 32,
) -> dict:
    cache = PixelCacheV1(root)
    spec = cache.spec
    if (
        spec.get("img"),
        spec.get("n_slices"),
        spec.get("crop_mm"),
        spec.get("dtype"),
        spec.get("n_studies"),
    ) != (224, 9, 130, "uint8", N_STUDIES):
        raise ValueError(f"unexpected SPEC geometry {spec.get('img')} {spec.get('n_slices')}")
    if spec.get("window") != [0.35, 0.65]:
        raise ValueError(f"unexpected window {spec.get('window')}")

    hashes = _spec_hashes(spec)
    npy_files = sorted(cache.root.glob("pixels-*.npy"))
    if len(npy_files) != N_SHARDS:
        raise ValueError(f"found {len(npy_files)} shards, expected {N_SHARDS}")

    shard_rows = spec.get("shards")
    if not isinstance(shard_rows, list) or len(shard_rows) != N_SHARDS:
        raise ValueError("SPEC.json shards list is incomplete")

    checked: list[str] = []
    pixel_bytes = 0
    to_hash = npy_files if max_shards is None else npy_files[:max_shards]
    hash_names = {path.name for path in to_hash}
    hash_names.update(["studies.csv", "slot_mask.npy", "AUDIT.json"])

    for shard in shard_rows:
        path = cache.root / shard["file"]
        if path.stat().st_size != int(shard["bytes"]):
            raise ValueError(f"size mismatch {path.name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (int(shard["n_studies"]), 6, 9, 224, 224)
        if array.dtype != np.uint8 or array.shape != expected_shape:
            raise ValueError(f"{path.name} shape/dtype {array.shape} {array.dtype}")
        pixel_bytes += int(array.nbytes)
        if path.name in hash_names:
            got = sha256_file(path)
            if got != hashes[path.name]:
                raise ValueError(f"hash mismatch {path.name}")
            checked.append(path.name)
        del array

    if pixel_bytes != spec["pixel_bytes"] or pixel_bytes != N_STUDIES * 6 * 9 * 224 * 224:
        raise ValueError("pixel byte count mismatch")

    for name in ("studies.csv", "slot_mask.npy", "AUDIT.json"):
        got = sha256_file(cache.root / name)
        if got != hashes[name]:
            raise ValueError(f"hash mismatch {name}")
        if name not in checked:
            checked.append(name)

    present = int(np.asarray(cache.mask).sum())
    absent = int(cache.mask.size - present)
    if present != EXPECTED_PRESENT_SLOTS or absent != EXPECTED_ABSENT_SLOTS:
        raise ValueError(f"slot occupancy {present}/{absent}")

    uid0 = str(cache.index.iloc[0]["StudyInstanceUID"])
    pix, mask = cache.get(uid0)
    if int(mask.sum()) == 0:
        raise ValueError(f"{uid0} has no present slots")

    zero_present = 0
    n_check = min(check_present_slots, len(cache))
    for i in range(n_check):
        uid = str(cache.index.iloc[i]["StudyInstanceUID"])
        study, slot_mask = cache.get(uid)
        maxima = study.max(axis=(1, 2, 3))
        if np.any(maxima[slot_mask == 0] != 0):
            raise ValueError(f"{uid} absent slot is not zero")
        zero_present += int(np.sum(maxima[slot_mask == 1] == 0))
    if zero_present:
        raise ValueError(f"{zero_present} present slots among first {n_check} studies are blank")

    return {
        "root": str(cache.root),
        "n_studies": len(cache),
        "n_shards": len(npy_files),
        "pixel_bytes": pixel_bytes,
        "pixel_gib": pixel_bytes / 1024**3,
        "sample_uid": uid0,
        "sample_shape": list(pix.shape),
        "sample_dtype": str(pix.dtype),
        "sample_mask_sum": int(mask.sum()),
        "present_slots": present,
        "absent_slots": absent,
        "hashes_checked": checked,
        "full_shard_hashes": max_shards is None,
    }


def smoke(root: Path | None = None, n_studies: int = 32) -> dict:
    cache = PixelCacheV1(root)
    folds_csv = default_folds_csv()
    if sha256_file(folds_csv) != FOLDS_SHA256:
        raise ValueError("FOLDS_V1 hash mismatch")
    table = cache.join_folds(folds_csv)
    if len(table) != N_STUDIES:
        raise ValueError("fold join lost studies")

    unlabeled = []
    try:
        labels_csv = default_pilkwang_csv()
    except FileNotFoundError:
        labels_csv = None
    if labels_csv is not None:
        labeled = cache.join_labels(labels_csv)
        unlabeled = labeled.loc[labeled[TARGETS[0]].isna(), "StudyInstanceUID"].astype(str).tolist()
        if len(unlabeled) > 1:
            raise ValueError(f"unexpected unlabeled count {len(unlabeled)}")
        table = table.merge(
            labeled[["StudyInstanceUID"] + TARGETS],
            on="StudyInstanceUID",
            how="left",
            validate="one_to_one",
        )

    loaded = []
    for uid in table["StudyInstanceUID"].astype(str).head(n_studies):
        pixels, mask = cache.get(uid)
        loaded.append(
            {
                "uid": uid,
                "shape": list(pixels.shape),
                "dtype": str(pixels.dtype),
                "mask_sum": int(mask.sum()),
                "mean_u8": float(pixels.mean()),
            }
        )

    return {
        "n_studies": len(cache),
        "fold_sizes": {str(k): int(v) for k, v in table["fold"].value_counts().sort_index().items()},
        "unlabeled_pilkwang": unlabeled,
        "smoke_n": len(loaded),
        "first_study": loaded[0],
        "last_smoke_study": loaded[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PIXEL_CACHE_V1 mmap reader / verify / smoke")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-full", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()

    if not args.verify and not args.verify_full and not args.smoke:
        args.verify = True
        args.smoke = True

    payload: dict = {
        "key": "PIXEL_CACHE_V1",
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if args.verify or args.verify_full:
        payload["verify"] = verify_spec(
            args.root,
            max_shards=None if args.verify_full else args.max_shards,
        )
    if args.smoke:
        payload["smoke"] = smoke(args.root)
    print(json.dumps(payload, indent=2))
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
