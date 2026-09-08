"""Study-level dataset over PIXEL_CACHE_V1 + FOLDS_V1 + weak labels.

Pixels stay uint8 until the caller copies a batch. Convert to float32 in
collate/on-device so the 11 GiB corpus is not inflated to ~44 GiB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from pixel_cache_v1 import TARGETS, PixelCacheV1, default_folds_csv, default_pilkwang_csv

PLANE_FLUID_SLOTS = (0, 1, 2)
RGB_SLICE_IDX = (1, 4, 7)


def default_label_csv() -> Path:
    return default_pilkwang_csv()


def build_train_table(
    cache: PixelCacheV1,
    folds_csv: Path | None = None,
    labels_csv: Path | None = None,
) -> pd.DataFrame:
    table = cache.join_folds(folds_csv or default_folds_csv())
    labels = cache.join_labels(labels_csv or default_label_csv())
    table = table.merge(
        labels[["StudyInstanceUID"] + TARGETS],
        on="StudyInstanceUID",
        how="left",
        validate="one_to_one",
    )
    return table


def plane_rgb_float(pixels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Three plane images, each 3-channel from fluid-slot slices 1/4/7.

    Returns:
      x: float32 (3, 3, 224, 224) in [0, 1]
      plane_mask: float32 (3,) 1 if that fluid slot exists
    """
    if pixels.shape != (6, 9, 224, 224):
        raise ValueError(f"unexpected pixels shape {pixels.shape}")
    x = np.zeros((3, 3, 224, 224), dtype=np.float32)
    plane_mask = np.zeros(3, dtype=np.float32)
    for plane, slot in enumerate(PLANE_FLUID_SLOTS):
        if mask[slot]:
            x[plane] = pixels[slot, list(RGB_SLICE_IDX)].astype(np.float32) / 255.0
            plane_mask[plane] = 1.0
    return x, plane_mask


class KneePixelDataset:
    def __init__(
        self,
        cache: PixelCacheV1,
        table: pd.DataFrame,
        fold: int | None = None,
        holdout: bool = False,
        targets: list[str] | None = None,
    ):
        self.cache = cache
        self.targets = targets or TARGETS
        selected = table
        if fold is not None:
            if holdout:
                selected = table.loc[table["fold"] == fold]
            else:
                selected = table.loc[table["fold"] != fold]
        self.table = selected.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        uid = str(row["StudyInstanceUID"])
        pixels, mask = self.cache.get(uid)
        x, plane_mask = plane_rgb_float(pixels, mask)
        y = row[self.targets].to_numpy(dtype=np.float32, copy=True)
        labeled = np.isfinite(y).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0).astype(np.float32)
        return {
            "uid": uid,
            "x": x,
            "y": y,
            "labeled": labeled,
            "plane_mask": plane_mask,
            "slot_mask": mask.astype(np.float32),
            "fold": int(row["fold"]) if "fold" in row.index and pd.notna(row["fold"]) else -1,
        }


def numpy_collate(batch: list[dict]) -> dict:
    return {
        "uid": [item["uid"] for item in batch],
        "x": np.stack([item["x"] for item in batch], axis=0),
        "y": np.stack([item["y"] for item in batch], axis=0),
        "labeled": np.stack([item["labeled"] for item in batch], axis=0),
        "plane_mask": np.stack([item["plane_mask"] for item in batch], axis=0),
        "slot_mask": np.stack([item["slot_mask"] for item in batch], axis=0),
        "fold": np.asarray([item["fold"] for item in batch], dtype=np.int32),
    }
