"""Study-level dataset over PIXEL_CACHE_V1 + FOLDS_V1 + weak labels.

Pixels stay uint8 until the caller copies a batch. Convert to float32 in
collate/on-device so the 11 GiB corpus is not inflated to ~44 GiB.

Each plane is a 6-channel 2.5D image: fluid RGB (slices 1/4/7) then struct RGB.
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

PLANE_SLOTS = ((0, 3), (1, 4), (2, 5))
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


def plane_rgb6(
    pixels: np.ndarray,
    mask: np.ndarray,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Three plane images, each 6-channel (fluid RGB + struct RGB).

    Returns:
      x: float32 (3, 6, 224, 224) in [0, 1]
      plane_mask: float32 (3,) 1 if that plane has at least one present slot
    """
    if pixels.shape != (6, 9, 224, 224):
        raise ValueError(f"unexpected pixels shape {pixels.shape}")
    x = np.zeros((3, 6, 224, 224), dtype=np.float32)
    plane_mask = np.zeros(3, dtype=np.float32)
    for plane, (fluid, struct) in enumerate(PLANE_SLOTS):
        present = False
        if mask[fluid]:
            x[plane, 0:3] = pixels[fluid, list(RGB_SLICE_IDX)].astype(np.float32) / 255.0
            present = True
        if mask[struct]:
            x[plane, 3:6] = pixels[struct, list(RGB_SLICE_IDX)].astype(np.float32) / 255.0
            present = True
        plane_mask[plane] = 1.0 if present else 0.0
    if augment:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() < 0.5:
            x = x[:, :, :, ::-1].copy()
        scale = float(rng.uniform(0.85, 1.15))
        shift = float(rng.uniform(-0.05, 0.05))
        x = np.clip(x * scale + shift, 0.0, 1.0)
    return x, plane_mask


def plane_vol9(
    pixels: np.ndarray,
    mask: np.ndarray,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Three plane volumes: channels = fluid, struct, mean; time = 9 slices.

    Returns:
      x: float32 (3, 3, 9, 224, 224) in [0, 1]
      plane_mask: float32 (3,)
    """
    if pixels.shape != (6, 9, 224, 224):
        raise ValueError(f"unexpected pixels shape {pixels.shape}")
    x = np.zeros((3, 3, 9, 224, 224), dtype=np.float32)
    plane_mask = np.zeros(3, dtype=np.float32)
    for plane, (fluid, struct) in enumerate(PLANE_SLOTS):
        present = False
        if mask[fluid]:
            x[plane, 0] = pixels[fluid].astype(np.float32) / 255.0
            present = True
        if mask[struct]:
            x[plane, 1] = pixels[struct].astype(np.float32) / 255.0
            present = True
        if present:
            x[plane, 2] = 0.5 * (x[plane, 0] + x[plane, 1])
            plane_mask[plane] = 1.0
    if augment:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() < 0.5:
            x = x[:, :, :, :, ::-1].copy()
        scale = float(rng.uniform(0.85, 1.15))
        shift = float(rng.uniform(-0.05, 0.05))
        x = np.clip(x * scale + shift, 0.0, 1.0)
    return x, plane_mask


class KneePixelDataset:
    def __init__(
        self,
        cache: PixelCacheV1,
        table: pd.DataFrame,
        fold: int | None = None,
        holdout: bool = False,
        targets: list[str] | None = None,
        augment: bool = False,
        seed: int = 2026,
        layout: str = "rgb6",
    ):
        self.cache_root = str(cache.root)
        self.cache = cache
        self.targets = targets or TARGETS
        self.augment = augment
        self.seed = int(seed)
        if layout not in {"rgb6", "vol9"}:
            raise ValueError(f"unknown layout {layout}")
        self.layout = layout
        selected = table
        if fold is not None:
            if holdout:
                selected = table.loc[table["fold"] == fold]
            else:
                selected = table.loc[table["fold"] != fold]
        self.table = selected.reset_index(drop=True)

    def __getstate__(self) -> dict:
        return {
            "cache_root": self.cache_root,
            "targets": self.targets,
            "augment": self.augment,
            "seed": self.seed,
            "layout": self.layout,
            "table": self.table,
        }

    def __setstate__(self, state: dict) -> None:
        self.cache_root = state["cache_root"]
        self.cache = PixelCacheV1(Path(self.cache_root))
        self.targets = state["targets"]
        self.augment = state["augment"]
        self.seed = state["seed"]
        self.layout = state.get("layout", "rgb6")
        self.table = state["table"]

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        uid = str(row["StudyInstanceUID"])
        pixels, mask = self.cache.get(uid)
        rng = None
        if self.augment:
            rng = np.random.default_rng(self.seed + index * 10007)
        if self.layout == "vol9":
            x, plane_mask = plane_vol9(pixels, mask, augment=self.augment, rng=rng)
        else:
            x, plane_mask = plane_rgb6(pixels, mask, augment=self.augment, rng=rng)
        y = row[self.targets].to_numpy(dtype=np.float32, copy=True)
        labeled = np.isfinite(y).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0).astype(np.float32)
        return {
            "uid": uid,
            "x": np.ascontiguousarray(x),
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
