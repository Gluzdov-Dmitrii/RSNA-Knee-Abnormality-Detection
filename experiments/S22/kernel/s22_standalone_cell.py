# S22_LOCAL_CNN_BAG
# PIXEL_CACHE_V1 live decode + 5-fold ResNet-18 bag. No A0/Raptor mix.
# Writes submission.csv as rank-percentile of the bag (monotonic; AUC-preserving).
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from torchvision.models import resnet18

T0 = time.time()
IMG = 224
N_SLICES = 9
CROP_MM = 130.0
WINDOW = (0.35, 0.65)
RGB_SLICE_IDX = (1, 4, 7)
PLANE_SLOTS = ((0, 3), (1, 4), (2, 5))
SLOTS = [
    ("SAG_FLUID", "Sagittal", 1),
    ("COR_FLUID", "Coronal", 1),
    ("AX_FLUID", "Axial", 1),
    ("SAG_STRUCT", "Sagittal", 0),
    ("COR_STRUCT", "Coronal", 0),
    ("AX_STRUCT", "Axial", 0),
]
LABELS = [
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
HEADER = ["StudyInstanceUID", *LABELS]
WORK = Path("/kaggle/working")
WEIGHT_ROOTS = [
    Path("/kaggle/input/rsna-s22-resnet18-25d-folds"),
    Path("/kaggle/input/datasets/dmitriigluzdov/rsna-s22-resnet18-25d-folds"),
]


def log(msg: str) -> None:
    print(f"[S22 {time.time() - T0:.0f}s] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def competition_root() -> Path:
    for path in (
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    ):
        if (path / "sample_submission.csv").is_file():
            return path
    raise FileNotFoundError("competition sample_submission.csv is not mounted")


def weight_root() -> Path:
    for path in WEIGHT_ROOTS:
        if (path / "manifest.json").is_file() and (path / "fold0.pt").is_file():
            return path
    raise FileNotFoundError("S22 fold dataset is not mounted")


def finite_vector(value, n: int):
    try:
        array = np.asarray(value, dtype=float)
        return array if array.shape == (n,) and np.isfinite(array).all() else None
    except (ValueError, TypeError):
        return None


def order_files(paths: list[Path]) -> tuple[list[Path], float | None]:
    headers = [
        pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "SliceLocation", "InstanceNumber", "PixelSpacing"],
            force=True,
        )
        for path in paths
    ]
    normal = None
    for ds in headers:
        ori = finite_vector(getattr(ds, "ImageOrientationPatient", None), 6)
        if ori is not None:
            vec = np.cross(ori[:3], ori[3:])
            if np.linalg.norm(vec) > 1e-6:
                normal = vec / np.linalg.norm(vec)
                break
    positions = [finite_vector(getattr(ds, "ImagePositionPatient", None), 3) for ds in headers]
    keys = None
    if normal is not None and all(pos is not None for pos in positions):
        keys = [float(pos @ normal) for pos in positions]
    else:
        for tag in ("SliceLocation", "InstanceNumber"):
            vals = [finite_vector([getattr(ds, tag, None)], 1) for ds in headers]
            if all(val is not None for val in vals):
                keys = [float(val[0]) for val in vals]
                break
    if keys is None:
        if len(paths) == 1:
            keys = [0.0]
        else:
            raise ValueError("unordered series")
    idx = sorted(range(len(paths)), key=lambda i: (keys[i], paths[i].name))
    spacing = finite_vector(getattr(headers[idx[0]], "PixelSpacing", None), 2)
    px = None if spacing is None else float(spacing[0])
    return [paths[i] for i in idx], px


def sample_indices(n: int, n_slices: int, window) -> list[int]:
    if n < 1 or n_slices < 3 or n_slices % 3:
        raise ValueError("need nonempty series and groups of three")
    lo, hi = [int(frac * (n - 1)) for frac in window]
    if hi <= lo:
        lo, hi = 0, n - 1
    anchors = np.linspace(lo, hi, n_slices // 3).astype(int) if n_slices > 3 else np.array([(lo + hi) // 2])
    indices: list[int] = []
    for anchor in anchors:
        start = int(np.clip(anchor - 1, 0, max(0, n - 3)))
        indices.extend(range(start, min(start + 3, n)))
    indices += [indices[-1]] * (n_slices - len(indices))
    return indices[:n_slices]


def series_dir(comp: Path, uid: str, sid: str) -> Path | None:
    for split in ("test_series", "train_series"):
        path = comp / split / uid / sid
        if path.is_dir():
            return path
    return None


def load_pixels(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(path, force=True)
    array = ds.pixel_array.astype(np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(path.name)
    array = array * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        array = array.max() - array
    return array


def render_slot(rec: dict) -> np.ndarray:
    idx = sample_indices(rec["n"], N_SLICES, WINDOW)
    vol = np.stack([rec["pixels"][i] for i in idx])
    px = rec["spacing"]
    if px is not None and np.isfinite(px) and px > 0:
        want = int(round(CROP_MM / px))
        height, width = vol.shape[-2:]
        if 16 < want < min(height, width):
            half = want // 2
            vol = vol[:, height // 2 - half : height // 2 + half, width // 2 - half : width // 2 + half]
    low, high = np.percentile(vol, [1, 99])
    vol = np.clip((vol - low) / max(high - low, 1e-6), 0, 1)
    kwargs = dict(order=1, mode="nearest", prefilter=False)
    try:
        vol = zoom(vol, (1, IMG / vol.shape[1], IMG / vol.shape[2]), grid_mode=True, **kwargs)
    except TypeError:
        vol = zoom(vol, (1, IMG / vol.shape[1], IMG / vol.shape[2]), **kwargs)
    result = np.clip(np.rint(vol * 255), 0, 255).astype(np.uint8)
    if result.shape != (N_SLICES, IMG, IMG):
        raise RuntimeError(f"bad render {result.shape}")
    return result


def decode_study(comp: Path, frame: pd.DataFrame, uid: str) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.zeros((6, N_SLICES, IMG, IMG), dtype=np.uint8)
    mask = np.zeros(6, dtype=np.uint8)
    cand = frame.loc[frame["StudyInstanceUID"].astype(str) == uid]
    for slot, (name, plane, fluid) in enumerate(SLOTS):
        rows = cand[(cand["Anatomical_Plane"] == plane) & (cand["Fluid_Sensitive"].astype(int) == fluid)]
        choices = []
        for row in rows.itertuples(index=False):
            root = series_dir(comp, uid, str(row.SeriesInstanceUID))
            paths = sorted(root.glob("*.dcm")) if root is not None else []
            choices.append((len(paths), paths, str(row.SeriesInstanceUID)))
        if not choices:
            continue
        _, paths, _ = max(choices, key=lambda item: item[0])
        if not paths:
            continue
        try:
            paths, spacing = order_files(paths)
            needed = sample_indices(len(paths), N_SLICES, WINDOW)
            rec = {"n": len(paths), "pixels": {i: load_pixels(paths[i]) for i in sorted(set(needed))}, "spacing": spacing}
            pixels[slot] = render_slot(rec)
            mask[slot] = 1
        except Exception:
            continue
    return pixels, mask


def plane_rgb6(pixels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((3, 6, IMG, IMG), dtype=np.float32)
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
    return x, plane_mask


class PlaneHead(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            trunk = resnet18(weights=None)
        except TypeError:
            trunk = resnet18(pretrained=False)
        conv1 = trunk.conv1
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = conv1.weight
            new_conv.weight[:, 3:6] = conv1.weight
            new_conv.weight *= 0.5
        trunk.conv1 = new_conv
        trunk.fc = nn.Identity()
        self.trunk = trunk

    def forward(self, x):
        return self.trunk(x)


class S22ResNet18(nn.Module):
    def __init__(self, n_targets: int = 12):
        super().__init__()
        self.planes = nn.ModuleList([PlaneHead() for _ in range(3)])
        self.dropout = nn.Dropout(0.2)
        self.fusion = nn.Linear(3 * 512, n_targets)

    def forward(self, x, plane_mask):
        feats = []
        for plane, head in enumerate(self.planes):
            feat = head(x[:, plane])
            feats.append(feat * plane_mask[:, plane].unsqueeze(1))
        return self.fusion(self.dropout(torch.cat(feats, dim=1)))


def load_folds(root: Path, device: torch.device) -> list[nn.Module]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("n_targets") != 12:
        raise RuntimeError("S22 manifest n_targets drift")
    models = []
    for fold in range(5):
        path = root / f"fold{fold}.pt"
        expected = next(item["sha256"] for item in manifest["folds"] if item["fold"] == fold)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"fold{fold} sha256 drift {actual}")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        model = S22ResNet18().to(device)
        model.load_state_dict({key: value.float() for key, value in state.items()})
        model.eval()
        models.append(model)
    return models


def rank_pct(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(np.float64)


def load_series_table(comp: Path, uids: list[str]) -> pd.DataFrame:
    uid_set = set(uids)
    for name in ("test_series.csv", "train_series.csv"):
        path = comp / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame["StudyInstanceUID"].astype(str).isin(uid_set).any():
            return frame
    raise FileNotFoundError("test_series.csv/train_series.csv missing for mounted UIDs")


comp = competition_root()
sample = pd.read_csv(comp / "sample_submission.csv", dtype={"StudyInstanceUID": str})
if sample.columns.tolist() != HEADER:
    raise RuntimeError(f"sample header drift {sample.columns.tolist()}")
uids = sample["StudyInstanceUID"].astype(str).tolist()
if not uids or len(set(uids)) != len(uids):
    raise RuntimeError("sample UID identity drift")

weights = weight_root()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"weights={weights} device={device} n={len(uids)}")
models = load_folds(weights, device)
series = load_series_table(comp, uids)
threads = min(8, os.cpu_count() or 2)

s22 = np.full((len(uids), 12), 0.5, dtype=np.float64)
present = np.zeros(len(uids), dtype=np.int32)
chunk = 16
with torch.no_grad():
    for start in range(0, len(uids), chunk):
        batch_uids = uids[start : start + chunk]
        with ThreadPoolExecutor(max_workers=threads) as pool:
            decoded = list(pool.map(lambda uid: decode_study(comp, series, uid), batch_uids))
        xs = []
        masks = []
        for offset, (pixels, mask) in enumerate(decoded):
            x, plane_mask = plane_rgb6(pixels, mask)
            xs.append(x)
            masks.append(plane_mask)
            present[start + offset] = int(mask.sum() > 0)
        x_t = torch.from_numpy(np.stack(xs, axis=0)).to(device)
        m_t = torch.from_numpy(np.stack(masks, axis=0)).to(device)
        probs = []
        for model in models:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x_t, m_t)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        s22[start : start + len(decoded)] = np.mean(probs, axis=0)
        log(f"decoded {start + len(decoded)}/{len(uids)}")

if int(present.sum()) == 0:
    raise RuntimeError("S22 decoded zero studies with a present slot")

out = sample.copy()
for col, label in enumerate(LABELS):
    ranked = rank_pct(s22[:, col])
    if not np.isfinite(ranked).all() or ranked.min() < 0 or ranked.max() > 1:
        raise RuntimeError(f"invalid bag rank: {label}")
    if len(uids) >= 3 and np.unique(ranked).size < 2:
        raise RuntimeError(f"{label} bag is constant")
    out[label] = ranked

tmp_path = WORK / ".submission.tmp.csv"
out.to_csv(tmp_path, index=False)
tmp_path.replace(WORK / "submission.csv")

gate = pd.read_csv(WORK / "submission.csv", dtype={"StudyInstanceUID": str})
if gate.columns.tolist() != HEADER:
    raise RuntimeError(f"submission header drift {gate.columns.tolist()}")
if gate["StudyInstanceUID"].astype(str).tolist() != uids:
    raise RuntimeError("final UID order drift")
values = gate[LABELS].to_numpy(np.float64)
if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
    raise RuntimeError("final predictions outside finite [0,1]")

receipt = {
    "schema_version": "s22_local_cnn_bag_v1",
    "status": "S22_LOCAL_CNN_PASS",
    "experiment_id": "S22",
    "n_studies": len(uids),
    "present_studies": int(present.sum()),
    "decode_threads": threads,
    "device": str(device),
    "weight_root": str(weights),
    "weight_manifest_sha256": sha256_file(weights / "manifest.json"),
    "submission_sha256": sha256_file(WORK / "submission.csv"),
    "seconds": round(time.time() - T0, 1),
    "internet_expected_off": True,
    "blend": "none_local_cnn_only",
}
(WORK / "s22_local_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
log("S22_LOCAL_CNN_PASS " + receipt["submission_sha256"][:12] + f" seconds={receipt['seconds']}")
