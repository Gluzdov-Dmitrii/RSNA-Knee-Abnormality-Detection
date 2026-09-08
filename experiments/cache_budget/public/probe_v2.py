from __future__ import annotations

# === CELL: setup ===
import matplotlib

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

try:
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:
    sk_ssim = None

T0 = time.time()

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
SLOTS_PUBLIC = [
    ("SAG_FLUID", "Sagittal", True),
    ("COR_FLUID", "Coronal", True),
    ("AX_FLUID", "Axial", True),
    ("SAG_STRUCT", "Sagittal", False),
    ("COR_STRUCT", "Coronal", False),
    ("AX_STRUCT", "Axial", False),
]
N_SLOT = 6
GROUP = 3
ORDER_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "SliceLocation", "InstanceNumber"]
N_FULL = 4407
N_PER_FOLD = 40
SEED = 2026
PIX_THREADS = 8
N_BOOT = 800
EXPECTED_FOLDS_SHA256 = "3086df3341333f44adb883292da386857c3230eaa2d501514ddf827a2da11b1a"
STEVEN_GIB = N_FULL * N_SLOT * 9 * 224 * 224 / (1024**3)


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def full_gib(img: int, n_slices: int) -> float:
    return N_FULL * N_SLOT * n_slices * img * img / (1024**3)


# === CELL: variants ===
# Resolution family is decoded once at 336² x 9, crop 130 mm, then downsampled.
# Slice and crop families are native decodes. Crop does not change GiB.
RES_IMGS = [128, 160, 192, 224, 256, 288, 336]
SLICE_COUNTS = [3, 6, 9, 12, 15]
CROP_MMS = [110.0, 130.0, 160.0]


def variant_row(vid: str, family: str, img: int, n_slices: int, crop_mm: float, window: str, source: str) -> dict:
    return {
        "id": vid,
        "family": family,
        "img": img,
        "n_slices": n_slices,
        "n_group": n_slices // GROUP,
        "crop_mm": crop_mm,
        "window": window,
        "full_corpus_gib": round(full_gib(img, n_slices), 3),
        "source": source,
    }


def planned_variants() -> pd.DataFrame:
    rows = []
    for img in RES_IMGS:
        rows.append(
            variant_row(
                f"res_{img}x9_c130",
                "resolution",
                img,
                9,
                130.0,
                "0.35,0.65",
                "decode_336x9_c130" if img == 336 else "downsample_from_336x9_c130",
            )
        )
    for n_slices in SLICE_COUNTS:
        if n_slices == 9:
            continue
        rows.append(
            variant_row(
                f"slc_224x{n_slices}_c130",
                "slices",
                224,
                n_slices,
                130.0,
                "0.35,0.65",
                "native",
            )
        )
    for crop in CROP_MMS:
        if crop == 130.0:
            continue
        crop_i = int(crop)
        rows.append(
            variant_row(
                f"crp_224x9_c{crop_i}",
                "crop",
                224,
                9,
                crop,
                "0.35,0.65",
                "native",
            )
        )
    rows.append(variant_row("tiny_160x3_c130", "tiny", 160, 3, 130.0, "0.40,0.60", "native"))
    rows.append(variant_row("tiny_128x3_c130", "tiny", 128, 3, 130.0, "0.40,0.60", "downsample_from_160x3_c130"))
    return pd.DataFrame(rows)


# === CELL: io ===
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_comp_root() -> Path:
    for c in [
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    ]:
        if (c / "train.csv").is_file():
            return c
    raise FileNotFoundError("competition mount not found")


def find_labels() -> Path:
    hits = list(Path("/kaggle/input").glob("**/report_labels_v2.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError("Pilkwang report_labels_v2.csv not mounted")


def find_folds() -> Path:
    for c in [
        Path("/kaggle/working/folds.csv"),
        Path("folds.csv"),
        Path("experiments/cache_budget/kernel/folds.csv"),
        Path("ops/assets/FOLDS_V1/folds.csv"),
    ]:
        if c.is_file():
            return c
    raise FileNotFoundError("FOLDS_V1 folds.csv not found")


def working_dir() -> Path:
    kaggle_out = Path("/kaggle/working")
    if kaggle_out.exists():
        return kaggle_out
    out = Path("experiments/cache_budget/public")
    out.mkdir(parents=True, exist_ok=True)
    return out


# === CELL: decode ===
def order_slices(directory: str, files: list[str]) -> list[str]:
    if len(files) < 2:
        return list(files)
    positions, locations, instances = [], [], []
    normal = None
    for name in files:
        try:
            ds = pydicom.dcmread(
                os.path.join(directory, name),
                stop_before_pixels=True,
                force=True,
                specific_tags=ORDER_TAGS,
            )
        except Exception:
            positions.append(None)
            locations.append(None)
            instances.append(None)
            continue
        pos = getattr(ds, "ImagePositionPatient", None)
        orient = getattr(ds, "ImageOrientationPatient", None)
        if normal is None and orient is not None and len(orient) == 6:
            try:
                row_dir = np.array([float(v) for v in orient[:3]])
                col_dir = np.array([float(v) for v in orient[3:]])
                normal = np.cross(row_dir, col_dir)
            except Exception:
                normal = None
        try:
            positions.append(np.array([float(v) for v in pos]) if pos is not None else None)
        except Exception:
            positions.append(None)
        loc = getattr(ds, "SliceLocation", None)
        locations.append(float(loc) if loc is not None else None)
        num = getattr(ds, "InstanceNumber", None)
        instances.append(float(num) if num is not None else None)

    def sorted_by(values):
        return [f for _, f in sorted(zip(values, files), key=lambda pair: pair[0])]

    if normal is not None and all(p is not None for p in positions):
        return sorted_by([float(p @ normal) for p in positions])
    if all(loc is not None for loc in locations):
        return sorted_by(locations)
    if all(num is not None for num in instances):
        return sorted_by(instances)
    return list(files)


def pixel_spacing(ds) -> float | None:
    ps = getattr(ds, "PixelSpacing", None)
    if ps is None:
        return None
    try:
        return float(ps[0])
    except Exception:
        return None


def index_series(directory: str, files: list[str]) -> dict:
    files = [f for f in files if f.endswith(".dcm")]
    files = order_slices(directory, files)
    px = None
    if files:
        try:
            ds = pydicom.dcmread(
                os.path.join(directory, files[0]),
                stop_before_pixels=True,
                force=True,
                specific_tags=["PixelSpacing"],
            )
            px = pixel_spacing(ds)
        except Exception:
            px = None
    return {"dir": directory, "files": files, "px": px}


def n_dcm(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix == ".dcm")


def pick_series(root: Path, series_df: pd.DataFrame, study: str) -> dict[str, str]:
    sub = series_df[series_df["StudyInstanceUID"] == study]
    chosen: dict[str, str] = {}
    for name, plane, fluid in SLOTS_PUBLIC:
        cand = sub[(sub["Anatomical_Plane"] == plane) & (sub["Fluid_Sensitive"].astype(int) == int(fluid))]
        if cand.empty:
            continue
        best, best_n = None, -1
        for sid in cand["SeriesInstanceUID"]:
            n = n_dcm(root / "train_series" / study / sid)
            if n > best_n:
                best, best_n = sid, n
        if best:
            chosen[name] = best
    return chosen


def read_slot(rec: dict, n_slice: int, out_size: int, crop_mm: float, window: str) -> np.ndarray | None:
    files, directory, px = rec["files"], rec["dir"], rec["px"]
    n = len(files)
    if n == 0:
        return None
    lo_f, hi_f = (float(x) for x in window.split(","))
    lo, hi = int(lo_f * (n - 1)), int(hi_f * (n - 1))
    if hi <= lo:
        lo, hi = 0, n - 1
    n_anchor = max(1, n_slice // GROUP)
    anchors = np.linspace(lo, hi, n_anchor).astype(int) if n_anchor > 1 else np.array([(lo + hi) // 2])
    idx: list[int] = []
    for centre in anchors:
        start = int(np.clip(centre - GROUP // 2, 0, max(0, n - GROUP)))
        idx.extend(range(start, min(start + GROUP, n)))
    while len(idx) < n_slice:
        idx.append(idx[-1] if idx else 0)

    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(directory, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
            if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
                a = a.max() - a
            if px is None:
                px = pixel_spacing(ds)
            planes.append(a)
        except Exception:
            planes.append(None)
    shp = next((p.shape for p in planes if p is not None), None)
    if shp is None:
        return None
    planes = [p if (p is not None and p.shape == shp) else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)
    if px and np.isfinite(px) and px > 0:
        want = int(round(crop_mm / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            vol = vol[:, max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()


def downsample_slot(vol: np.ndarray | None, img: int) -> np.ndarray | None:
    if vol is None:
        return None
    if vol.shape[-1] == img and vol.shape[-2] == img:
        return vol
    t = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0)
    t = F.interpolate(t, size=(img, img), mode="bilinear", align_corners=False)
    return t.squeeze(0).round().clamp(0, 255).to(torch.uint8).numpy()


def decode_study(recs: dict[str, dict], n_slices: int, img: int, crop_mm: float, window: str) -> dict[str, np.ndarray | None]:
    out: dict[str, np.ndarray | None] = {}
    for name, _, _ in SLOTS_PUBLIC:
        rec = recs.get(name)
        out[name] = None if not rec else read_slot(rec, n_slices, img, crop_mm, window)
    return out


def decode_many(uids: list[str], recs_by_uid: dict[str, dict], n_slices: int, img: int, crop_mm: float, window: str) -> list[dict[str, np.ndarray | None]]:
    def _one(uid: str):
        return decode_study(recs_by_uid[uid], n_slices, img, crop_mm, window)

    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        return list(pool.map(_one, uids))


def downsample_cache(cache: list[dict[str, np.ndarray | None]], img: int) -> list[dict[str, np.ndarray | None]]:
    out = []
    for slots in cache:
        out.append({name: downsample_slot(slots.get(name), img) for name, _, _ in SLOTS_PUBLIC})
    return out


# === CELL: metrics ===
def slot_features(vol: np.ndarray | None) -> np.ndarray:
    feats = np.zeros(8, dtype=np.float32)
    if vol is None:
        feats[7] = 1.0
        return feats
    x = vol.astype(np.float32)
    feats[0] = x.mean()
    feats[1] = x.std()
    feats[2] = np.percentile(x, 10)
    feats[3] = np.percentile(x, 90)
    mid = x[x.shape[0] // 2]
    feats[4] = mid.mean()
    if x.shape[0] > 1:
        feats[5] = np.abs(np.diff(x, axis=0)).mean()
    gy = np.abs(np.diff(mid, axis=0)).mean() if mid.shape[0] > 1 else 0.0
    gx = np.abs(np.diff(mid, axis=1)).mean() if mid.shape[1] > 1 else 0.0
    feats[6] = gy + gx
    return feats


def study_features(slots: dict[str, np.ndarray | None]) -> np.ndarray:
    return np.concatenate([slot_features(slots.get(name)) for name, _, _ in SLOTS_PUBLIC], axis=0)


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) < 2:
            continue
        vals.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(vals)) if vals else float("nan")


def oof_gbdt(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[float, np.ndarray]:
    n, k = y.shape
    pred = np.zeros((n, k), dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        tr, va = folds != fold, folds == fold
        for j in range(k):
            ytr = y[tr, j]
            if len(np.unique(ytr)) < 2:
                pred[va, j] = float(ytr.mean())
                continue
            clf = HistGradientBoostingClassifier(
                max_depth=3,
                max_iter=80,
                learning_rate=0.08,
                random_state=SEED,
            )
            try:
                clf.fit(X[tr], ytr)
                proba = clf.predict_proba(X[va])
                pred[va, j] = proba[:, 1] if proba.shape[1] == 2 else float(ytr[0])
            except Exception:
                pred[va, j] = float(ytr.mean())
    return macro_auc(y, pred), pred


def bootstrap_macro(y: np.ndarray, pred: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(y)
    stats = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        stats.append(macro_auc(y[idx], pred[idx]))
    stats = np.array(stats, dtype=np.float64)
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


def simple_ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def resize_vol(vol: np.ndarray, n_slice: int, hw: int) -> np.ndarray:
    t = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(n_slice, hw, hw), mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def mean_ssim(a: dict[str, np.ndarray | None], b: dict[str, np.ndarray | None]) -> float:
    vals = []
    for name, _, _ in SLOTS_PUBLIC:
        va, vb = a.get(name), b.get(name)
        if va is None or vb is None:
            continue
        n = min(va.shape[0], vb.shape[0], 3)
        aa = resize_vol(va, n, 64)
        bb = resize_vol(vb, n, 64)
        for i in range(n):
            if sk_ssim is not None:
                vals.append(sk_ssim(aa[i], bb[i], data_range=255.0))
            else:
                vals.append(simple_ssim(aa[i], bb[i]))
    return float(np.mean(vals)) if vals else float("nan")


def cache_ssim(cache, ref) -> float:
    return float(np.nanmean([mean_ssim(a, b) for a, b in zip(cache, ref)]))


def score_cache(cache, y, fold_ids, rng) -> tuple[float, float, float, np.ndarray]:
    X = np.stack([study_features(s) for s in cache])
    auc, pred = oof_gbdt(X, y, fold_ids)
    lo, hi = bootstrap_macro(y, pred, rng)
    return auc, lo, hi, pred


# === CELL: run ===
def main() -> None:
    plan = planned_variants()
    log("planned variants")
    print(plan.to_string(index=False))

    root = find_comp_root()
    labels_path = find_labels()
    folds_path = find_folds()
    folds_sha = sha256_file(folds_path)
    if folds_sha != EXPECTED_FOLDS_SHA256:
        raise SystemExit(f"FOLDS_V1 hash mismatch: {folds_sha}")
    log(f"root={root}")
    log(f"labels={labels_path}")
    log(f"folds={folds_path} sha={folds_sha[:12]}")
    log(f"steven_published_gib={STEVEN_GIB:.3f}")

    folds = pd.read_csv(folds_path)
    labels = pd.read_csv(labels_path, usecols=["StudyInstanceUID"] + TARGETS)
    series = pd.read_csv(root / "train_series.csv")
    rng = np.random.default_rng(SEED)
    picked = []
    for fold in range(5):
        uids = folds.loc[folds["fold"] == fold, "StudyInstanceUID"].tolist()
        rng.shuffle(uids)
        picked.extend(uids[:N_PER_FOLD])
    subset = pd.DataFrame({"StudyInstanceUID": picked}).merge(folds, on="StudyInstanceUID")
    subset = subset.merge(labels, on="StudyInstanceUID", how="left")
    subset[TARGETS] = subset[TARGETS].fillna(0)
    y = (subset[TARGETS].to_numpy(dtype=np.float64) >= 0.5).astype(np.int32)
    fold_ids = subset["fold"].to_numpy()
    uids = subset["StudyInstanceUID"].tolist()
    log(f"subset={len(subset)}")

    log("index series")
    recs_by_uid: dict[str, dict] = {}
    for i, uid in enumerate(uids):
        chosen = pick_series(root, series, uid)
        recs = {}
        for name, sid in chosen.items():
            d = root / "train_series" / uid / sid
            files = sorted(p.name for p in d.iterdir() if p.suffix == ".dcm") if d.is_dir() else []
            recs[name] = index_series(str(d), files)
        recs_by_uid[uid] = recs
        if (i + 1) % 20 == 0 or i + 1 == len(uids):
            log(f"  indexed {i + 1}/{len(uids)}")

    rows = []
    rng_boot = np.random.default_rng(SEED + 1)

    def record(meta: dict, cache, ref_cache) -> None:
        n_ok = sum(v is not None for slots in cache for v in slots.values())
        auc, lo, hi, _pred = score_cache(cache, y, fold_ids, rng_boot)
        ssim = cache_ssim(cache, ref_cache) if ref_cache is not None else float("nan")
        row = dict(meta)
        row.update(
            {
                "subset_n": len(subset),
                "slots_decoded": int(n_ok),
                "gbdt_macro_auc": auc,
                "gbdt_ci95_lo": lo,
                "gbdt_ci95_hi": hi,
                "ssim_vs_336x9_c130": ssim,
            }
        )
        rows.append(row)
        log(f"  {meta['id']}: {meta['full_corpus_gib']:.3f} GiB  AUC={auc:.4f} [{lo:.4f},{hi:.4f}]  SSIM={ssim:.3f}")

    log("decode resolution parent 336x9 c130")
    cache_336 = decode_many(uids, recs_by_uid, 9, 336, 130.0, "0.35,0.65")
    for img in RES_IMGS:
        cache = cache_336 if img == 336 else downsample_cache(cache_336, img)
        meta = variant_row(
            f"res_{img}x9_c130",
            "resolution",
            img,
            9,
            130.0,
            "0.35,0.65",
            "decode_336x9_c130" if img == 336 else "downsample_from_336x9_c130",
        )
        record(meta, cache, cache_336)
        if img != 336:
            del cache
    ref = cache_336

    for n_slices in SLICE_COUNTS:
        if n_slices == 9:
            continue
        log(f"decode slices 224x{n_slices} c130")
        cache = decode_many(uids, recs_by_uid, n_slices, 224, 130.0, "0.35,0.65")
        meta = variant_row(f"slc_224x{n_slices}_c130", "slices", 224, n_slices, 130.0, "0.35,0.65", "native")
        record(meta, cache, ref)
        del cache

    for crop in CROP_MMS:
        if crop == 130.0:
            continue
        crop_i = int(crop)
        log(f"decode crop 224x9 c{crop_i}")
        cache = decode_many(uids, recs_by_uid, 9, 224, crop, "0.35,0.65")
        meta = variant_row(f"crp_224x9_c{crop_i}", "crop", 224, 9, crop, "0.35,0.65", "native")
        record(meta, cache, ref)
        del cache

    log("decode tiny 160x3 c130")
    cache_tiny = decode_many(uids, recs_by_uid, 3, 160, 130.0, "0.40,0.60")
    record(variant_row("tiny_160x3_c130", "tiny", 160, 3, 130.0, "0.40,0.60", "native"), cache_tiny, ref)
    record(
        variant_row("tiny_128x3_c130", "tiny", 128, 3, 130.0, "0.40,0.60", "downsample_from_160x3_c130"),
        downsample_cache(cache_tiny, 128),
        ref,
    )
    del cache_tiny
    del cache_336

    out = pd.DataFrame(rows)
    out_dir = working_dir()
    metrics_path = out_dir / "cache_budget_v2_metrics.csv"
    out.to_csv(metrics_path, index=False)
    spec = {
        "kernel": "dmitriigluzdov/rsna-knee-on-a-storage-budget",
        "source_geometry": "stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache",
        "folds": "FOLDS_V1",
        "folds_sha256": folds_sha,
        "labels": "LABEL_PILKWANG_V1",
        "n_subset": int(len(subset)),
        "n_per_fold": N_PER_FOLD,
        "seed": SEED,
        "n_boot": N_BOOT,
        "slot_scheme": "public_train_series_flags",
        "steven_published_gib": STEVEN_GIB,
        "probe_model": "HistGradientBoostingClassifier slot stats, not DINOv2",
        "rows": rows,
    }
    spec_path = out_dir / "cache_budget_v2_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    plot_curve(out, out_dir)
    log(f"wrote {metrics_path}")
    print(out.to_string(index=False))


# === CELL: plots ===
def plot_curve(out: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    ax = axes[0]
    families = [
        ("resolution", "o-", "resolution, 9 slices, crop 130 mm"),
        ("slices", "s--", "slice count, 224², crop 130 mm"),
        ("tiny", "^:", "tiny, 3 slices, crop 130 mm"),
    ]
    for fam, fmt, label in families:
        d = out[out["family"] == fam].copy()
        if fam == "slices":
            extra = out[out["id"] == "res_224x9_c130"].copy()
            if not extra.empty:
                extra["family"] = "slices"
                extra["id"] = "slc_224x9_c130"
                d = pd.concat([d, extra], ignore_index=True)
        d = d.sort_values("full_corpus_gib")
        if d.empty:
            continue
        yerr = np.vstack([d["gbdt_macro_auc"] - d["gbdt_ci95_lo"], d["gbdt_ci95_hi"] - d["gbdt_macro_auc"]])
        ax.errorbar(d["full_corpus_gib"], d["gbdt_macro_auc"], yerr=yerr, fmt=fmt, label=label, capsize=3)
        for _, r in d.iterrows():
            ax.annotate(r["id"].replace("_c130", ""), (r["full_corpus_gib"], r["gbdt_macro_auc"]), fontsize=7)
    ax.axvline(STEVEN_GIB, color="0.4", ls=":", label="Steven 11.1 GiB design")
    ax.set_xlabel("Full-corpus cache size (GiB, uint8)")
    ax.set_ylabel("Subset OOF macro AUC (GBDT slot stats)")
    ax.set_title("Quality vs size  ·  crop 130 mm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    crop = pd.concat(
        [out[out["id"] == "res_224x9_c130"], out[out["family"] == "crop"]]
    ).drop_duplicates("id")
    crop = crop.sort_values("crop_mm")
    ax.bar([str(int(v)) for v in crop["crop_mm"]], crop["gbdt_macro_auc"], color="#4c78a8")
    for i, (_, r) in enumerate(crop.iterrows()):
        ax.plot([i, i], [r["gbdt_ci95_lo"], r["gbdt_ci95_hi"]], color="black")
        ax.scatter([i], [r["gbdt_macro_auc"]], color="black", zorder=3)
    ax.set_xlabel("Physical crop (mm) at 224² × 9, same GiB")
    ax.set_ylabel("Subset OOF macro AUC")
    ax.set_title("Crop is not a size knob")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    png = out_dir / "cache_budget_v2_curve.png"
    fig.savefig(png, dpi=150)
    plt.show()

    fig2, ax = plt.subplots(figsize=(7.5, 4.6))
    for fam, fmt, label in [
        ("resolution", "o-", "resolution, 9 slices"),
        ("slices", "s--", "slice count, 224²"),
        ("tiny", "^:", "tiny, 3 slices"),
    ]:
        d = out[out["family"] == fam].copy()
        if fam == "slices":
            extra = out[out["id"] == "res_224x9_c130"].copy()
            if not extra.empty:
                extra["id"] = "slc_224x9_c130"
                d = pd.concat([d, extra], ignore_index=True)
        d = d.sort_values("full_corpus_gib")
        if d.empty:
            continue
        ax.plot(d["full_corpus_gib"], d["ssim_vs_336x9_c130"], fmt, label=label)
        for _, r in d.iterrows():
            ax.annotate(r["id"].replace("_c130", ""), (r["full_corpus_gib"], r["ssim_vs_336x9_c130"]), fontsize=7)
    ax.set_xlabel("Full-corpus cache size (GiB, uint8)")
    ax.set_ylabel("Mean SSIM vs 336² × 9 crop 130 mm")
    ax.set_title("Fidelity to the densest crop-130 cache, not to raw DICOM")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    png2 = out_dir / "cache_budget_v2_ssim.png"
    fig2.savefig(png2, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
