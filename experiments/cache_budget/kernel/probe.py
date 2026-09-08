from __future__ import annotations

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
import torch.nn as nn
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
N_PER_FOLD = 32
SEED = 2026
PIX_THREADS = 8
EXPECTED_FOLDS_SHA256 = "3086df3341333f44adb883292da386857c3230eaa2d501514ddf827a2da11b1a"
STEVEN_GIB = N_FULL * N_SLOT * 9 * 224 * 224 / (1024**3)

VARIANTS = [
    {"id": "tiny_160x3", "img": 160, "n_group": 1, "crop_mm": 130.0, "window": "0.40,0.60"},
    {"id": "steven_224x9_c130", "img": 224, "n_group": 3, "crop_mm": 130.0, "window": "0.35,0.65"},
    {"id": "steven_224x9_c160", "img": 224, "n_group": 3, "crop_mm": 160.0, "window": "0.35,0.65"},
    {"id": "wide_224x15", "img": 224, "n_group": 5, "crop_mm": 130.0, "window": "0.10,0.90"},
    {"id": "hi_336x9_c130", "img": 336, "n_group": 3, "crop_mm": 130.0, "window": "0.35,0.65"},
]


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def full_gib(img: int, n_group: int) -> float:
    return N_FULL * N_SLOT * (GROUP * n_group) * img * img / (1024**3)


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
    candidates = [
        Path("/kaggle/working/folds.csv"),
        Path("folds.csv"),
        Path("experiments/cache_budget/kernel/folds.csv"),
        Path("ops/assets/FOLDS_V1/folds.csv"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    hits = list(Path("/kaggle/input").glob("**/folds.csv")) if Path("/kaggle/input").exists() else []
    if hits:
        return hits[0]
    raise FileNotFoundError("FOLDS_V1 folds.csv not found")


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


def oof_gbdt(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[float, dict[str, float | None]]:
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
    per: dict[str, float | None] = {}
    for j, t in enumerate(TARGETS):
        if len(np.unique(y[:, j])) < 2:
            per[t] = None
        else:
            per[t] = float(roc_auc_score(y[:, j], pred[:, j]))
    return macro_auc(y, pred), per


class TinySlotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32 * N_SLOT, 12)

    def forward(self, x):
        z = [self.conv(x[:, s]).flatten(1) for s in range(N_SLOT)]
        return self.head(torch.cat(z, dim=1))


def oof_tiny_cnn(caches: list[dict[str, np.ndarray | None]], y: np.ndarray, folds: np.ndarray) -> float:
    def pack(i: int) -> np.ndarray:
        slots = caches[i]
        arr = np.zeros((N_SLOT, 3, 64, 64), dtype=np.float32)
        for si, (name, _, _) in enumerate(SLOTS_PUBLIC):
            vol = slots.get(name)
            if vol is None:
                continue
            v = vol.astype(np.float32) / 255.0
            if v.shape[0] >= 3:
                mid = v.shape[0] // 2
                trip = v[max(0, mid - 1) : mid + 2]
                if trip.shape[0] < 3:
                    trip = np.stack([v[0]] * 3)
            else:
                trip = np.stack([v[min(0, v.shape[0] - 1)]] * 3)
            t = torch.from_numpy(trip).unsqueeze(0)
            t = F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)
            arr[si] = t.squeeze(0).numpy()
        return arr

    xs = np.stack([pack(i) for i in range(len(caches))])
    n = len(caches)
    pred = np.zeros((n, 12), dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        tr, va = np.where(folds != fold)[0], np.where(folds == fold)[0]
        torch.manual_seed(SEED + int(fold))
        np.random.seed(SEED + int(fold))
        model = TinySlotNet()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        bce = nn.BCEWithLogitsLoss()
        xt = torch.from_numpy(xs[tr])
        yt = torch.from_numpy(y[tr].astype(np.float32))
        model.train()
        for _ in range(6):
            perm = np.random.permutation(len(tr))
            for start in range(0, len(tr), 16):
                bidx = perm[start : start + 16]
                opt.zero_grad()
                loss = bce(model(xt[bidx]), yt[bidx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pred[va] = torch.sigmoid(model(torch.from_numpy(xs[va]))).numpy()
    return macro_auc(y, pred)


def resize_vol(vol: np.ndarray, n_slice: int, hw: int) -> np.ndarray:
    t = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(n_slice, hw, hw), mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def simple_ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


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


def decode_study(recs: dict[str, dict], variant: dict) -> dict[str, np.ndarray | None]:
    n_slice = GROUP * variant["n_group"]
    out: dict[str, np.ndarray | None] = {}
    for name, _, _ in SLOTS_PUBLIC:
        rec = recs.get(name)
        if not rec:
            out[name] = None
            continue
        out[name] = read_slot(rec, n_slice, variant["img"], variant["crop_mm"], variant["window"])
    return out


def working_dir() -> Path:
    kaggle_out = Path("/kaggle/working")
    if kaggle_out.exists():
        return kaggle_out
    return Path("experiments/cache_budget/kernel")


def main() -> None:
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
    recs_by_uid: dict[str, dict[str, dict]] = {}
    for i, uid in enumerate(uids):
        chosen = pick_series(root, series, uid)
        recs: dict[str, dict] = {}
        for name, sid in chosen.items():
            d = root / "train_series" / uid / sid
            files = sorted(p.name for p in d.iterdir() if p.suffix == ".dcm") if d.is_dir() else []
            recs[name] = index_series(str(d), files)
        recs_by_uid[uid] = recs
        if (i + 1) % 16 == 0 or i + 1 == len(uids):
            log(f"  indexed {i + 1}/{len(uids)}")

    rows = []
    caches: dict[str, list[dict[str, np.ndarray | None]]] = {}
    for variant in VARIANTS:
        log(f"decode {variant['id']}")

        def _one(uid: str, variant=variant):
            return decode_study(recs_by_uid[uid], variant)

        with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
            decoded = list(pool.map(_one, uids))
        caches[variant["id"]] = decoded
        n_ok = sum(v is not None for slots in decoded for v in slots.values())
        X = np.stack([study_features(s) for s in decoded])
        gbdt_macro, per = oof_gbdt(X, y, fold_ids)
        cnn_macro = oof_tiny_cnn(decoded, y, fold_ids)
        gib = full_gib(variant["img"], variant["n_group"])
        rows.append(
            {
                "id": variant["id"],
                "img": variant["img"],
                "n_slices": GROUP * variant["n_group"],
                "crop_mm": variant["crop_mm"],
                "window": variant["window"],
                "full_corpus_gib": round(gib, 3),
                "subset_n": len(subset),
                "slots_decoded": int(n_ok),
                "gbdt_macro_auc": gbdt_macro,
                "tiny_cnn_macro_auc": cnn_macro,
                **{f"gbdt_{k}": v for k, v in per.items()},
            }
        )
        log(f"  gib={gib:.3f} slots={n_ok} gbdt={gbdt_macro:.4f} cnn={cnn_macro:.4f}")

    ref = caches["hi_336x9_c130"]
    for row, variant in zip(rows, VARIANTS):
        ss = [mean_ssim(a, b) for a, b in zip(caches[variant["id"]], ref)]
        row["ssim_vs_336"] = float(np.nanmean(ss))

    out = pd.DataFrame(rows)
    out_dir = working_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cache_budget_metrics.csv"
    out.to_csv(out_path, index=False)
    spec = {
        "source": "stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache",
        "folds": "FOLDS_V1",
        "folds_sha256": folds_sha,
        "labels": "LABEL_PILKWANG_V1",
        "n_subset": int(len(subset)),
        "n_per_fold": N_PER_FOLD,
        "seed": SEED,
        "slot_scheme": "public_train_series_flags",
        "steven_published_gib": STEVEN_GIB,
        "rows": rows,
    }
    spec_path = out_dir / "cache_budget_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(out["full_corpus_gib"], out["gbdt_macro_auc"], "o-", label="GBDT slot stats")
    ax.plot(out["full_corpus_gib"], out["tiny_cnn_macro_auc"], "s--", label="tiny 2.5D CNN")
    ax.axvline(STEVEN_GIB, color="gray", ls=":", label="Steven 11.13 GiB design")
    for _, r in out.iterrows():
        ax.annotate(r["id"], (r["full_corpus_gib"], r["gbdt_macro_auc"]), fontsize=8)
    ax.set_xlabel("Full-corpus cache size (GiB, uint8)")
    ax.set_ylabel("Subset OOF macro AUC (Pilkwang ≥0.5, FOLDS_V1)")
    ax.set_title("RSNA Knee on a storage budget — probe, not LB")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = out_dir / "cache_budget_curve.png"
    fig.savefig(png, dpi=140)
    plt.show()
    log(f"wrote {out_path} {png}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
