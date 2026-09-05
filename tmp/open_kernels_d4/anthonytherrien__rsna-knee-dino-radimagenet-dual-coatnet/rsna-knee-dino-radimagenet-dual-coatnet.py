# Enable postponed evaluation of annotations for the whole script
from __future__ import annotations
# Import the standard library modules used by every stage
import base64
import contextlib
import gc
import glob
import hashlib
import json
import os
import re
import shutil
import threading
import time
import traceback
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
# Keep every model library offline before those libraries are imported
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Import the third party modules used by every stage
import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydicom.pixel_data_handlers.util import apply_modality_lut
from torchvision.models import resnet50
# Hide library warnings and keep OpenCV single threaded inside worker processes
warnings.filterwarnings("ignore")
cv2.setNumThreads(1)
# Let cuDNN benchmark the fixed convolution shapes and allow TF32 matmuls
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
# Define the error raised when a weights bundle does not match this pipeline
class WeightsError(RuntimeError):
    pass
# Resolve the competition mount point
def resolve_competition_root():
    # List both mount points a competition dataset can appear under
    candidates = (
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "/kaggle/input/rsna-knee-abnormality-detection",
    )
    # Return the first candidate that exists on disk
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    # Fail loudly when neither mount is present
    raise RuntimeError("competition data not found under /kaggle/input")
# Resolve the competition root once for every stage
COMPETITION_ROOT = resolve_competition_root()
# Point at the competition data, the reproduction assets and the output folder
ROOT = Path(COMPETITION_ROOT)
ASSET = Path("/kaggle/input/rsna-knee-bend-dinov3-0917-repro-assets")
DINO = Path("/kaggle/input/models/metaresearch/dinov2/pytorch/small/1")
WORKING = Path("/kaggle/working")
# Record the start time shared by every log line
T0 = time.time()
# List every visible CUDA device
DEVS = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
# Fix the seed shared by fingerprints and jitter
SEED = 2026
# Name the twelve official findings in submission order
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
# Set the geometry of the transformer branch cache
CROP_MM = 130.0
CACHE_IMG = 336
IMG = CACHE_IMG
GROUP = 3
N_GROUP_MAX = 1
# Set how much host memory the cache may claim
CACHE_FRACTION = 0.45
CACHE_BUDGET_MAX_GB = 24.0
CACHE_BUDGET_GB = 12.0
TEST_SHARE = 0.3
# Set the thread counts used while probing, ordering and decoding DICOM data
HDR_THREADS = 16
PIX_THREADS = 12
ORDER_THREADS = 32
ORDER_BUDGET_S = 5400
# Set the augmentation strengths used by the optional jitter pass
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.1
# Set the laterality and slice band rules
LAT_MIN_OFFSET_MM = 20.0
SLICE_BAND = (0.2, 0.8)
LEGACY_LAT_OFFSET_MM = 5.0
# Describe the native pixel rules
RULES_NATIVE = {
    "order": "normal",
    "lat": "centre",
    "slot_fallback": False,
    "decode_fill": "nearest",
}
# Describe the legacy pixel rules
RULES_LEGACY = {
    "order": "dominant_axis",
    "lat": "corner_x",
    "slot_fallback": True,
    "decode_fill": "zero",
}
# Start from the native rules
RULES = dict(RULES_NATIVE)
# Set the evaluation batch size and the wall clock budget
EVAL_BATCH = 8
TIME_BUDGET = 8.0 * 3600
# Declare the recovered slot scheme
SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
# Declare the public slot scheme
SLOTS_PUBLIC = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]
# Choose the slot scheme from the environment
SLOT_SCHEME = os.environ.get("SLOT_SCHEME", "recovered")
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == "public" else SLOTS_RECOVERED
N_SLOT = len(SLOTS)
# Record how many feature parts each pooling mode produces
POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}
# Map each finding to the slots that usually show it
SLOT_PRIOR_TABLE = {
    "ACL": (0, 3, 5),
    "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4),
    "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5),
    "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5),
    "Effusion": (0, 2),
    "Synovitis": (0, 2),
    "Baker's": (0,),
    "Contusion": (0, 1, 2),
    "Fracture": (0, 1, 2, 4, 5),
}
# Set how strongly the slot prior biases attention
SLOT_PRIOR_STRENGTH = 0.55
# List the scan options that mark fat saturation
FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
# Compile the patterns used to read a series description
_SEP = re.compile("[_\\-.]")
_FATSAT_RX = re.compile(
    "\\bfs\\b|fatsat|fat sat|\\bstir\\b|\\bspair\\b|\\bspir\\b|\\bwe\\b|"
    "water excit|\\btirm\\b|\\bsting\\b|\\bfatsup\\b"
)
_T1_RX = re.compile("\\bt1\\b|\\bt1w\\b")
_T2_RX = re.compile("\\bt2\\b|\\bt2w\\b")
_PD_RX = re.compile("\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens")
# List the DICOM header tags read from every series
HDR_TAGS = [
    "SeriesDescription",
    "SequenceName",
    "ScanOptions",
    "ScanningSequence",
    "RepetitionTime",
    "EchoTime",
    "Laterality",
    "PixelSpacing",
    "Rows",
    "Columns",
    "RescaleSlope",
    "RescaleIntercept",
    "ImagePositionPatient",
    "ImageOrientationPatient",
]
# List the tags read while ordering slices
ORDER_TAGS = [(32, 50), (32, 55), (32, 19)]
# Collect the series that failed to decode
DECODE_FAILED = []
# Read the optional slice order cache path
ORDER_CACHE = os.environ.get("RSNA_ORDER_CACHE") or None
# Set the tolerance used when checking a stored fingerprint
FINGERPRINT_TOL = 0.002
# Set the window level test time augmentation policy
TTA_OVERLAP = True
TTA_POOL = "prob"
# Pool the public frontier windows per finding
PUBLIC_FRONTIER_TARGET_POOL = {
    "Fracture": "max",
    "Contusion": "max",
    "Medial Meniscus": "max",
    "Lateral Meniscus": "max",
    "ACL": "top2",
    "MCL": "top2",
    "Baker's": "max",
}
# Pool the primary windows per finding
TTA_TARGET_POOL = {**PUBLIC_FRONTIER_TARGET_POOL, "Synovitis": "original_mean"}
# Set the legacy soft pooling temperature per finding
LEGACY_FOLD_SOFTPOOL_BETA = {
    "ACL": 6.0,
    "MCL": 6.0,
    "Medial Meniscus": 8.0,
    "Lateral Meniscus": 8.0,
    "Baker's": 8.0,
    "Contusion": 8.0,
    "Fracture": 10.0,
}
# Set how much of the legacy soft pool enters the blend per finding
LEGACY_FOLD_SOFTPOOL_ALPHA = {
    "ACL": 0.2,
    "MCL": 0.2,
    "Medial Meniscus": 0.25,
    "Lateral Meniscus": 0.25,
    "Baker's": 0.2,
    "Contusion": 0.2,
    "Fracture": 0.15,
}
# Serialise model building and shared state across the device workers
BUILD_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
# Start from the largest cache the configuration allows
N_GROUP = N_GROUP_MAX
CACHE_SLICES = GROUP * N_GROUP
# Report the calibration flags produced by the RadImageNet stage
V18_CALIBRATOR_APPLIED = False
V18_CAL_GATE = tuple()
V18_TRANSFORMER_RAW = None
V18_TRANSFORMER_CAL = None
# Print a message with the elapsed run time
def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)
# Read how much host memory is still available
def available_gb():
    # Parse the kernel memory report
    try:
        with open("/proc/meminfo") as fh:
            info = {k.strip(): v for k, v in (l.split(":", 1) for l in fh if ":" in l)}
        return int(info["MemAvailable"].split()[0]) / 1024 ** 2
    # Fall back to the configured budget when the report is unreadable
    except Exception:
        return CACHE_BUDGET_GB / CACHE_FRACTION
# Decide how many slice groups the cache can hold
def plan_cache(n_study, n_test=0):
    # Measure the memory the cache may claim
    avail = available_gb()
    budget = min(avail * CACHE_FRACTION, CACHE_BUDGET_MAX_GB)
    # Estimate the number of studies the run has to hold
    n_total = n_study + max(n_test, int(TEST_SHARE * n_study))
    # Work out how many slices that budget affords
    per_slice = n_total * N_SLOT * IMG * IMG
    afford = int(budget * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(N_GROUP_MAX, afford // GROUP))
    # Report the sizing decision
    log(
        f"memory: {avail:.1f} GB available, {budget:.1f} GB to the cache; "
        f"sizing for {n_study} train + {n_total - n_study} test studies -> "
        f"{groups} group(s) of {GROUP} = {groups * GROUP} slices per slot"
        + (f" (wanted {N_GROUP_MAX})" if groups < N_GROUP_MAX else "")
    )
    return groups
# Size the transformer cache from the competition tables
def plan_cache_globals():
    # Rebind the cache sizing globals
    global N_GROUP, CACHE_SLICES
    # Count the studies on both sides of the split
    n_train = len(pd.read_csv(ROOT / "train.csv"))
    n_test = len(pd.read_csv(ROOT / "test.csv"))
    # Store the sizing decision
    N_GROUP = plan_cache(n_train, n_test)
    CACHE_SLICES = GROUP * N_GROUP
# Parse a pipe separated header value into a vector
def _hdr_vec(s, n):
    # Reject anything that is not a string
    if not isinstance(s, str):
        return None
    # Parse the numeric parts
    try:
        v = [float(x) for x in s.split("|")]
    except ValueError:
        return None
    # Keep the vector only when it is long enough
    return np.array(v) if len(v) >= n else None
# Infer the scanned side from the slice centre
def side_from_geometry(h):
    # Collect the centre x coordinate of every slice
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
        iop = _hdr_vec(getattr(r, "ImageOrientationPatient", None), 6)
        ps = _hdr_vec(getattr(r, "PixelSpacing", None), 2)
        rows, cols = (getattr(r, "Rows", None), getattr(r, "Columns", None))
        # Skip a series with an incomplete geometry
        if ipp is None or iop is None or ps is None or (not rows) or (not cols):
            continue
        # Project the image centre into patient space
        try:
            c = (
                ipp[:3]
                + iop[:3] * ps[1] * float(cols) / 2
                + iop[3:6] * ps[0] * float(rows) / 2
            )
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    # Turn the median offset into a side
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else "R" if m < 0 else "L"
    return out
# Infer the scanned side from the slice corner
def side_from_corner_x(h):
    # Work one study at a time
    out = {}
    for st, g in h.groupby("StudyInstanceUID"):
        # Collect the corner x coordinate of every slice
        xs = []
        for r in g.itertuples(index=False):
            ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        # Leave the side unresolved when no coordinate survived
        if not xs:
            out[st] = None
            continue
        # Turn the median offset into a side
        x = float(np.median(xs))
        out[st] = None if abs(x) < LEGACY_LAT_OFFSET_MM else "R" if x < 0 else "L"
    return out
# Resolve the scanned side of every study
def lat_of(h, tag=""):
    # Choose the geometric rule in force
    geo = side_from_corner_x(h) if RULES["lat"] == "corner_x" else side_from_geometry(h)
    # Prefer the header tag and fall back to geometry
    d, n_tag, n_geo, n_none, n_disagree = ({}, 0, 0, 0, 0)
    for st, g in h.groupby("StudyInstanceUID"):
        v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        # Read the legacy image laterality tag as well
        if RULES["lat"] == "corner_x" and "ImageLaterality" in g.columns:
            v += [str(x).strip().upper() for x in g["ImageLaterality"].dropna()]
        v = [x[0] for x in v if x and x[0] in ("L", "R")]
        side = v[0] if v else None
        # Count how often the tag and the geometry agree
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        # Fall back to the geometric side
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    # Report how the sides were resolved
    log(
        f"{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, "
        f"{n_none} unresolved; tag and geometry disagree on {n_disagree} "
        f"({n_disagree / max(n_tag, 1):.1%} of the tagged)"
    )
    return d
# Read the header of one series
def probe(item):
    # Unpack the work item
    split, study, series, path = item
    row = {
        "split": split,
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "dir": path,
    }
    # Read one representative slice header
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        # Stop early when the series holds no slice
        if not files:
            return row
        ds = pydicom.dcmread(
            os.path.join(path, files[len(files) // 2]),
            stop_before_pixels=True,
            force=True,
        )
        # Flatten every requested tag into the row
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t] = "|".join(str(x) for x in v)
            else:
                row[t] = str(v)
    # Record the failure and keep going
    except Exception as exc:
        row["err"] = str(exc)[:120]
    return row
# Read the headers of every series in one split
def walk(split):
    # Return an empty frame when the split is absent
    base = ROOT / split
    items = []
    if not base.is_dir():
        return pd.DataFrame(
            columns=[
                "split",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "dir",
                "files",
                "n_slices",
            ]
            + HDR_TAGS
        )
    # Collect every series directory
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    # Probe the headers in parallel
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)
# Derive the contrast of every series from its header
def annotate(df):
    # Normalise the free text description
    desc = df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna("")
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)
    # Read the scan options
    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    # Mark the fat saturated series
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs
    # Read the timing tags
    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = (
        desc.str.contains(_T1_RX),
        desc.str.contains(_T2_RX),
        desc.str.contains(_PD_RX),
    )
    # Decide the weighting from the text first and the timing second
    df["weight"] = np.where(
        t1 & ~t2 & ~pdw,
        "T1",
        np.where(
            t2 & ~pdw,
            "T2",
            np.where(
                pdw,
                "PD",
                np.where(
                    gre,
                    "GRE",
                    np.where(
                        tr < 800,
                        "T1",
                        np.where(te > 60, "T2", np.where(tr >= 800, "PD", "UNK")),
                    ),
                ),
            ),
        ),
    )
    # Mark the fluid sensitive series
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    # Read the in plane pixel spacing
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce",
    )
    return df
# Choose one series per slot for every study
def pick_slots(series_df, plane_map):
    # Attach the declared plane to every series
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    # Fill the slots one study at a time
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            # Select the candidates that match the slot
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            if fluid is not None:
                sel &= g["fluid"] == fluid
            cand = g[sel]
            # Relax the fluid requirement when the legacy fallback is on
            if len(cand) == 0 and RULES["slot_fallback"] and (fluid is False):
                cand = g[(g["plane"] == plane) & ~g["fatsat"]]
            # Keep the longest matching series
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out
# Build a sort key that keeps embedded numbers in order
def _natural_key(name):
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split("(\\d+)", str(name)))
# Order the slices of a series along its dominant axis
def _order_dominant_axis(rec):
    # Read the position and instance number of every slice
    files, d = (rec["files"], rec["dir"])
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(
                os.path.join(d, f),
                force=True,
                stop_before_pixels=True,
                specific_tags=["ImagePositionPatient", "InstanceNumber"],
            )
            raw = getattr(ds, "ImagePositionPatient", None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, "InstanceNumber", None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))
    # Require most slices to carry a position before trusting geometry
    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    # Sort along the axis that varies most
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(
            key=lambda r: (
                float(r[1][axis]) if r[1] is not None else spare,
                r[2] if r[2] is not None else float("inf"),
                r[3],
            )
        )
    # Fall back to the instance number
    elif sum(r[2] is not None for r in rows) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float("inf"), r[3]))
    # Fall back to the file name
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return ([r[0] for r in rows], True)
# Order the slices of a series
def order_slices(rec):
    # Use the legacy rule when it is in force
    if RULES["order"] == "dominant_axis":
        return _order_dominant_axis(rec)
    # Project every slice onto the slice normal
    files, d = (rec["files"], rec["dir"])
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(
                os.path.join(d, f),
                force=True,
                stop_before_pixels=True,
                specific_tags=ORDER_TAGS,
            )
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    # Keep the file order when any slice lacks a key
    if any(k is None for k, _ in keyed):
        return (files, False)
    return ([f for _, f in sorted(keyed, key=lambda t: t[0])], True)
# Decode the slices of one slot into a fixed size stack
def read_slot(rec, n_slice=None, out_size=None):
    # Fall back to the configured geometry
    n_slice = GROUP if n_slice is None else n_slice
    out_size = IMG if out_size is None else out_size
    files, d, px = (rec.get("ordered") or rec["files"], rec["dir"], rec["px"])
    n = len(files)
    # Give up on an empty series
    if n == 0:
        return None
    # Pick evenly spaced slices from the central band
    lo, hi = (int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1)))
    idx = (
        np.unique(np.linspace(lo, hi, n_slice).astype(int))
        if hi > lo
        else np.array([n // 2])
    )
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])
    # Decode every chosen slice
    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
        except Exception:
            a = None
        planes.append(a)
    # Record which slices decoded
    got = [k for k, p in enumerate(planes) if p is not None]
    # Fill a failed slice with zeros under the legacy rule
    if RULES["decode_fill"] == "zero":
        if not got:
            DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        planes = [
            np.zeros((out_size, out_size), np.float32) if p is None else p
            for p in planes
        ]
        got = list(range(len(planes)))
    # Drop the slot when nothing decoded
    if not got:
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        return None
    # Fill a failed slice with its nearest neighbour
    if len(got) < len(planes):
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        for k, p in enumerate(planes):
            if p is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]
    # Force every slice to the same shape
    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)
    # Crop a fixed millimetre box around the centre
    if px and np.isfinite(px) and (px > 0):
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = (h // 2, w // 2)
            half = want // 2
            vol = vol[:, max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]
    # Scale the intensities into the unit range
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-06), 0, 1)
    # Resize the stack and return it as bytes
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)
# Flip a right knee so every study faces the same way
def normalise_laterality(img, plane, lat):
    # Leave a left knee untouched
    if lat != "R":
        return img
    # Mirror in plane for the two transverse views
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    # Reverse the slice order for a sagittal view
    return torch.flip(img, dims=[0])
# Decode every slot of every study into one array
def build_cache(slot_map, plane_map, lat_map, tag):
    # Allocate the cache and the slot mask
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f"{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB")
    # List every filled slot as a job
    jobs = [
        (st, k, plane, slot_map[st][name])
        for st in studies
        for k, (name, plane, _, _) in enumerate(SLOTS)
        if name in slot_map[st]
    ]
    n_job = len(jobs)
    t_ord = time.time()
    n_slice_total = sum(len(j[3]["files"]) for j in jobs)
    log(f"{tag}: ordering {len(jobs)} slot-series ({n_slice_total} slice headers)")
    ok = done = 0
    CHUNK_O = 1024
    seen = {}
    # Reuse a previously computed slice order
    if ORDER_CACHE and Path(ORDER_CACHE).is_file():
        try:
            seen = json.loads(Path(ORDER_CACHE).read_text())
        except (OSError, ValueError):
            seen = {}
        hit = 0
        for _, _, _, rec in jobs:
            e = seen.get(rec["SeriesInstanceUID"])
            if e and len(e["files"]) == len(rec["files"]):
                rec["ordered"] = e["files"]
                ok += int(e["good"])
                hit += 1
        jobs = [j for j in jobs if "ordered" not in j[3]]
        log(f"{tag}: {hit} slot-series ordered from {ORDER_CACHE}, {len(jobs)} to read")
    # Order the remaining series within the time budget
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK_O):
            block = jobs[c0 : c0 + CHUNK_O]
            for (_, _, _, rec), (files, good) in zip(
                block, pool.map(lambda j: order_slices(j[3]), block)
            ):
                rec["ordered"] = files
                ok += int(good)
                done += 1
                if ORDER_CACHE:
                    seen[rec["SeriesInstanceUID"]] = {"files": files, "good": bool(good)}
            budget = min(
                ORDER_BUDGET_S, max(60.0, (TIME_BUDGET - (time.time() - T0)) * 0.35)
            )
            if time.time() - t_ord > budget:
                log(
                    f"{tag}: ordering budget spent at {done}/{len(jobs)}; "
                    "the rest keep file order"
                )
                break
    # Persist the slice order for the next run
    if ORDER_CACHE and done:
        _t = Path(ORDER_CACHE).with_suffix(".tmp")
        _t.write_text(json.dumps(seen))
        _t.replace(Path(ORDER_CACHE))
    log(
        f"{tag}: ordered {ok}/{n_job} by geometry ({n_job - ok} kept arbitrary) "
        f"in {time.time() - t_ord:.0f}s"
    )
    # Rebuild the job list now that every record carries its order
    jobs = [
        (st, k, plane, slot_map[st][name])
        for st in studies
        for k, (name, plane, _, _) in enumerate(SLOTS)
        if name in slot_map[st]
    ]
    log(f"{tag}: decoding {len(jobs)} slot-series")
    n_failed_before = len(DECODE_FAILED)
    CHUNK = 512
    done = 0
    # Decode every slot into the cache
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            block = jobs[c0 : c0 + CHUNK]
            for (st, k, plane, _), img in zip(
                block, pool.map(lambda j: read_slot(j[3], CACHE_SLICES, IMG), block)
            ):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(
                    img, plane, lat_map.get(st)
                ).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < CHUNK:
                log(f"  {tag} {done}/{len(jobs)}")
            if time.time() - T0 > TIME_BUDGET:
                log(f"  {tag}: time budget reached during decode")
                break
    # Report how many slots were filled
    n_failed = len(DECODE_FAILED) - n_failed_before
    log(
        f"{tag}: {int(mask.sum())}/{len(jobs)} slots filled"
        + (
            f"; {n_failed} series had a slice that would not decode"
            if n_failed
            else ""
        )
    )
    gc.collect()
    return (studies, cache, mask)
# Attend over the slots with one query per finding
class SlotHead(nn.Module):
    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        # Project the pooled backbone features
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        # Learn one embedding per slot and one query per finding
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        # Build the slot prior that biases attention towards the useful views
        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == len(SLOTS) and (n_out == len(TARGETS)):
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in TARGETS:
                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior
        if prior:
            self.register_buffer("slot_prior", p_)
    def forward(self, x, mask):
        # Add the slot embedding to every projected feature
        h = self.proj(x) + self.slot_emb
        # Score every slot against every finding query
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        # Bias the scores with the slot prior
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        # Mask the missing slots and normalise
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -10000.0).softmax(-1)
        # Read out one logit per finding
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
# Wrap a frozen transformer backbone with the slot head
class Model(nn.Module):
    def __init__(self, backbone, dim, pool="cls_mean", prior=False):
        super().__init__()
        # Keep the backbone and the pooling mode
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        # Store the ImageNet normalisation constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, imgs, mask, img_size=None):
        # Flatten the slots into the batch dimension
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        # Resize when the member was fitted at another resolution
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        # Normalise the batch
        x = (x - self.mean) / self.std
        # Encode every slot
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        # Pool the class token and the patch tokens
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        # Restore the slot dimension and read out the findings
        feat = torch.cat(parts, dim=1).reshape(B, S, -1)
        return self.head(feat, mask)
# Locate the attached DINOv2 weights
def find_dinov2(variant="small"):
    # Fail when the weights were not attached
    if not (DINO / "config.json").is_file():
        raise FileNotFoundError(DINO)
    return DINO
# Build one ensemble member
def build_model(unfreeze_last, source=None, variant="small", pool="cls_mean", prior=False):
    # Import the backbone loader lazily so the run stays offline
    from transformers import AutoModel
    # Resolve the backbone weights
    p = source if source is not None else find_dinov2(variant)
    if p is None:
        raise FileNotFoundError("DINOv2 weights not attached")
    bb = AutoModel.from_pretrained(str(p))
    # Freeze everything except the last blocks and the final norm
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last) :]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    # Report the trainable size of the backbone
    dim = bb.config.hidden_size
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    log(
        f"backbone: {n_layer} blocks, last {unfreeze_last} trainable "
        f"({trainable / 1000000.0:.1f}M params), feature dim {dim * POOL_PARTS[pool]}"
    )
    return Model(bb, dim, pool=pool, prior=prior)
# Compute the deterministic output of a member on fixed noise
def fingerprint(model, dev, img_size, n_slot=None, group=None, seed=None):
    # Fall back to the configured geometry
    n_slot = N_SLOT if n_slot is None else n_slot
    group = GROUP if group is None else group
    seed = SEED if seed is None else seed
    # Build the same random input every time
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(
        0, 256, (2, n_slot, group, img_size, img_size), generator=g, dtype=torch.uint8
    ).to(dev)
    mask = torch.ones(2, n_slot, device=dev)
    mask[1, -1] = 0.0
    # Run the member in evaluation mode and restore its previous mode
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(imgs, mask, img_size).float().cpu().numpy()
    if was_training:
        model.train()
    return out
# Check a member against the fingerprint stored with its weights
def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag=""):
    # Compare the shapes first
    got = fingerprint(model, dev, img_size)
    exp = np.asarray(expected, np.float32)
    if got.shape != exp.shape:
        raise WeightsError(
            f"{tag}fingerprint shape {got.shape} != stored {exp.shape}: "
            "the architecture is not the one these weights were fitted to"
        )
    # Compare the values
    d = float(np.abs(got - exp).max())
    if d > tol:
        raise WeightsError(
            f"{tag}fingerprint differs by {d:.4g} (tolerance {tol:g}). The weights "
            "load but do not compute what they computed when fitted - preprocessing, "
            "resolution or architecture has moved between the two runs."
        )
    log(f"{tag}fingerprint matches within {d:.2g}")
    return d
# List the window start positions used at test time
def window_starts(n_slice, group, overlap=None):
    # Fall back to the configured overlap policy
    overlap = TTA_OVERLAP if overlap is None else overlap
    # Slide the window one slice at a time when overlapping
    if overlap and n_slice >= group:
        return list(range(n_slice - group + 1))
    # Otherwise take disjoint windows
    return [g * group for g in range(max(n_slice // group, 1))]
# Pool the windows of a finding with its declared rule
def apply_target_window_pool(values, probs, logits, original_probs, mapping, target_idx):
    # Apply one rule per finding
    for target, mode in mapping.items():
        j = target_idx[target]
        if mode == "max":
            values[:, j] = probs[:, :, j].max(0).values
        elif mode == "mean":
            values[:, j] = probs[:, :, j].mean(0)
        elif mode == "logit_mean":
            values[:, j] = torch.sigmoid(logits[:, :, j].mean(0))
        elif mode == "original_mean":
            values[:, j] = original_probs[:, :, j].mean(0)
        elif mode in ("top2", "top3"):
            k = min(int(mode[3:]), probs.shape[0])
            values[:, j] = probs[:, :, j].topk(k, dim=0).values.mean(0)
        else:
            raise ValueError(f"unknown TTA pooling mode for {target}: {mode}")
    return values
# Pool the windows with the legacy soft maximum
def legacy_fold_soft_window_pool(original_probs, target_idx):
    # Start from the plain window mean
    values = original_probs.mean(0).clone()
    # Sharpen the findings that declare a temperature
    for target, beta in LEGACY_FOLD_SOFTPOOL_BETA.items():
        j = target_idx[target]
        x = original_probs[:, :, j]
        weight = torch.softmax(float(beta) * x, dim=0)
        values[:, j] = (weight * x).sum(0)
    return values
# Score one member over every window of every study
@torch.no_grad()
def predict_member(
    model,
    cache,
    mask,
    idx,
    dev,
    img_size,
    group=None,
    pool=None,
    starts=None,
    jitter=False,
    jitter_seed=SEED,
    return_public_frontier=False,
):
    # Fall back to the configured windowing
    group = GROUP if group is None else group
    pool = TTA_POOL if pool is None else pool
    starts = window_starts(cache.shape[2], group) if starts is None else list(starts)
    if not starts:
        raise ValueError("predict_member was given no windows to average over")
    # Reject a pooling table that names an unknown finding
    target_idx = {t: j for j, t in enumerate(TARGETS)}
    unknown = (set(TTA_TARGET_POOL) | set(PUBLIC_FRONTIER_TARGET_POOL)) - set(target_idx)
    if unknown:
        raise ValueError(f"unknown target(s) in TTA_TARGET_POOL: {unknown}")
    # Seed the jitter generator for this member
    jitter_gen = torch.Generator(device=dev)
    jitter_gen.manual_seed(int(jitter_seed) % (2 ** 63 - 1))
    model.eval()
    out, public_frontier_out, public_soft_out = ([], [], [])
    # Score the studies in batches
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b : b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        win_probs, win_logits, win_original_probs = ([], [], [])
        # Score every window position
        for st in starts:
            rows = torch.from_numpy(
                np.ascontiguousarray(cache[sel, :, st : st + group])
            ).to(dev)
            views = [rows] + ([augment(rows, generator=jitter_gen)] if jitter else [])
            view_probs, view_logits = ([], [])
            # Average the plain view with the jittered view
            for view in views:
                with torch.autocast("cuda", enabled=dev.type == "cuda"):
                    z = model(view, m, img_size).float()
                view_logits.append(z)
                view_probs.append(torch.sigmoid(z))
            win_logits.append(torch.stack(view_logits).mean(0))
            win_probs.append(torch.stack(view_probs).mean(0))
            win_original_probs.append(view_probs[0])
        # Pool the windows into one prediction
        probs = torch.stack(win_probs)
        logits = torch.stack(win_logits)
        original_probs = torch.stack(win_original_probs)
        v = torch.sigmoid(logits.mean(0)) if pool == "logit" else probs.mean(0)
        v = apply_target_window_pool(
            v, probs, logits, original_probs, TTA_TARGET_POOL, target_idx
        )
        out.append(v.cpu().numpy())
        # Produce the public frontier votes as well
        if return_public_frontier:
            public_v = apply_target_window_pool(
                original_probs.mean(0),
                original_probs,
                logits,
                original_probs,
                PUBLIC_FRONTIER_TARGET_POOL,
                target_idx,
            )
            public_frontier_out.append(public_v.cpu().numpy())
            public_soft = legacy_fold_soft_window_pool(original_probs, target_idx)
            public_soft_out.append(public_soft.cpu().numpy())
    # Concatenate the primary prediction
    primary = (
        np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)
    )
    if not return_public_frontier:
        return primary
    # Concatenate the two public branches
    public_frontier = (
        np.concatenate(public_frontier_out)
        if public_frontier_out
        else np.zeros((0, len(TARGETS)), np.float32)
    )
    public_soft = (
        np.concatenate(public_soft_out)
        if public_soft_out
        else np.zeros((0, len(TARGETS)), np.float32)
    )
    return (primary, public_frontier, public_soft)
# Load one member, verify it and score the test cache with it
def _run_member(path, m, dev, Cte, Mte, idx, starts, jitter):
    # Build the member under the shared build lock
    t0 = time.time()
    with BUILD_LOCK:
        if "state" in m:
            state, fp = (m["state"], None)
        else:
            ck = torch.load(Path(path) / m["file"], map_location="cpu", weights_only=False)
            state, fp = (ck["model"], ck.get("fingerprint"))
        model = build_model(
            int(m["config"]["unfreeze_last"]),
            variant=m["config"]["variant"],
            pool=m["config"].get("pool", "cls_mean"),
            prior=bool(m["config"].get("prior", False)),
        ).to(dev)
        model.load_state_dict(state)
        # Verify the member against its stored fingerprint
        if fp is not None:
            check_fingerprint(model, dev, IMG, fp, tag=f"{m['id']}: ")
        else:
            log(
                f"  {m['id']}: no stored fingerprint (legacy bundle) "
                "-- accepted at reduced weight"
            )
    t_ready = time.time()
    # Derive a stable jitter seed from the member identity
    jitter_seed = SEED + int(hashlib.sha256(str(m["id"]).encode()).hexdigest()[:8], 16)
    public_member = "state" not in m
    # Score the test cache
    predicted = predict_member(
        model,
        Cte,
        Mte,
        idx,
        dev,
        IMG,
        starts=starts,
        jitter=jitter,
        jitter_seed=jitter_seed,
        return_public_frontier=public_member,
    )
    # Split the returned branches
    if public_member:
        p, public_p, public_soft = predicted
    else:
        p, public_p, public_soft = (predicted, None, None)
    t_done = time.time()
    # Release the member before the next one is built
    del model, state
    gc.collect()
    if dev.type == "cuda":
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    passes = len(starts) * (2 if jitter else 1)
    return (p, public_p, public_soft, (t_ready - t0, (t_done - t_ready) / max(passes, 1)))
# Combine the banked members into one weighted rank mean
def _combine(per_member):
    # Index every study seen by any member
    all_ids = sorted({s for m in per_member for s in m["ids"]})
    pos = {s: i for i, s in enumerate(all_ids)}
    acc = np.zeros((len(all_ids), len(TARGETS)), np.float64)
    tot = np.zeros(len(TARGETS), np.float64)
    # Accumulate one rank matrix per member
    for m in per_member:
        target_weight = m.get("target_weight")
        w = np.asarray(
            target_weight
            if target_weight is not None
            else [float(m.get("weight", 1.0))] * len(TARGETS),
            dtype=np.float64,
        )
        if w.shape != (len(TARGETS),) or np.any(w < 0):
            raise ValueError(f"invalid target weights for {m.get('id')}: {w}")
        r = pd.DataFrame(m["pred"]).rank(pct=True).to_numpy()
        acc[[pos[s] for s in m["ids"]]] += r * w[None, :]
        tot += w
    # Refuse a finding that nobody voted on
    if np.any(tot <= 0):
        raise ValueError(f"at least one target has no ensemble vote: {tot}")
    return (all_ids, acc / tot[None, :])
# Combine the public members fold by fold
def combine_public_members_by_fold(per_member, pred_key="pred"):
    # Index every study
    all_ids = sorted({study for member in per_member for study in member["ids"]})
    position = {study: i for i, study in enumerate(all_ids)}
    # Group the members by fold
    groups = {}
    for i, member in enumerate(per_member):
        fold = member.get("fold")
        key = f"fold_{fold}" if fold is not None else f"member_{i}"
        groups.setdefault(key, []).append(member)
    fold_ranks, diagnostics = ([], [])
    # Rank the mean prediction of every fold
    for key, members_in_fold in sorted(groups.items()):
        matrices = []
        for member in members_in_fold:
            values = np.full((len(all_ids), len(TARGETS)), np.nan, np.float64)
            values[[position[study] for study in member["ids"]]] = np.asarray(
                member[pred_key], np.float64
            )
            if np.isnan(values).any():
                raise WeightsError(f"{member.get('id')}: incomplete {pred_key} coverage")
            matrices.append(values)
        raw_fold_mean = np.mean(matrices, axis=0)
        fold_ranks.append(
            pd.DataFrame(raw_fold_mean).rank(method="average", pct=True).to_numpy(np.float64)
        )
        diagnostics.append({"ensemble_group": key, "members": len(members_in_fold)})
    # Require the five folds the legacy branch was fitted with
    if len(fold_ranks) != 5:
        raise WeightsError(f"legacy branch requires five folds, found {len(fold_ranks)}")
    return (all_ids, np.mean(fold_ranks, axis=0), pd.DataFrame(diagnostics))
# Blend the hard and soft legacy pools per finding
def blend_legacy_frontier_and_soft(frontier_rank, soft_rank):
    # Start from the hard pooled ranks
    output = np.asarray(frontier_rank, np.float64).copy()
    # Mix in the soft pool where a weight is declared
    for j, target in enumerate(TARGETS):
        alpha = float(LEGACY_FOLD_SOFTPOOL_ALPHA.get(target, 0.0))
        if alpha:
            output[:, j] = (1.0 - alpha) * frontier_rank[:, j] + alpha * soft_rank[:, j]
    return output
# Adopt the pixel configuration recorded with a group of members
def adopt_config_globals(cfg):
    # Rebind every global the cache builder reads
    global IMG, CACHE_IMG, GROUP, CACHE_SLICES, N_GROUP, CROP_MM, SLICE_BAND, RULES
    CACHE_IMG = IMG = int(cfg["img"])
    GROUP = int(cfg["group"])
    CACHE_SLICES = int(cfg["slices"])
    N_GROUP = max(CACHE_SLICES // GROUP, 1)
    CROP_MM = float(cfg["crop_mm"])
    SLICE_BAND = tuple(float(x) for x in cfg["band"])
    # Reject any pixel rule this pipeline cannot reproduce
    rules = cfg.get("rules") or RULES_NATIVE
    unknown = {
        k: v
        for k, v in rules.items()
        if k not in RULES_NATIVE or v not in (RULES_NATIVE[k], RULES_LEGACY[k])
    }
    if unknown:
        raise WeightsError(
            f"the members record pixel rules this pipeline cannot reproduce: {unknown}"
        )
    RULES = {**RULES_NATIVE, **rules}
    # Reject a slot scheme the members were not fitted on
    if [s[0] for s in SLOTS] != list(cfg["slots"]):
        raise WeightsError(
            f"the members were fitted on slots {cfg['slots']} and this pipeline "
            f"defines {[s[0] for s in SLOTS]}; a weight would be read against the wrong slot"
        )
# Apply a random affine and intensity jitter to a batch of windows
def augment(imgs, generator=None):
    # Flatten the leading dimensions
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float()
    n, dev = (x.shape[0], x.device)
    # Draw one transform per window
    rot = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * (AUG_ROT_DEG * np.pi / 180)
    sc = 1.0 + torch.rand(n, device=dev, generator=generator) * AUG_SCALE
    tx = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * AUG_SHIFT
    # Build the affine matrices
    cos, sin = (torch.cos(rot) / sc, torch.sin(rot) / sc)
    theta = torch.zeros(n, 2, 3, device=dev, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = (cos, -sin, tx)
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = (sin, cos, ty)
    # Resample the windows
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    # Scale the intensities
    scale = 1.0 + (torch.rand(n, 1, 1, 1, device=dev, generator=generator) - 0.5) * 2 * AUG_INTENSITY
    x = (x * scale).clamp(0, 255)
    return x.reshape(*lead, *x.shape[-3:]).to(imgs.dtype)
# Write a rank submission aligned to the test table
def write_submission(pred, studies, test_df, path):
    # Rank every finding into the unit range
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", studies)
    # Align the rows to the test table and fill any gap
    sub = test_df[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv(path, index=False)
    return sub
# List the extra members held outside the weights package
def legacy_group_members():
    return {}
# Score every member of a weights package and write the submission
def infer_from_package(path, dev=None):
    # Read the package manifest
    man = json.loads((Path(path) / "manifest.json").read_text())
    members = man["members"]
    log(f"weights package: {len(members)} member(s) from {path}; {len(DEVS)} device(s)")
    # Read the test tables and the declared planes
    test_df = pd.read_csv(ROOT / "test.csv")
    test_series = pd.read_csv(ROOT / "test_series.csv")
    plane_map = dict(zip(test_series["SeriesInstanceUID"], test_series["Anatomical_Plane"]))
    hte = annotate(walk("test_series"))
    log(f"test header pass: {len(hte)} series")
    # Group the members by the pixel configuration they need
    groups = {}
    for m in members:
        groups.setdefault(m["pixel_group"], []).append(m)
    groups.update(legacy_group_members())
    per_member, public_frontier_members = ([], [])
    est = {"fixed": None, "win": None}
    # Bank one member and rewrite the running submission
    def bank(m, ids, pred, starts, jitter, public_pred=None, public_soft=None):
        # Drop a member that predicts a constant
        if float(np.std(pred)) < 1e-09:
            log(f"  {m['id']}: degenerate predictions; not banked")
            return
        with STATE_LOCK:
            per_member.append(
                {
                    "id": m["id"],
                    "fold": m.get("fold"),
                    "ids": ids,
                    "pred": pred,
                    "weight": m.get("weight", 1.0),
                    "target_weight": m.get("target_weight"),
                    "holdout": m.get("holdout"),
                }
            )
            # Bank the public frontier vote only when every window ran
            if public_pred is not None and len(starts) == len(starts_full):
                if float(np.std(public_pred)) < 1e-09:
                    raise WeightsError(f"{m['id']}: degenerate public-frontier prediction")
                public_frontier_members.append(
                    {
                        "id": m["id"],
                        "fold": m.get("fold"),
                        "ids": ids,
                        "pred": public_pred,
                        "soft_pred": public_soft,
                    }
                )
            elif public_pred is not None:
                log(
                    f"  {m['id']}: public-frontier vote omitted because only "
                    f"{len(starts)} / {len(starts_full)} windows completed"
                )
            # Rewrite the submission after every banked member
            all_ids, acc = _combine(per_member)
            write_submission(acc, all_ids, test_df, WORKING / "submission.csv")
            log(
                f"  banked {m['id']} fold {m.get('fold', '?')} "
                f"({len(starts)} window(s){(', jitter' if jitter else '')}); "
                f"submission.csv = weighted rank mean of {len(per_member)} member(s)"
            )
    # Work through the pixel groups one at a time
    for gi, (key, gm) in enumerate(groups.items(), 1):
        cfg = json.loads(key)
        adopt_config_globals(cfg)
        log(
            f"decode group {gi}/{len(groups)}: {cfg['img']}px x {cfg['slices']} slices, "
            f"crop {cfg['crop_mm']} mm -> {len(gm)} member(s)"
        )
        # Decode the test cache this group needs
        st_te, Cte, Mte = build_cache(
            pick_slots(hte, plane_map), plane_map, lat_of(hte, "test "), f"test g{gi}"
        )
        idx = np.arange(len(st_te))
        starts_full = window_starts(Cte.shape[2], GROUP)
        pending = sorted(gm, key=lambda m: -(m.get("holdout") or 0))
        left_after = sum(len(g) for j, (_, g) in enumerate(groups.items(), 1) if j > gi)
        # Hand out the next member that still fits the time budget
        def pop_next():
            with STATE_LOCK:
                # Stop when the queue is empty
                if not pending:
                    return (None, None, False)
                # Work out how much time this member may take
                left = TIME_BUDGET - (time.time() - T0)
                remaining = len(pending) + left_after
                slots_left = -(-remaining // len(DEVS))
                starts, jit = (starts_full, False)
                if est["fixed"] is not None and est["win"] is not None:
                    afford = max(left * 0.9, 0.0)
                    room = afford / max(slots_left, 1)
                    # Surrender the rest when not even one member fits
                    if est["fixed"] + est["win"] > room:
                        log(
                            f"  {left / 60:.0f} min left: surrendering "
                            f"{len(pending)} member(s); not one more fits"
                        )
                        pending.clear()
                        return (None, None, False)
                    # Add jitter only when it comfortably fits
                    jit = est["fixed"] + 2 * len(starts_full) * est["win"] <= room * 0.6
                    per_win = est["win"] * (2 if jit else 1)
                    n_win = int((room - est["fixed"]) / per_win) if per_win > 0 else len(starts_full)
                    n_win = max(1, min(len(starts_full), n_win))
                    # Keep the central windows when the budget is short
                    if n_win < len(starts_full):
                        mid = (len(starts_full) - n_win) // 2
                        starts = starts_full[mid : mid + n_win]
                return (pending.pop(0), starts, jit)
        # Run members on one device until the queue is empty
        def worker(dev):
            others = [d for d in DEVS if d is not dev]
            while True:
                m, starts, jit = pop_next()
                if m is None:
                    return
                # Retry a failed member once on a peer device
                for attempt, d in enumerate([dev] + others[:1]):
                    try:
                        p, public_p, public_soft, (fs, ws) = _run_member(
                            path, m, d, Cte, Mte, idx, starts, jit
                        )
                        with STATE_LOCK:
                            est["fixed"], est["win"] = (fs, ws)
                        bank(m, st_te, p, starts, jit, public_p, public_soft)
                        break
                    except Exception as exc:
                        log(
                            f"  MEMBER {m['id']} failed on {d} "
                            f"({type(exc).__name__}: {exc}); "
                            + (
                                "retrying on peer device"
                                if attempt == 0 and others
                                else "dropped -- costs one vote, not the run"
                            )
                        )
                        if d.type == "cuda":
                            with torch.cuda.device(d):
                                torch.cuda.empty_cache()
        # Run one worker per device
        threads = [threading.Thread(target=worker, args=(d,)) for d in DEVS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Release the cache before the next group
        del Cte, Mte
        gc.collect()
    # Refuse to finish when no member survived
    if not per_member:
        raise WeightsError("no member produced predictions; submission stays at 0.5")
    # Write the primary submission
    all_ids, acc = _combine(per_member)
    sub = write_submission(acc, all_ids, test_df, WORKING / "submission.csv")
    log(
        f"final submission.csv = weighted rank mean of {len(per_member)} member(s); "
        f"{sub.shape}; nulls {int(sub[TARGETS].isna().sum().sum())}"
    )
    # Write the public branches when every member produced them
    if len(public_frontier_members) == len(members):
        frontier_ids, frontier_acc = _combine(public_frontier_members)
        frontier_sub = write_submission(
            frontier_acc, frontier_ids, test_df, WORKING / "submission_public_0899.csv"
        )
        log(
            "submission_public_0899.csv = exact no-jitter public-frontier rank mean of "
            f"{len(public_frontier_members)} member(s); {frontier_sub.shape}; "
            f"nulls {int(frontier_sub[TARGETS].isna().sum().sum())}"
        )
        # Blend the hard and soft legacy pools fold by fold
        fold_ids, fold_frontier, fold_diagnostics = combine_public_members_by_fold(
            public_frontier_members, "pred"
        )
        soft_ids, fold_soft, _ = combine_public_members_by_fold(
            public_frontier_members, "soft_pred"
        )
        if fold_ids != soft_ids:
            raise WeightsError("legacy hard/soft study order mismatch")
        legacy_prediction = blend_legacy_frontier_and_soft(fold_frontier, fold_soft)
        legacy_sub = write_submission(
            legacy_prediction, fold_ids, test_df, WORKING / "submission_legacy_fold_blend.csv"
        )
        fold_diagnostics.to_csv(WORKING / "legacy_fold_diagnostics.csv", index=False)
        log(f"legacy DINO aggregation written from five folds; {legacy_sub.shape}")
    else:
        log(
            f"public-frontier fallback not emitted: {len(public_frontier_members)} / "
            f"{len(members)} required public members completed"
        )
    return sub
# Run the DINOv2 transformer stage
def run_dinov2_stage():
    # Size the cache from the competition tables
    plan_cache_globals()
    # Score every member of the attached weights package
    path = ASSET / "rsna-knee-weights"
    infer_from_package(path, DEVS[0])
    # Promote the public frontier branch to the primary submission
    public = WORKING / "submission_public_0899.csv"
    if not public.is_file():
        raise RuntimeError("public DINOv2 frontier was not produced")
    public.replace(WORKING / "submission.csv")
    # Remove the legacy diagnostics this checkpoint does not submit
    for name in ("submission_legacy_fold_blend.csv", "legacy_fold_diagnostics.csv"):
        candidate = WORKING / name
        if candidate.is_file():
            candidate.unlink()
# Set the geometry of the fold checkpoint branch
FOLD_CROP_MM = 130.0
FOLD_SIZE = 336
FOLD_SLICE_BAND = (0.12, 0.88)
FOLD_N_SLICE = 16
FOLD_INTENSITY = "slice"
# Declare the plane and fat suppression of every fold slot
FOLD_SLOTS = [
    ("Sagittal", 1),
    ("Sagittal", 0),
    ("Coronal", 1),
    ("Coronal", 0),
    ("Axial", 1),
    ("Axial", 0),
]
FOLD_N_SLOT = len(FOLD_SLOTS)
# Reuse the official finding order
FOLD_LABELS = list(TARGETS)
# Point at the fold checkpoints and the series directory
FOLD_CKPT = ASSET / "knee-mri-fold-weights"
FOLD_DEV = "cuda" if torch.cuda.is_available() else "cpu"
FOLD_SERIES_ROOT = (
    ROOT / "test_series" if (ROOT / "test_series").exists() else ROOT / "train_series"
)
# Declare the slot embedding table size and its padding index
FOLD_N_SLOT_TYPES = 6
FOLD_MASK_IDX = 0
# Store the ImageNet normalisation constants used by the depth stem
FOLD_IMAGENET_MEAN = (0.485, 0.456, 0.406)
FOLD_IMAGENET_STD = (0.229, 0.224, 0.225)
# Declare how many planes and contrasts the mixer conditions on
FOLD_N_PLANE = 3
FOLD_N_CONTRAST = 2
# Set the autocast preference and the batching of the fold branch
FOLD_AMP_PREF = "bf16"
FOLD_AMP_DTYPE = torch.float32
FOLD_AMP_ENABLED = False
FOLD_WORKERS = max(1, min(4, os.cpu_count() or 4))
FOLD_CHUNK = 48
FOLD_MICRO = 8
# Set how much of the fold branch enters the submission
FOLD_BLEND_WEIGHT = 0.45
# Hold the loaded fold checkpoints and their shared configuration
FOLD_MODELS = []
FOLD_CFG = {}
# Order the slices of a series by instance number
def fold_ordered_files(sdir, cap=64):
    # Read the instance number of every slice
    keyed = []
    for f in sdir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            keyed.append((int(ds.InstanceNumber), str(f)))
        except Exception:
            continue
        # Stop once enough slices are known
        if len(keyed) >= cap * 4:
            break
    return [f for _, f in sorted(keyed)]
# Read the patient x coordinate of one slice
def fold_series_side(path):
    # Fall back to the midline when the tag is missing
    try:
        return float(pydicom.dcmread(path, stop_before_pixels=True).ImagePositionPatient[0])
    except Exception:
        return 0.0
# Crop a fixed millimetre box out of one slice
def fold_read_crop(path):
    # Decode the pixel data
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None
    # Read the pixel spacing and fall back to the frame size
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = FOLD_CROP_MM / max(arr.shape)
    # Cut the box out of the centre
    half = int(round(FOLD_CROP_MM / ps / 2))
    cy, cx = (arr.shape[0] // 2, arr.shape[1] // 2)
    y0, y1 = (max(0, cy - half), min(arr.shape[0], cy + half))
    x0, x1 = (max(0, cx - half), min(arr.shape[1], cx + half))
    crop = arr[y0:y1, x0:x1]
    return None if crop.size == 0 else crop
# Window one crop into the unit range and resize it
def fold_window(crop, lo, hi, flip):
    # Clip the intensities into the window
    c = np.clip((crop - lo) / max(hi - lo, 1e-06), 0, 1)
    # Resize to the model resolution
    img = cv2.resize(c, (FOLD_SIZE, FOLD_SIZE), interpolation=cv2.INTER_AREA)
    # Mirror a right knee
    return img[:, ::-1].copy() if flip else img
# Window one slice with its own percentiles
def fold_render(path, flip):
    # Crop the slice first
    crop = fold_read_crop(path)
    if crop is None:
        return None
    # Take the window from a subsampled copy
    lo, hi = np.percentile(crop[::4, ::4], [1, 99])
    return fold_window(crop, lo, hi, flip)
# Build the slot stack of one study
def fold_build_study(args):
    # Unpack the work item
    idx, study, recs = args
    out = np.zeros((FOLD_N_SLOT, FOLD_N_SLICE, FOLD_SIZE, FOLD_SIZE), np.uint8)
    mask = np.zeros(FOLD_N_SLOT, np.uint8)
    rows = pd.DataFrame(recs)
    # Fill every slot the study can serve
    if len(rows):
        for s_i, (plane, fs) in enumerate(FOLD_SLOTS):
            # Select the first series that matches the slot
            sub = rows[(rows.Anatomical_Plane == plane) & (rows.Fat_Suppression == fs)]
            if sub.empty:
                continue
            files = fold_ordered_files(FOLD_SERIES_ROOT / study / sub.iloc[0].SeriesInstanceUID)
            if not files:
                continue
            # Mirror a right knee on the transverse views
            flip = plane != "Sagittal" and fold_series_side(files[0]) < 0
            # Pick evenly spaced slices from the band
            lo, hi = FOLD_SLICE_BAND
            i0 = int(round(lo * (len(files) - 1)))
            i1 = int(round(hi * (len(files) - 1)))
            avail = list(range(i0, i1 + 1))
            if len(avail) >= FOLD_N_SLICE:
                picks = [
                    avail[int(round(t))]
                    for t in np.linspace(0, len(avail) - 1, FOLD_N_SLICE)
                ]
                off = 0
            else:
                picks, off = (avail, (FOLD_N_SLICE - len(avail)) // 2)
            # Window the whole series against shared percentiles
            if FOLD_INTENSITY == "series":
                crops = [fold_read_crop(files[p]) for p in picks]
                got = [x for x in crops if x is not None]
                if got:
                    samp = np.concatenate([x[::4, ::4].ravel() for x in got])
                    lo_, hi_ = np.percentile(samp, [1, 99])
                    for c, x in enumerate(crops):
                        if x is None:
                            x = fold_read_crop(files[min(len(files) - 1, picks[c] + 1)])
                        if x is not None:
                            out[s_i, off + c] = (fold_window(x, lo_, hi_, flip) * 255).astype(np.uint8)
            # Window every slice against its own percentiles
            else:
                for c, p in enumerate(picks):
                    img = fold_render(files[p], flip)
                    if img is None:
                        img = fold_render(files[min(len(files) - 1, p + 1)], flip)
                    if img is not None:
                        out[s_i, off + c] = (img * 255).astype(np.uint8)
            mask[s_i] = len(picks)
    return (idx, out, mask)
# Normalise attention scores inside every study segment
def segment_softmax(scores, sidx, B):
    # Take the per study maximum
    T, K = scores.shape
    idx = sidx.unsqueeze(1).expand(-1, K)
    m = torch.full((B, K), float("-inf"), device=scores.device, dtype=scores.dtype)
    m = m.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    # Exponentiate and normalise inside each segment
    e = (scores - m[sidx]).exp()
    s = torch.zeros(B, K, device=scores.device, dtype=scores.dtype).index_add_(0, sidx, e)
    return e / s[sidx].clamp(min=1e-06)
# Pool the slot features with a mean and a maximum
class MeanMaxPool(nn.Module):
    def forward(self, f, sidx, B, slot=None, return_attn=False):
        # Average the features of every study
        D = f.shape[1]
        cnt = torch.zeros(B, device=f.device, dtype=f.dtype).index_add_(
            0, sidx, torch.ones(f.shape[0], device=f.device, dtype=f.dtype)
        )
        mean = torch.zeros(B, D, device=f.device, dtype=f.dtype).index_add_(0, sidx, f)
        mean = mean / cnt.clamp(min=1).unsqueeze(1)
        # Take the maximum of every study
        mx = torch.full((B, D), -10000.0, device=f.device, dtype=f.dtype)
        mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), f, reduce="amax", include_self=True)
        return (torch.cat([mean, mx], 1), None)
# Pool the slot features with one attention query per finding
class LabelAttentionPool(nn.Module):
    def __init__(self, d, n_labels=12, n_heads=4, slot_bias=True):
        super().__init__()
        # Learn one query, key and value projection per finding
        self.d, self.k, self.h = (d, n_labels, n_heads)
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.key, self.val = (nn.Linear(d, d), nn.Linear(d, d))
        # Bias the attention per slot type
        self.slot_bias = (
            nn.Parameter(torch.zeros(n_labels, FOLD_N_SLOT_TYPES + 1)) if slot_bias else None
        )
    def forward(self, f, sidx, B, slot=None, return_attn=False):
        # Score every feature against every finding query
        scores = self.key(f) @ self.q.t() / self.d ** 0.5
        if self.slot_bias is not None and slot is not None:
            scores = scores + self.slot_bias.t()[slot]
        # Normalise inside every study and pool the values
        a = segment_softmax(scores, sidx, B)
        out = torch.zeros(B, self.k, self.d, device=f.device, dtype=f.dtype)
        out = out.index_add_(0, sidx, a.unsqueeze(-1) * self.val(f).unsqueeze(1))
        return (out, a)
# Pool the patch tokens with cross attention
class TokenXAttnPool(nn.Module):
    def __init__(self, d, n_labels=12, n_heads=6, dropout=0.2):
        super().__init__()
        # Learn one query per finding and one embedding per slot
        self.d, self.k = (d, n_labels)
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.slot_emb = nn.Embedding(FOLD_N_SLOT_TYPES + 1, d, padding_idx=0)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
    def forward(self, tok, sidx, B, slot=None, return_attn=False):
        # Pad the ragged slot dimension into a dense batch
        T, N, D = tok.shape
        cnt = torch.bincount(sidx, minlength=B)
        S = int(cnt.max().item())
        starts = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(T, device=tok.device) - starts[sidx]
        kv = tok + self.slot_emb(slot).unsqueeze(1)
        pad = tok.new_zeros(B, S, N, D)
        pad[sidx, pos] = kv
        keep = torch.zeros(B, S, dtype=torch.bool, device=tok.device)
        keep[sidx, pos] = True
        kpm = ~keep.repeat_interleave(N, dim=1)
        # Attend from the finding queries to every token
        pad = self.kv_norm(pad.reshape(B, S * N, D))
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(
            q, pad, pad, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True
        )
        # Add the pooled class token as a baseline
        cls = tok[:, 0]
        mean = torch.zeros(B, D, device=tok.device, dtype=tok.dtype).index_add_(
            0, sidx, cls
        ) / cnt.clamp(min=1).unsqueeze(1)
        mx = torch.full((B, D), -10000.0, device=tok.device, dtype=tok.dtype)
        mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), cls, reduce="amax", include_self=True)
        base = torch.cat([mean, mx], 1).unsqueeze(1).expand(-1, self.k, -1)
        return (torch.cat([att, base], -1), w)
# Insert a slot token into a timm vision transformer
class ViTSlotToken(nn.Module):
    def __init__(self, vit, n_cat, dim=None):
        super().__init__()
        # Learn one embedding per slot type
        self.vit = vit
        d = dim or vit.embed_dim
        self.tok = nn.Embedding(n_cat + 1, d, padding_idx=FOLD_MASK_IDX)
        self.num_features = vit.num_features
        # Tell the backbone that one more prefix token is present
        self._orig_prefix = getattr(vit, "num_prefix_tokens", 1)
        vit.num_prefix_tokens = self._orig_prefix + 1
        for blk in vit.blocks:
            a = getattr(blk, "attn", None)
            if a is not None and hasattr(a, "num_prefix_tokens"):
                a.num_prefix_tokens = a.num_prefix_tokens + 1
    @staticmethod
    def _maybe(mod, x):
        return x if mod is None else mod(x)
    def forward_features(self, x, cat):
        # Embed the patches and add the position encoding
        v = self.vit
        x = v.patch_embed(x)
        pos = v._pos_embed(x)
        rope = None
        if isinstance(pos, tuple):
            x, rope = pos
        else:
            x = pos
        # Apply the optional patch dropout and pre norm
        x = self._maybe(getattr(v, "patch_drop", None), x)
        x = self._maybe(getattr(v, "norm_pre", None), x)
        # Splice the slot token in behind the class token
        npt = self._orig_prefix
        tok = self.tok(cat).unsqueeze(1)
        x = torch.cat([x[:, :npt], tok, x[:, npt:]], dim=1)
        # Run the blocks with or without rotary embeddings
        if rope is not None:
            if getattr(v, "rope_mixed", False):
                for i, blk in enumerate(v.blocks):
                    x = blk(x, rope=rope[i])
            else:
                for blk in v.blocks:
                    x = blk(x, rope=rope)
        else:
            x = v.blocks(x)
        return v.norm(x)
    def forward_head(self, x, pre_logits=True):
        return self.vit.forward_head(x, pre_logits=pre_logits)
# Mix the slices of one slot with a gated depthwise block
class _GatedDepthBlock(nn.Module):
    def __init__(self, n_slice, dropout=0.0, ls_init=0.1):
        super().__init__()
        # Build the gated projection
        self.norm = nn.GroupNorm(1, n_slice)
        self.v = nn.Conv2d(n_slice, n_slice, 1)
        self.g = nn.Conv2d(n_slice, n_slice, 1)
        self.out = nn.Conv2d(n_slice, n_slice, 1)
        # Scale the residual with a learned gain
        self.gamma = nn.Parameter(torch.full((n_slice, 1, 1), ls_init))
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()
    def forward(self, x):
        # Add the gated branch to the input
        z = self.norm(x)
        return x + self.gamma * self.drop(self.out(self.v(z) * F.silu(self.g(z))))
# Compress a slice stack into three channels
class DepthCompress(nn.Module):
    def __init__(self, n_slice=16, out_ch=3, depth=1, dropout=0.0, ls_init=0.1, imagenet=True, proj_noise=0.25):
        super().__init__()
        # Stack the gated blocks and the output projection
        self.imagenet = imagenet
        self.blocks = nn.ModuleList(
            [_GatedDepthBlock(n_slice, dropout, ls_init) for _ in range(depth)]
        )
        self.proj = nn.Conv2d(n_slice, out_ch, 1, bias=True)
        # Store the ImageNet statistics
        if imagenet:
            self.register_buffer("mu", torch.tensor(FOLD_IMAGENET_MEAN).view(1, -1, 1, 1))
            self.register_buffer("sd", torch.tensor(FOLD_IMAGENET_STD).view(1, -1, 1, 1))
    def forward(self, x):
        # Remember which pixels carry data
        keep = (x.amax(dim=1, keepdim=True) > 0).to(x.dtype)
        # Mix the slices and project to three channels
        z = x
        for b in self.blocks:
            z = b(z)
        z = self.proj(z)
        # Normalise and blank the padding
        if self.imagenet:
            z = (z - self.mu.to(z.dtype)) / self.sd.to(z.dtype)
        return z * keep
# Read the plane index of a slot
def fold_plane_of(slot):
    return torch.clamp(slot - 1, 0, 5) // 2
# Read the contrast index of a slot
def fold_contrast_of(slot):
    return torch.clamp(slot - 1, 0, 5) % 2
# Blur along the slice axis with a slot conditioned kernel
class SlotDepthMixer(nn.Module):
    def __init__(self, n_slice=16, ksize=5, alpha_max=0.25):
        super().__init__()
        # Store the kernel geometry
        self.n_slice, self.ksize, self.r = (n_slice, ksize, ksize // 2)
        self.alpha_max = alpha_max
        # Start from a binomial kernel in log space
        b = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        self.register_buffer("base", b.log()[self.r :])
        # Learn a shared and a conditioned kernel offset
        n_u = self.r + 1
        self.shared = nn.Parameter(torch.zeros(n_u))
        self.plane_k = nn.Parameter(torch.zeros(FOLD_N_PLANE, n_u))
        self.contrast_k = nn.Parameter(torch.zeros(FOLD_N_CONTRAST, n_u))
        # Learn how much mixing each slot receives
        self.g0 = nn.Parameter(torch.zeros(()))
        self.gate_p = nn.Parameter(torch.zeros(FOLD_N_PLANE))
        self.gate_c = nn.Parameter(torch.zeros(FOLD_N_CONTRAST))
        # Precompute the slice offsets
        idx = torch.arange(n_slice)
        self.register_buffer("off", idx[None, :] - idx[:, None])
    def kernel(self, slot):
        # Build a symmetric kernel from the conditioned half
        p, c = (fold_plane_of(slot), fold_contrast_of(slot))
        half = self.base + self.shared + self.plane_k[p] + self.contrast_k[c]
        full = torch.cat([half.flip(-1)[..., : self.r], half], dim=-1)
        return F.softmax(full, dim=-1)
    def alpha(self, slot):
        # Bound the mixing strength of this slot
        p, c = (fold_plane_of(slot), fold_contrast_of(slot))
        return self.alpha_max * torch.tanh(self.g0 + self.gate_p[p] + self.gate_c[c])
    def forward(self, x, slot, vmask):
        # Require the padding mask
        T, S, H, W = x.shape
        if vmask is None:
            raise ValueError("stem=mixer requires the padding mask")
        # Expand the kernel into a slice mixing matrix
        k = self.kernel(slot)
        v = vmask.to(k.dtype)
        d = self.off + self.r
        inb = (d >= 0) & (d < self.ksize)
        kk = k[:, d.clamp(0, self.ksize - 1)] * inb
        M = kk * v[:, None, :]
        # Renormalise the matrix over the present slices only
        den = M.sum(-1, keepdim=True)
        eye = torch.eye(S, device=x.device, dtype=M.dtype).expand(T, S, S)
        ok = (den > 1e-06) & v[:, :, None].bool()
        M = torch.where(ok, M / den.clamp(min=1e-06), eye)
        # Interpolate between the identity and the mixing matrix
        a = self.alpha(slot)[:, None, None]
        Aop = ((1.0 - a) * eye + a * M).to(x.dtype)
        # Apply the operator in the memory layout the tensor already has
        if x.is_contiguous(memory_format=torch.channels_last) and (not x.is_contiguous()):
            y = torch.bmm(x.permute(0, 2, 3, 1).reshape(T, H * W, S), Aop.transpose(1, 2))
            return y.reshape(T, H, W, S).permute(0, 3, 1, 2)
        return torch.bmm(Aop, x.reshape(T, S, H * W)).reshape(T, S, H, W)
# Pool a feature matrix into a mean and a maximum per study
def _seg_mean_max(v, sidx, B):
    # Average the features of every study
    D = v.shape[1]
    cnt = torch.zeros(B, device=v.device, dtype=v.dtype).index_add_(
        0, sidx, torch.ones(v.shape[0], device=v.device, dtype=v.dtype)
    )
    mean = torch.zeros(B, D, device=v.device, dtype=v.dtype).index_add_(0, sidx, v)
    mean = mean / cnt.clamp(min=1).unsqueeze(1)
    # Take the maximum of every study
    mx = torch.full((B, D), -10000.0, device=v.device, dtype=v.dtype)
    mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), v, reduce="amax", include_self=True)
    return torch.cat([mean, mx], 1)
# Pad the ragged slot dimension into a dense key value batch
def _pad_kv(x, sidx, B, norm):
    # Work out how many slots the largest study holds
    T, P, D = x.shape
    cnt = torch.bincount(sidx, minlength=B)
    S = int(cnt.max().item())
    starts = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(T, device=x.device) - starts[sidx]
    # Scatter the tokens into the padded batch
    pad = x.new_zeros(B, S, P, D)
    pad[sidx, pos] = x
    keep = torch.zeros(B, S, dtype=torch.bool, device=x.device)
    keep[sidx, pos] = True
    return (norm(pad.reshape(B, S * P, D)), ~keep.repeat_interleave(P, dim=1))
# Produce a gated cross attention correction per finding
class _GatedDelta(nn.Module):
    def __init__(self, d, n_labels, n_heads, dropout):
        super().__init__()
        # Learn one query per finding
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        # Learn how the attended feature becomes a correction
        self.d_norm = nn.LayerNorm(d)
        self.dw = nn.Parameter(torch.randn(n_labels, d) * (1.0 / d ** 0.5))
        self.db = nn.Parameter(torch.zeros(n_labels))
        self.gate = nn.Parameter(torch.zeros(n_labels))
    def delta(self, pat, sidx, B, return_attn):
        # Attend from the finding queries to the patch tokens
        kv, kpm = _pad_kv(pat, sidx, B, self.kv_norm)
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(
            q, kv, kv, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True
        )
        return ((self.d_norm(att) * self.dw).sum(-1) + self.db, w)
# Read out from the pooled patch tokens plus a gated correction
class TokenResidualPool(_GatedDelta):
    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        # Build the linear baseline
        self.base = nn.Sequential(
            nn.LayerNorm(2 * d + pe), nn.Dropout(dropout), nn.Linear(2 * d + pe, n_labels)
        )
    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        # Score the pooled patch tokens
        base = self.base(torch.cat([_seg_mean_max(tok[:, 1:].mean(1), sidx, B), pres], 1))
        # Add the gated attention correction
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return (base + self.gate * d_, w)
# Read out from the pooled class token plus a gated correction
class CodexResidualPool(_GatedDelta):
    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        # Build the linear baseline
        self.base = nn.Sequential(
            nn.LayerNorm(2 * d + pe), nn.Dropout(dropout), nn.Linear(2 * d + pe, n_labels)
        )
    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        # Score the pooled class token
        base = self.base(torch.cat([_seg_mean_max(tok[:, 0], sidx, B), pres], 1))
        # Add the gated attention correction
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return (base + self.gate * d_, w)
# Read out from the class token and the patch tokens together
class ClsAddPool(nn.Module):
    def __init__(self, d, n_labels=12, pe=64, dropout=0.2):
        super().__init__()
        # Build the linear readout
        self.net = nn.Sequential(
            nn.LayerNorm(4 * d + pe), nn.Dropout(dropout), nn.Linear(4 * d + pe, n_labels)
        )
    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        # Concatenate both pooled views and the presence code
        return (
            self.net(
                torch.cat(
                    [
                        _seg_mean_max(tok[:, 1:].mean(1), sidx, B),
                        _seg_mean_max(tok[:, 0], sidx, B),
                        pres,
                    ],
                    1,
                )
            ),
            None,
        )
# Turn the encoder features of one study into twelve logits
class Readout(nn.Module):
    def __init__(self, pool, d, n_labels=12, pe=64):
        super().__init__()
        # Encode which slots the study actually holds
        self.pool_kind, self.k = (pool, n_labels)
        self.pres_emb = nn.Embedding(FOLD_N_SLOT_TYPES + 1, pe, padding_idx=0)
        # Build the requested token pooling head
        if pool in ("xres", "clsadd", "xcodex"):
            self.pool = {
                "xres": TokenResidualPool,
                "clsadd": ClsAddPool,
                "xcodex": CodexResidualPool,
            }[pool](d, n_labels, pe=pe)
        # Build the requested attention pooling head
        elif pool in ("attn", "xattn"):
            if pool == "xattn":
                self.pool = TokenXAttnPool(d, n_labels)
                wd = 3 * d + pe
            else:
                self.pool = LabelAttentionPool(d, n_labels)
                wd = d + pe
            self.norm = nn.LayerNorm(wd)
            self.w = nn.Parameter(torch.randn(n_labels, wd) * (1.0 / wd ** 0.5))
            self.b = nn.Parameter(torch.zeros(n_labels))
        # Fall back to the mean and maximum head
        else:
            self.pool = MeanMaxPool()
            self.net = nn.Sequential(
                nn.LayerNorm(2 * d + pe), nn.Dropout(0.2), nn.Linear(2 * d + pe, n_labels)
            )
        self.drop = nn.Dropout(0.2)
    def forward(self, f, slot, sidx, B, return_attn=False):
        # Sum the slot presence codes of every study
        pe = self.pres_emb(slot)
        pres = torch.zeros(B, pe.shape[1], device=f.device, dtype=f.dtype).index_add_(0, sidx, pe)
        # Score with the token pooling heads
        if self.pool_kind in ("xres", "clsadd", "xcodex"):
            return self.pool(f, slot, sidx, B, pres)[0]
        pooled, attn = self.pool(f, sidx, B, slot=slot, return_attn=return_attn)
        # Score with the attention pooling heads
        if self.pool_kind in ("attn", "xattn"):
            x = torch.cat([pooled, pres.unsqueeze(1).expand(-1, self.k, -1)], -1)
            x = self.drop(self.norm(x))
            return (x * self.w).sum(-1) + self.b
        # Score with the mean and maximum head
        return self.net(torch.cat([pooled, pres], 1))
# Wrap the encoder, the optional stem and the readout
class Net(nn.Module):
    def __init__(self, enc, cond, n_meta=0, pool="mean_max", stem="native", n_slice=16):
        super().__init__()
        # Keep the encoder and the conditioning mode
        self.enc, self.cond = (enc, cond)
        # Build the requested depth stem
        self.compress = DepthCompress(n_slice, 3) if stem == "compress" else None
        self.mixer = SlotDepthMixer(n_slice) if stem == "mixer" else None
        self.tokens = pool in ("xattn", "xres", "clsadd", "xcodex")
        D = enc.num_features
        # Build the optional metadata branch
        self.meta_mlp = (
            nn.Sequential(
                nn.LayerNorm(n_meta), nn.Linear(n_meta, 128), nn.GELU(), nn.Linear(128, D)
            )
            if n_meta > 0
            else None
        )
        self.readout = Readout(pool, D)
        # Learn a slot embedding added after the encoder
        if cond == "post":
            self.slot_emb = nn.Embedding(FOLD_N_SLOT_TYPES + 1, D, padding_idx=FOLD_MASK_IDX)
    def forward(self, im, slot, smeta, sidx, B, vm=None):
        # Apply the depth stem
        if self.mixer is not None:
            im = self.mixer(im, slot, vm)
        if self.compress is not None:
            im = self.compress(im)
        # Encode every slot
        f = self.enc.forward_features(im, slot) if self.cond == "token" else self.enc.forward_features(im)
        # Keep the tokens or pool them into one vector
        if self.tokens:
            inner = getattr(self.enc, "vit", self.enc)
            orig = getattr(self.enc, "_orig_prefix", getattr(inner, "num_prefix_tokens", 1))
            f = torch.cat([f[:, :1], f[:, orig:]], 1)
        else:
            f = self.enc.forward_head(f, pre_logits=True)
            if f.dim() > 2:
                f = f.flatten(1)
        # Add the slot embedding after the encoder
        ex = (lambda v: v.unsqueeze(1)) if self.tokens else lambda v: v
        if self.cond == "post":
            f = f + ex(self.slot_emb(slot))
        # Add the metadata branch
        if self.meta_mlp is not None and smeta.shape[1] > 0:
            mt = self.meta_mlp(smeta)
            f = torch.cat([f, mt.unsqueeze(1)], 1) if self.tokens else f + mt
        return self.readout(f, slot, sidx, B)
# Choose the autocast dtype for one device
def fold_amp_for(dev):
    # Keep full precision on the host
    if not str(dev).startswith("cuda"):
        return (torch.float32, False)
    cc = torch.cuda.get_device_capability(dev)
    # Honour an explicit preference
    if FOLD_AMP_PREF == "bf16":
        return (torch.bfloat16, True)
    if FOLD_AMP_PREF == "fp16":
        return (torch.float16, True)
    if FOLD_AMP_PREF == "fp32":
        return (torch.float32, False)
    # Otherwise pick what the device supports
    return (torch.bfloat16 if cc >= (8, 0) else torch.float16, True)
# Load every fold checkpoint attached to the run
def fold_load_models():
    # Rebind the fold branch globals
    global FOLD_MODELS, FOLD_CFG, FOLD_AMP_DTYPE, FOLD_AMP_ENABLED
    # Report the devices the branch will use
    print(f"competition : {ROOT}")
    print(f"checkpoints : {FOLD_CKPT}")
    print(f"device      : {FOLD_DEV}")
    for i in range(torch.cuda.device_count() if FOLD_DEV == "cuda" else 0):
        cc = torch.cuda.get_device_capability(i)
        print(
            f"  gpu{i}       : {torch.cuda.get_device_name(i)} sm_{cc[0]}{cc[1]}, "
            f"{torch.cuda.get_device_properties(i).total_memory / 2 ** 30:.0f} GiB, "
            f"native bf16={cc >= (8, 0)}"
        )
    print("series root:", FOLD_SERIES_ROOT)
    # Rebuild every checkpoint from its stored configuration
    models = []
    cfg = {}
    for ckpt_path in sorted(FOLD_CKPT.glob("*_f*.pt")):
        z = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = z["cfg"]
        _stem = cfg.get("stem", "native")
        _in = 3 if _stem == "compress" else cfg.get("n_slice", 16)
        enc = timm.create_model(
            cfg["backbone"],
            pretrained=False,
            num_classes=0,
            in_chans=_in,
            **{"img_size": cfg["img"]} if "vit_" in cfg["backbone"] else {},
        )
        # Insert the slot token when the checkpoint was conditioned that way
        if cfg["cond"] == "token":
            enc = ViTSlotToken(enc, FOLD_N_SLOT_TYPES)
        m = Net(
            enc,
            cfg["cond"],
            cfg.get("n_meta", 0),
            cfg["pool"],
            stem=_stem,
            n_slice=cfg.get("n_slice", 16),
        )
        # Refuse a checkpoint that does not match this architecture
        missing, unexpected = m.load_state_dict(z["state_dict"], strict=False)
        assert not missing, f"missing {missing[:5]}"
        assert not unexpected, f"unexpected {unexpected[:5]}"
        models.append(m.eval())
        print(
            f"loaded {ckpt_path.name}  fold {z['fold']}  {cfg['backbone']} "
            f"pool={cfg['pool']} meta={cfg['meta']}"
        )
    # Refuse a checkpoint that expects metadata this branch does not build
    FOLD_CFG = cfg
    assert FOLD_CFG.get("n_meta", 0) == 0, (
        f"checkpoint expects {FOLD_CFG['n_meta']} metadata features -- build slot_meta "
        "for the TEST studies and pass it to fold_predict() before submitting"
    )
    print(f"\n{len(models)} fold models ready | input norm: {FOLD_CFG.get('norm', 'none')}")
    # Move every checkpoint onto the device
    FOLD_AMP_DTYPE, FOLD_AMP_ENABLED = fold_amp_for(FOLD_DEV)
    FOLD_MODELS = [m.to(FOLD_DEV).eval() for m in models]
    print(
        f"device {FOLD_DEV} | amp {str(FOLD_AMP_DTYPE).split('.')[-1]} "
        f"(on={FOLD_AMP_ENABLED}) | workers {FOLD_WORKERS} | chunk {FOLD_CHUNK} | "
        f"micro {FOLD_MICRO}"
    )
# Normalise one batch the way the checkpoints expect
def fold_normalise(im):
    # Standardise inside the non empty pixels
    k = FOLD_CFG.get("norm", "none")
    if k == "zscore":
        m = (im > 0).float()
        n = m.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        mu = (im * m).sum(dim=(1, 2, 3), keepdim=True) / n
        var = (((im - mu) * m) ** 2).sum(dim=(1, 2, 3), keepdim=True) / n
        return (im - mu) / (var.sqrt() + 1e-06) * m
    # Apply the single channel ImageNet statistics
    if k == "imagenet":
        m = (im > 0).float()
        return (im - 0.485) / 0.229 * m
    # Leave the batch untouched
    return im
# Score one micro batch of studies with every fold
@torch.no_grad()
def fold_micro_predict(images, masks):
    # Flatten the present slots of every study
    dev = FOLD_DEV
    ims, slots, sidx, vms = ([], [], [], [])
    for b in range(len(masks)):
        present = np.nonzero(masks[b] > 0)[0]
        if len(present) == 0:
            continue
        blk = images[b][present]
        ims.append(torch.from_numpy(blk))
        vms.append(torch.from_numpy(blk.reshape(blk.shape[0], blk.shape[1], -1).max(2) > 0))
        slots.append(torch.from_numpy(present + 1).long())
        sidx.append(torch.full((len(present),), b, dtype=torch.long))
    # Return nothing when no study carries a slot
    out = np.full((len(FOLD_MODELS), len(masks), len(FOLD_LABELS)), np.nan, np.float32)
    if not ims:
        return out
    # Move the batch onto the device
    im = fold_normalise(torch.cat(ims).to(dev, non_blocking=True).float().div_(255.0))
    sl = torch.cat(slots).to(dev)
    si = torch.cat(sidx).to(dev)
    vm = torch.cat(vms).to(dev)
    sm = torch.zeros(len(sl), FOLD_CFG.get("n_meta", 0), device=dev)
    # Score every fold
    per = torch.zeros(
        len(FOLD_MODELS), len(masks), len(FOLD_LABELS), device=dev, dtype=torch.float32
    )
    with torch.autocast(
        "cuda" if str(dev).startswith("cuda") else "cpu",
        dtype=FOLD_AMP_DTYPE,
        enabled=FOLD_AMP_ENABLED,
    ):
        for fold_index, model in enumerate(FOLD_MODELS):
            per[fold_index] = torch.sigmoid(model(im, sl, sm, si, len(masks), vm=vm).float())
    # Keep only the studies that carried a slot
    got = per.cpu().numpy()
    keep = np.array([(masks[b] > 0).any() for b in range(len(masks))])
    out[:, keep] = got[:, keep]
    return out
# Score a block of studies in micro batches
def fold_predict(images, masks):
    # Score the block one micro batch at a time
    out = np.full((len(FOLD_MODELS), len(masks), len(FOLD_LABELS)), np.nan, np.float32)
    for a in range(0, len(masks), FOLD_MICRO):
        b = min(a + FOLD_MICRO, len(masks))
        out[:, a:b] = fold_micro_predict(images[a:b], masks[a:b])
    return out
# Run the fold checkpoint stage and blend it into the submission
def run_fold_stage():
    # Load the checkpoints
    fold_load_models()
    # Read the submission template and the series table
    sub_df = pd.read_csv(ROOT / "sample_submission.csv")
    ser_csv = pd.read_csv(ROOT / "test_series.csv")
    if not (ROOT / "test_series").exists():
        ser_csv = pd.read_csv(ROOT / "train_series.csv")
    ser_csv = ser_csv.loc[:, ~ser_csv.columns.duplicated()]
    # Group the series records by study
    studies = sub_df.StudyInstanceUID.tolist()
    by = {
        s: g.to_dict("records")
        for s, g in ser_csv[ser_csv.StudyInstanceUID.isin(set(studies))].groupby(
            "StudyInstanceUID"
        )
    }
    print(f"{len(studies):,} test studies, {len(by):,} with series metadata")
    # Score every study block while the workers decode the next one
    preds = np.full((len(FOLD_MODELS), len(studies), len(FOLD_LABELS)), np.nan, np.float32)
    t0, done = (time.time(), 0)
    with ProcessPoolExecutor(max_workers=FOLD_WORKERS) as ex:
        for c0 in range(0, len(studies), FOLD_CHUNK):
            block = studies[c0 : c0 + FOLD_CHUNK]
            imgs = np.zeros(
                (len(block), FOLD_N_SLOT, FOLD_N_SLICE, FOLD_SIZE, FOLD_SIZE), np.uint8
            )
            msks = np.zeros((len(block), FOLD_N_SLOT), np.uint8)
            futs = [ex.submit(fold_build_study, (i, s, by.get(s, []))) for i, s in enumerate(block)]
            # Collect the decoded studies
            for f in as_completed(futs):
                try:
                    i, a, k = f.result()
                    imgs[i], msks[i] = (a, k)
                except Exception as e:
                    print(f"  study failed: {type(e).__name__}: {e}")
            preds[:, c0 : c0 + len(block)] = fold_predict(imgs, msks)
            # Report the progress and the estimated remaining time
            done += len(block)
            el = time.time() - t0
            print(
                f"  {done:,}/{len(studies):,}  {el / 60:.1f}m  "
                f"eta {el / done * (len(studies) - done) / 60:.1f}m",
                flush=True,
            )
            del imgs, msks
            gc.collect()
    print(f"\ninference done in {(time.time() - t0) / 60:.1f} min")
    # Average the fold ranks over the studies that scored
    fold_ok = np.isfinite(preds).all(axis=(0, 2))
    fold_rank_mean = np.zeros((len(studies), len(FOLD_LABELS)), np.float64)
    for fold_index in range(preds.shape[0]):
        fold = preds[fold_index][fold_ok]
        ordinal = fold.argsort(0).argsort(0).astype(np.float64)
        fold_rank_mean[fold_ok] += ordinal / max(len(fold) - 1, 1)
    fold_rank_mean /= preds.shape[0]
    fold_rank_mean[~fold_ok] = np.nan
    fold_preds = dict(zip(sub_df["StudyInstanceUID"].astype(str), fold_rank_mean.astype(np.float32)))
    # Refuse to blend into a submission with another schema
    fold_sub = pd.read_csv(WORKING / "submission.csv", dtype={"StudyInstanceUID": str})
    assert fold_sub.columns.tolist()[1:] == FOLD_LABELS, "submission schema drift"
    # Blend this branch into the running submission
    if FOLD_BLEND_WEIGHT > 0:
        fold_ours = np.stack([fold_preds[u] for u in fold_sub["StudyInstanceUID"].astype(str)])
        fold_base_rank = fold_sub[FOLD_LABELS].rank(method="average", pct=True)
        fold_ours_rank = pd.DataFrame(
            fold_ours, columns=FOLD_LABELS, index=fold_sub.index
        ).rank(method="average", pct=True)
        fold_sub[FOLD_LABELS] = (
            1.0 - FOLD_BLEND_WEIGHT
        ) * fold_base_rank + FOLD_BLEND_WEIGHT * fold_ours_rank
        assert np.isfinite(fold_sub[FOLD_LABELS].to_numpy()).all()
        fold_sub.to_csv(WORKING / "submission.csv", index=False)
# Reuse the official finding order for the RadImageNet branch
_RAD_LABELS = list(TARGETS)
# Set how much of the RadImageNet branch enters the blend
_RAD_ALPHA = 0.5
# Keep two findings out of the first RadImageNet blend
_RAD_EXCLUDE = ("Baker's", "Fracture")
# Pin the checkpoints this branch is allowed to load
_RAD_REFERENCE_HEADS_SHA256 = "0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa"
_RAD_ENCODER_SHA256 = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
_RAD_E13_HEADS_SHA256 = "ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a"
# Set the weight of the diverse head bundle and of the second pass
_RAD_E13_MEMBER_WEIGHT = 0.5
_RAD_V48_SECOND_ALPHA = 0.15
# Carry the compressed calibration payload used by the transformer gate
_RAD_CAL_PAYLOAD = 'eNrtmk1vI8cRhv9KsJdcKKE/q6tzc4z4ZCMBcjQWhrCRDSG2ZEjaIEGQ/57n7RlRQ3KG4jqLJAcDS4o709NdXR9vvVU9/3z30+3N/bvffRuuawgxlm7eq2ePeffrpV8v/V9euvLraD3k6ilW6zn126vYd+U6eKmx9RiaF7OSx+X1weE67K7SdSgpVSauuaecUxr3rtp1LK0HyyV7y9Gmy/E6pBhTL61Zt2gWx+WtSew6JmOoM5AHutXp+ro8+ZrlsnvJOfDdfRL+Kl4zd2bqllNmga7rvrva2OzGNFuynGxpmnxrS/069hCtpt6T1dLLOQVsTbJ1fWunG2qXAX8NiU+7VK8rdqvNU89uwQwjpWyeLdTCnVy84ua9pthSjznHXLjDpZxbLe4WPRAQCT+LrSVcGGdLzNX0XKhesC2eV5uZ67lQPDlmSzUhS2+61ELtoTOyWGjdPudc73fvnj7c/Hg7ElpyzYXjdwIZ32m7X3INZ0crtvtc833ua6fyoUYUGf4H13rJpX+GK5aa5VbeWO9ye/03bLi97i8SpaSCl49nc4gdxI2p1dZyVmQnfL56jBH/j70GqSq6dyMROLHQGvCpcSnkYsyeohNErnVjbamVjs47wOpBa7BAsa61SXulDfSIwAbAkQqDmSK38WwImKK1kFJ3d11CpFqR1mPHlj1pWVAHeDfyU2hk0vGogz0GqCBbKmFMx9xIWEAbQ8I4PUvmAplQCeuij7GNXGMk3yhBsFIfEnvVE6HFYMHDtFuLzKwpya9gisaZduAkcyAvmw1ROjmd7OnBYytpOBqJjTyK4KzT0Pi0tQJYslc0nGqcNZV7dEaRvfcSewg9h4p41iwO+4CWMkNG1326VBmHFUoB7VoakzE+JAtYN/NknS9lLJ+9S5qR56Q2kLA4qgTuplGNmUJAkbXiGbrU0HCIsgtYaGOUVXbayNeykLfpGv8lulBfgiFEmx61xm3LTlrOPoa51Ng7IyJiT4tWxrE0ZmnJRnhawZsyLgjXIC9MRu0tRAgImkJ7bUyHKk1eUgIewtRjXGDJqO1mNJrHfNUty1HRW2SSoV3FAVIA//hrUXKANxhbRbEkgqAFKiCC43qOEXsPTaKtKCdkqx3/koMYsuP+mh4HrHoQtqSIw/h4DM7EpRL5zRirAcHiUDj2SUlr9fFAsElHBX/zWgJCQlw+83Qksw8Pt9+Ty0hmOBQBh9XQAr7fxjRQ2IkGTT/w8cAiBKwRc1jwcMh+WKtMgFShNaDGJWRAechNSOMUldXTynNIUPAaEKKkCv1cLFzxaowBOmX8vU8Pj1voWa5JTPFcGN4QXm+PpZPjA8QQYYp9Qab9rZiIAuLX8AA8f/jX4kHgI+BXCtBEeOTFvNPapIyC92YAEOmHypJcCSiFOpQi712M73IL8KiBX8Hz62pDNsAH8MK/rMR+tNSRIRJwAkIIY8Rn/fD2kB2V9I7BMX+KSJ/XJtNAbwFglmQZbu/90CZcMhAeZKudKAvlSBLwNwP1aA88BYpPdJRkbADGeBzWN2IOvySVELxyU0Lx0JHQHsFpEQRsCB9iO1qT3IOHGUjMDqeczX4BiIbvCjkiLD6t6G239p+gAML1KQuAdaLLdidDV6Y6uPR+9+3LT8KrlYgzAsjg7ok+VJakimvgAjn5qUFmtTGJKXGxSadg2UuLkWok4EvGJ0Hdsr+DmpVlGsUMZkxtuTI8vDAVaIsnSIDbK0DhVSpAFlu4iRtgrLYWRaArADOcFDAfErEdwBS8dWpNqHupW3o7nIukTmxlmASq4L/51ESrznqJTSkgCzslVSMncekb8659ljeyYttxK0oX9k50n3h2kDqoapToWt8S9/yO3tTXyteLu+19rkPViKkIblLnzJhJUhYENUJCtdfWKkEFe9WTsWin6azpqJFIfEwNyLNes+PZXFV3h+tIOXjb5mzIRnCLaWKuhnPtjsCVPAOwdkhQhBSMfWLVruTgwEM/WXt1FYgv5AN4IsyBiHrOSscCBjIomksgZCPFn9grqDrE9UXDQCSPfs5RXyeuQl3gwcEIdn5JzADhpG34s4uF5NQ24eh1GWISkEmYA6iFiNezYAbkKNGJhIk+l9XNXK3r7AgLT4YnRUuHicJS0NRLLG0KDrF1IbkaMzj2MoeKzAH/kTo9KsnWJe8gDZUuxs1iPi0CayABMdLIT4qQCbfENfCiAsujIsj9KMUQ1kXwXUBFZspHZm9Zj/dB/SrVy1qOJg9ApKAlmSQNpr6SjibnNgVoqZQCr8gllgsTiFAOCFSeDcYWuKOChZQXi5dPxouF5Oo6ogVTaxDwnVE8KfQFI6Zomwgb/oI4VG2Ae1MdsTQCwQuZp5ZROZPcdudWHrZRfwe4cPE1cv9L/FDuwRwGNYOttGllMRItS1K3sFDQDAwQMvgpfIzyIeQj2yTFOhomGLsBVPtAASGLqiXKnChASW/4ILUTGlftQlUVl8FDjifbKYIzLKXmco4anNPK6ZVl8MhVBs+D5YgTo8HXuIEDQDJF4wHEYL5PvEExn4PIt7x0JumDeUgjxHeFIZDt+6xN5mALCc6pMjTZ7oy8EomCUA09MTRvvuCLxyFC+sEWCGiqjE+HIAJ4g39Bi5sadSN9MJZaFWLbpubBNBvlmwrPAONJWS1DpaIgbyKuZSbKg7qZzgNKsqnwIX8RAOvINdgfRg0jPNJBWE+5xAQ9qDaAPh4nKInagjAL9otj26U+sAnbUERTF0QYjHV8j8OEA+mohtH9SLLuoUIh6uAWHI6yAw54uEsqL6zufMAAt3yW+6xSFLmKwEmYgIPlXHYbPAo3AvKovCk/0KR5m3fmKkpFz2W3ng58h7KzqVVHqVYUpfGlAJHvQ2dN7QKmXoQITggHUz9HGZhSty5smYHeSL4pFdvbXlXZcSJV7GCHKPpsePEmVAOm41WxbuW3Y+qkrhYhisbrSBqbGiGdAkBwd5gzNehxGUVhzu5JCKa2VAyL2pfYgYgUtcTwu3RKZzO2JublhfCDbO1Qqypu8SdTWO0501Z6iAAH68g21mECvjX8bZZ7yvpViqq9BlHCnhj7DFuiXm8qEwgrAkK9hpfbuU/nN/i4l7I/qQEnpR4qVHUxgDjblWuM2UZFhOpzp+apb8i4ua8gnyHrqPMHs839gK7gaaYDDtILqNTWOF8kxVYdMxlVDwE68/F8DR2CgpCSkkzkEvKY3x9GoRqjpn8ZLj62TyEoRVV1dxrV2GtNOMosPAjqSC6pm6yRKRJFM5wnixbhh6uLl9FdjElsUvX7WxR61T1gGohOkQaej27MxiR+DQjAndQ5SgL4Yb+VMKT2UncGK4se5hfeR7Jr6nygbTJcPGK7QSpXOoBJkuMXlFTYQX0aRsObYEQn22BNrDQdKwmZ1DDaZNcANxumdkGs3pZIJcaCcxBuPSlGziTf+UNZIgqmLMTWS1/XYNBxFowH5FchZb7uT5EQqervUK+Rc15pP55mBpKra6Wju5MCbZwiKOSS/HFuZyVRBFeeIoVrc35oNnVt1ARpKDaHF2BO1z3DLILOrOVZdhIG52ojyFxsVa1k6Bq+sOS7AA0upIqpiCt9Om8+1SsIoI4/ZYFKZq9tdwnhCzrVRpoCxpqSo+0ua1DNLCOMc3X19vGSZZ9znfEAMTUpymSGqYW/kpXU6FWV6+pah9OGLveaTkopOV1d3U9oWfAhMiyK46fRgPeL9LTSDAtUjj5cFN4D8S5zmaCjmaTjJ/Kvxb3MaFrHVI27QS2vRfcMCmeqcUXY+NX6652okyj19OEC3SaFrdWyeyJMxo7gPrANn17GjQ5RgtqvregNjzr1hYOOxATeos4b/IItGRxEFZC4aNruVL1ZbGx8ojpE5moLiGNavazB+baxPoXr/gK56zjI1VGbjtG8nA3Xg3SpFl2gtqzqsM/oGgb5Axj4w+XD1g48MBFrMAnR+TRxMci1+8h+uCC1e1nmC7XEWhf2zEdIw5SsCe3Tcc18XibsU/PKhJhd7a380s67RLEvQQWUw9KKjmps7qefVeub/IZQoLbK0qxnHRFP7qlOy8BURC7Fy2LDPk7N1F1VEm1lrZbSmSWVSoNk63DQXgvIohAjYcjXU7b/zI8u7RVvPiRChSqVjkiU6YQjrT1ImVAAN/WoKEp62V2aQGAdAuQKhSW5qouyIdQ4jNe5NS6I8v10hLIeqIIMqplDn71o0dYl4fEFkRZ4LmhFJNUSQqZCNlBT2fIWnKzJ9XWwG48bOzGZWEIffcVx/q23xAziBV83gf1cE4/+mvI8/EFV3QVmPW0a6rB7ZD314ploTllOSlFYqX3X4u7TJg6jmxLRWxit1FaPtArKoHfHeb3jwPmNGDqfvS+Iv9Ns14Vv6p9rfyft+HGiFqmJIQSU++rlbegPBqDumli9zoOGSptec8O6GAWqtaAgkOgqZ3P1WhQQ08aj2KGONuEIqdZetzuLFB6qQniid8HkZTnlyGsPGnA4lY5Qu1456WnbokG9IupwU0NoPhHTwRRUqynZU0HObV8QUyZuo/lb6tFhsdxb7bCoU9qWe8vLxBwj2laVzgS+bIZGMfee1Qaldl87gBZ5jjr3Y4rq2Xaf3LatohHwxzbqBKtv4d8bHnh1eUqfVRx07jc4zfzySrj8IGV9NMFX4mBKrq6FXc4Jz154/3737u7++fbxw+3Pz9N7eg0X0mHCeHfHbHrNhrCIpdXRA/LRRC4WR9sU/ZqJB46XJsJouwU1rPT+il6uKHqNAdbJ2InUJp0wVWBV7w8NuldU7uMyajZqh2N6UfeoM6ym1zLGizF6T6AnNQZiHO/KZMijZMOV2/yKQFKv0TSXq0Hb5teE9F4HLp50QKJ3OX64edZ7ie+++PLrd7t339z+5e7mx9/88Qt+f82dx5f//Omr6e8fvv/+49Pdwz0/f3/z19vH3z7x68uH++fpKhP+/Pjw/PDh4cfv+Hz86f5Jk99/93T7eHepsfef/fnmh/H3y4fH8feLv9/x96ub5+l7vq9f0wj9msf8+HH6fhnDr3kMvzRGG3p8+PizVt3vaXy/ysjox5sPzx8fbxn+7cuWv7m9v3v68PFpsfH9pcWwLc1oyEI3f/7H/cPf7p7vnhZ6ev/+X/8GYIe3xg=='
# Set how much of the calibrated prediction enters the gated findings
_RAD_CAL_W = 0.40
# Set the feature width of the encoder and of the head
_RAD_TOKEN_DIM, _RAD_HEAD_DIM = (2048, 512)
# Declare the slots of the second pass
_RAD_E11_SLOTS = [
    ("SAG_NOFS", "Sagittal", None, False),
    ("COR_NOFS", "Coronal", None, False),
    ("AX_NOFS", "Axial", None, False),
    ("SAG_FS", "Sagittal", None, True),
]
_RAD_E11_CROP_MM = 130.0
# Declare the slots and geometry of the diverse head bundle
_RAD_E13_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
    ("SAG_NOFS", "Sagittal", None, False),
]
_RAD_E13_CROP_MM = 130.0
_RAD_E13_CACHE_SLICES = 8
_RAD_E13_IMG = 224
# Hash one file in streaming chunks
def _rad_sha256(path, chunk=8 << 20):
    # Digest the file block by block
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()
# Locate one pinned checkpoint and verify its hash
def _rad_find_file(name, expected_sha=None, explicit_env=None):
    # Map every accepted hash to its path
    files = {
        _RAD_ENCODER_SHA256: ASSET / "resnet-50-radimagenet-marwan/ResNet50.pt",
        _RAD_REFERENCE_HEADS_SHA256: ASSET
        / "rsna-knee-e9-radimagenet-heads-v15/v52_radimagenet_heads.pt",
        _RAD_E13_HEADS_SHA256: ASSET
        / "kernel-sources/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt",
    }
    # Fail when the file is absent
    path = files.get(expected_sha)
    if path is None or not path.is_file():
        raise FileNotFoundError(name)
    # Fail when the file is not the pinned one
    if _rad_sha256(path) != expected_sha:
        raise RuntimeError(f"hash mismatch for {path}")
    return path
# Encode one slice with the RadImageNet ResNet50
class _RadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep the convolutional trunk only
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(self, image):
        # Average the final feature map
        return self.backbone(image).mean(dim=(2, 3))
# Read twelve findings out of the encoded slice tokens
class _RadHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Project the encoder features
        self.project = nn.Sequential(
            nn.LayerNorm(_RAD_TOKEN_DIM), nn.Linear(_RAD_TOKEN_DIM, _RAD_HEAD_DIM), nn.GELU()
        )
        # Learn a plane and a position code
        self.plane = nn.Parameter(torch.randn(N_SLOT, _RAD_HEAD_DIM) * 0.01)
        self.position = nn.Parameter(torch.randn(CACHE_SLICES, _RAD_HEAD_DIM) * 0.01)
        # Learn one query per finding
        self.query = nn.Parameter(torch.randn(len(_RAD_LABELS), _RAD_HEAD_DIM) * 0.02)
        self.attn = nn.MultiheadAttention(_RAD_HEAD_DIM, 8, dropout=0.1, batch_first=True)
        # Fuse the attended token with the pooled token
        self.fuse = nn.Sequential(
            nn.LayerNorm(_RAD_HEAD_DIM * 4),
            nn.Linear(_RAD_HEAD_DIM * 4, _RAD_HEAD_DIM),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.weight = nn.Parameter(torch.randn(len(_RAD_LABELS), _RAD_HEAD_DIM) * 0.02)
        self.bias = nn.Parameter(torch.zeros(len(_RAD_LABELS)))
    def forward(self, feature, mask):
        # Project the features and add the plane and position codes
        token = self.project(feature.float())
        token = token.view(len(token), N_SLOT, CACHE_SLICES, _RAD_HEAD_DIM)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        # Keep at least one key for a study with no token
        key_padding = mask <= 0
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        # Attend from the finding queries to the tokens
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(
            query, token, token, key_padding_mask=key_padding, need_weights=False
        )[0]
        # Pool the present tokens as a baseline
        denominator = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdim=True) / denominator
        mean = mean.expand(-1, len(_RAD_LABELS), -1)
        # Fuse both views into one logit per finding
        fused = self.fuse(
            torch.cat(
                [attended, mean, torch.abs(attended - mean), attended * mean], dim=-1
            )
        )
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias
# Load the five public heads and check their contract
def _rad_load_public_heads(device, expected_sha):
    # Read the pinned bundle
    heads_path = _rad_find_file("v52_radimagenet_heads.pt", expected_sha)
    payload = torch.load(heads_path, map_location="cpu", weights_only=True)
    # Refuse a bundle that was fitted under another contract
    expected = {
        "version": "v52-radimagenet-resnet50-official-1",
        "targets": _RAD_LABELS,
        "encoder_sha256": _RAD_ENCODER_SHA256,
        "encoder_source_commit": "0ce16f7375db4236e646829d1eca61cdb4282133",
        "img": 224,
        "slices_per_plane": 8,
        "feature": "global_average_pool",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"public-v15 head contract drift for {key}")
    # Refuse a bundle that does not hold the five folds
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("public-v15 bundle requires exactly five heads")
    if sorted(int(record.get("fold", -1)) for record in folds) != list(range(5)):
        raise RuntimeError("public-v15 fold identity drift")
    # Rebuild every head
    heads = []
    for record in folds:
        head = _RadHead().to(device).eval()
        head.load_state_dict(record["state_dict"], strict=True)
        heads.append(head)
    return (heads, str(heads_path))
# Load the five diverse heads and check their contract
def _rad_load_e13_heads(device):
    # Read the pinned bundle
    heads_path = _rad_find_file("v52_e11_heads.pt", _RAD_E13_HEADS_SHA256)
    payload = torch.load(heads_path, map_location="cpu", weights_only=False)
    # Refuse a bundle that was fitted under another contract
    expected = {
        "version": "e11-radimagenet-resnet50-diverse-1",
        "targets": _RAD_LABELS,
        "encoder_sha256": _RAD_ENCODER_SHA256,
        "slots": [list(slot) for slot in _RAD_E13_SLOTS],
        "crop_mm": _RAD_E13_CROP_MM,
        "img": _RAD_E13_IMG,
        "slices_per_plane": _RAD_E13_CACHE_SLICES,
        "feature": "global_average_pool",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"E13 head contract drift for {key}")
    # Refuse a bundle that does not hold the five folds
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("E13 bundle requires exactly five heads")
    if sorted(int(record.get("fold", -1)) for record in folds) != list(range(5)):
        raise RuntimeError("E13 fold identity drift")
    # Rebuild every head
    heads = []
    for record in folds:
        head = _RadHead().to(device).eval()
        head.load_state_dict(record["state_dict"], strict=True)
        heads.append(head)
    return (heads, str(heads_path))
# Encode every present slice of the cache
@torch.inference_mode()
def _rad_encode(encoder, pixels, slot_mask, device):
    # Expand the slot mask over the slices
    n, slots, slices, height, width = pixels.shape
    features = np.zeros((n, slots * slices, _RAD_TOKEN_DIM), np.float16)
    token_mask = np.repeat(slot_mask[:, :, None], slices, axis=2).reshape(n, -1)
    valid = np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = pixels.reshape(-1, height, width)
    # Size the batch to the device
    batch = (
        192
        if device.type == "cuda" and torch.cuda.device_count() > 1
        else 96
        if device.type == "cuda"
        else 8
    )
    # Encode the present slices in batches
    for start in range(0, len(valid), batch):
        indices = valid[start : start + batch]
        image = torch.from_numpy(flat[indices]).to(device).float().div_(127.5).sub_(1.0)
        image = image.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        amp = torch.autocast("cuda") if device.type == "cuda" else contextlib.nullcontext()
        with amp:
            feature = encoder(image)
        # Refuse a feature that is not finite
        values = feature.float().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("V36 non-finite RadImageNet feature")
        features.reshape(-1, _RAD_TOKEN_DIM)[indices] = values.astype(np.float16)
    return (features, token_mask.astype(np.float32))
# Score every study with one head
@torch.inference_mode()
def _rad_predict_head(head, features, masks, device, batch=64):
    # Score the studies in batches
    predictions = []
    for start in range(0, len(features), batch):
        image = torch.from_numpy(features[start : start + batch]).to(device)
        mask = torch.from_numpy(masks[start : start + batch]).to(device)
        amp = torch.autocast("cuda") if device.type == "cuda" else contextlib.nullcontext()
        with amp:
            predictions.append(torch.sigmoid(head(image, mask)).float().cpu())
    return torch.cat(predictions).numpy()
# Rank every column into the unit range
def _rad_rank_columns(values):
    return (
        pd.DataFrame(np.asarray(values, dtype=np.float64))
        .rank(method="average", pct=True)
        .to_numpy(np.float64)
    )
# Refuse a submission that drifted in schema, identity or range
def _rad_validate(frame, expected_ids):
    # Check the column layout
    if frame.columns.tolist() != ["StudyInstanceUID", *_RAD_LABELS]:
        raise RuntimeError("V36 submission schema drift")
    # Check the study identity and order
    ids = frame["StudyInstanceUID"].astype(str).tolist()
    if ids != list(map(str, expected_ids)) or len(ids) != len(set(ids)):
        raise RuntimeError("V36 submission study identity/order drift")
    # Check the value range
    values = frame[_RAD_LABELS].to_numpy(np.float64)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError("V36 invalid submission values")
# Count the protocol of every study for the calibrator
def _v18_cal_protocol(uids):
    # Read the series table
    frame = pd.read_csv(
        ROOT / "test_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    # Index the table by the requested studies
    index = pd.Index([str(uid) for uid in uids], name="StudyInstanceUID")
    table = pd.DataFrame(index=index)
    # Count every series of a study
    table["n_series"] = frame.groupby("StudyInstanceUID").size().reindex(index).fillna(0)
    # Count the series of every plane
    for plane in ("Sagittal", "Coronal", "Axial"):
        part = frame[frame["Anatomical_Plane"].astype(str).eq(plane)]
        table[f"n_{plane[:3]}"] = (
            part.groupby("StudyInstanceUID").size().reindex(index).fillna(0)
        )
    # Count the marked series overall and per plane
    for flag in ("Fat_Suppression", "Fluid_Sensitive"):
        marked = frame[pd.to_numeric(frame[flag], errors="coerce").fillna(0) > 0]
        prefix = flag[:3]
        table[prefix] = marked.groupby("StudyInstanceUID").size().reindex(index).fillna(0)
        for plane in ("Sagittal", "Coronal", "Axial"):
            part = marked[marked["Anatomical_Plane"].astype(str).eq(plane)]
            table[f"{prefix}_{plane[:3]}"] = (
                part.groupby("StudyInstanceUID").size().reindex(index).fillna(0)
            )
    return table
# Calibrate the transformer branch against the three rank views
def _v18_calibrate_transformer(branch, baseline_rank, public_rank, pass2_rank, expected_ids):
    # Read the fitted calibrator
    payload = json.loads(zlib.decompress(base64.b64decode(_RAD_CAL_PAYLOAD)).decode())
    gate = set(payload["gate"])
    # Refuse a protocol table the calibrator was not fitted on
    protocol = _v18_cal_protocol(expected_ids)
    if protocol.columns.tolist() != list(payload["protocol_columns"]):
        raise RuntimeError("V18 calibration protocol layout mismatch")
    # Build the rank blocks the calibrator reads
    mean_rank = (baseline_rank + public_rank + pass2_rank) / 3.0
    blocks = [
        baseline_rank,
        public_rank,
        pass2_rank,
        public_rank - baseline_rank,
        pass2_rank - baseline_rank,
        mean_rank,
    ]
    # Add the mean rank of every declared finding group
    for group in payload["groups"]:
        columns = [_RAD_LABELS.index(target) for target in group]
        blocks.append(mean_rank[:, columns].mean(axis=1, keepdims=True))
    # Add the protocol counts
    blocks.append(protocol.to_numpy(np.float64))
    x = np.concatenate(blocks, axis=1)
    # Read the fitted standardisation and coefficients
    centre = np.asarray(payload["mean"], np.float64)
    spread = np.asarray(payload["scale"], np.float64)
    coef = np.asarray(payload["coef"], np.float64)
    bias = np.asarray(payload["intercept"], np.float64)
    # Refuse a feature matrix of the wrong width
    if x.shape[1] != 88 or coef.shape != (len(_RAD_LABELS), 88):
        raise RuntimeError(
            f"V18 calibration feature drift: x={x.shape}, coef={coef.shape}"
        )
    # Apply the calibrator and rank its output
    spread = np.where(np.abs(spread) > 1e-8, spread, 1.0)
    adjusted = _rad_rank_columns(((x - centre) / spread) @ coef.T + bias)
    # Blend the calibrated ranks into the gated findings only
    output = branch.copy()
    values = output[_RAD_LABELS].to_numpy(np.float64).copy()
    for index, target in enumerate(_RAD_LABELS):
        if target in gate:
            values[:, index] = (1.0 - _RAD_CAL_W) * values[:, index] + _RAD_CAL_W * adjusted[:, index]
    output[_RAD_LABELS] = _rad_rank_columns(values)
    _rad_validate(output, expected_ids)
    return output, gate
# Run the RadImageNet stage and rewrite the submission
def _rad_main():
    # Report the calibration outcome through the module globals
    global V18_TRANSFORMER_RAW, V18_TRANSFORMER_CAL, V18_CALIBRATOR_APPLIED, V18_CAL_GATE
    # Read the running submission and the expected study order
    primary = WORKING / "submission.csv"
    test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
    expected_ids = test.StudyInstanceUID.astype(str).tolist()
    baseline = pd.read_csv(primary, dtype={"StudyInstanceUID": str})
    _rad_validate(baseline, expected_ids)
    # Read the declared plane of every series
    device = torch.device("cuda:0")
    test_series = pd.read_csv(
        ROOT / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str}
    )
    plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
    # Decode one cache under a given slot scheme
    def cache(slots, crop, tag, threshold):
        # Adopt the geometry this cache needs
        globals().update(
            SLOTS=list(slots),
            N_SLOT=len(slots),
            CACHE_SLICES=8,
            IMG=224,
            CACHE_IMG=224,
            CROP_MM=float(crop),
            RULES=dict(RULES_LEGACY),
        )
        # Decode every study
        headers = annotate(walk("test_series"))
        studies, pixels, masks = build_cache(
            pick_slots(headers, plane), plane, lat_of(headers, tag + " "), tag
        )
        # Refuse a cache that misses a study
        positions = {str(uid): index for index, uid in enumerate(studies)}
        missing = [uid for uid in expected_ids if uid not in positions]
        if missing:
            raise RuntimeError(f"{len(missing)} studies absent from {tag}")
        # Reorder the cache to the submission order
        order = np.asarray([positions[uid] for uid in expected_ids], dtype=np.int64)
        pixels, masks = (pixels[order], masks[order])
        # Refuse a cache that decoded too little
        tokens = int(np.repeat(masks[:, :, None], CACHE_SLICES, axis=2).sum())
        if tokens < int(threshold * len(test) * N_SLOT * CACHE_SLICES):
            raise RuntimeError(f"insufficient slices for {tag}: {tokens}")
        return (pixels, masks)
    # Decode the fluid sensitive cache of the reference heads
    public_slots = [
        ("SAG_FS", "Sagittal", None, True),
        ("COR_FS", "Coronal", None, True),
        ("AX_FS", "Axial", None, True),
    ]
    pixels, masks = cache(public_slots, 10000.0, "test-e10", 0.85)
    # Load the pinned encoder
    encoder_path = _rad_find_file("ResNet50.pt", _RAD_ENCODER_SHA256)
    encoder = _RadEncoder()
    encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu", weights_only=True), strict=True
    )
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    # Spread the encoder over every device
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder, device_ids=list(range(torch.cuda.device_count())))
    # Score the reference heads
    reference_heads, _ = _rad_load_public_heads(device, _RAD_REFERENCE_HEADS_SHA256)
    features, token_mask = _rad_encode(encoder, pixels, masks, device)
    reference_predictions = [
        _rad_predict_head(head, features, token_mask, device) for head in reference_heads
    ]
    reference_probability = np.mean(np.stack(reference_predictions), axis=0)
    reference_rank = _rad_rank_columns(reference_probability)
    # Release the reference pass
    del reference_predictions, reference_heads
    del reference_probability, features, token_mask, pixels, masks
    gc.collect()
    torch.cuda.empty_cache()
    # Score the diverse heads on their own slot scheme
    globals().update(
        SLOTS=list(_RAD_E13_SLOTS),
        N_SLOT=len(_RAD_E13_SLOTS),
        CACHE_SLICES=_RAD_E13_CACHE_SLICES,
        IMG=_RAD_E13_IMG,
        CACHE_IMG=_RAD_E13_IMG,
        CROP_MM=_RAD_E13_CROP_MM,
        RULES=dict(RULES_LEGACY),
    )
    e13_heads, _ = _rad_load_e13_heads(device)
    pixels, masks = cache(_RAD_E13_SLOTS, _RAD_E13_CROP_MM, "test-e13", 0.85)
    features, token_mask = _rad_encode(encoder, pixels, masks, device)
    e13_predictions = [
        _rad_predict_head(head, features, token_mask, device) for head in e13_heads
    ]
    e13_probability = np.mean(np.stack(e13_predictions), axis=0)
    e13_rank = _rad_rank_columns(e13_probability)
    # Merge both head bundles into one RadImageNet rank
    reference_rank = _rad_rank_columns(
        (1.0 - _RAD_E13_MEMBER_WEIGHT) * reference_rank + _RAD_E13_MEMBER_WEIGHT * e13_rank
    )
    # Release the diverse pass
    del e13_predictions, e13_probability, e13_rank
    del features, token_mask, pixels, masks
    gc.collect()
    torch.cuda.empty_cache()
    # Blend the RadImageNet rank into every finding it is trusted on
    baseline_rank = _rad_rank_columns(baseline[_RAD_LABELS].to_numpy())
    e10 = baseline.copy()
    for index, target in enumerate(_RAD_LABELS):
        if target not in _RAD_EXCLUDE:
            e10[target] = (1.0 - _RAD_ALPHA) * baseline_rank[:, index] + _RAD_ALPHA * reference_rank[:, index]
    _rad_validate(e10, expected_ids)
    # Score the diverse heads a second time on the structural slots
    pixels, masks = cache(_RAD_E11_SLOTS, _RAD_E11_CROP_MM, "test-v48-pass2", 0.55)
    features, token_mask = _rad_encode(encoder, pixels, masks, device)
    pass2_predictions = [
        _rad_predict_head(head, features, token_mask, device) for head in e13_heads
    ]
    pass2_probability = np.mean(np.stack(pass2_predictions), axis=0)
    pass2_rank = _rad_rank_columns(pass2_probability)
    # Add the second pass to the blend
    final = e10.copy()
    final[_RAD_LABELS] = (1.0 - _RAD_V48_SECOND_ALPHA) * _rad_rank_columns(
        e10[_RAD_LABELS].to_numpy()
    ) + _RAD_V48_SECOND_ALPHA * pass2_rank
    final[_RAD_LABELS] = _rad_rank_columns(final[_RAD_LABELS].to_numpy())
    _rad_validate(final, expected_ids)
    # Record the uncalibrated branch
    V18_TRANSFORMER_RAW = final.copy()
    V18_CALIBRATOR_APPLIED = False
    V18_CAL_GATE = tuple()
    # Calibrate the gated findings when the calibrator loads
    try:
        calibrated, gate = _v18_calibrate_transformer(
            final, baseline_rank, reference_rank, pass2_rank, expected_ids
        )
        final = calibrated
        V18_TRANSFORMER_CAL = final.copy()
        V18_CALIBRATOR_APPLIED = True
        V18_CAL_GATE = tuple(sorted(gate))
        print(
            "[V18] 88-feature transformer calibration applied to: "
            + ", ".join(sorted(gate)),
            flush=True,
        )
    # Keep the raw branch when the calibrator cannot run
    except Exception as exc:
        print(
            "[V18] calibration skipped safely; raw transformer kept: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
    # Write the branch back over the submission
    _rad_validate(final, expected_ids)
    final.to_csv(primary, index=False)
# Run the RadImageNet stage under its declared slot scheme
def run_radimagenet_stage():
    # Adopt the fluid sensitive slots this stage starts from
    globals().update(
        SLOTS=[
            ("SAG_FS", "Sagittal", None, True),
            ("COR_FS", "Coronal", None, True),
            ("AX_FS", "Axial", None, True),
        ],
        N_SLOT=3,
        CACHE_SLICES=8,
    )
    # Score the branch and rewrite the submission
    _rad_main()
# Set the geometry of the CoAtNet branch
COAT_IMG = 336
COAT_CROP_MM = 140.0
# Fill five fixed slots per study for a stack of sixty four images
COAT_SLOTS = [
    ("Sagittal", 1, 18),
    ("Sagittal", 0, 14),
    ("Coronal", 1, 12),
    ("Coronal", 0, 8),
    ("Axial", -1, 12),
]
COAT_MAXS = sum(s[2] for s in COAT_SLOTS)
# Score every window position the volume holds
COAT_K_EVAL = 62
COAT_NORM = "imagenet"
# Reuse the official finding order
COAT_LABELS = list(TARGETS)
# Store the ImageNet normalisation constants
COAT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
COAT_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# Declare the primary MaxSpan checkpoint
COAT_ARMS = [
    {
        "file": "raptor_ft_coatnet_v5_full_swa.pt",
        "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
        "res": 384,
        "w": 1.0,
    },
]
# Declare the complementary WideDense checkpoint
COAT_LEGACY_ARM = {
    "file": "raptor_ft_coatnet_v4_full.pt",
    "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "res": 384,
    "k_eval": 42,
    "span_lo": 0.06,
    "span_hi": 0.94,
}
# Set the slice span each checkpoint was fitted on
COAT_PRIMARY_SPAN_LO = 0.02
COAT_PRIMARY_SPAN_HI = 0.98
# Build the timm backbone the checkpoint declares
def coat_build_backbone(arch, pretrained=False):
    # Route the conv attention hybrids away from the transformer path
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(
        k in arch for k in ("vit", "deit", "dinov2", "eva", "beit")
    )
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    # Pool the class token for a transformer and the feature map otherwise
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)
# Attend over the windows with separate weights per finding
class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        # Keep the backbone and normalise its features
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        # Score every window against every finding
        self.att = nn.Sequential(
            nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n)
        )
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n
    def encode(self, x):
        # Encode every window of the study
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)
    def head(self, feats):
        # Pool the windows with one attention map per finding
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb
    def forward(self, x):
        return self.head(self.encode(x))
# Load one CoAtNet arm from its checkpoint
def coat_load_model(pt_path, arch_default, res_default, device, ngpu=1):
    # Read the architecture and resolution the checkpoint declares
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", arch_default)
    ck_res = int(ck.get("res", res_default))
    # Rebuild the arm and load its weights
    bb = coat_build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    # Release the checkpoint before the next arm is built
    del ck
    gc.collect()
    return model, ck_res
# List the window centres of one volume
def coat_eval_centers(mask, D, k):
    # Work out which slices carry data
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    # Keep the centres that hold a full three slice window
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    # Space the requested number of centres evenly
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]
# Stack three neighbouring slices into every window
def coat_eval_windows(vol, mask, k, res, norm=COAT_NORM):
    # Choose the window centres
    D = vol.shape[0]
    cs = coat_eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    # Build every window at the model resolution
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear", align_corners=False)[0]
        wins[j] = t.numpy()
    # Normalise the stack
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - COAT_MEAN) / COAT_STD
    return x
# Score one study and retry in full precision when half precision fails
@torch.no_grad()
def coat_infer_probs(model, xwins, device):
    # Move the windows onto the device
    x = xwins.unsqueeze(0).to(device, non_blocking=True)
    use_cuda = str(device).startswith("cuda")
    # Run the arm once
    def _forward():
        return torch.sigmoid(model(x).float())[0].cpu().numpy()
    # Prefer half precision on a device
    if use_cuda:
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                return _forward()
        # Fall back to full precision after clearing the cache
        except RuntimeError as error:
            try:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print(
                f"[DINOsaur V4.2] {device} fp16 retry in fp32: {type(error).__name__}",
                flush=True,
            )
            return _forward()
    return _forward()
# Rank every column into the unit range
def coat_rankpct(x):
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))
# Build the DICOM reading closures of the CoAtNet branch
def coat_make_reader():
    # Order the slices of a series and read its median spacing
    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm")
        recs = []
        ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, "ImageOrientationPatient", None)
                ipp = getattr(h, "ImagePositionPatient", None)
                # Project the slice onto its normal when the geometry is present
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], float)
                    c = np.array(iop[3:], float)
                    n = np.cross(r, c)
                    pos = float(np.dot(np.array(ipp, float), n))
                # Fall back to the instance number
                else:
                    pos = float(getattr(h, "InstanceNumber", 0) or 0)
                ps = getattr(h, "PixelSpacing", None)
                ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps)
                recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps
    # Decode one slice through its modality lookup table
    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        # Invert an inverted photometric interpretation
        if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
            a = a.max() - a
        return a
    # Crop a fixed millimetre box and resize it
    def mm_crop_resize(a, ps):
        h, w = a.shape
        cpx = int(round(COAT_CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w))
        y0 = (h - cpx) // 2
        x0 = (w - cpx) // 2
        a = a[y0 : y0 + cpx, x0 : x0 + cpx]
        return cv2.resize(a, (COAT_IMG, COAT_IMG), interpolation=cv2.INTER_AREA)
    return order_and_meta, read_px, mm_crop_resize
# Choose the series that fills one CoAtNet slot
def coat_pick_series_for_slot(rows, plane, fluid, used):
    # Keep the unused series of the right plane
    cands = [
        r
        for r in rows
        if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used
    ]
    # Prefer the requested fluid sensitivity
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None
# Fill one span variant of the volume from the shared pixel cache
def coat_fill_variant_volume(target_volume, offset, picks, pixel_cache, files, med_ps, mm_crop_resize):
    # Collect the chosen slices and their spacing
    arrays = []
    spacings = []
    for position in picks:
        position = min(int(position), len(files) - 1)
        file_path, spacing = files[position]
        arrays.append(pixel_cache.get(position))
        spacings.append(spacing if spacing > 0 else med_ps)
    # Take the window from every slice this variant uses
    valid = [array for array in arrays if array is not None]
    if valid:
        all_pixels = np.concatenate([array.ravel() for array in valid])
        low, high = np.percentile(all_pixels, [2.0, 98.0])
    else:
        low, high = 0.0, 1.0
    # Write every slice into the volume
    for local_index, (array, spacing) in enumerate(zip(arrays, spacings)):
        output_index = offset + local_index
        if output_index >= COAT_MAXS:
            break
        if array is None:
            continue
        normalized = np.clip((array - low) / (high - low + 1e-6), 0, 1)
        normalized = mm_crop_resize(normalized, spacing)
        target_volume[output_index] = (normalized * 255).astype(np.uint8)
# Build the MaxSpan and legacy volumes of one study from a single DICOM pass
def coat_build_study_pair(sid, ser_records, tsdir, reader):
    # Unpack the reading closures
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    # Allocate both variants
    primary_volume = np.zeros((COAT_MAXS, COAT_IMG, COAT_IMG), np.uint8)
    legacy_volume = np.zeros_like(primary_volume)
    used = set()
    offset = 0
    # Fill every slot in the declared order
    for plane, fluid, count in COAT_SLOTS:
        record = coat_pick_series_for_slot(rows, plane, fluid, used)
        # Leave the slot empty when no series matches
        if record is None:
            offset += count
            continue
        used.add(record["SeriesInstanceUID"])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{record['SeriesInstanceUID']}")
        if not files:
            offset += count
            continue
        number = len(files)
        # Space the primary picks across the wide span
        primary_low = int(number * COAT_PRIMARY_SPAN_LO)
        primary_high = int(number * COAT_PRIMARY_SPAN_HI) - 1
        primary_high = max(primary_high, primary_low)
        # Space the legacy picks across the narrow span
        legacy_low = int(number * float(COAT_LEGACY_ARM["span_lo"]))
        legacy_high = int(number * float(COAT_LEGACY_ARM["span_hi"])) - 1
        legacy_high = max(legacy_high, legacy_low)
        # Choose the slices of both variants
        if number > 1:
            primary_picks = np.linspace(primary_low, primary_high, count).round().astype(int)
            legacy_picks = np.linspace(legacy_low, legacy_high, count).round().astype(int)
        else:
            primary_picks = np.zeros(count, dtype=int)
            legacy_picks = np.zeros(count, dtype=int)
        # Decode every slice either variant needs exactly once
        required_positions = sorted(set(primary_picks.tolist() + legacy_picks.tolist()))
        pixel_cache = {}
        for position in required_positions:
            position = min(int(position), number - 1)
            file_path, _ = files[position]
            try:
                pixel_cache[position] = read_px(file_path)
            except Exception:
                pixel_cache[position] = None
        # Fill both variants from the shared cache
        coat_fill_variant_volume(
            primary_volume, offset, primary_picks, pixel_cache, files, med_ps, mm_crop_resize
        )
        coat_fill_variant_volume(
            legacy_volume, offset, legacy_picks, pixel_cache, files, med_ps, mm_crop_resize
        )
        offset += count
        if offset >= COAT_MAXS:
            break
    # Mark which slices carry data
    primary_mask = (primary_volume.reshape(COAT_MAXS, -1).sum(1) > 0).astype(np.uint8)
    legacy_mask = (legacy_volume.reshape(COAT_MAXS, -1).sum(1) > 0).astype(np.uint8)
    return (primary_volume, primary_mask, legacy_volume, legacy_mask)
# Locate the test root of the CoAtNet branch
def coat_find_test_root():
    # Prefer the two standard mounts
    cands = [
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "/kaggle/input/rsna-knee-abnormality-detection",
    ]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    # Fall back to any mount that holds a test table and its images
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (
            os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")
        ):
            return d
    # Fall back to any mount that holds a test table
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")
# Locate one arm checkpoint under the attached datasets
def coat_find_weight_file(fname, required=True):
    # Try the known dataset paths first
    direct = [
        f"/kaggle/input/raptor-knee-arms/{fname}",
        f"/kaggle/input/raptor-knee-arms/1/{fname}",
        f"/kaggle/input/raptor-cnn336/{fname}",
    ]
    for path in direct:
        if os.path.exists(path):
            return path
    # Search every attached dataset except the competition itself
    for directory in sorted(glob.glob("/kaggle/input/*/")):
        if "competition" in directory.lower():
            continue
        hits = glob.glob(os.path.join(directory, "**", fname), recursive=True)
        if hits:
            return hits[0]
    # Fail only when the arm is required
    if required:
        raise RuntimeError(f"{fname} not found under /kaggle/input")
    return None
# Run the CoAtNet branch and write its own submission
def run_coatnet_stage():
    # Report the devices this branch will use
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device {dev} | gpus {ngpu} | torch {torch.__version__}", flush=True)
    # Locate the test tables and the series directory
    coat_root = coat_find_test_root()
    tsdir = coat_root + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = coat_root + "/test_images"
    print("test root:", coat_root, "| series dir:", tsdir, flush=True)
    # Read the test tables
    test = pd.read_csv(coat_root + "/test.csv")
    test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(coat_root + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)
    # Take the column order from the sample submission when it is present
    sub_cols = ["StudyInstanceUID"] + COAT_LABELS
    ssub = os.path.join(coat_root, "sample_submission.csv")
    if os.path.exists(ssub):
        sub_cols = list(pd.read_csv(ssub, nrows=1).columns)
    # Prepare the reader and the prediction tables
    reader = coat_make_reader()
    number_studies = len(test_ids)
    primary_predictions = np.full((number_studies, len(COAT_LABELS)), 0.5, np.float32)
    legacy_predictions = np.full_like(primary_predictions, 0.5)
    legacy_success = np.zeros(number_studies, dtype=np.bool_)
    # Put the primary arm on the first device
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        primary_device = torch.device("cuda:0")
    else:
        primary_device = torch.device("cpu")
    # Run the complement only when a second device is free
    legacy_path = coat_find_weight_file(COAT_LEGACY_ARM["file"], required=False)
    legacy_enabled = (
        legacy_path is not None
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
    )
    # Load the primary arm
    primary_path = coat_find_weight_file(COAT_ARMS[0]["file"], required=True)
    primary_model, primary_res = coat_load_model(
        primary_path, COAT_ARMS[0]["arch"], COAT_ARMS[0]["res"], primary_device
    )
    print(
        f"[DINOsaur V4.2] primary {COAT_ARMS[0]['file']} on {primary_device}",
        flush=True,
    )
    legacy_model = None
    legacy_device = None
    legacy_res = None
    # Load the complementary arm on the second device
    if legacy_enabled:
        legacy_device = torch.device("cuda:1")
        try:
            legacy_model, legacy_res = coat_load_model(
                legacy_path, COAT_LEGACY_ARM["arch"], COAT_LEGACY_ARM["res"], legacy_device
            )
            print(
                f"[DINOsaur V4.2] complement {COAT_LEGACY_ARM['file']} on {legacy_device}",
                flush=True,
            )
        # Keep the primary arm alone when the complement will not load
        except Exception as error:
            legacy_enabled = False
            legacy_model = None
            print(
                "[DINOsaur V4.2] legacy checkpoint disabled safely: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
    else:
        print(
            "[DINOsaur V4.2] legacy complement unavailable or second GPU absent; "
            "exact 0.935 Raptor retained",
            flush=True,
        )
    # Run both arms in parallel when the complement is enabled
    executor = ThreadPoolExecutor(max_workers=2) if legacy_enabled else None
    # Score every study
    for study_index, study_id in enumerate(test_ids):
        try:
            # Build both span variants from one DICOM pass
            primary_volume, primary_mask, legacy_volume, legacy_mask = coat_build_study_pair(
                study_id, SER, tsdir, reader
            )
            primary_windows = coat_eval_windows(
                primary_volume, primary_mask, k=COAT_K_EVAL, res=primary_res, norm=COAT_NORM
            )
            # Score both arms at once
            if legacy_enabled:
                legacy_windows = coat_eval_windows(
                    legacy_volume,
                    legacy_mask,
                    k=int(COAT_LEGACY_ARM["k_eval"]),
                    res=legacy_res,
                    norm=COAT_NORM,
                )
                primary_future = executor.submit(
                    coat_infer_probs, primary_model, primary_windows, primary_device
                )
                legacy_future = executor.submit(
                    coat_infer_probs, legacy_model, legacy_windows, legacy_device
                )
                primary_prediction = primary_future.result()
                # Fall back to the primary arm when the complement fails
                try:
                    legacy_prediction = legacy_future.result()
                    legacy_success[study_index] = True
                except Exception as legacy_error:
                    legacy_prediction = primary_prediction.copy()
                    print(
                        f"[DINOsaur V4.2] legacy study {study_index} fallback: "
                        f"{type(legacy_error).__name__}: {legacy_error}",
                        flush=True,
                    )
                del legacy_windows
            # Score the primary arm alone
            else:
                primary_prediction = coat_infer_probs(
                    primary_model, primary_windows, primary_device
                )
                legacy_prediction = primary_prediction.copy()
            # Bank both predictions
            primary_predictions[study_index] = primary_prediction
            legacy_predictions[study_index] = legacy_prediction
            del (
                primary_volume,
                primary_mask,
                legacy_volume,
                legacy_mask,
                primary_windows,
                primary_prediction,
                legacy_prediction,
            )
        # Leave a failed study at its neutral prediction
        except Exception as error:
            print(
                f"[DINOsaur V4.2] study {study_index} {study_id[:16]} FALLBACK "
                f"({type(error).__name__}: {error})",
                flush=True,
            )
        # Report the progress every hundred studies
        if (study_index + 1) % 100 == 0 or study_index + 1 == number_studies:
            print(
                f"[DINOsaur V4.2] {study_index + 1}/{number_studies} | "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
    # Release both arms
    if executor is not None:
        executor.shutdown(wait=True)
    del primary_model
    if legacy_model is not None:
        del legacy_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Rank the primary predictions
    primary_rank = coat_rankpct(np.clip(primary_predictions, 0, 1))
    raptor_rank = primary_rank.copy()
    legacy_fraction = float(legacy_success.mean()) if legacy_enabled else 0.0
    # Mix the complement into the findings the primary arm did not gain on
    if legacy_enabled and legacy_fraction >= 0.98:
        legacy_rank = coat_rankpct(np.clip(legacy_predictions, 0, 1))
        complement_weight = {
            "MCL": 0.16,
            "Medial OA": 0.10,
            "PF OA": 0.12,
            "Effusion": 0.10,
            "Synovitis": 0.16,
            "Baker's": 0.12,
            "Contusion": 0.12,
        }
        complement_log = []
        # Damp the weight when the two arms agree too much or too little
        for target_index, target in enumerate(COAT_LABELS):
            weight = float(complement_weight.get(target, 0.0))
            if weight <= 0:
                continue
            correlation = float(
                np.corrcoef(primary_rank[:, target_index], legacy_rank[:, target_index])[0, 1]
            )
            if not np.isfinite(correlation):
                weight = 0.0
            elif correlation > 0.992:
                weight *= 0.50
            elif correlation < 0.65:
                weight *= 0.40
            if weight <= 0:
                continue
            raptor_rank[:, target_index] = (1.0 - weight) * primary_rank[
                :, target_index
            ] + weight * legacy_rank[:, target_index]
            complement_log.append((target, weight, correlation))
        raptor_rank = coat_rankpct(raptor_rank)
        print(
            "[DINOsaur V4.2] legacy complement: "
            + "; ".join(
                f"{target}=w{weight:.3f},corr={correlation:.3f}"
                for (target, weight, correlation) in complement_log
            ),
            flush=True,
        )
    # Keep the primary arm alone when the complement did not cover the test set
    else:
        print(
            f"[DINOsaur V4.2] legacy success={legacy_fraction:.3f}; exact primary Raptor used",
            flush=True,
        )
    # Repair any non finite rank before writing
    ranks = raptor_rank
    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5
    # Write the CoAtNet submission
    sub = pd.DataFrame(ranks.astype(np.float32), columns=COAT_LABELS)
    sub.insert(0, "StudyInstanceUID", test_ids)
    sub = sub[sub_cols]
    assert list(sub.columns) == sub_cols, "column order drift"
    assert sub["StudyInstanceUID"].tolist() == test_ids, "row identity drift"
    assert np.isfinite(sub[COAT_LABELS].values).all()
    out = str(WORKING / "submission_coatnet.csv")
    sub.to_csv(out, index=False)
    print("wrote", out, "|", len(sub), "rows x", len(sub.columns), "cols", flush=True)
    print(sub.head().to_string(index=False), flush=True)
    print(f"DONE {time.time() - t0:.0f}s", flush=True)
# Run the CoAtNet branch without letting it end the run
def run_coatnet_stage_safely():
    # Keep the transformer submission when this branch fails
    try:
        run_coatnet_stage()
    except Exception as coat_exc:
        print(
            "CoAtNet branch failed; retaining transformer submission: "
            f"{type(coat_exc).__name__}: {coat_exc}",
            flush=True,
        )
        traceback.print_exc()
# Blend the CoAtNet branch into the transformer submission
def blend_coatnet_into_submission():
    # Stop when the CoAtNet branch produced nothing
    blend_transformer_path = WORKING / "submission.csv"
    blend_coatnet_path = WORKING / "submission_coatnet.csv"
    if not blend_coatnet_path.is_file():
        print(
            "CoAtNet output unavailable; submission.csv remains the validated "
            "transformer ensemble",
            flush=True,
        )
        return
    # Read both submissions
    blend_transformer = pd.read_csv(blend_transformer_path, dtype={"StudyInstanceUID": str})
    blend_coatnet = pd.read_csv(blend_coatnet_path, dtype={"StudyInstanceUID": str})
    blend_labels = [c for c in blend_transformer.columns if c != "StudyInstanceUID"]
    # Refuse to blend two submissions that do not line up
    if blend_coatnet.columns.tolist() != blend_transformer.columns.tolist():
        raise RuntimeError("CoAtNet/transformer submission schema mismatch")
    if blend_coatnet["StudyInstanceUID"].tolist() != blend_transformer["StudyInstanceUID"].tolist():
        raise RuntimeError("CoAtNet/transformer study order mismatch")
    # Rank both branches
    blend_tr = blend_transformer[blend_labels].rank(method="average", pct=True)
    blend_cr = blend_coatnet[blend_labels].rank(method="average", pct=True)
    blend_output = blend_transformer.copy()
    # Start from an even split on every finding
    coatnet_weight = {label: 0.50 for label in blend_labels}
    # Lean on the CoAtNet checkpoint where its gains were measured
    if V18_CALIBRATOR_APPLIED:
        coatnet_weight.update(
            {
                "ACL": 0.53,
                "Medial Meniscus": 0.56,
                "Lateral Meniscus": 0.61,
                "Lateral OA": 0.55,
                "Fracture": 0.61,
            }
        )
    # Keep the proven weights as the risk fallback
    base_0935 = {label: 0.50 for label in blend_labels}
    base_0935.update(
        {"Medial Meniscus": 0.52, "Lateral Meniscus": 0.54, "Fracture": 0.54}
    )
    # Pull a weight back towards the fallback when the branches agree too much or too little
    for label in blend_labels:
        correlation = float(
            np.corrcoef(
                blend_tr[label].to_numpy(np.float64), blend_cr[label].to_numpy(np.float64)
            )[0, 1]
        )
        if not np.isfinite(correlation):
            coatnet_weight[label] = base_0935[label]
        elif correlation > 0.992:
            coatnet_weight[label] = 0.65 * coatnet_weight[label] + 0.35 * base_0935[label]
        elif correlation < 0.60:
            coatnet_weight[label] = 0.50 * coatnet_weight[label] + 0.50 * base_0935[label]
    # Blend every finding at its own weight
    for label in blend_labels:
        cw = float(coatnet_weight[label])
        blend_output[label] = (1.0 - cw) * blend_tr[label] + cw * blend_cr[label]
    blend_output[blend_labels] = blend_output[blend_labels].rank(method="average", pct=True)
    # Report the weights that moved off the even split
    print(
        "[DINOsaur V4.2] CoAtNet target weights: "
        + ", ".join(
            f"{label}={coatnet_weight[label]:.2f}"
            for label in blend_labels
            if coatnet_weight[label] != 0.50
        ),
        flush=True,
    )
    # Refuse to write an invalid prediction
    blend_values = blend_output[blend_labels].to_numpy(np.float64)
    if not np.isfinite(blend_values).all() or blend_values.min() < 0 or blend_values.max() > 1:
        raise RuntimeError("invalid blended prediction values")
    blend_output.to_csv(blend_transformer_path, index=False)
    print(
        f"final submission.csv = DINOsaur V4.2 dual-checkpoint target fusion; "
        f"{blend_output.shape}",
        flush=True,
    )
    # Remove the intermediate submissions
    for v18_temp in (
        WORKING / "submission_coatnet.csv",
        WORKING / "submission_transformer_0920.csv",
    ):
        try:
            if v18_temp.is_file():
                v18_temp.unlink()
        except OSError:
            pass
# Fuse the branches behind a fail-safe that never erases a valid submission
def run_final_fusion():
    # Back the current submission up
    d42_primary = WORKING / "submission.csv"
    d42_backup = WORKING / ".d42_transformer_backup.csv"
    if d42_primary.is_file():
        shutil.copy2(d42_primary, d42_backup)
    # Blend the two independently validated rank predictors
    try:
        blend_coatnet_into_submission()
    # Restore the backup when the fusion fails
    except Exception as d42_error:
        print(
            "[DINOsaur V4.2] final fusion failed; restoring calibrated transformer "
            f"submission: {type(d42_error).__name__}: {d42_error}",
            flush=True,
        )
        traceback.print_exc()
        if d42_backup.is_file():
            shutil.copy2(d42_backup, d42_primary)
    # Remove the backup either way
    finally:
        try:
            if d42_backup.is_file():
                d42_backup.unlink()
        except OSError:
            pass
    # Refuse to end the run without a submission
    if not d42_primary.is_file():
        raise RuntimeError("submission.csv missing after V4.2 fail-safe")
# Define the main function
def main():
    # Score the DINOv2 transformer ensemble
    run_dinov2_stage()
    # Blend the knee MRI fold checkpoints into the submission
    run_fold_stage()
    # Blend the RadImageNet branches and calibrate the gated findings
    run_radimagenet_stage()
    # Score the CoAtNet Raptor arms into their own submission
    run_coatnet_stage_safely()
    # Fuse the CoAtNet branch into the final submission
    run_final_fusion()
# Call the main function
if __name__ == "__main__":
    main()