"""Frozen report-free geometry; credit Steven Lee, Apache2.0, see NOTICE.md.

Adapted from experiments/cache_budget/recheck/pipeline.py. Only directory selection
is generalized for live test data; numeric cache preprocessing is unchanged.
"""
import numpy as np
import pydicom
from scipy.ndimage import zoom
SLOTS=[("SAG_FLUID","Sagittal",1),("COR_FLUID","Coronal",1),("AX_FLUID","Axial",1),
       ("SAG_STRUCT","Sagittal",0),("COR_STRUCT","Coronal",0),("AX_STRUCT","Axial",0)]

def finite_vector(value, n):
    try:
        a = np.asarray(value, dtype=float)
        return a if a.shape == (n,) and np.isfinite(a).all() else None
    except (ValueError, TypeError):
        return None


def order_files(paths, audit):
    """Physical projection, then SliceLocation, then InstanceNumber; fail if unordered."""
    headers = [pydicom.dcmread(p, stop_before_pixels=True, specific_tags=[
        "ImagePositionPatient", "ImageOrientationPatient", "SliceLocation",
        "InstanceNumber", "PixelSpacing"], force=True) for p in paths]
    normal = None
    for ds in headers:
        ori = finite_vector(getattr(ds, "ImageOrientationPatient", None), 6)
        if ori is not None:
            v = np.cross(ori[:3], ori[3:])
            if np.linalg.norm(v) > 1e-6:
                normal = v / np.linalg.norm(v)
                break
    positions = [finite_vector(getattr(d, "ImagePositionPatient", None), 3) for d in headers]
    keys = None
    if normal is not None and all(p is not None for p in positions):
        keys = [float(p @ normal) for p in positions]
        audit["order_geometry"] += 1
    else:
        for tag in ["SliceLocation", "InstanceNumber"]:
            vals = [finite_vector([getattr(d, tag, None)], 1) for d in headers]
            if all(v is not None for v in vals):
                keys = [float(v[0]) for v in vals]
                audit["order_" + tag] += 1
                break
    if keys is None:
        if len(paths) == 1:
            keys = [0.]
            audit["single_slice_series"] += 1
        else:
            raise ValueError("Series lacks all physical-order fallback tags")
    idx = sorted(range(len(paths)), key=lambda i: (keys[i], paths[i].name))
    spacing = finite_vector(getattr(headers[idx[0]], "PixelSpacing", None), 2)
    if spacing is not None and not np.isclose(spacing[0], spacing[1], rtol=.001):
        audit["anisotropic_spacing_series"] += 1
    return [paths[i] for i in idx], None if spacing is None else float(spacing[0])


def sample_indices(n, n_slices, window):
    if n < 1 or n_slices < 3 or n_slices % 3:
        raise ValueError("Require nonempty series and groups of three")
    lo, hi = [int(f*(n-1)) for f in window]
    if hi <= lo:
        lo, hi = 0, n-1
    anchors = (np.linspace(lo, hi, n_slices//3).astype(int) if n_slices > 3
               else np.array([(lo+hi)//2]))
    indices = []
    for anchor in anchors:
        start = int(np.clip(anchor-1, 0, max(0,n-3)))
        indices.extend(range(start, min(start+3,n)))
    indices += [indices[-1]] * (n_slices-len(indices))
    return indices[:n_slices]


def load_series(root, frame, audit, configs, series_directory="train_series"):
    """Decode the union of required slices once. Never read reports or free text."""
    records = {}
    for name, plane, fluid in SLOTS:
        cand = frame[(frame.Anatomical_Plane == plane) & (frame.Fluid_Sensitive.astype(int) == fluid)]
        choices = []
        for row in cand.itertuples(index=False):
            d = root / series_directory / row.StudyInstanceUID / row.SeriesInstanceUID
            paths = sorted(d.glob("*.dcm")) if d.is_dir() else []
            # Keep CSV order for equal counts, matching the original locked probe.
            choices.append((len(paths), paths, str(row.SeriesInstanceUID)))
        if not choices:
            audit["absent_slots"] += 1
            continue
        _, paths, sid = max(choices, key=lambda x: x[0])
        if not paths:
            raise FileNotFoundError(f"Metadata series has no DICOM files: {sid}")
        paths, spacing = order_files(paths, audit)
        needed = sorted(set(i for c in configs for i in sample_indices(len(paths), c["n_slices"], c["window"])))
        pixels = {}
        for i in needed:
            ds = pydicom.dcmread(paths[i], force=True)
            a = ds.pixel_array.astype(np.float32)
            if a.ndim != 2 or not np.isfinite(a).all():
                raise ValueError(f"Unsupported or nonfinite pixels: {paths[i].name}")
            a = a * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
            if str(getattr(ds,"PhotometricInterpretation", "")).strip() == "MONOCHROME1":
                a = a.max() - a
                audit["mono1_inverted"] += 1
            pixels[i] = a
            audit["slices_read"] += 1
        records[name] = dict(n=len(paths), pixels=pixels, spacing=spacing, series_uid=sid)
    return records


def render_slot(rec, cfg, audit):
    idx = sample_indices(rec["n"], cfg["n_slices"], cfg["window"])
    vol = np.stack([rec["pixels"][i] for i in idx])
    px = rec["spacing"]
    if px is not None and np.isfinite(px) and px > 0:
        want = int(round(cfg["crop_mm"] / px))
        h,w = vol.shape[-2:]
        if 16 < want < min(h,w):
            half = want//2
            vol = vol[:,h//2-half:h//2+half,w//2-half:w//2+half]
            audit["crop_applied"] += 1
        else:
            audit["crop_skipped_fov"] += 1
    else:
        audit["crop_missing_spacing"] += 1
    # Same sampled-series percentile scope as the previous probe and credited geometry.
    low,high = np.percentile(vol,[1,99])
    vol = np.clip((vol-low)/max(high-low,1e-6),0,1)
    img = cfg["img"]
    vol = zoom(vol, (1,img/vol.shape[1],img/vol.shape[2]), order=1,
               mode="nearest", prefilter=False, grid_mode=True)
    result = np.clip(np.rint(vol*255),0,255).astype(np.uint8)
    assert result.shape == (cfg["n_slices"],img,img)
    return result
