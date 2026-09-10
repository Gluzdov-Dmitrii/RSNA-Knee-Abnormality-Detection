"""Write 20-40 compressed JPEG slices labeled by pathology for visual review.

Runs on NSU against PIXEL_CACHE_V1. Do not commit the JPEGs: they are derived
competition MRI and must stay local / gitignored.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pixel_cache_v1 import TARGETS, PixelCacheV1, default_pilkwang_csv

SLOT = ["SAG_FLUID", "COR_FLUID", "AX_FLUID", "SAG_STRUCT", "COR_STRUCT", "AX_STRUCT"]
PLANE_FOR = {
    "ACL": 0,
    "MCL": 1,
    "Medial Meniscus": 1,
    "Lateral Meniscus": 1,
    "Medial OA": 1,
    "Lateral OA": 1,
    "PF OA": 2,
    "Effusion": 0,
    "Synovitis": 2,
    "Baker's": 0,
    "Contusion": 0,
    "Fracture": 0,
    "Healthy": 0,
    "TypicalMixed": 0,
}
MID = 4
SAFE = {
    "ACL": "ACL",
    "MCL": "MCL",
    "Medial Meniscus": "MedialMeniscus",
    "Lateral Meniscus": "LateralMeniscus",
    "Medial OA": "MedialOA",
    "Lateral OA": "LateralOA",
    "PF OA": "PFOA",
    "Effusion": "Effusion",
    "Synovitis": "Synovitis",
    "Baker's": "Bakers",
    "Contusion": "Contusion",
    "Fracture": "Fracture",
    "Healthy": "Healthy",
    "TypicalMixed": "TypicalMixed",
}


def to_u8(plane: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(plane)


def montage(pixels: np.ndarray, mask: np.ndarray, highlight: int) -> Image.Image:
    panels = []
    for slot in range(3):
        if mask[slot]:
            img = Image.fromarray(to_u8(pixels[slot, MID]), mode="L").convert("RGB")
        else:
            img = Image.new("RGB", (224, 224), (20, 20, 20))
        if slot == highlight:
            draw = ImageDraw.Draw(img)
            draw.rectangle([1, 1, 222, 222], outline=(255, 210, 40), width=3)
        panels.append(img)
    canvas = Image.new("RGB", (224 * 3 + 8, 224 + 36), (8, 8, 8))
    canvas.paste(panels[0], (0, 36))
    canvas.paste(panels[1], (228, 36))
    canvas.paste(panels[2], (456, 36))
    return canvas


def caption(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 34], fill=(8, 8, 8))
    draw.text((6, 8), text[:110], fill=(240, 240, 240), font=font)
    return img


def findings(row: dict) -> list[str]:
    hits = []
    for name in TARGETS:
        try:
            if float(row[name]) >= 0.5:
                hits.append(name)
        except (TypeError, ValueError):
            continue
    return hits


def pick_rows(table) -> list[dict]:
    chosen: list[dict] = []
    used: set[str] = set()

    def take(mask, label: str, n: int) -> None:
        subset = table.loc[mask]
        for _, row in subset.iterrows():
            uid = str(row["StudyInstanceUID"])
            if uid in used:
                continue
            chosen.append({"uid": uid, "label": label, "row": row})
            used.add(uid)
            if sum(1 for item in chosen if item["label"] == label) >= n:
                return

    n_pos = np.zeros(len(table), dtype=int)
    for name in TARGETS:
        n_pos += (table[name].to_numpy(dtype=float) >= 0.5).astype(int)
    table = table.copy()
    table["_n_pos"] = n_pos

    for name in TARGETS:
        clean = (table[name] >= 0.5) & (table["_n_pos"] <= 3)
        take(clean, name, 2)
        if sum(1 for item in chosen if item["label"] == name) < 2:
            take(table[name] >= 0.5, name, 2)
    take(table["_n_pos"] == 0, "Healthy", 4)
    take(table["_n_pos"] >= 6, "TypicalMixed", 4)
    return chosen[:40]


def main() -> None:
    out = Path("/tmp/rsna-pathology-gallery")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    cache = PixelCacheV1()
    labels = cache.join_labels(default_pilkwang_csv())
    table = labels.dropna(subset=TARGETS)
    picked = pick_rows(table)
    index_rows = []
    for i, item in enumerate(picked, start=1):
        uid = item["uid"]
        label = item["label"]
        pixels, mask = cache.get(uid)
        hits = findings(item["row"].to_dict())
        highlight = PLANE_FOR.get(label, 0)
        if not mask[highlight]:
            for alt in range(3):
                if mask[alt]:
                    highlight = alt
                    break
        img = montage(pixels, mask, highlight)
        other = [SAFE.get(name, name) for name in hits if name != label]
        other_s = ",".join(other[:4]) if other else "none"
        title = f"{SAFE.get(label, label)} | also {other_s} | {SLOT[highlight]} mid"
        img = caption(img, title)
        fname = f"{i:02d}_{SAFE.get(label, label)}.jpg"
        dest = out / fname
        img.save(dest, format="JPEG", quality=82, optimize=True)
        index_rows.append(
            {
                "file": fname,
                "label": label,
                "highlight": SLOT[highlight],
                "findings": ";".join(hits) if hits else "none",
                "n_findings": len(hits),
                "uid_tail": uid[-12:],
            }
        )
        print(fname, flush=True)
    (out / "index.csv").write_text(
        "file,label,highlight,n_findings,findings,uid_tail\n"
        + "".join(
            f"{r['file']},{r['label']},{r['highlight']},{r['n_findings']},{r['findings']},{r['uid_tail']}\n"
            for r in index_rows
        ),
        encoding="utf-8",
    )
    readme = [
        "# Pathology slice gallery",
        "",
        "PIXEL_CACHE_V1 224px, crop 130 mm, mid slice (index 4 of 9).",
        "Left-to-right: sagittal / coronal / axial **fluid** series.",
        "Yellow box = preferred plane for that finding.",
        "Labels are Pilkwang weak labels from reports, not expert pixel maps.",
        "Do not publish or commit these JPEGs.",
        "",
        f"n={len(index_rows)}",
        "",
    ]
    (out / "README.txt").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({"n": len(index_rows), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
