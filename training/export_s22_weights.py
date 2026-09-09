"""Strip S22 fold checkpoints to fp16 state_dicts and verify a CUDA forward pass.

Reads the Quadro run `best.pt` files (full training blobs) and writes fold0.pt
.. fold4.pt plus manifest.json. Tokens stay in queue files; do not print them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection")
DEFAULT_SRC = PROJECT / "runs/20260908T1635Z-s22-resnet18-25d/checkpoints"
DEFAULT_OUT = PROJECT / "export/s22-resnet18-25d-folds"
CODE_CANDIDATES = [
    PROJECT / "code/pixel-cache-v1",
    PROJECT / "training",
    Path(__file__).resolve().parent,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    for code in CODE_CANDIDATES:
        if (code / "s22_resnet18_25d.py").is_file() and str(code) not in sys.path:
            sys.path.insert(0, str(code))
            break
    from s22_resnet18_25d import TARGETS, build_model

    import torch

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = []
    for fold in range(5):
        ckpt_path = src / f"fold{fold}" / "best.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = blob["model"]
        fp16 = {key: value.detach().cpu().half().contiguous() for key, value in state.items()}
        dest = out / f"fold{fold}.pt"
        torch.save(fp16, dest)
        model = build_model(n_targets=12, pretrained=False).to(device)
        model.load_state_dict({key: value.float() for key, value in fp16.items()})
        model.eval()
        with torch.no_grad():
            dummy_x = torch.zeros(2, 3, 6, 224, 224, device=device)
            dummy_mask = torch.ones(2, 3, device=device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(dummy_x, dummy_mask)
        if tuple(logits.shape) != (2, 12) or not torch.isfinite(logits).all():
            raise RuntimeError(f"fold{fold} forward failed: {tuple(logits.shape)}")
        folds.append(
            {
                "fold": fold,
                "file": dest.name,
                "sha256": sha256_file(dest),
                "bytes": dest.stat().st_size,
                "src_val_macro_auc": blob.get("val_macro_auc"),
                "src_epoch": blob.get("epoch"),
                "n_tensors": len(fp16),
            }
        )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "bytes": dest.stat().st_size,
                    "sha256": folds[-1]["sha256"],
                    "device": str(device),
                    "logit_mean": float(logits.float().mean().cpu()),
                }
            ),
            flush=True,
        )

    manifest = {
        "schema_version": "s22_resnet18_25d_folds_fp16_v1",
        "experiment_id": "S22",
        "backbone": "torchvision.resnet18",
        "in_channels": 6,
        "n_targets": 12,
        "targets": TARGETS,
        "img": 224,
        "n_slices": 9,
        "crop_mm": 130,
        "window": [0.35, 0.65],
        "rgb_slice_idx": [1, 4, 7],
        "dropout": 0.2,
        "seed": 2026,
        "dtype": "float16",
        "source_run": str(src.parent),
        "exported_at_utc": utc_now(),
        "verify_device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "folds": folds,
    }
    man_path = out / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(out), "manifest_sha256": sha256_file(man_path)}, indent=2))


if __name__ == "__main__":
    main()
