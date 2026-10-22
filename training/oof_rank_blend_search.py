"""Rank-blend existing OOF predictions and report local weak-label AUC.

This is an analysis utility, not a submission recipe by itself. Target-wise
weight search optimizes on the same OOF labels and is therefore marked as
optimistic unless wrapped by a higher-level nested protocol.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "ops" / "tools" / "pixel_cache_v1.py").is_file() else HERE
TOOLS = ROOT / "ops" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from pixel_cache_v1 import TARGETS, sha256_file  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_member(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("members must be NAME=path")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("empty member name")
    return name, Path(path)


def parse_named_float(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("weights must be NAME=value")
    name, value = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("empty weight name")
    return name, float(value)


def auc_for(y: np.ndarray, p: np.ndarray) -> float | None:
    mask = np.isfinite(y)
    yt = (y[mask] >= 0.5).astype(np.int32)
    if mask.sum() < 8 or len(np.unique(yt)) < 2:
        return None
    return float(roc_auc_score(yt, p[mask]))


def evaluate(labels: pd.DataFrame, pred: pd.DataFrame) -> dict:
    merged = labels.merge(pred, on="StudyInstanceUID", suffixes=("_y", "_p"), how="inner", validate="one_to_one")
    scores = {}
    for target in TARGETS:
        scores[target] = auc_for(
            merged[f"{target}_y"].to_numpy(dtype=np.float64),
            merged[f"{target}_p"].to_numpy(dtype=np.float64),
        )
    values = [v for v in scores.values() if v is not None]
    return {"n_rows": int(len(merged)), "macro_auc": float(np.mean(values)), "per_target_auc": scores}


def evaluate_matrix(y: np.ndarray, p: np.ndarray) -> dict:
    scores = {}
    for idx, target in enumerate(TARGETS):
        scores[target] = auc_for(y[:, idx], p[:, idx])
    values = [v for v in scores.values() if v is not None]
    return {"n_rows": int(len(y)), "macro_auc": float(np.mean(values)), "per_target_auc": scores}


def rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["StudyInstanceUID"]].copy()
    for target in TARGETS:
        out[target] = frame[target].rank(method="average", pct=True).to_numpy(dtype=np.float64)
    return out


def weighted_rank(ranked: dict[str, pd.DataFrame], weights: dict[str, float], target_weights: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    names = list(ranked)
    base = ranked[names[0]][["StudyInstanceUID"]].copy()
    for target in TARGETS:
        if target_weights is None:
            w = np.asarray([weights[name] for name in names], dtype=np.float64)
        else:
            w = np.asarray([target_weights[target][name] for name in names], dtype=np.float64)
        denom = float(w.sum())
        if denom <= 0:
            raise ValueError(f"zero blend weight for {target}")
        vals = np.stack([ranked[name][target].to_numpy(dtype=np.float64) for name in names], axis=0)
        base[target] = (w[:, None] * vals).sum(axis=0) / denom
    return base


def simplex_grid(n: int, step: float) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    out = []
    for cuts in itertools.product(range(units + 1), repeat=n - 1):
        if sum(cuts) > units:
            continue
        last = units - sum(cuts)
        out.append(tuple([c / units for c in cuts] + [last / units]))
    return out


def local_simplex_grid(names: list[str], step: float, center: dict[str, float], radius: float) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    radius_units = int(round(radius / step))
    if radius_units < 1:
        raise ValueError("--radius must be at least one --step")
    missing = [name for name in names if name not in center]
    extra = [name for name in center if name not in names]
    if missing or extra:
        raise ValueError(f"center weights mismatch; missing={missing}, extra={extra}")
    center_units = [int(round(center[name] * units)) for name in names]
    ranges = [
        range(max(0, unit - radius_units), min(units, unit + radius_units) + 1)
        for unit in center_units
    ]
    out = []
    for combo in itertools.product(*ranges):
        if sum(combo) == units:
            out.append(tuple(unit / units for unit in combo))
    if not out:
        raise ValueError("local simplex grid is empty; increase --radius")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--member", action="append", type=parse_member, required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--skip-targetwise", action="store_true")
    parser.add_argument("--center-weight", action="append", type=parse_named_float)
    parser.add_argument("--radius", type=float, default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels)
    labels["StudyInstanceUID"] = labels["StudyInstanceUID"].astype(str)
    labels = labels[["StudyInstanceUID"] + TARGETS]

    frames = {}
    member_receipts = {}
    uid_order: list[str] | None = None
    for name, path in args.member:
        frame = pd.read_csv(path)
        frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
        frame = frame[["StudyInstanceUID"] + TARGETS].sort_values("StudyInstanceUID").reset_index(drop=True)
        if uid_order is None:
            uid_order = frame["StudyInstanceUID"].tolist()
        elif frame["StudyInstanceUID"].tolist() != uid_order:
            raise ValueError(f"UID order mismatch for {name}")
        frames[name] = frame
        member_receipts[name] = {"path": str(path), "sha256": sha256_file(path), **evaluate(labels, frame)}

    if uid_order is None:
        raise ValueError("no OOF members were provided")
    label_uids = set(labels["StudyInstanceUID"])
    common_order = [uid for uid in uid_order if uid in label_uids]
    if not common_order:
        raise ValueError("OOF members and labels have no overlapping StudyInstanceUID values")
    excluded_unlabeled = [uid for uid in uid_order if uid not in label_uids]
    labels = labels.set_index("StudyInstanceUID").loc[common_order].reset_index()
    frames = {
        name: frame.set_index("StudyInstanceUID").loc[common_order].reset_index()
        for name, frame in frames.items()
    }
    uid_order = common_order

    ranked = {name: rank_frame(frame) for name, frame in frames.items()}
    names = list(ranked)
    y_matrix = labels[TARGETS].to_numpy(dtype=np.float64)
    rank_stack = np.stack(
        [ranked[name][TARGETS].to_numpy(dtype=np.float64) for name in names],
        axis=0,
    )
    center_weights = dict(args.center_weight or [])
    if center_weights:
        grid = local_simplex_grid(names, args.step, center_weights, args.radius or args.step)
        grid_kind = "local"
    else:
        grid = simplex_grid(len(names), args.step)
        grid_kind = "full_simplex"

    global_rows = []
    best_global = None
    for combo in grid:
        weights = dict(zip(names, combo))
        pred_matrix = np.tensordot(np.asarray(combo, dtype=np.float64), rank_stack, axes=(0, 0))
        result = evaluate_matrix(y_matrix, pred_matrix)
        row = {"weights": weights, **result}
        global_rows.append(row)
        if best_global is None or row["macro_auc"] > best_global["macro_auc"]:
            best_global = row

    global_pred = weighted_rank(ranked, best_global["weights"])
    global_oof_path = args.out / "best_global_rankblend_oof.csv"
    global_pred.to_csv(global_oof_path, index=False)

    target_report = None
    target_result = None
    target_oof_path = None
    if not args.skip_targetwise:
        target_weights: dict[str, dict[str, float]] = {}
        target_rows = {}
        for target in TARGETS:
            best = None
            target_idx = TARGETS.index(target)
            for combo in grid:
                weights = dict(zip(names, combo))
                vals = np.tensordot(np.asarray(combo, dtype=np.float64), rank_stack[:, :, target_idx], axes=(0, 0))
                score = auc_for(y_matrix[:, target_idx], vals)
                row = {"target": target, "weights": weights, "auc": score}
                if best is None or (score is not None and (best["auc"] is None or score > best["auc"])):
                    best = row
            target_rows[target] = best
            target_weights[target] = best["weights"]

        target_pred = weighted_rank(ranked, {}, target_weights=target_weights)
        target_result = evaluate(labels, target_pred)
        target_oof_path = args.out / "targetwise_rankblend_oof.csv"
        target_pred.to_csv(target_oof_path, index=False)
        target_report = {
            "optimism_warning": "weights were selected on the same OOF labels; use as local upper-bound/search evidence, not unbiased promotion evidence",
            "result": target_result,
            "per_target_best": target_rows,
            "oof_csv": str(target_oof_path),
        }

    pair_corr = {}
    for a, b in itertools.combinations(names, 2):
        pair_corr[f"{a}__{b}"] = {}
        for target in TARGETS:
            pair_corr[f"{a}__{b}"][target] = float(np.corrcoef(
                ranked[a][target].to_numpy(dtype=np.float64),
                ranked[b][target].to_numpy(dtype=np.float64),
            )[0, 1])

    report = {
        "checked_at_utc": utc_now(),
        "labels": {"path": str(args.labels), "sha256": sha256_file(args.labels)},
        "members": member_receipts,
        "alignment": {
            "n_common_rows": len(common_order),
            "n_excluded_unlabeled_oof_rows": len(excluded_unlabeled),
            "excluded_unlabeled_uids": excluded_unlabeled,
        },
        "grid": {
            "kind": grid_kind,
            "step": args.step,
            "n_candidates": len(grid),
            "center_weights": center_weights or None,
            "radius": args.radius,
        },
        "skip_targetwise": args.skip_targetwise,
        "best_global": best_global,
        "best_global_oof_csv": str(global_oof_path),
        "targetwise": target_report,
        "rank_correlations": pair_corr,
    }
    (args.out / "rankblend_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        "best_global_macro_auc": best_global["macro_auc"],
        "best_global_oof_csv": str(global_oof_path),
    }
    if target_result is not None:
        summary["targetwise_macro_auc"] = target_result["macro_auc"]
        summary["targetwise_oof_csv"] = str(target_oof_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
