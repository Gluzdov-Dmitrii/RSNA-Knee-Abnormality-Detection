"""Audit RSNA study predictions without training or optimizing ensemble weights.

Python standard library only. ROC-AUC uses the Mann-Whitney rank statistic with
average ranks for ties. Run --help for the required provenance declarations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
WARNINGS = [
    "These scores use Pilkwang weak labels, not hidden competition ground truth; they are not LB estimates.",
    "This evaluator checks file consistency, not whether each checkpoint excluded the evaluated studies during training.",
    "FOLDS_V1 separates StudyInstanceUID; patient and site independence have not been demonstrated.",
    "Legacy AUC thresholds soft labels at >=0.5, thereby treating UNK=0.28 as negative.",
    "Known-only AUC uses only YES/NO verdicts; this changes case mix. Compare it together with target coverage.",
    "Existing S22/S24 fold checkpoints were selected by validation epoch on the same fold; this may add selection optimism.",
    "Existing rebuilt_oof files are not OOF predictions of the SPRINT943 public ensemble or CF01.",
    "Public competition checkpoints evaluated on train studies must not be called OOF without fold-specific exclusion evidence.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def read_table(path: Path, required: list[str]) -> tuple[dict[str, dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate CSV column names")
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = {}
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: malformed CSV row")
            key = row[UID]
            if not key or key.strip() != key:
                raise ValueError(f"{path}:{line}: blank or whitespace-padded UID")
            if key in rows:
                raise ValueError(f"{path}:{line}: duplicate UID {key}")
            rows[key] = row
    if not rows:
        raise ValueError(f"{path}: empty table")
    return rows, fields


def finite_number(value: str, context: str, probability: bool = False) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{context}: not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: non-finite value")
    if probability and not 0 <= number <= 1:
        raise ValueError(f"{context}: probability/label outside [0, 1]: {number}")
    return number


def fold_number(value: str, context: str) -> int:
    number = finite_number(value, context)
    if number < 0 or not number.is_integer():
        raise ValueError(f"{context}: fold must be a nonnegative integer")
    return int(number)


def auc_binary(truth: list[bool], scores: list[float]) -> float | None:
    """Exact binary ROC-AUC including tied predictions; undefined => None."""
    if len(truth) != len(scores):
        raise ValueError("truth and score lengths differ")
    positives = sum(truth)
    negatives = len(truth) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        rank_sum += average_rank * sum(truth[order[i]] for i in range(start, end))
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def macro_summary(values: list[float | None]) -> dict:
    defined = [value for value in values if value is not None]
    mean = sum(defined) / len(defined) if defined else None
    return {
        "macro_auc": mean if len(defined) == len(TARGETS) else None,
        "macro_auc_defined_targets_only": mean,
        "defined_targets": len(defined),
        "total_targets": len(TARGETS),
    }


def metric_block(keys: list[str], predictions: dict, labels: dict) -> dict:
    per_target = {}
    for target in TARGETS:
        legacy_y = [labels[key][target] >= 0.5 for key in keys]
        all_scores = [predictions[key][target] for key in keys]
        known_keys = [key for key in keys if labels[key][target + "__verdict"] in {"YES", "NO"}]
        known_y = [labels[key][target + "__verdict"] == "YES" for key in known_keys]
        known_scores = [predictions[key][target] for key in known_keys]
        verdicts = Counter(labels[key][target + "__verdict"] for key in keys)
        per_target[target] = {
            "legacy_auc": auc_binary(legacy_y, all_scores),
            "legacy_n": len(keys),
            "legacy_positives": sum(legacy_y),
            "legacy_negatives": len(keys) - sum(legacy_y),
            "known_only_auc": auc_binary(known_y, known_scores),
            "known_n": len(known_keys),
            "known_positives": verdicts["YES"],
            "known_negatives": verdicts["NO"],
            "unknown_n": verdicts["UNK"],
            "known_coverage": len(known_keys) / len(keys) if keys else None,
        }
    return {
        "labeled_studies": len(keys),
        "legacy": macro_summary([row["legacy_auc"] for row in per_target.values()]),
        "known_only": macro_summary([row["known_only_auc"] for row in per_target.values()]),
        "per_target": per_target,
    }


def evaluate(args: argparse.Namespace) -> dict:
    if not args.provenance.strip():
        raise ValueError("--provenance must describe the model/checkpoint source and prediction procedure")
    prediction_rows, prediction_columns = read_table(args.predictions, [UID, "fold", *TARGETS])
    fold_rows, _ = read_table(args.folds, [UID, "fold"])
    label_rows, _ = read_table(args.labels, [UID, *TARGETS, *[t + "__verdict" for t in TARGETS]])
    fold_ids = {key: fold_number(row["fold"], f"folds/{key}") for key, row in fold_rows.items()}
    unexpected = set(prediction_rows) - set(fold_rows)
    if unexpected:
        raise ValueError(f"predictions contain {len(unexpected)} UIDs absent from folds: {sorted(unexpected)[:3]}")
    missing_predictions = sorted(set(fold_rows) - set(prediction_rows))
    if missing_predictions and not args.allow_partial:
        raise ValueError(f"predictions miss {len(missing_predictions)} fold UIDs; use --allow-partial only for declared screens")
    for key, row in prediction_rows.items():
        if fold_number(row["fold"], f"predictions/{key}") != fold_ids[key]:
            raise ValueError(f"prediction fold mismatch for {key}")
        for target in TARGETS:
            row[target] = finite_number(row[target], f"predictions/{key}/{target}", args.score_type == "probability")
    for key, row in label_rows.items():
        for target in TARGETS:
            row[target] = finite_number(row[target], f"labels/{key}/{target}", True)
            verdict = row[target + "__verdict"]
            if verdict not in {"YES", "NO", "UNK"}:
                raise ValueError(f"labels/{key}/{target}: unexpected verdict {verdict!r}")
            if verdict in {"YES", "NO"} and (row[target] >= 0.5) != (verdict == "YES"):
                raise ValueError(f"labels/{key}/{target}: numeric label and verdict disagree")
    unlabeled = sorted(set(prediction_rows) - set(label_rows))
    unauthorized_unlabeled = set(unlabeled) - set(args.allow_unlabeled_uid)
    if unauthorized_unlabeled:
        raise ValueError(f"missing label UIDs need explicit --allow-unlabeled-uid: {sorted(unauthorized_unlabeled)}")
    keys = sorted(set(prediction_rows) & set(label_rows))
    if not keys:
        raise ValueError("no labeled predictions to score")
    folds = {}
    for fold in sorted(set(fold_ids.values())):
        fold_keys = [key for key in keys if fold_ids[key] == fold]
        block = metric_block(fold_keys, prediction_rows, label_rows)
        block["expected_studies"] = sum(value == fold for value in fold_ids.values())
        block["predicted_studies"] = sum(fold_ids[key] == fold for key in prediction_rows)
        folds[str(fold)] = block
    warnings = list(WARNINGS)
    if args.prediction_kind != "fold-out":
        warnings.append(f"Declared prediction kind is {args.prediction_kind}; these metrics must not be reported as OOF performance.")
    if missing_predictions:
        warnings.append("PARTIAL COVERAGE: this report evaluates a declared subset, not complete OOF.")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_kind_declared": args.prediction_kind,
        "provenance_declared": args.provenance.strip(),
        "training_exclusion_independently_verified": False,
        "patient_independence_verified": False,
        "inputs": {
            name: {"path": path_label(path), "sha256": sha256(path)}
            for name, path in [("predictions", args.predictions), ("labels", args.labels), ("folds", args.folds), ("evaluator", Path(__file__))]
        },
        "protocol": {
            "target_order": TARGETS,
            "legacy_positive_threshold": 0.5,
            "known_only_positive_verdict": "YES",
            "known_only_negative_verdict": "NO",
            "known_only_excluded_verdict": "UNK",
            "macro_weighting": "equal weight per target; null if any target AUC is undefined",
            "fold_aggregation": "pooled study AUC is primary; per-fold AUC is diagnostic, not averaged",
            "score_type": args.score_type,
            "blend_weights_optimized": False,
        },
        "integrity": {
            "fold_studies": len(fold_rows),
            "prediction_studies": len(prediction_rows),
            "labeled_studies": len(keys),
            "full_fold_coverage": not missing_predictions,
            "missing_prediction_uids": missing_predictions,
            "excluded_unlabeled_uids": unlabeled,
            "allowed_unlabeled_uids": sorted(set(args.allow_unlabeled_uid)),
            "label_studies_outside_folds": len(set(label_rows) - set(fold_rows)),
            "ignored_prediction_columns": sorted(set(prediction_columns) - {UID, "fold", *TARGETS}),
            "uid_uniqueness": "passed for predictions, labels and folds",
            "fold_match": "passed",
            "finite_values": "passed for every prediction and numeric label",
        },
        "pooled": metric_block(keys, prediction_rows, label_rows),
        "folds": folds,
        "warnings": warnings,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--predictions", type=Path, required=True, help="CSV: StudyInstanceUID,fold,12 target columns")
    result.add_argument("--provenance", required=True, help="Explicit source of weights, training split and prediction procedure; stored verbatim")
    result.add_argument("--prediction-kind", choices=["fold-out", "in-sample", "unknown"], required=True, help="Declaration, not an automatic proof of exclusion")
    result.add_argument("--labels", type=Path, default=ROOT / "data/labels/pilkwang/report_labels_v2.csv")
    result.add_argument("--folds", type=Path, default=ROOT / "ops/assets/FOLDS_V1/folds.csv")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--score-type", choices=["probability", "logit", "rank"], default="probability")
    result.add_argument("--allow-partial", action="store_true", help="Permit incomplete fold coverage; prominently marks the report")
    result.add_argument("--allow-unlabeled-uid", action="append", default=[], help="Explicit UID allowed to have no label row; repeat for each UID")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        report = evaluate(args)
        if args.output.resolve() in {args.predictions.resolve(), args.labels.resolve(), args.folds.resolve(), Path(__file__).resolve()}:
            raise ValueError("output must not overwrite an input or the evaluator")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2
    pooled = report["pooled"]
    print(json.dumps({
        "report": path_label(args.output),
        "predicted_studies": report["integrity"]["prediction_studies"],
        "labeled_studies": pooled["labeled_studies"],
        "full_fold_coverage": report["integrity"]["full_fold_coverage"],
        "legacy_macro_auc": pooled["legacy"]["macro_auc"],
        "known_only_macro_auc": pooled["known_only"]["macro_auc"],
    }, indent=2))
    print("WARNING: weak-label validation is not the LB or the public ensemble's OOF; study folds do not prove patient independence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
