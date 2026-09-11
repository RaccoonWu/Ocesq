#!/usr/bin/env python3
"""Score downstream model predictions against raw-intervention ground truth."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


STATUS_LABELS = ("supported", "conflicting", "missing", "not_applicable", "unchanged", "unknown")
COMPONENT_LABELS = ("provenance", "entity", "time", "effect", "state", "none")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def macro_f1(gold: list[str], predicted: list[str], labels: tuple[str, ...]) -> float:
    scores = []
    for label in labels:
        if label not in gold:
            continue
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return mean(scores) if scores else 0.0


def acceptable_pointer_sets(gold: dict[str, Any]) -> list[set[int]]:
    expected = set(gold["source_event_ids"])
    allowed = set(gold.get("allowed_equivalent_source_ids", []))
    if not allowed:
        return [expected]
    source_members = expected & allowed
    fixed = expected - source_members
    return [fixed | {source} for source in allowed]


def source_scores(gold: dict[str, Any], prediction: dict[str, Any]) -> dict[str, float | bool]:
    predicted = {int(value) for value in prediction.get("source_event_ids", []) if isinstance(value, int)}
    candidates = acceptable_pointer_sets(gold)
    best = None
    for expected in candidates:
        overlap = len(predicted & expected)
        precision = overlap / len(predicted) if predicted else (1.0 if not expected else 0.0)
        recall = overlap / len(expected) if expected else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = (f1, precision, recall, predicted == expected)
        if best is None or row > best:
            best = row
    assert best is not None
    return {"f1": best[0], "precision": best[1], "recall": best[2], "exact": best[3]}


def score_one(gold: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    source = source_scores(gold, prediction)
    valid_ids = set(gold.get("valid_source_event_ids", []))
    cited = [value for value in prediction.get("source_event_ids", []) if isinstance(value, int)]
    invalid = sum(value not in valid_ids for value in cited)
    expected_status = gold["expected_relation_status"]
    predicted_status = prediction.get("relation_status", "unknown")
    expected_component = gold["changed_component"]
    predicted_component = prediction.get("changed_component", "none")
    return {
        "case_token": gold["case_token"],
        "status_gold": expected_status,
        "status_predicted": predicted_status,
        "status_correct": predicted_status == expected_status,
        "component_gold": expected_component,
        "component_predicted": predicted_component,
        "component_correct": predicted_component == expected_component,
        "change_present_gold": gold["change_present"],
        "change_present_predicted": bool(prediction.get("change_present", predicted_component != "none")),
        "source_exact": source["exact"],
        "source_exact_fault": source["exact"] if gold["change_present"] else None,
        "source_precision": source["precision"],
        "source_recall": source["recall"],
        "source_f1": source["f1"],
        "source_f1_fault": source["f1"] if gold["change_present"] else None,
        "invalid_source_ids": invalid,
        "cited_source_ids": len(cited),
        "absence_case": bool(gold["scope_bound_absence_required"]),
        "absence_scope_correct": (
            prediction.get("absence_is_scope_bound") is True
            if gold["scope_bound_absence_required"] else None
        ),
        "input_tokens": prediction.get("input_tokens"),
        "output_tokens": prediction.get("output_tokens"),
        "latency_ms": prediction.get("latency_ms"),
        "cost_usd": prediction.get("cost_usd"),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_gold = [row["status_gold"] for row in rows]
    status_pred = [row["status_predicted"] for row in rows]
    component_gold = [row["component_gold"] for row in rows]
    component_pred = [row["component_predicted"] for row in rows]
    benign = [row for row in rows if not row["change_present_gold"]]
    faults = [row for row in rows if row["change_present_gold"]]
    absence = [row for row in rows if row["absence_case"]]
    cited_total = sum(row["cited_source_ids"] for row in rows)
    return {
        "cases": len(rows),
        "status_accuracy": mean(row["status_correct"] for row in rows),
        "status_macro_f1": macro_f1(status_gold, status_pred, STATUS_LABELS),
        "changed_component_accuracy": mean(row["component_correct"] for row in rows),
        "changed_component_macro_f1": macro_f1(component_gold, component_pred, COMPONENT_LABELS),
        "source_event_exact": mean(row["source_exact"] for row in rows),
        "source_event_exact_on_faults": mean(row["source_exact"] for row in faults),
        "source_event_precision": mean(row["source_precision"] for row in rows),
        "source_event_recall": mean(row["source_recall"] for row in rows),
        "source_event_f1": mean(row["source_f1"] for row in rows),
        "source_event_f1_on_faults": mean(row["source_f1"] for row in faults),
        "benign_false_positive_rate": safe_mean([
            row["change_present_predicted"] for row in benign
        ]),
        "invalid_source_id_rate": sum(row["invalid_source_ids"] for row in rows) / cited_total if cited_total else 0.0,
        "scoped_absence_accuracy": safe_mean([row["absence_scope_correct"] for row in absence]),
        "mean_input_tokens": safe_mean([row["input_tokens"] for row in rows if isinstance(row["input_tokens"], (int, float))]),
        "mean_output_tokens": safe_mean([row["output_tokens"] for row in rows if isinstance(row["output_tokens"], (int, float))]),
        "mean_latency_ms": safe_mean([row["latency_ms"] for row in rows if isinstance(row["latency_ms"], (int, float))]),
        "total_cost_usd": sum(row["cost_usd"] for row in rows if isinstance(row["cost_usd"], (int, float))),
    }


def bootstrap_difference(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    tokens = sorted(set(baseline) & set(candidate))
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        eligible = [
            token for token in tokens
            if isinstance(baseline[token].get(metric), (int, float))
            and isinstance(candidate[token].get(metric), (int, float))
        ]
        selected = [eligible[rng.randrange(len(eligible))] for _ in eligible]
        differences.append(mean(candidate[token][metric] - baseline[token][metric] for token in selected))
    ordered = sorted(differences)
    return {
        "mean_difference": mean(differences),
        "ci95_low": ordered[int(0.025 * (samples - 1))],
        "ci95_high": ordered[int(0.975 * (samples - 1))],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="NAME=JSONL")
    parser.add_argument("--case-source", type=Path, help="Restrict scoring to case tokens present in this JSONL")
    parser.add_argument("--baseline")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    gold = {row["case_token"]: row for row in read_jsonl(args.ground_truth)}
    if args.case_source:
        selected_tokens = {row["case_token"] for row in read_jsonl(args.case_source)}
        gold = {token: row for token, row in gold.items() if token in selected_tokens}
    scored: dict[str, list[dict[str, Any]]] = {}
    for spec in args.prediction:
        name, sep, path_text = spec.partition("=")
        if not sep:
            raise ValueError(f"invalid prediction spec: {spec}")
        predictions = {
            row["case_token"]: row for row in read_jsonl(Path(path_text))
            if row["case_token"] in gold
        }
        missing = sorted(set(gold) - set(predictions))
        extra = sorted(set(predictions) - set(gold))
        if missing or extra:
            raise ValueError(f"{name}: prediction coverage mismatch; missing={len(missing)}, extra={len(extra)}")
        scored[name] = [score_one(gold[token], predictions[token]) for token in sorted(gold)]

    summary = {
        "protocol": "downstream-fact-recovery-evaluation-v1",
        "ground_truth_boundary": "raw mutation and benign-rewrite manifests only",
        "model_outputs_are_ground_truth": False,
        "case_selection": str(args.case_source) if args.case_source else "all ground-truth cases",
        "representations": {name: aggregate(rows) for name, rows in scored.items()},
        "paired_bootstrap": {},
    }
    if args.baseline:
        if args.baseline not in scored:
            raise ValueError(f"unknown baseline: {args.baseline}")
        baseline = {row["case_token"]: row for row in scored[args.baseline]}
        for name, rows in scored.items():
            if name == args.baseline:
                continue
            candidate = {row["case_token"]: row for row in rows}
            summary["paired_bootstrap"][name] = {
                metric: bootstrap_difference(baseline, candidate, metric, args.bootstrap_samples, args.seed)
                for metric in ("status_correct", "component_correct", "source_exact_fault", "source_f1_fault")
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in scored.items():
        with (args.output_dir / f"{name}.scored.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
