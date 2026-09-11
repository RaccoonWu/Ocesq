#!/usr/bin/env python3
"""Differentially test production query semantics against an independent reference."""

from __future__ import annotations

import argparse
import ast
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from ocesq.behavior_graph.query_oracle import evaluate as evaluate_production
from ocesq.behavior_graph.query_reference import evaluate_reference


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_set(answer: dict[str, Any], collection: str) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    return {
        (tuple(map(str, row.get("node_ids", []))), tuple(sorted(map(str, row.get("violations", [])))))
        for row in answer.get(collection, [])
    }


def compare(production: dict[str, Any], reference: dict[str, Any]) -> dict[str, bool]:
    status = production.get("status") == reference.get("status")
    witnesses = _path_set(production, "witnesses") == _path_set(reference, "witnesses")
    conflicts = _path_set(production, "conflicts") == _path_set(reference, "conflicts")
    if reference.get("status") == "missing":
        left = production.get("certificate") or {}
        right = reference.get("certificate") or {}
        certificate = (
            left.get("scope") == right.get("scope")
            and left.get("complete") is True
            and right.get("complete") is True
            and left.get("candidate_count") == right.get("candidate_count")
            and left.get("searched_node_ids") == right.get("searched_node_ids")
            and left.get("searched_edge_ids") == right.get("searched_edge_ids")
            and left.get("failed_constraints") == right.get("failed_constraints")
        )
    else:
        certificate = production.get("certificate") == reference.get("certificate")
    return {
        "status_exact": status,
        "witness_exact": witnesses,
        "conflict_exact": conflicts,
        "certificate_exact": certificate,
        "answer_exact": status and witnesses and conflicts and certificate,
    }


def _mutate(answer: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    changed = deepcopy(answer)
    if changed.get("witnesses"):
        changed["witnesses"] = changed["witnesses"][:-1]
        return "remove_witness", changed
    if changed.get("conflicts"):
        changed["conflicts"] = changed["conflicts"][:-1]
        return "remove_conflict", changed
    if changed.get("certificate"):
        changed["certificate"]["complete"] = False
        return "invalidate_certificate", changed
    changed["status"] = "supported" if changed.get("status") != "supported" else "missing"
    return "change_status", changed


def _reference_import_audit(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = [name for name in imports if any(token in name for token in ("query_oracle", "ocesq", "ocei", "material"))]
    return {"reference_file": str(path), "imports": sorted(imports), "forbidden_imports": forbidden, "passed": not forbidden}


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        values.setdefault(str(row.get(key) or "unspecified"), []).append(row)
    return {
        name: {
            "cases": len(group),
            "answer_exact": mean(row["comparison"]["answer_exact"] for row in group),
            "expected_status_match": mean(row["expected_status_match"] for row in group),
            "negative_control_detected": mean(row["negative_control_detected"] for row in group),
        }
        for name, group in sorted(values.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=Path("data/query_oracle_v5"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/deterministic_query_v2"))
    args = parser.parse_args()
    manifest = _read(args.workload / "manifest.json")
    rows = []
    for item in manifest["cases"]:
        graph = _read(args.workload / item["oracle_graph"])
        query = _read(args.workload / item["oracle_query"])
        production = evaluate_production(graph, query)
        reference = evaluate_reference(graph, query)
        comparison = compare(production, reference)
        mutation, corrupted = _mutate(production)
        negative_detected = not compare(corrupted, reference)["answer_exact"]
        expected_status = item.get("expected_status")
        rows.append({
            "case_id": item["case_id"],
            "split": item.get("split"),
            "family": item.get("family"),
            "tool_schema": item.get("tool_schema"),
            "path_depth": item.get("path_depth"),
            "comparison": comparison,
            "production_status": production.get("status"),
            "reference_status": reference.get("status"),
            "witness_count": len(reference.get("witnesses", [])),
            "conflict_count": len(reference.get("conflicts", [])),
            "expected_status": expected_status,
            "expected_status_match": expected_status is None or reference.get("status") == expected_status,
            "negative_control": mutation,
            "negative_control_detected": negative_detected,
        })

    reference_path = Path(__file__).resolve().parents[1] / "ocesq" / "behavior_graph" / "query_reference.py"
    dependency_audit = _reference_import_audit(reference_path)
    metrics = ("status_exact", "witness_exact", "conflict_exact", "certificate_exact", "answer_exact")
    summary = {
        "experiment_id": "deterministic-query-differential-v2",
        "workload": str(args.workload),
        "cases": len(rows),
        **{metric: mean(row["comparison"][metric] for row in rows) for metric in metrics},
        "expected_status_match": mean(row["expected_status_match"] for row in rows),
        "negative_control_detection_rate": mean(row["negative_control_detected"] for row in rows),
        "dependency_audit": dependency_audit,
        "by_split": _group(rows, "split"),
        "by_family": _group(rows, "family"),
        "by_tool_schema": _group(rows, "tool_schema"),
        "by_path_depth": _group(rows, "path_depth"),
        "failed_cases": [row["case_id"] for row in rows if not row["comparison"]["answer_exact"]],
        "expected_status_failures": [row["case_id"] for row in rows if not row["expected_status_match"]],
        "max_witness_count": max(row["witness_count"] for row in rows),
        "mixed_support_conflict_cases": sum(row["witness_count"] > 0 and row["conflict_count"] > 0 for row in rows),
        "interpretation": "Graph-relative formal correctness only; no claim about open-domain semantic completeness.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if summary["answer_exact"] == 1.0 and summary["expected_status_match"] == 1.0 and summary["negative_control_detection_rate"] == 1.0 and dependency_audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
