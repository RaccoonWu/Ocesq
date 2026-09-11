#!/usr/bin/env python3
"""Verify graph-query materialization against independent reference semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from mandrel.behavior_graph.query_oracle import certificate_aware_materialize, witness_union_materialize
from mandrel.behavior_graph.query_reference import evaluate_reference


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _signature(answer: dict[str, Any]) -> tuple[Any, ...]:
    paths = lambda key: tuple(sorted((tuple(map(str, row.get("node_ids", []))), tuple(sorted(map(str, row.get("violations", []))))) for row in answer.get(key, [])))
    certificate = json.dumps(answer.get("certificate"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return answer.get("status"), paths("witnesses"), paths("conflicts"), certificate


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _subgraph(graph: dict[str, Any], keep: set[str]) -> dict[str, Any]:
    return {
        **graph,
        "nodes": [node for node in graph.get("nodes", []) if str(node["id"]) in keep],
        "edges": [edge for edge in graph.get("edges", []) if str(edge["source"]) in keep and str(edge["target"]) in keep],
    }


def _all_deciding_union(graph: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    answer = evaluate_reference(graph, query)
    keep = {str(query.get("root_selector", {}).get("action_id"))}
    for row in [*answer.get("witnesses", []), *answer.get("conflicts", [])]:
        keep.update(map(str, row.get("node_ids", [])))
    return _subgraph(graph, keep)


def _graph_valid(full_answer: dict[str, Any], graph: dict[str, Any], query: dict[str, Any]) -> bool:
    return _signature(full_answer) == _signature(evaluate_reference(graph, query))


def _package_valid(full_graph: dict[str, Any], query: dict[str, Any], result: dict[str, Any]) -> bool:
    expected = evaluate_reference(full_graph, query)
    supplied = result.get("answer", {})
    if _signature(expected) != _signature(supplied):
        return False
    if expected["status"] == "missing":
        return (
            result.get("certificate_source_snapshot") == full_graph.get("graph_snapshot")
            and result.get("certificate_digest") == _digest(expected.get("certificate"))
        )
    return _graph_valid(expected, result.get("graph", {}), query)


def _exact_minimum_package(
    graph: dict[str, Any],
    query: dict[str, Any],
    max_optional: int,
) -> dict[str, Any] | None:
    answer = evaluate_reference(graph, query)
    root = str(query.get("root_selector", {}).get("action_id"))
    if answer["status"] == "missing":
        compact = _subgraph(graph, {root})
        return {"graph": compact, "answer": answer, "certificate_digest": _digest(answer.get("certificate")), "certificate_source_snapshot": graph.get("graph_snapshot")}
    node_ids = [str(node["id"]) for node in graph.get("nodes", [])]
    optional = [node_id for node_id in node_ids if node_id != root]
    if len(optional) > max_optional:
        return None
    for size in range(len(optional) + 1):
        for selected in combinations(optional, size):
            candidate = _subgraph(graph, {root, *selected})
            if _graph_valid(answer, candidate, query):
                return {"graph": candidate, "answer": answer, "certificate_digest": None, "certificate_source_snapshot": graph.get("graph_snapshot")}
    return None


def _metrics(graph: dict[str, Any], valid: bool) -> dict[str, Any]:
    return {
        "valid": valid,
        "nodes": len(graph.get("nodes", [])),
        "edges": len(graph.get("edges", [])),
        "bytes": len(json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=Path("data/query_oracle_v6"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/answer_materialization_v2"))
    parser.add_argument("--exact-max-optional", type=int, default=12)
    args = parser.parse_args()
    manifest = _read(args.workload / "manifest.json")
    rows = []
    for item in manifest["cases"]:
        graph = _read(args.workload / item["oracle_graph"])
        query = _read(args.workload / item["oracle_query"])
        full_answer = evaluate_reference(graph, query)
        production_union = witness_union_materialize(graph, query)
        deciding_union = _all_deciding_union(graph, query)
        packaged = certificate_aware_materialize(graph, query)
        exact = _exact_minimum_package(graph, query, args.exact_max_optional)
        methods = {
            "production_witness_union": _metrics(production_union, _graph_valid(full_answer, production_union, query)),
            "all_deciding_union": _metrics(deciding_union, _graph_valid(full_answer, deciding_union, query)),
            "certificate_aware": _metrics(packaged["graph"], _package_valid(graph, query, packaged)),
        }
        if exact is not None:
            methods["exact_minimum"] = _metrics(exact["graph"], _package_valid(graph, query, exact))
            methods["certificate_aware"]["minimality_gap_nodes"] = methods["certificate_aware"]["nodes"] - methods["exact_minimum"]["nodes"]
        rows.append({
            "case_id": item["case_id"],
            "split": item.get("split"),
            "family": item.get("family"),
            "path_depth": item.get("path_depth"),
            "status": full_answer["status"],
            "full_nodes": len(graph.get("nodes", [])),
            "full_edges": len(graph.get("edges", [])),
            "methods": methods,
        })
    names = ("production_witness_union", "all_deciding_union", "certificate_aware")
    summary: dict[str, Any] = {
        "experiment_id": "answer-preserving-materialization-v2",
        "cases": len(rows),
        "methods": {},
        "exact_cases": sum("exact_minimum" in row["methods"] for row in rows),
        "interpretation": "Preservation is relative to formal graph-query answers, including conflicting alternatives and missing certificates.",
    }
    for name in names:
        values = [row["methods"][name] for row in rows]
        valid_values = [value for value in values if value["valid"]]
        summary["methods"][name] = {
            "valid_rate": mean(value["valid"] for value in values),
            "mean_nodes_valid_only": mean(value["nodes"] for value in valid_values) if valid_values else None,
            "mean_edges_valid_only": mean(value["edges"] for value in valid_values) if valid_values else None,
            "mean_bytes_valid_only": mean(value["bytes"] for value in valid_values) if valid_values else None,
            "invalid_cases": [row["case_id"] for row in rows if not row["methods"][name]["valid"]],
        }
    gap_rows = [row["methods"]["certificate_aware"]["minimality_gap_nodes"] for row in rows if "minimality_gap_nodes" in row["methods"]["certificate_aware"]]
    summary["certificate_aware_exact_subset"] = {"cases": len(gap_rows), "mean_minimality_gap_nodes": mean(gap_rows) if gap_rows else None, "max_minimality_gap_nodes": max(gap_rows) if gap_rows else None}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if summary["methods"]["certificate_aware"]["valid_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
