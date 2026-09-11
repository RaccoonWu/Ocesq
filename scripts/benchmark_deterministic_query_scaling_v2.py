#!/usr/bin/env python3
"""Benchmark deterministic graph queries across controlled scaling axes."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
import tracemalloc
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

from mandrel.behavior_graph.query_oracle import build_query_index, evaluate, evaluate_indexed
from mandrel.behavior_graph.query_reference import evaluate_reference


def _scenario_graph(nodes_target: int, depth: int, witnesses: int, branches: int, status: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if depth < 2:
        raise ValueError("depth must include root and at least one path node")
    graph: dict[str, Any] = {
        "trajectory_id": f"scale-n{nodes_target}-d{depth}-w{witnesses}-b{branches}-{status}",
        "graph_snapshot": "scale@0",
        "schema_version": "scale-v2",
        "nodes": [{"id": "action", "type": "Action", "attrs": {"event_index": nodes_target + 1}}],
        "edges": [],
    }
    effective_witnesses = witnesses if status == "supported" else 0
    for witness in range(effective_witnesses):
        previous = "action"
        for position in range(1, depth):
            node_id = f"w{witness}_p{position}"
            attrs = {"event_index": witness * depth + position}
            if position == depth - 1:
                attrs["match"] = True
            graph["nodes"].append({"id": node_id, "type": f"L{position}", "attrs": attrs})
            graph["edges"].append({"source": previous, "target": node_id, "type": f"step_{position}"})
            previous = node_id
    if depth > 2:
        for branch in range(branches):
            node_id = f"dead_{branch}"
            graph["nodes"].append({"id": node_id, "type": "L1", "attrs": {"event_index": nodes_target + branch + 2}})
            graph["edges"].append({"source": "action", "target": node_id, "type": "step_1"})
    while len(graph["nodes"]) < nodes_target:
        index = len(graph["nodes"])
        node_id = f"noise_{index}"
        graph["nodes"].append({"id": node_id, "type": "Note", "attrs": {"event_index": index}})
        if index % 4 == 0:
            graph["edges"].append({"source": "action", "target": node_id, "type": "context"})
    query = {
        "query_id": graph["trajectory_id"],
        "root_selector": {"action_id": "action"},
        "path_pattern": ["Action", *[f"L{position}" for position in range(1, depth)]],
        "edge_constraints": [f"step_{position}" for position in range(1, depth)],
        "state_constraints": [{"id": "terminal_match", "node_type": f"L{depth - 1}", "field": "match", "equals": True}],
        "entity_constraints": [],
        "temporal_constraints": [],
    }
    return graph, query


def _scenarios() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for nodes in (32, 128, 512, 2048, 8192):
        rows.append({"sweep": "graph_size", "nodes": nodes, "depth": 3, "witnesses": 4, "branches": min(32, nodes // 16), "status": "supported"})
        rows.append({"sweep": "missing_scope", "nodes": nodes, "depth": 3, "witnesses": 0, "branches": min(32, nodes // 16), "status": "missing"})
    for depth in (2, 3, 4, 6, 8):
        rows.append({"sweep": "path_depth", "nodes": 512, "depth": depth, "witnesses": 4, "branches": 0 if depth == 2 else 32, "status": "supported"})
    for witnesses in (1, 2, 4, 8, 16, 32, 64):
        rows.append({"sweep": "witness_count", "nodes": 2048, "depth": 4, "witnesses": witnesses, "branches": 32, "status": "supported"})
    for branches in (0, 8, 32, 128, 512):
        rows.append({"sweep": "branching", "nodes": 2048, "depth": 4, "witnesses": 4, "branches": branches, "status": "supported"})
    return rows


def _measure(fn: Callable[[], Any], warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {"mean_ms": mean(values), "p50_ms": median(values), "p95_ms": p95}


def _index_memory(graph: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    tracemalloc.start()
    index = build_query_index(graph)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    payload = len(json.dumps(index, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return index, payload, peak


def _answer_signature(answer: dict[str, Any]) -> tuple[Any, ...]:
    paths = lambda key: tuple(sorted((tuple(row.get("node_ids", [])), tuple(sorted(row.get("violations", [])))) for row in answer.get(key, [])))
    certificate = json.dumps(answer.get("certificate"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return answer.get("status"), paths("witnesses"), paths("conflicts"), certificate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results/deterministic_scaling_v2"))
    args = parser.parse_args()
    records = []
    for spec in _scenarios():
        graph, query = _scenario_graph(spec["nodes"], spec["depth"], spec["witnesses"], spec["branches"], spec["status"])
        index, index_bytes, peak_bytes = _index_memory(graph)
        production = evaluate_indexed(graph, query, index)
        reference = evaluate_reference(graph, query)
        exact = _answer_signature(production) == _answer_signature(reference) and reference["status"] == spec["status"]
        if not exact:
            raise AssertionError(f"reference mismatch: {spec}")
        repeats = min(args.repeats, 30) if len(graph["nodes"]) >= 8192 else args.repeats
        build = _measure(lambda: build_query_index(graph), min(args.warmup, 5), min(repeats, 30))
        unindexed = _measure(lambda: evaluate(graph, query), args.warmup, repeats)
        indexed = _measure(lambda: evaluate_indexed(graph, query, index), args.warmup, repeats)
        saved = unindexed["p50_ms"] - indexed["p50_ms"]
        records.append({
            **spec,
            "actual_nodes": len(graph["nodes"]),
            "actual_edges": len(graph["edges"]),
            "reference_exact": exact,
            "answer_witnesses": len(reference["witnesses"]),
            "answer_conflicts": len(reference["conflicts"]),
            "repeats": repeats,
            "index_build": build,
            "unindexed": unindexed,
            "indexed": indexed,
            "index_payload_bytes": index_bytes,
            "index_peak_allocated_bytes": peak_bytes,
            "p50_speedup": unindexed["p50_ms"] / max(indexed["p50_ms"], 1e-12),
            "amortization_queries_p50": math.ceil(build["p50_ms"] / saved) if saved > 0 else None,
        })
    by_sweep = {}
    for sweep in sorted({row["sweep"] for row in records}):
        rows = [row for row in records if row["sweep"] == sweep]
        by_sweep[sweep] = {
            "scenarios": len(rows),
            "reference_exact": mean(row["reference_exact"] for row in rows),
            "mean_p50_speedup": mean(row["p50_speedup"] for row in rows),
            "min_p50_speedup": min(row["p50_speedup"] for row in rows),
            "max_p50_speedup": max(row["p50_speedup"] for row in rows),
        }
    report = {
        "experiment_id": "deterministic-query-scaling-v2",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "timer": "perf_counter_ns"},
        "scenario_count": len(records),
        "reference_exact": mean(row["reference_exact"] for row in records),
        "repeats_default": args.repeats,
        "warmup": args.warmup,
        "by_sweep": by_sweep,
        "records": records,
        "interpretation": "Single-process local microbenchmark; timings and Python allocation peaks are environment-specific.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("scenario_count", "reference_exact", "by_sweep", "environment")}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["reference_exact"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
