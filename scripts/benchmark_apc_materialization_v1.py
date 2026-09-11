#!/usr/bin/env python3
"""Benchmark paired APC compilation, query, and materialize-and-validate cost."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from evaluate_apc_fault_materialization_v1 import METHODS, evaluate_materializer, sources
from mandrel.behavior_graph.ocesq import build_ocei_instance, run_ocesq
from mandrel.behavior_graph.ocesq_contract import adapt_ocesq_result
from mandrel.behavior_graph.trace_compiler import TraceCompiler, load_jsonl


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def timed_instance(path: str, target_id: str, repeats: int) -> tuple[dict[str, Any], dict[str, float]]:
    events = load_jsonl(Path(path))
    start = time.perf_counter_ns()
    graph = TraceCompiler(events).compile().graph
    compile_ms = (time.perf_counter_ns() - start) / 1_000_000
    action = next(node for node in graph.nodes if node.type == "ToolAction" and node.attrs.get("tool_use_id") == target_id)
    start = time.perf_counter_ns()
    result = run_ocesq(graph, action.id, budget={"max_nodes": 24, "max_edges": 36}, ocei=build_ocei_instance(graph))
    contract = adapt_ocesq_result(graph, result)
    query_ms = (time.perf_counter_ns() - start) / 1_000_000

    method_times = {}
    for method in METHODS:
        evaluate_materializer(graph, action, result, contract, method)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            evaluate_materializer(graph, action, result, contract, method)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        method_times[method] = median(samples)
    return {
        "compile_ms": compile_ms,
        "query_and_adapt_ms": query_ms,
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
    }, method_times


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "instances": len(values),
        "mean_ms": mean(values),
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def summarize(rows: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    datasets = sorted({row["dataset"] for row in rows})

    def group(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "instances": len(selected),
            "compile": distribution([row["compile_ms"] for row in selected]),
            "query_and_adapt": distribution([row["query_and_adapt_ms"] for row in selected]),
            "materialize_and_validate": {
                method: distribution([row["method_ms"][method] for row in selected])
                for method in METHODS
            },
            "mean_graph_nodes": mean(row["graph_nodes"] for row in selected),
            "mean_graph_edges": mean(row["graph_edges"] for row in selected),
        }

    return {
        "protocol": "apc-materialization-efficiency-v1",
        "pairs": len(rows) // 2,
        "instances": len(rows),
        "repeats_per_method_instance": repeats,
        "timing_unit": "milliseconds",
        "aggregation": "median over repeats per instance, then distribution across instances",
        "overall": group(rows),
        "by_dataset": {dataset: group([row for row in rows if row["dataset"] == dataset]) for dataset in datasets},
        "scope": "single-process warm-cache CPU timing; materializer timing includes package construction and preservation validation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="NAME=MANIFEST:v2|expanded")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("results/apc_materialization_efficiency_v1"))
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    rows = []
    for item in sources(args.source):
        for variant, path in (("baseline", item["baseline_trace"]), ("variant", item["variant_trace"])):
            system, method_times = timed_instance(path, item["target_tool_use_id"], args.repeats)
            rows.append({
                "dataset": item["dataset"], "case_id": item["case_id"], "family": item["family"],
                "variant": variant, **system, "method_ms": method_times,
            })
    summary = summarize(rows, args.repeats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
