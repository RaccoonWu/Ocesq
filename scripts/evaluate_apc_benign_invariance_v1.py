#!/usr/bin/env python3
"""Evaluate normalized APC-answer invariance under raw benign rewrites."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from mandrel.behavior_graph.ocesq import build_ocei_instance, run_ocesq
from mandrel.behavior_graph.ocesq_contract import adapt_ocesq_result, graph_snapshot
from mandrel.behavior_graph.trace_compiler import TraceCompiler, load_jsonl


SEMANTIC_ANSWER_CHECKS = {
    "status_vector_exact",
    "obligations_exact",
    "candidate_paths_exact",
    "supporting_paths_exact",
    "conflicting_paths_exact",
    "missing_descriptors_exact",
    "source_pointers_exact",
    "compact_result_exact",
    "normalized_contract_semantics_exact",
    "baseline_certificates_valid",
    "variant_certificates_valid",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compile_query(path: str, tool_use_id: str) -> tuple[Any, Any, dict[str, Any]]:
    graph = TraceCompiler(load_jsonl(Path(path))).compile().graph
    actions = [
        node for node in graph.nodes
        if node.type == "ToolAction" and node.attrs.get("tool_use_id") == tool_use_id
    ]
    if len(actions) != 1:
        raise ValueError(f"expected one compiled target action {tool_use_id}, found {len(actions)}")
    result = run_ocesq(
        graph,
        actions[0].id,
        budget={"max_nodes": 24, "max_edges": 36},
        ocei=build_ocei_instance(graph),
    )
    return graph, result, adapt_ocesq_result(graph, result)


def result_projection(result: Any) -> dict[str, Any]:
    row = result.to_dict()
    row.pop("timings", None)
    return row


def path_projection(paths: list[Any]) -> list[dict[str, Any]]:
    return [path.to_dict() for path in paths]


def contract_projection(contract: dict[str, Any]) -> dict[str, Any]:
    """Remove snapshot-specific values while retaining certificate semantics."""
    row = deepcopy(contract)
    row.pop("graph_snapshot", None)
    for obligation in row.get("obligations", []):
        certificate = obligation.get("missing_certificate")
        if certificate:
            certificate.pop("graph_snapshot", None)
            certificate.pop("searched_node_ids", None)
            certificate.pop("searched_edge_refs", None)
    return row


def certificates_valid(graph: Any, contract: dict[str, Any]) -> bool:
    expected_nodes = sorted(node.id for node in graph.nodes)
    expected_edges = sorted(f"{edge.source}->{edge.type}->{edge.target}" for edge in graph.edges)
    for obligation in contract.get("obligations", []):
        if not obligation.get("required") or obligation.get("status") != "missing":
            continue
        certificate = obligation.get("missing_certificate") or {}
        descriptor = certificate.get("descriptor") or {}
        if not (
            certificate.get("complete") is True
            and certificate.get("scope") == "full_behavior_graph"
            and certificate.get("graph_snapshot") == graph_snapshot(graph)
            and certificate.get("searched_node_ids") == expected_nodes
            and certificate.get("searched_edge_refs") == expected_edges
            and descriptor.get("obligation_id") == obligation.get("id")
            and contract.get("root_action_id") in descriptor.get("anchor_node_ids", [])
        ):
            return False
    return True


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    target_id = case["target_action"]["tool_use_id"]
    base_graph, base, base_contract = compile_query(case["baseline_trace"], target_id)
    variant_graph, variant, variant_contract = compile_query(case["variant_trace"], target_id)

    checks = {
        "status_vector_exact": base.obligation_statuses == variant.obligation_statuses,
        "obligations_exact": canonical(base.to_dict()["obligations"]) == canonical(variant.to_dict()["obligations"]),
        "candidate_paths_exact": canonical(path_projection(base.candidate_paths)) == canonical(path_projection(variant.candidate_paths)),
        "supporting_paths_exact": canonical(path_projection(base.supporting_paths)) == canonical(path_projection(variant.supporting_paths)),
        "conflicting_paths_exact": canonical(path_projection(base.conflicting_paths)) == canonical(path_projection(variant.conflicting_paths)),
        "missing_descriptors_exact": canonical([item.to_dict() for item in base.missing_path_descriptors]) == canonical([item.to_dict() for item in variant.missing_path_descriptors]),
        "source_pointers_exact": base.source_event_pointers == variant.source_event_pointers,
        "compact_result_exact": canonical(base.compact_evidence_subgraph) == canonical(variant.compact_evidence_subgraph),
        "coverage_exact": base.coverage == variant.coverage,
        "redundancy_exact": base.redundancy == variant.redundancy,
        "normalized_contract_semantics_exact": canonical(contract_projection(base_contract)) == canonical(contract_projection(variant_contract)),
        "baseline_certificates_valid": certificates_valid(base_graph, base_contract),
        "variant_certificates_valid": certificates_valid(variant_graph, variant_contract),
    }
    # These artifact-level diagnostics include graph-relative compactness
    # ratios. They are intentionally not part of <status, W+, W-, certificate>.
    checks["root_result_exact"] = canonical(result_projection(base)) == canonical(result_projection(variant))
    semantic_invariance = all(checks[name] for name in SEMANTIC_ANSWER_CHECKS)
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "integrity_passed": all(case["integrity"].values()),
        "checks": checks,
        "benign_complete_answer_invariance": semantic_invariance,
        "operational_metadata_invariance": checks["coverage_exact"] and checks["redundancy_exact"],
        "root_artifact_exact": checks["root_result_exact"],
        "baseline_graph_snapshot": graph_snapshot(base_graph),
        "variant_graph_snapshot": graph_snapshot(variant_graph),
        "compiled_graph_snapshot_changed": graph_snapshot(base_graph) != graph_snapshot(variant_graph),
        "baseline_graph_size": [len(base_graph.nodes), len(base_graph.edges)],
        "variant_graph_size": [len(variant_graph.nodes), len(variant_graph.edges)],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    check_names = sorted({name for row in rows for name in row["checks"]})
    return {
        "pairs": len(rows),
        "raw_integrity_rate": mean(row["integrity_passed"] for row in rows),
        "benign_complete_answer_invariance": mean(row["benign_complete_answer_invariance"] for row in rows),
        "operational_metadata_invariance": mean(row["operational_metadata_invariance"] for row in rows),
        "root_artifact_exact": mean(row["root_artifact_exact"] for row in rows),
        "compiled_graph_snapshot_change_rate": mean(row["compiled_graph_snapshot_changed"] for row in rows),
        "component_rates": {
            name: mean(row["checks"][name] for row in rows)
            for name in check_names
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({row["family"] for row in rows})
    failures = [
        {
            "case_id": row["case_id"],
            "failed_checks": [name for name in sorted(SEMANTIC_ANSWER_CHECKS) if not row["checks"][name]],
        }
        for row in rows if not row["benign_complete_answer_invariance"]
    ]
    operational_differences = [
        {
            "case_id": row["case_id"],
            "changed_fields": [
                name for name in ("coverage_exact", "redundancy_exact", "root_result_exact")
                if not row["checks"][name]
            ],
        }
        for row in rows if not row["operational_metadata_invariance"] or not row["root_artifact_exact"]
    ]
    return {
        "protocol": "apc-benign-invariance-v1",
        "overall": aggregate(rows),
        "by_family": {
            family: aggregate([row for row in rows if row["family"] == family])
            for family in families
        },
        "failures": failures,
        "operational_differences": operational_differences,
        "oracle_boundary": "raw transform integrity is independent; graph/query outputs are prediction-side only",
        "snapshot_policy": "graph digests may change; each scope certificate must validate against its own compiled snapshot",
        "metric_boundary": "semantic invariance covers <status, W+, W-, certificate>; graph-relative redundancy ratios are reported separately",
        "human_labels_used": False,
        "model_labels_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/apc_benign_invariance_v1/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/apc_benign_invariance_v1"))
    args = parser.parse_args()
    rows = [evaluate_case(case) for case in read_jsonl(args.manifest)]
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
