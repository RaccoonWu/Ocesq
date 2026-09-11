#!/usr/bin/env python3
"""Validate invariant and answer-changing graph-query transformations."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from ocesq.behavior_graph.query_oracle import evaluate as evaluate_production
from ocesq.behavior_graph.query_reference import evaluate_reference


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_set(answer: dict[str, Any], key: str) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    return {
        (tuple(map(str, row.get("node_ids", []))), tuple(sorted(map(str, row.get("violations", [])))))
        for row in answer.get(key, [])
    }


def _answer_exact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("status") == right.get("status")
        and _path_set(left, "witnesses") == _path_set(right, "witnesses")
        and _path_set(left, "conflicts") == _path_set(right, "conflicts")
        and left.get("certificate") == right.get("certificate")
    )


def _add_d3_path(graph: dict[str, Any], version: str, suffix: str) -> None:
    observation, entity = f"meta_obs_{suffix}", f"meta_entity_{suffix}"
    graph["nodes"].extend([
        {"id": observation, "type": "Observation", "attrs": {"event_index": 5, "version": version}},
        {"id": entity, "type": "Entity", "attrs": {"event_index": 5, "entity_id": "E1"}},
    ])
    graph["edges"].extend([
        {"source": "action", "target": observation, "type": "depends_on"},
        {"source": observation, "target": entity, "type": "resolves"},
    ])


def _noise(graph: dict[str, Any], query: dict[str, Any]) -> None:
    graph["nodes"].append({"id": "meta_noise", "type": "Note", "attrs": {"event_index": 999}})


def _wrapper(graph: dict[str, Any], query: dict[str, Any]) -> None:
    graph["nodes"].append({"id": "meta_wrapper", "type": "Message", "attrs": {"event_index": 9}})
    graph["edges"].append({"source": "action", "target": "meta_wrapper", "type": "context"})


def _reorder_irrelevant(graph: dict[str, Any], query: dict[str, Any]) -> None:
    for node in graph["nodes"]:
        if node.get("type") in {"Note", "Message"}:
            node.setdefault("attrs", {})["event_index"] = -100


def _delete_witness(graph: dict[str, Any], query: dict[str, Any]) -> None:
    graph["edges"] = [edge for edge in graph["edges"] if edge.get("type") != "resolves"]


def _add_valid(graph: dict[str, Any], query: dict[str, Any]) -> None:
    _add_d3_path(graph, "v1", "valid")


def _add_conflict(graph: dict[str, Any], query: dict[str, Any]) -> None:
    _add_d3_path(graph, "v0", "conflict")


def _version_conflict(graph: dict[str, Any], query: dict[str, Any]) -> None:
    next(node for node in graph["nodes"] if node.get("id") == "obs_1")["attrs"]["version"] = "v0"


def _entity_conflict(graph: dict[str, Any], query: dict[str, Any]) -> None:
    next(node for node in graph["nodes"] if node.get("id") == "entity_1")["attrs"]["entity_id"] = "E2"


def _snapshot_change(graph: dict[str, Any], query: dict[str, Any]) -> None:
    graph["graph_snapshot"] = f"{graph['trajectory_id']}@1"


def _wrong_root_type(graph: dict[str, Any], query: dict[str, Any]) -> None:
    next(node for node in graph["nodes"] if node.get("id") == "action")["type"] = "Observation"


def _add_reverse_witness(graph: dict[str, Any], query: dict[str, Any]) -> None:
    graph["nodes"].append({"id": "meta_reader", "type": "Observation", "attrs": {"event_index": 1, "version": "v1"}})
    graph["edges"].append({"source": "meta_reader", "target": "entity", "type": "resolves"})


TRANSFORMS: list[dict[str, Any]] = [
    {"id": "disconnected_noise", "base": "v6_d3_witness_2", "kind": "invariant", "fn": _noise, "status": "supported", "witness_delta": 0, "conflict_delta": 0, "certificate": "same"},
    {"id": "transparent_wrapper", "base": "v6_d3_witness_2", "kind": "invariant", "fn": _wrapper, "status": "supported", "witness_delta": 0, "conflict_delta": 0, "certificate": "same"},
    {"id": "irrelevant_event_reorder", "base": "v6_d3_noise_invariant", "kind": "invariant", "fn": _reorder_irrelevant, "status": "supported", "witness_delta": 0, "conflict_delta": 0, "certificate": "same"},
    {"id": "delete_unique_witness", "base": "v6_d3_witness_1", "kind": "changing", "fn": _delete_witness, "status": "missing", "witness_delta": -1, "conflict_delta": 0, "certificate": "created"},
    {"id": "add_witness_to_missing", "base": "v6_d3_missing_dead_prefix", "kind": "changing", "fn": _add_valid, "status": "supported", "witness_delta": 1, "conflict_delta": 0, "certificate": "removed"},
    {"id": "add_second_witness", "base": "v6_d3_witness_1", "kind": "changing", "fn": _add_valid, "status": "supported", "witness_delta": 1, "conflict_delta": 0, "certificate": "same"},
    {"id": "version_conflict", "base": "v6_d3_witness_1", "kind": "changing", "fn": _version_conflict, "status": "conflicting", "witness_delta": -1, "conflict_delta": 1, "certificate": "same"},
    {"id": "entity_conflict", "base": "v6_d3_witness_1", "kind": "changing", "fn": _entity_conflict, "status": "conflicting", "witness_delta": -1, "conflict_delta": 1, "certificate": "same"},
    {"id": "inject_conflicting_alternative", "base": "v6_d3_witness_1", "kind": "changing", "fn": _add_conflict, "status": "conflicting", "witness_delta": 0, "conflict_delta": 1, "certificate": "same"},
    {"id": "missing_snapshot_change", "base": "v6_d3_missing_dead_prefix", "kind": "changing", "fn": _snapshot_change, "status": "missing", "witness_delta": 0, "conflict_delta": 0, "certificate": "changed"},
    {"id": "wrong_root_type", "base": "v6_d3_witness_1", "kind": "changing", "fn": _wrong_root_type, "status": "missing", "witness_delta": -1, "conflict_delta": 0, "certificate": "created"},
    {"id": "add_reverse_witness", "base": "v6_d4_witness_1", "kind": "changing", "fn": _add_reverse_witness, "status": "supported", "witness_delta": 1, "conflict_delta": 0, "certificate": "same"},
]


def _certificate_relation(base: dict[str, Any], changed: dict[str, Any]) -> str:
    left, right = base.get("certificate"), changed.get("certificate")
    if left is None and right is None:
        return "same"
    if left is None:
        return "created"
    if right is None:
        return "removed"
    return "same" if left == right else "changed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=Path("data/query_oracle_v6"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/query_metamorphic_v2"))
    args = parser.parse_args()
    manifest = {row["case_id"]: row for row in _read(args.workload / "manifest.json")["cases"]}
    rows = []
    for spec in TRANSFORMS:
        item = manifest[spec["base"]]
        graph = _read(args.workload / item["oracle_graph"])
        query = _read(args.workload / item["oracle_query"])
        base_reference = evaluate_reference(graph, query)
        changed_graph, changed_query = deepcopy(graph), deepcopy(query)
        transform: Callable[[dict[str, Any], dict[str, Any]], None] = spec["fn"]
        transform(changed_graph, changed_query)
        changed_reference = evaluate_reference(changed_graph, changed_query)
        changed_production = evaluate_production(changed_graph, changed_query)
        witness_delta = len(changed_reference["witnesses"]) - len(base_reference["witnesses"])
        conflict_delta = len(changed_reference["conflicts"]) - len(base_reference["conflicts"])
        certificate_relation = _certificate_relation(base_reference, changed_reference)
        invariant_correct = spec["kind"] != "invariant" or _answer_exact(base_reference, changed_reference)
        relation_correct = (
            changed_reference["status"] == spec["status"]
            and witness_delta == spec["witness_delta"]
            and conflict_delta == spec["conflict_delta"]
            and certificate_relation == spec["certificate"]
            and invariant_correct
        )
        rows.append({
            "transformation": spec["id"],
            "kind": spec["kind"],
            "base_case": spec["base"],
            "base_status": base_reference["status"],
            "transformed_status": changed_reference["status"],
            "witness_delta": witness_delta,
            "conflict_delta": conflict_delta,
            "certificate_relation": certificate_relation,
            "reference_relation_correct": relation_correct,
            "production_reference_exact": _answer_exact(changed_production, changed_reference),
        })
    summary = {
        "experiment_id": "query-metamorphic-v2",
        "transformations": len(rows),
        "invariant_cases": sum(row["kind"] == "invariant" for row in rows),
        "changing_cases": sum(row["kind"] == "changing" for row in rows),
        "reference_relation_accuracy": mean(row["reference_relation_correct"] for row in rows),
        "production_reference_exact": mean(row["production_reference_exact"] for row in rows),
        "failed_relations": [row["transformation"] for row in rows if not row["reference_relation_correct"]],
        "production_failures": [row["transformation"] for row in rows if not row["production_reference_exact"]],
        "interpretation": "Exact metamorphic relations over formal graph queries; not open-domain semantic accuracy.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if summary["reference_relation_accuracy"] == 1.0 and summary["production_reference_exact"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
