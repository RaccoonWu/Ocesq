#!/usr/bin/env python3
"""Evaluate effect-removal and stale-state APC fault pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from mandrel.behavior_graph.ocesq import build_ocei_instance, run_ocesq
from mandrel.behavior_graph.ocesq_contract import adapt_ocesq_result, graph_snapshot
from mandrel.behavior_graph.trace_compiler import TraceCompiler, load_jsonl


IMPACT_CLOSURE = {
    "remove_unique_effect_result": {
        "parameter_provenance", "target_entity_support", "verification_evidence",
        "effect_evidence", "output_grounding", "entity_state_consistency",
    },
    "inject_stale_entity_version": {
        "parameter_provenance", "target_entity_support", "input_observation",
        "state_freshness", "entity_state_consistency", "output_grounding",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def raw_tool_use(events: list[dict[str, Any]], tool_use_id: str) -> int:
    matches = []
    for idx, event in enumerate(events):
        if event.get("type") != "message":
            continue
        if any(
            isinstance(part, dict) and part.get("type") == "tool_use" and part.get("id") == tool_use_id
            for part in event.get("message", {}).get("content", []) or []
        ):
            matches.append(idx)
    if len(matches) != 1:
        raise ValueError(f"expected one raw declaration {tool_use_id}, found {len(matches)}")
    return matches[0]


def raw_dispatch(events: list[dict[str, Any]], tool_use_id: str) -> int | None:
    matches = [
        idx for idx, event in enumerate(events)
        if event.get("type") == "tool_dispatch" and event.get("tool_use_id") == tool_use_id
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple raw dispatches for {tool_use_id}")
    return matches[0] if matches else None


def compile_query(path: str, target_id: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    graph = TraceCompiler(load_jsonl(Path(path))).compile().graph
    actions = [
        node for node in graph.nodes
        if node.type == "ToolAction" and node.attrs.get("tool_use_id") == target_id
    ]
    if len(actions) != 1:
        raise ValueError(f"expected one compiled action {target_id}, found {len(actions)}")
    result = run_ocesq(
        graph,
        actions[0].id,
        budget={"max_nodes": 24, "max_edges": 36},
        ocei=build_ocei_instance(graph),
    )
    return graph, actions[0], result, adapt_ocesq_result(graph, result)


def obligation_type(identifier: str) -> str:
    return identifier.rsplit(":", 1)[-1]


def paths(result: Any, obligation: str, kind: str | None = None) -> list[Any]:
    rows = [path for path in result.candidate_paths if obligation_type(path.obligation_id) == obligation]
    return [path for path in rows if kind is None or path.kind == kind]


def descriptor(result: Any, obligation: str) -> Any | None:
    return next(
        (item for item in result.missing_path_descriptors if obligation_type(item.obligation_id) == obligation),
        None,
    )


def certificate_valid(graph: Any, result: Any, contract: dict[str, Any], obligation: str) -> bool:
    normalized = next((item for item in contract["obligations"] if item["type"] == obligation), None)
    desc = descriptor(result, obligation)
    if normalized is None or desc is None:
        return False
    certificate = normalized.get("missing_certificate") or {}
    return bool(
        normalized.get("status") == "missing"
        and certificate.get("complete") is True
        and certificate.get("scope") == "full_behavior_graph"
        and certificate.get("graph_snapshot") == graph_snapshot(graph)
        and certificate.get("searched_node_ids") == sorted(node.id for node in graph.nodes)
        and certificate.get("searched_edge_refs") == sorted(
            f"{edge.source}->{edge.type}->{edge.target}" for edge in graph.edges
        )
        and (certificate.get("descriptor") or {}).get("obligation_id") == desc.obligation_id
        and result.action_id in desc.anchor_node_ids
    )


def component_signature(path: Any) -> tuple[str, str, str, str, str]:
    return (
        obligation_type(path.obligation_id),
        path.kind,
        path.summary,
        path.path_pattern,
        json.dumps(path.attrs, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def component_counter(result: Any, obligations: Iterable[str]) -> Counter[tuple[str, ...]]:
    selected = set(obligations)
    return Counter(
        component_signature(path)
        for path in result.candidate_paths
        if obligation_type(path.obligation_id) in selected
    )


def prf(predicted: set[int], expected: set[int]) -> tuple[float, float, float]:
    overlap = len(predicted & expected)
    precision = overlap / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = overlap / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    family = case["family"]
    target_id = case["target_action"]["tool_use_id"]
    baseline_events = load_jsonl(Path(case["baseline_trace"]))
    variant_events = load_jsonl(Path(case["variant_trace"]))
    base_graph, base_action, baseline, _ = compile_query(case["baseline_trace"], target_id)
    variant_graph, variant_action, variant, variant_contract = compile_query(case["variant_trace"], target_id)
    checks: dict[str, bool] = {}
    expected_pointers: set[int] = set()
    predicted_pointers: set[int] = set()

    if family == "remove_unique_effect_result":
        base_action_idx = raw_tool_use(baseline_events, target_id)
        base_dispatch_idx = raw_dispatch(baseline_events, target_id)
        assert base_dispatch_idx is not None
        expected_pointers = {base_action_idx, base_dispatch_idx}
        effect_paths = [
            path for path in paths(baseline, "effect_evidence", "supporting")
            if expected_pointers.issubset(path.source_event_pointers)
        ]
        for path in effect_paths:
            predicted_pointers.update(path.source_event_pointers)
        desc = descriptor(variant, "effect_evidence")
        checks = {
            "clean_effect_satisfied": baseline.obligation_statuses.get("effect_evidence") == "supported",
            "removed_effect_witness": bool(effect_paths) and not paths(variant, "effect_evidence", "supporting"),
            "effect_contract_unmet": variant.obligation_statuses.get("effect_evidence") == "missing",
            "effect_missing_descriptor": bool(desc and variant.action_id in desc.anchor_node_ids),
            "effect_scope_certificate_valid": certificate_valid(
                variant_graph, variant, variant_contract, "effect_evidence"
            ),
        }
    else:
        source_id = case["source"]["source_tool_use_id"]
        injected_id = case["transform"]["injected_tool_use_id"]
        source_idx = raw_dispatch(variant_events, source_id)
        injected_idx = raw_dispatch(variant_events, injected_id)
        action_idx = raw_tool_use(variant_events, target_id)
        assert source_idx is not None and injected_idx is not None
        expected_pointers = {source_idx, injected_idx, action_idx}
        equivalent_sources = set(case["source"].get("equivalent_source_event_indices", [source_idx]))
        fixed_pointers = {injected_idx, action_idx}
        stale_paths = [
            path for path in paths(variant, "entity_state_consistency", "conflicting")
            if "APC-STALE" in path.summary
        ]
        localized = [
            path for path in stale_paths
            if fixed_pointers.issubset(path.source_event_pointers)
            and equivalent_sources.intersection(path.source_event_pointers)
        ]
        for path in localized:
            predicted_pointers.update(path.source_event_pointers)
        original_source = next(
            node for node in variant_graph.nodes
            if node.type == "ToolObservation" and node.attrs.get("tool_use_id") == source_id
        )
        original_support = [
            path for path in paths(variant, "target_entity_support", "supporting")
            if original_source.source_event_idx in path.source_event_pointers
        ]
        checks = {
            "added_stale_state_conflict": bool(stale_paths),
            "stale_conflict_localized": bool(localized),
            "entity_state_contract_conflicted": variant.obligation_statuses.get("entity_state_consistency") == "conflicting",
            "original_identity_support_preserved": bool(original_support),
        }

    all_obligations = set(baseline.obligation_statuses) | set(variant.obligation_statuses)
    invariant = all_obligations - IMPACT_CLOSURE[family]
    unexpected_statuses = sorted(
        obligation for obligation in invariant
        if baseline.obligation_statuses.get(obligation) != variant.obligation_statuses.get(obligation)
    )
    before = component_counter(baseline, invariant)
    after = component_counter(variant, invariant)
    unexpected_components = list((before - after).elements()) + list((after - before).elements())
    if family == "inject_stale_entity_version" and localized:
        # One member of the raw-event equivalence class is sufficient; the
        # materializer need not choose the exact source transaction cloned by
        # the mutation generator.
        selected = set(localized[0].source_event_pointers)
        loc_p = loc_r = loc_f1 = 1.0 if (
            fixed_pointers.issubset(selected)
            and bool(equivalent_sources & selected)
            and selected.issubset(fixed_pointers | equivalent_sources)
        ) else 0.0
    else:
        loc_p, loc_r, loc_f1 = prf(predicted_pointers, expected_pointers)
    exact = bool(
        all(case["integrity"].values())
        and all(checks.values())
        and not unexpected_statuses
        and not unexpected_components
        and loc_p == loc_r == 1.0
    )
    return {
        "case_id": case["case_id"],
        "family": family,
        "target_tool_name": case["target_action"]["tool_name"],
        "raw_integrity_passed": all(case["integrity"].values()),
        "component_checks": checks,
        "expected_component_recall": mean(checks.values()),
        "expected_localization_pointers": sorted(expected_pointers),
        "allowed_equivalent_source_pointers": sorted(equivalent_sources) if family == "inject_stale_entity_version" else [],
        "predicted_localization_pointers": sorted(predicted_pointers),
        "localization_precision": loc_p,
        "localization_recall": loc_r,
        "localization_f1": loc_f1,
        "unexpected_status_changes": unexpected_statuses,
        "unexpected_component_changes": [list(item) for item in unexpected_components],
        "unexpected_change": bool(unexpected_statuses or unexpected_components),
        "contract_delta_exact": exact,
        "baseline_statuses": baseline.obligation_statuses,
        "variant_statuses": variant.obligation_statuses,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "raw_integrity_rate": mean(row["raw_integrity_passed"] for row in rows),
        "contract_delta_exact": mean(row["contract_delta_exact"] for row in rows),
        "expected_component_recall": mean(row["expected_component_recall"] for row in rows),
        "localization_precision": mean(row["localization_precision"] for row in rows),
        "localization_recall": mean(row["localization_recall"] for row in rows),
        "localization_f1": mean(row["localization_f1"] for row in rows),
        "unexpected_change_rate": mean(row["unexpected_change"] for row in rows),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    failed_checks = Counter(
        name for row in rows for name, passed in row["component_checks"].items() if not passed
    )
    return {
        "protocol": "apc-expanded-faults-v1",
        "overall": aggregate(rows),
        "by_family": {family: aggregate(selected) for family, selected in sorted(grouped.items())},
        "failed_component_counts": dict(sorted(failed_checks.items())),
        "failed_cases": [row["case_id"] for row in rows if not row["contract_delta_exact"]],
        "oracle_boundary": "raw mutation manifest and frozen OEG contract schema; graph/query outputs are prediction-side only",
        "human_labels_used": False,
        "model_labels_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/apc_expanded_faults_v1/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/apc_expanded_faults_v1"))
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
