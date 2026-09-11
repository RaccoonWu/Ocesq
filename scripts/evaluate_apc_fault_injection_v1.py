#!/usr/bin/env python3
"""Evaluate APC answer deltas on paired raw-trace fault injections.

The oracle side of this evaluator reads raw events and the frozen mutation
manifest only.  Compiled graphs and OCESQ results are used exclusively on the
prediction side.  In particular, a pre-existing ``conflicting`` status does
not hide an added conflict witness: answers are compared component by
component.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from mandrel.behavior_graph.ocesq import build_ocei_instance, run_ocesq
from mandrel.behavior_graph.ocesq_contract import adapt_ocesq_result, graph_snapshot
from mandrel.behavior_graph.trace_compiler import TraceCompiler, load_jsonl


FAMILY_SPEC: dict[str, dict[str, Any]] = {
    "delete_unique_entity_observation": {
        # The removed entity observation can also be the unique state/version
        # witness.  This follows the frozen fault model, rather than treating a
        # consequent state-contract change as a false positive.
        "affected": {
            "parameter_provenance",
            "target_entity_support",
            "input_observation",
            "state_freshness",
            "entity_state_consistency",
        },
        "removed_support": {"parameter_provenance", "target_entity_support", "input_observation"},
        "missing": {"parameter_provenance", "input_observation"},
        "conflict_reason": None,
    },
    "move_observation_after_action": {
        "affected": {
            "parameter_provenance",
            "target_entity_support",
            "input_observation",
            "state_freshness",
            "entity_state_consistency",
        },
        "removed_support": {"parameter_provenance", "target_entity_support", "input_observation"},
        "missing": {"parameter_provenance", "input_observation"},
        "conflict_reason": "late_source",
    },
    "replace_action_entity_only": {
        "affected": {"parameter_provenance", "target_entity_support"},
        "removed_support": set(),
        "missing": set(),
        "conflict_reason": "dispatch_mismatch",
    },
    "inject_conflicting_response": {
        # A second read can legitimately add provenance, input, freshness, and
        # entity-state components as well as the target-identity conflict.
        "affected": {
            "parameter_provenance",
            "target_entity_support",
            "input_observation",
            "entity_state_consistency",
            "state_freshness",
        },
        "removed_support": set(),
        "missing": set(),
        "conflict_reason": "injected_identity_conflict",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tool_use(events: list[dict[str, Any]], tool_use_id: str) -> tuple[int, dict[str, Any]]:
    matches = []
    for idx, event in enumerate(events):
        if event.get("type") != "message":
            continue
        for part in event.get("message", {}).get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "tool_use" and part.get("id") == tool_use_id:
                matches.append((idx, part))
    if len(matches) != 1:
        raise ValueError(f"expected one declaration for {tool_use_id}, found {len(matches)}")
    return matches[0]


def _dispatch(events: list[dict[str, Any]], tool_use_id: str) -> tuple[int, dict[str, Any]] | None:
    matches = [
        (idx, event)
        for idx, event in enumerate(events)
        if event.get("type") == "tool_dispatch" and event.get("tool_use_id") == tool_use_id
    ]
    if len(matches) > 1:
        raise ValueError(f"expected at most one dispatch for {tool_use_id}, found {len(matches)}")
    return matches[0] if matches else None


def _values(value: Any, key: str) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key.lower() == key.lower() and isinstance(child, (str, int, float, bool)):
                found.add(str(child))
            found.update(_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.update(_values(child, key))
    return found


def build_raw_oracle(case: dict[str, Any]) -> dict[str, Any]:
    """Build and validate expectations without compiling either trace."""
    clean = load_jsonl(Path(case["clean_trace"]))
    perturbed = load_jsonl(Path(case["perturbed_trace"]))
    intervention = case["intervention"]
    family = intervention["type"]
    if family not in FAMILY_SPEC:
        raise ValueError(f"unsupported fault family: {family}")
    target_id = case["target_action"]["tool_use_id"]
    source_id = intervention["source_tool_use_id"]
    clean_action_idx, clean_action = _tool_use(clean, target_id)
    pert_action_idx, pert_action = _tool_use(perturbed, target_id)
    clean_target_dispatch = _dispatch(clean, target_id)
    pert_target_dispatch = _dispatch(perturbed, target_id)
    clean_source = _dispatch(clean, source_id)
    pert_source = _dispatch(perturbed, source_id)
    if clean_target_dispatch is None or pert_target_dispatch is None or clean_source is None:
        raise ValueError(f"{case['case_id']}: required clean/target dispatch missing")

    clean_target_idx, clean_target_event = clean_target_dispatch
    pert_target_idx, pert_target_event = pert_target_dispatch
    clean_source_idx, _ = clean_source
    key = intervention["shared_identity_key"]
    preconditions: dict[str, bool] = {
        "target_declaration_preserved": target_id == case["target_action"]["tool_use_id"],
        "target_dispatch_preserved": clean_target_event.get("tool_use_id") == pert_target_event.get("tool_use_id") == target_id,
        "clean_source_precedes_action": clean_source_idx < clean_action_idx,
        "manifest_integrity_passed": all(
            bool(value) for name, value in case.get("integrity", {}).items() if not name.endswith("sha256")
        ),
    }
    localization: set[int]
    if family == "delete_unique_entity_observation":
        preconditions["source_removed"] = pert_source is None
        localization = {clean_source_idx, clean_action_idx}
    elif family == "move_observation_after_action":
        preconditions["source_preserved"] = pert_source is not None
        preconditions["source_moved_after_action"] = bool(pert_source and pert_source[0] > pert_action_idx)
        localization = {pert_source[0], pert_action_idx} if pert_source else set()
    elif family == "replace_action_entity_only":
        clean_declared = _values(clean_action.get("input", {}), key)
        clean_dispatched = _values(clean_target_event.get("request_body", {}), key)
        pert_declared = _values(pert_action.get("input", {}), key)
        pert_dispatched = _values(pert_target_event.get("request_body", {}), key)
        preconditions["clean_declaration_dispatch_agree"] = bool(clean_declared and clean_declared == clean_dispatched)
        preconditions["perturbed_declaration_dispatch_conflict"] = bool(
            pert_declared and pert_dispatched and pert_declared.isdisjoint(pert_dispatched)
        )
        localization = {pert_action_idx, pert_target_idx}
    else:
        injected_id = intervention.get("injected_tool_use_id")
        injected = _dispatch(perturbed, injected_id) if injected_id else None
        preconditions["original_source_preserved"] = pert_source is not None
        preconditions["injected_response_present"] = injected is not None
        preconditions["injected_response_precedes_action"] = bool(injected and injected[0] < pert_action_idx)
        localization = {injected[0], pert_action_idx} if injected else set()

    return {
        "case_id": case["case_id"],
        "family": family,
        "target_tool_use_id": target_id,
        "source_tool_use_id": source_id,
        "preconditions": preconditions,
        "preconditions_passed": all(preconditions.values()),
        "expected_localization_pointers": sorted(localization),
        "affected_contracts": sorted(FAMILY_SPEC[family]["affected"]),
        "oracle_boundary": "raw events, source order, identity equality, and frozen family specification only",
    }


def _reason(summary: str) -> str:
    text = summary.lower()
    patterns = (
        ("declared action entity differs", "dispatch_mismatch"),
        ("same source request returned identity values", "injected_identity_conflict"),
        ("matching source evidence occurs after", "late_source"),
        ("prior observation contains action parameter", "prior_entity"),
        ("pre-action observation supports", "prior_observation"),
        ("assembled from multiple prior trace observations", "composite_payload"),
        ("unmatched or ambiguous target", "unmatched_target"),
    )
    for phrase, label in patterns:
        if phrase in text:
            return label
    return re.sub(r"\s+", " ", text.strip())


def _obligation_type(identifier: str) -> str:
    return identifier.rsplit(":", 1)[-1]


def _component_signature(path: Any) -> tuple[str, str, str, str, tuple[str, ...]]:
    # Values are deliberately excluded: a value mutation is represented by a
    # witness delta, while stable structure outside the affected contracts
    # remains comparable across traces whose raw indices shift.
    return (
        _obligation_type(path.obligation_id),
        path.kind,
        _reason(path.summary),
        path.path_pattern,
        tuple(sorted(path.attrs.keys())),
    )


def _counter_for_contracts(result: Any, contracts: Iterable[str]) -> Counter[tuple[Any, ...]]:
    selected = set(contracts)
    return Counter(
        _component_signature(path)
        for path in result.candidate_paths
        if _obligation_type(path.obligation_id) in selected
    )


def _paths(result: Any, *, obligation: str, kind: str | None = None, reason: str | None = None) -> list[Any]:
    rows = [p for p in result.candidate_paths if _obligation_type(p.obligation_id) == obligation]
    if kind is not None:
        rows = [p for p in rows if p.kind == kind]
    if reason is not None:
        rows = [p for p in rows if _reason(p.summary) == reason]
    return rows


def _descriptor(result: Any, obligation: str) -> Any | None:
    return next(
        (item for item in result.missing_path_descriptors if _obligation_type(item.obligation_id) == obligation),
        None,
    )


def _strict_certificate_valid(graph: Any, result: Any, contract: dict[str, Any], obligation: str) -> bool:
    descriptor = _descriptor(result, obligation)
    normalized = next((item for item in contract["obligations"] if item["type"] == obligation), None)
    if descriptor is None or normalized is None or normalized["status"] != "missing":
        return False
    certificate = normalized.get("missing_certificate") or {}
    expected_nodes = sorted(node.id for node in graph.nodes)
    expected_edges = sorted(f"{edge.source}->{edge.type}->{edge.target}" for edge in graph.edges)
    cert_descriptor = certificate.get("descriptor") or {}
    return bool(
        certificate.get("complete") is True
        and certificate.get("scope") == "full_behavior_graph"
        and certificate.get("graph_snapshot") == graph_snapshot(graph)
        and certificate.get("searched_node_ids") == expected_nodes
        and certificate.get("searched_edge_refs") == expected_edges
        and cert_descriptor.get("obligation_id") == descriptor.obligation_id
        and result.action_id in descriptor.anchor_node_ids
    )


def _compile_query(trace_path: str, target_tool_use_id: str) -> tuple[Any, Any, dict[str, Any]]:
    graph = TraceCompiler(load_jsonl(Path(trace_path))).compile().graph
    action = next(
        node for node in graph.nodes
        if node.type == "ToolAction" and node.attrs.get("tool_use_id") == target_tool_use_id
    )
    result = run_ocesq(
        graph,
        action.id,
        budget={"max_nodes": 24, "max_edges": 36},
        ocei=build_ocei_instance(graph),
    )
    return graph, result, adapt_ocesq_result(graph, result)


def _prf(predicted: set[int], expected: set[int]) -> tuple[float, float, float]:
    overlap = len(predicted & expected)
    precision = overlap / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = overlap / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_case(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    oracle = build_raw_oracle(case)
    family = oracle["family"]
    spec = FAMILY_SPEC[family]
    target_id = oracle["target_tool_use_id"]
    clean_graph, clean, _ = _compile_query(case["clean_trace"], target_id)
    pert_graph, perturbed, pert_contract = _compile_query(case["perturbed_trace"], target_id)
    source_id = oracle["source_tool_use_id"]
    clean_source = next(
        node for node in clean_graph.nodes
        if node.type == "ToolObservation" and node.attrs.get("tool_use_id") == source_id
    )
    pert_source = next(
        (node for node in pert_graph.nodes if node.type == "ToolObservation" and node.attrs.get("tool_use_id") == source_id),
        None,
    )

    checks: dict[str, bool] = {}
    predicted_localization: set[int] = set()
    for obligation in sorted(spec["removed_support"]):
        clean_linked = [
            path for path in _paths(clean, obligation=obligation, kind="supporting")
            if clean_source.source_event_idx in path.source_event_pointers
        ]
        pert_linked = [
            path for path in _paths(perturbed, obligation=obligation, kind="supporting")
            if pert_source is not None and pert_source.source_event_idx in path.source_event_pointers
        ]
        checks[f"removed_source_support:{obligation}"] = bool(clean_linked) and not pert_linked
        for path in clean_linked if family == "delete_unique_entity_observation" else pert_linked:
            predicted_localization.update(path.source_event_pointers)

    certificate_rows = []
    for obligation in sorted(spec["missing"]):
        descriptor = _descriptor(perturbed, obligation)
        checks[f"unmet_status:{obligation}"] = perturbed.obligation_statuses.get(obligation) == "missing"
        checks[f"missing_descriptor:{obligation}"] = bool(
            descriptor is not None and perturbed.action_id in descriptor.anchor_node_ids
        )
        certificate_rows.append(_strict_certificate_valid(pert_graph, perturbed, pert_contract, obligation))

    reason = spec["conflict_reason"]
    if reason:
        conflict_paths = _paths(
            perturbed,
            obligation="target_entity_support",
            kind="conflicting",
            reason=reason,
        )
        expected_pointers = set(oracle["expected_localization_pointers"])
        localized = [path for path in conflict_paths if expected_pointers.issubset(path.source_event_pointers)]
        checks[f"added_conflict:{reason}"] = bool(localized)
        for path in localized:
            predicted_localization.update(path.source_event_pointers)

    if family == "inject_conflicting_response":
        clean_original = [
            path for path in _paths(clean, obligation="target_entity_support", kind="supporting", reason="prior_entity")
            if clean_source.source_event_idx in path.source_event_pointers
        ]
        assert pert_source is not None
        pert_original = [
            path for path in _paths(perturbed, obligation="target_entity_support", kind="supporting", reason="prior_entity")
            if pert_source.source_event_idx in path.source_event_pointers
        ]
        checks["preserved_original_support"] = bool(clean_original and pert_original)

    all_contracts = set(clean.obligation_statuses) | set(perturbed.obligation_statuses)
    invariant_contracts = all_contracts - set(spec["affected"])
    unexpected_status = sorted(
        obligation for obligation in invariant_contracts
        if clean.obligation_statuses.get(obligation) != perturbed.obligation_statuses.get(obligation)
    )
    clean_invariant = _counter_for_contracts(clean, invariant_contracts)
    pert_invariant = _counter_for_contracts(perturbed, invariant_contracts)
    unexpected_components = list((clean_invariant - pert_invariant).elements()) + list(
        (pert_invariant - clean_invariant).elements()
    )

    expected_localization = set(oracle["expected_localization_pointers"])
    loc_p, loc_r, loc_f1 = _prf(predicted_localization, expected_localization)
    component_recall = mean(checks.values()) if checks else 1.0
    certificate_validity = mean(certificate_rows) if certificate_rows else 1.0
    status_transition_correct = all(
        perturbed.obligation_statuses.get(obligation) == "missing" for obligation in spec["missing"]
    ) and (not reason or bool(_paths(
        perturbed, obligation="target_entity_support", kind="conflicting", reason=reason
    )))
    exact = bool(
        oracle["preconditions_passed"]
        and all(checks.values())
        and not unexpected_status
        and not unexpected_components
        and certificate_validity == 1.0
        and loc_p == loc_r == 1.0
    )
    row = {
        "case_id": case["case_id"],
        "family": family,
        "oracle_preconditions_passed": oracle["preconditions_passed"],
        "component_checks": checks,
        "expected_component_recall": component_recall,
        "status_transition_correct": status_transition_correct,
        "expected_localization_pointers": sorted(expected_localization),
        "predicted_localization_pointers": sorted(predicted_localization),
        "localization_precision": loc_p,
        "localization_recall": loc_r,
        "localization_f1": loc_f1,
        "certificate_validity": certificate_validity,
        "unexpected_status_changes": unexpected_status,
        "unexpected_component_changes": [list(item) for item in unexpected_components],
        "unexpected_component_change": bool(unexpected_status or unexpected_components),
        "contract_delta_exact": exact,
        "clean_statuses": clean.obligation_statuses,
        "perturbed_statuses": perturbed.obligation_statuses,
    }
    return oracle, row


def summarize(rows: list[dict[str, Any]], oracles: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    metrics = (
        "oracle_preconditions_passed",
        "contract_delta_exact",
        "expected_component_recall",
        "status_transition_correct",
        "localization_precision",
        "localization_recall",
        "localization_f1",
        "certificate_validity",
        "unexpected_component_change",
    )

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {"pairs": len(selected), **{metric: mean(row[metric] for row in selected) for metric in metrics}}

    failed_checks = Counter(
        name for row in rows for name, passed in row["component_checks"].items() if not passed
    )
    return {
        "protocol": "apc-fault-injection-v1-pilot",
        "overall": aggregate(rows),
        "by_family": {family: aggregate(selected) for family, selected in sorted(grouped.items())},
        "failure_taxonomy": {
            "failed_expected_components": dict(sorted(failed_checks.items())),
            "invalid_scope_certificates": sum(row["certificate_validity"] < 1.0 for row in rows),
            "imperfect_localization": sum(row["localization_f1"] < 1.0 for row in rows),
            "unexpected_component_changes": sum(row["unexpected_component_change"] for row in rows),
        },
        "oracle_rows": len(oracles),
        "human_labels_used": False,
        "model_labels_used": False,
        "ground_truth_boundary": "raw traces plus frozen mutation specifications; OCESQ is prediction-side only",
        "certificate_note": "strict validity checks graph digest, complete full-graph scope, enumerated nodes/edges, descriptor identity, and root anchor",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/audit_intervention_v2_pilot/manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/apc_fault_injection_v1_pilot"),
    )
    args = parser.parse_args()
    evaluated = [evaluate_case(case) for case in read_jsonl(args.manifest)]
    oracles = [item[0] for item in evaluated]
    rows = [item[1] for item in evaluated]
    summary = summarize(rows, oracles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("oracle.jsonl", oracles), ("results.jsonl", rows)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
