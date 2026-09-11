#!/usr/bin/env python3
"""Evaluate fixed-schema applicability on external traces without semantic labels."""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from mandrel.behavior_graph.ocesq import build_ocei_instance, high_impact_actions, run_ocesq
from mandrel.behavior_graph.trace_compiler import TraceCompiler, _is_effectful_tool, load_jsonl


SUPPORTED_EVENT_TYPES = {"trace_start", "message", "tool_dispatch", "audit_snapshot", "trace_end", "grading_result"}
EXCLUDED_LABEL_TYPES = {"atbench_label"}
COMPILER_IDENTITY_KEYS = {"id", "entity_id", "message_id", "contact_id", "customer_id", "ticket_id", "integration_id", "email", "to", "from", "name", "subject"}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def _tool_uses(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    rows = []
    for event in events:
        if event.get("type") != "message":
            continue
        for part in event.get("message", {}).get("content", []) or []:
            if part.get("type") == "tool_use":
                rows.append((str(part.get("id") or ""), str(part.get("name") or "")))
    return rows


def _anchor_fields(value: Any) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            tail = str(key).lower().rsplit(".", 1)[-1]
            if isinstance(child, (str, int)) and (tail == "id" or tail.endswith("_id") or tail.endswith("_ids") or tail in {"email", "to", "from", "path", "file", "recipient", "name", "subject"}):
                found.add((tail, str(child)))
            found.update(_anchor_fields(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_anchor_fields(child))
    return found


def _stable_identity_fields(value: Any) -> set[tuple[str, str]]:
    return {
        (key, item)
        for key, item in _anchor_fields(value)
        if (tail := re.sub(r"\[\d+\]$", "", key.lower())) in COMPILER_IDENTITY_KEYS
        or tail.endswith("_id")
        or tail.endswith("_ids")
    }


def _artifact_anchor_fields(value: Any) -> set[tuple[str, str]]:
    return {
        (key, item)
        for key, item in _anchor_fields(value)
        if re.sub(r"\[\d+\]$", "", key.lower()) in {"path", "file"}
    }


def _bootstrap_ci(rows: list[dict[str, Any]], field: str, samples: int, seed: int) -> list[float] | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return None
    rng = random.Random(seed)
    estimates = [mean(rng.choice(values) for _ in values) for _ in range(samples)]
    return [round(_percentile(estimates, 0.025) or 0.0, 4), round(_percentile(estimates, 0.975) or 0.0, 4)]


def _audit_trace(path: Path) -> dict[str, Any]:
    events = load_jsonl(path)
    event_counts = Counter(str(event.get("type")) for event in events)
    supported_events = sum(event_counts[event_type] for event_type in SUPPORTED_EVENT_TYPES)
    excluded_events = sum(event_counts[event_type] for event_type in EXCLUDED_LABEL_TYPES)
    unknown_events = len(events) - supported_events - excluded_events
    declarations = _tool_uses(events)
    dispatches = [(str(event.get("tool_use_id") or ""), str(event.get("tool_name") or "")) for event in events if event.get("type") == "tool_dispatch"]
    declaration_ids = Counter(tool_id for tool_id, _ in declarations if tool_id)
    dispatch_ids = Counter(tool_id for tool_id, _ in dispatches if tool_id)
    matched = sum((declaration_ids & dispatch_ids).values())
    failures: list[str] = []
    if unknown_events:
        failures.append("unknown_event_type")
    if matched < len(declarations):
        failures.append("declaration_without_dispatch")
    if matched < len(dispatches):
        failures.append("dispatch_without_declaration")

    compile_started = time.perf_counter_ns()
    try:
        compilation = TraceCompiler(events).compile()
    except Exception as exc:  # pragma: no cover - retained for external compatibility reporting
        return {
            "trace_file": str(path), "raw_events": len(events), "compile_success": 0.0,
            "compile_exception": f"{type(exc).__name__}: {exc}", "failure_types": [*failures, "compile_exception"],
            "event_contract_rate": _ratio(supported_events, supported_events + unknown_events), "query_instantiation_rate": None,
        }
    compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000
    graph = compilation.graph
    actions = [node for node in graph.nodes if node.type == "ToolAction"]
    observations = [node for node in graph.nodes if node.type == "ToolObservation"]
    event_nodes = [node for node in graph.nodes if node.type not in {"Requirement", "StructuralSignals"}]
    valid_pointers = sum(node.source_event_idx is not None and 0 <= node.source_event_idx < len(events) for node in event_nodes)
    if len(actions) != len(declarations):
        failures.append("compiled_action_count_mismatch")
    if len(observations) != len(dispatches):
        failures.append("compiled_observation_count_mismatch")
    if valid_pointers < len(event_nodes):
        failures.append("invalid_source_pointer")

    anchor_actions = [node for node in actions if _anchor_fields(node.attrs.get("input", {}))]
    artifact_anchor_actions = [node for node in actions if _artifact_anchor_fields(node.attrs.get("input", {}))]
    identity_actions = [node for node in actions if _stable_identity_fields(node.attrs.get("input", {}))]
    mentioned_actions = {
        edge.source for edge in graph.edges
        if edge.type == "mentions" and any(node.id == edge.source and node.type == "ToolAction" for node in actions)
    }
    identity_observations = [node for node in observations if _stable_identity_fields(node.attrs.get("response_body"))]
    resolved_observations = {edge.source for edge in graph.edges if edge.type == "resolves"}
    node_map = {node.id: node for node in graph.nodes}
    entity_observations = {
        edge.source for edge in graph.edges
        if edge.type == "observes"
        and node_map.get(edge.source) and node_map[edge.source].type == "ToolObservation"
        and node_map.get(edge.target) and node_map[edge.target].type == "ServiceEntity"
    }
    anchor_bound_actions = {
        edge.source for edge in graph.edges
        if node_map.get(edge.source) and node_map[edge.source].type == "ToolAction"
        and node_map.get(edge.target) and node_map[edge.target].type in {"ServiceEntity", "FileArtifact", "FileVersion", "EntityVersion"}
    }
    artifact_bound_actions = {
        edge.source for edge in graph.edges
        if node_map.get(edge.source) and node_map[edge.source].type == "ToolAction"
        and node_map.get(edge.target) and node_map[edge.target].type in {"FileArtifact", "FileVersion"}
    }
    observations_by_call = {str(node.attrs.get("tool_use_id") or ""): node for node in observations}
    identity_action_tools = Counter(str(node.attrs.get("tool_name") or "unknown") for node in identity_actions)
    bound_identity_action_tools = Counter(str(node.attrs.get("tool_name") or "unknown") for node in identity_actions if node.id in mentioned_actions)
    unbound_action_reasons: Counter[str] = Counter()
    for node in identity_actions:
        if node.id in mentioned_actions:
            continue
        observation = observations_by_call.get(str(node.attrs.get("tool_use_id") or ""))
        request_has_identity = bool(observation and _stable_identity_fields(observation.attrs.get("request_body", {})))
        unbound_action_reasons["compiler_missed_dispatch_identity" if request_has_identity else "dispatch_request_lacks_identity"] += 1
    identity_observation_tools = Counter(str(node.attrs.get("tool_name") or "unknown") for node in identity_observations)
    resolved_identity_observation_tools = Counter(str(node.attrs.get("tool_name") or "unknown") for node in identity_observations if node.id in resolved_observations)
    resolution_outcomes: Counter[str] = Counter()
    eligible_unique_resolution_observations = []
    for node in identity_observations:
        tool_name = str(node.attrs.get("tool_name") or "")
        result_count = int(node.attrs.get("result_count") or 0)
        is_effectful = _is_effectful_tool(tool_name)
        if not is_effectful and result_count == 1:
            eligible_unique_resolution_observations.append(node)
        if node.id in resolved_observations:
            resolution_outcomes["resolved_singleton_read"] += 1
        elif is_effectful:
            resolution_outcomes["effect_result_reference"] += 1
        elif result_count > 1:
            resolution_outcomes["ambiguous_candidate_set"] += 1
        else:
            resolution_outcomes["unsupported_response_shape"] += 1
    targets = high_impact_actions(graph)
    query_successes = 0
    query_source_valid = 0
    query_times = []
    obligation_applicability: Counter[str] = Counter()
    obligation_statuses: Counter[str] = Counter()
    try:
        ocei = build_ocei_instance(graph)
    except Exception:
        ocei = None
        if targets:
            failures.append("index_build_exception")
    for action in targets:
        started = time.perf_counter_ns()
        try:
            result = run_ocesq(graph, action.id, ocei=ocei)
        except Exception:
            failures.append("query_exception")
            continue
        query_times.append((time.perf_counter_ns() - started) / 1_000_000)
        query_successes += 1
        if all(0 <= pointer < len(events) for pointer in result.source_event_pointers):
            query_source_valid += 1
        for obligation in result.obligations:
            obligation_statuses[f"{obligation.type}:{obligation.status}"] += 1
            if obligation.status != "not_applicable":
                obligation_applicability[obligation.type] += 1

    return {
        "trace_file": str(path),
        "raw_events": len(events),
        "supported_events": supported_events,
        "excluded_label_events": excluded_events,
        "unknown_events": unknown_events,
        "unknown_event_types": sorted(set(event_counts) - SUPPORTED_EVENT_TYPES - EXCLUDED_LABEL_TYPES),
        "event_contract_rate": _ratio(supported_events, supported_events + unknown_events),
        "compile_success": 1.0,
        "compile_ms": compile_ms,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "declarations": len(declarations),
        "dispatches": len(dispatches),
        "matched_tool_calls": matched,
        "actions": len(actions),
        "observations": len(observations),
        "action_compilation_rate": _ratio(len(actions), len(declarations)),
        "observation_compilation_rate": _ratio(len(observations), len(dispatches)),
        "source_pointer_valid_rate": _ratio(valid_pointers, len(event_nodes)),
        "identity_actions": len(identity_actions),
        "identity_bound_actions": sum(node.id in mentioned_actions for node in identity_actions),
        "identity_binding_rate": _ratio(sum(node.id in mentioned_actions for node in identity_actions), len(identity_actions)),
        "anchor_actions": len(anchor_actions),
        "anchor_bound_actions": sum(node.id in anchor_bound_actions for node in anchor_actions),
        "artifact_anchor_actions": len(artifact_anchor_actions),
        "artifact_bound_actions": sum(node.id in artifact_bound_actions for node in artifact_anchor_actions),
        "artifact_anchor_materialization_rate": _ratio(sum(node.id in artifact_bound_actions for node in artifact_anchor_actions), len(artifact_anchor_actions)),
        "identity_observations": len(identity_observations),
        "captured_identity_observations": sum(node.id in entity_observations for node in identity_observations),
        "identity_observation_capture_rate": _ratio(sum(node.id in entity_observations for node in identity_observations), len(identity_observations)),
        "eligible_unique_resolution_observations": len(eligible_unique_resolution_observations),
        "eligible_resolved_observations": sum(node.id in resolved_observations for node in eligible_unique_resolution_observations),
        "eligible_unique_resolution_fidelity": _ratio(sum(node.id in resolved_observations for node in eligible_unique_resolution_observations), len(eligible_unique_resolution_observations)),
        "resolved_identity_observations": sum(node.id in resolved_observations for node in identity_observations),
        "resolution_outcomes": dict(resolution_outcomes),
        "high_impact_actions": len(targets),
        "query_successes": query_successes,
        "query_source_valid": query_source_valid,
        "audited_action_selection_rate": _ratio(len(targets), len(actions)),
        "query_instantiation_rate": _ratio(query_successes, len(targets)),
        "query_source_valid_rate": _ratio(query_source_valid, query_successes),
        "query_p50_ms": median(query_times) if query_times else None,
        "query_p95_ms": _percentile(query_times, 0.95),
        "obligation_applicability": dict(obligation_applicability),
        "obligation_statuses": dict(obligation_statuses),
        "identity_action_tools": dict(identity_action_tools),
        "bound_identity_action_tools": dict(bound_identity_action_tools),
        "unbound_action_reasons": dict(unbound_action_reasons),
        "identity_observation_tools": dict(identity_observation_tools),
        "resolved_identity_observation_tools": dict(resolved_identity_observation_tools),
        "failure_types": sorted(set(failures)),
    }


def _summarize(name: str, rows: list[dict[str, Any]], bootstrap_samples: int) -> dict[str, Any]:
    failures = Counter(failure for row in rows for failure in row.get("failure_types", []))
    unknown_types = Counter(event_type for row in rows for event_type in row.get("unknown_event_types", []))
    obligations = Counter()
    statuses = Counter()
    identity_action_tools = Counter()
    bound_identity_action_tools = Counter()
    unbound_action_reasons = Counter()
    identity_observation_tools = Counter()
    resolved_identity_observation_tools = Counter()
    resolution_outcomes = Counter()
    for row in rows:
        obligations.update(row.get("obligation_applicability", {}))
        statuses.update(row.get("obligation_statuses", {}))
        identity_action_tools.update(row.get("identity_action_tools", {}))
        bound_identity_action_tools.update(row.get("bound_identity_action_tools", {}))
        unbound_action_reasons.update(row.get("unbound_action_reasons", {}))
        identity_observation_tools.update(row.get("identity_observation_tools", {}))
        resolved_identity_observation_tools.update(row.get("resolved_identity_observation_tools", {}))
        resolution_outcomes.update(row.get("resolution_outcomes", {}))
    total = lambda field: sum(int(row.get(field) or 0) for row in rows)
    compiled = [row for row in rows if row.get("compile_success") == 1.0]
    query_times = [float(row["query_p50_ms"]) for row in compiled if row.get("query_p50_ms") is not None]
    rate_fields = ("event_contract_rate", "action_compilation_rate", "observation_compilation_rate", "source_pointer_valid_rate", "identity_binding_rate", "artifact_anchor_materialization_rate", "identity_observation_capture_rate", "eligible_unique_resolution_fidelity", "audited_action_selection_rate", "query_instantiation_rate", "query_source_valid_rate")
    macro = {field: round(mean(float(row[field]) for row in rows if row.get(field) is not None), 4) if any(row.get(field) is not None for row in rows) else None for field in rate_fields}
    ci = {field: _bootstrap_ci(rows, field, bootstrap_samples, 20260721 + index) for index, field in enumerate(rate_fields)}
    micro = {
        "event_contract_rate": _ratio(total("supported_events"), total("supported_events") + total("unknown_events")),
        "declaration_dispatch_link_rate": _ratio(total("matched_tool_calls"), total("declarations")),
        "dispatch_declaration_link_rate": _ratio(total("matched_tool_calls"), total("dispatches")),
        "action_compilation_rate": _ratio(total("actions"), total("declarations")),
        "observation_compilation_rate": _ratio(total("observations"), total("dispatches")),
        "identity_binding_rate": _ratio(total("identity_bound_actions"), total("identity_actions")),
        "artifact_anchor_materialization_rate": _ratio(total("artifact_bound_actions"), total("artifact_anchor_actions")),
        "identity_observation_capture_rate": _ratio(total("captured_identity_observations"), total("identity_observations")),
        "eligible_unique_resolution_fidelity": _ratio(total("eligible_resolved_observations"), total("eligible_unique_resolution_observations")),
        "audited_action_selection_rate": _ratio(total("high_impact_actions"), total("actions")),
        "query_instantiation_rate": _ratio(total("query_successes"), total("high_impact_actions")),
        "query_source_valid_rate": _ratio(total("query_source_valid"), total("query_successes")),
    }
    return {
        "source": name,
        "traces": len(rows),
        "compile_success_rate": mean(row.get("compile_success", 0.0) for row in rows) if rows else None,
        "raw_events": total("raw_events"),
        "excluded_label_events": total("excluded_label_events"),
        "unknown_events": total("unknown_events"),
        "unknown_event_types": dict(unknown_types),
        "actions": total("actions"),
        "high_impact_actions": total("high_impact_actions"),
        "query_successes": total("query_successes"),
        "avg_nodes": mean(row["nodes"] for row in compiled) if compiled else None,
        "avg_edges": mean(row["edges"] for row in compiled) if compiled else None,
        "compiler_p50_ms": _percentile([float(row["compile_ms"]) for row in compiled], 0.5),
        "compiler_p95_ms": _percentile([float(row["compile_ms"]) for row in compiled], 0.95),
        "trace_query_p50_median_ms": median(query_times) if query_times else None,
        "trace_query_p50_p95_ms": _percentile(query_times, 0.95),
        "micro_rates": {key: round(value, 4) if value is not None else None for key, value in micro.items()},
        "macro_trace_rates": macro,
        "macro_trace_bootstrap95": ci,
        "pipeline_failure_taxonomy": dict(failures),
        "unbound_action_reason_counts": dict(unbound_action_reasons),
        "identity_action_tools": dict(identity_action_tools),
        "bound_identity_action_tools": dict(bound_identity_action_tools),
        "identity_observation_tools": dict(identity_observation_tools),
        "resolved_identity_observation_tools": dict(resolved_identity_observation_tools),
        "applicability_counts": {
            "stable_identity_actions": total("identity_actions"),
            "stable_identity_bound_actions": total("identity_bound_actions"),
            "artifact_anchor_actions": total("artifact_anchor_actions"),
            "artifact_bound_actions": total("artifact_bound_actions"),
            "identity_bearing_observations": total("identity_observations"),
            "captured_identity_observations": total("captured_identity_observations"),
            "eligible_unique_resolution_observations": total("eligible_unique_resolution_observations"),
            "eligible_resolved_observations": total("eligible_resolved_observations"),
        },
        "identity_observation_outcomes": dict(resolution_outcomes),
        "obligation_applicability_counts": dict(obligations),
        "registry_status_counts_descriptive_only": dict(statuses),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="NAME=GLOB_PATTERN")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/external_applicability_v2"))
    args = parser.parse_args()
    reports = {}
    all_rows = []
    for raw in args.source:
        name, separator, pattern = raw.partition("=")
        if not separator:
            raise ValueError(f"invalid source: {raw}")
        paths = [Path(path) for path in sorted(glob.glob(pattern, recursive=True))]
        if args.limit > 0:
            paths = paths[: args.limit]
        rows = [_audit_trace(path) for path in paths]
        reports[name] = _summarize(name, rows, args.bootstrap_samples)
        for row in rows:
            all_rows.append({"source": name, **row})
    result = {
        "protocol": "external-applicability-v2",
        "schema_and_registry_frozen": True,
        "label_usage": "none",
        "semantic_accuracy_computed": False,
        "sample_rule": "all glob-matched traces" if args.limit == 0 else f"lexicographically first {args.limit} per source",
        "bootstrap": {"unit": "trace", "samples": args.bootstrap_samples, "seed_family": 20260721},
        "sources": reports,
        "interpretation": "External schema/query applicability and system compatibility only; registry statuses are descriptive, not gold labels.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({name: {key: report[key] for key in ("traces", "compile_success_rate", "actions", "high_impact_actions", "query_successes", "unknown_events", "micro_rates", "pipeline_failure_taxonomy")} for name, report in reports.items()}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
