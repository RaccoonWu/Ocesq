"""Frozen action-evidence atoms used by the V1 raw-trace pilot."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .schema import BehaviorGraph

QUERY_ID = "action_execution_evidence_v1"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_atom(
    atom_type: str,
    semantic_key: str,
    value: Any,
    origin_event_id: str,
    source_pointers: list[int],
    temporal_position: str,
    source_class: str,
    polarity: str = "support",
) -> dict[str, Any]:
    value_digest = canonical_digest(value)
    identity = [atom_type, semantic_key, value_digest, origin_event_id, polarity]
    atom_id = "ea_" + canonical_digest(identity)[:20]
    return {
        "atom_id": atom_id,
        "atom_type": atom_type,
        "semantic_key": semantic_key,
        "value_digest": value_digest,
        "origin_event_id": origin_event_id,
        "source_pointers": sorted(set(source_pointers)),
        "temporal_position": temporal_position,
        "source_class": source_class,
        "polarity": polarity,
    }


def _tool_use(events: list[dict[str, Any]], tool_use_id: str) -> tuple[int, dict[str, Any]]:
    for idx, event in enumerate(events):
        if event.get("type") != "message":
            continue
        for part in event.get("message", {}).get("content", []) or []:
            if part.get("type") == "tool_use" and part.get("id") == tool_use_id:
                return idx, part
    raise ValueError(f"tool_use_id not declared: {tool_use_id}")


def _dispatch(events: list[dict[str, Any]], tool_use_id: str) -> tuple[int, dict[str, Any]] | None:
    return next(
        ((idx, event) for idx, event in enumerate(events) if event.get("type") == "tool_dispatch" and event.get("tool_use_id") == tool_use_id),
        None,
    )


def _task_event(events: list[dict[str, Any]]) -> tuple[int, dict[str, Any]] | None:
    return next(
        ((idx, event) for idx, event in enumerate(events) if event.get("type") == "message" and event.get("message", {}).get("role") == "user"),
        None,
    )


def _report_event(events: list[dict[str, Any]], after: int) -> tuple[int, dict[str, Any]] | None:
    rows = [
        (idx, event)
        for idx, event in enumerate(events)
        if idx > after and event.get("type") == "message" and event.get("message", {}).get("role") == "assistant"
    ]
    return rows[-1] if rows else None


def extract_raw_atoms(
    events: list[dict[str, Any]],
    target_tool_use_id: str,
    *,
    source_tool_use_id: str | None,
    intervention_type: str | None,
    injected_tool_use_id: str | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    """Extract gold atoms using raw events only; no graph/query output is read."""
    action_idx, action = _tool_use(events, target_tool_use_id)
    target_dispatch = _dispatch(events, target_tool_use_id)
    if target_dispatch is None:
        raise ValueError(f"target dispatch missing: {target_tool_use_id}")
    dispatch_idx, dispatch = target_dispatch
    key = f"tool_use:{target_tool_use_id}"
    atoms: list[dict[str, Any]] = []

    task = _task_event(events)
    if task:
        task_idx, task_event = task
        atoms.append(make_atom("task_constraint", key, task_event.get("message", {}).get("content", []), f"event:{task_idx}", [task_idx], "before", "user_constraint"))
    atoms.append(make_atom("declared_action", key, {"tool_name": action.get("name"), "input": action.get("input", {})}, target_tool_use_id, [action_idx], "at_action", "assistant_declaration"))
    atoms.append(make_atom("dispatch_request", key, dispatch.get("request_body", {}), target_tool_use_id, [dispatch_idx], "after", "executed_dispatch"))
    atoms.append(make_atom("tool_response", key, {"status": dispatch.get("response_status"), "body": dispatch.get("response_body")}, target_tool_use_id, [dispatch_idx], "after", "dispatch_response"))
    atoms.append(make_atom("external_effect", key, {"tool_name": dispatch.get("tool_name"), "request": dispatch.get("request_body"), "response": dispatch.get("response_body")}, target_tool_use_id, [dispatch_idx], "after", "external_effect"))

    declared = action.get("input", {})
    executed = dispatch.get("request_body", {})
    agrees = declared == executed
    atoms.append(make_atom("dispatch_correspondence", key, {"declared": declared, "executed": executed, "agrees": agrees}, target_tool_use_id, [action_idx, dispatch_idx], "cross_boundary", "identity_check", "support" if agrees else "conflict"))

    source_dispatch = _dispatch(events, source_tool_use_id) if source_tool_use_id else None
    if source_dispatch:
        source_idx, source = source_dispatch
        temporal = "before" if source_idx < action_idx else "after"
        atoms.append(make_atom("prior_observation", f"source:{source_tool_use_id}", {"status": source.get("response_status"), "body": source.get("response_body")}, source_tool_use_id or "", [source_idx], temporal, "independent_tool_call"))
        atoms.append(make_atom("temporal_relation", f"{source_tool_use_id}->{target_tool_use_id}", {"source_index": source_idx, "action_index": action_idx, "causal": source_idx < action_idx}, f"{source_tool_use_id}->{target_tool_use_id}", [source_idx, action_idx], "cross_boundary", "event_order", "support" if source_idx < action_idx else "conflict"))

    if injected_tool_use_id:
        injected = _dispatch(events, injected_tool_use_id)
        if injected:
            conflict_idx, conflict = injected
            atoms.append(make_atom("conflict", f"source:{source_tool_use_id}", {"status": conflict.get("response_status"), "body": conflict.get("response_body")}, injected_tool_use_id, [conflict_idx], "before" if conflict_idx < action_idx else "after", "independent_tool_call", "conflict"))

    if not agrees:
        atoms.append(make_atom("conflict", key, {"declared": declared, "executed": executed}, target_tool_use_id, [action_idx, dispatch_idx], "cross_boundary", "dispatch_mismatch", "conflict"))

    report = _report_event(events, dispatch_idx)
    if report:
        report_idx, report_event = report
        atoms.append(make_atom("agent_report", key, report_event.get("message", {}).get("content", []), f"event:{report_idx}", [report_idx], "after", "assistant_report"))

    if intervention_type == "delete_unique_entity_observation":
        status = "missing"
        absence = {"scope": "trace_prefix", "start_index": 0, "end_index": action_idx - 1, "predicate": f"matching observation from {source_tool_use_id}", "complete": True}
    elif intervention_type in {"replace_action_entity_only", "move_observation_after_action", "inject_conflicting_response"}:
        status = "conflicting"
        absence = None
    else:
        status = "supported"
        absence = None
    return atoms, status, absence


def extract_graph_atoms(
    graph: BehaviorGraph,
    target_tool_use_id: str,
    *,
    source_tool_use_id: str | None,
    injected_tool_use_id: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Extract the same atom projection from a compiled behavior graph."""
    nodes = graph.nodes
    action = next((node for node in nodes if node.type == "ToolAction" and node.attrs.get("tool_use_id") == target_tool_use_id), None)
    if action is None or action.source_event_idx is None:
        return [], "missing"
    observation = next((node for node in nodes if node.type == "ToolObservation" and node.attrs.get("tool_use_id") == target_tool_use_id), None)
    if observation is None or observation.source_event_idx is None:
        return [], "missing"
    key = f"tool_use:{target_tool_use_id}"
    atoms: list[dict[str, Any]] = []
    task = next((node for node in nodes if node.type == "Message" and node.attrs.get("role") == "user"), None)
    if task is not None and task.source_event_idx is not None:
        atoms.append(make_atom("task_constraint", key, task.attrs.get("content", []), f"event:{task.source_event_idx}", [task.source_event_idx], "before", "user_constraint"))
    atoms.append(make_atom("declared_action", key, {"tool_name": action.attrs.get("tool_name"), "input": action.attrs.get("input", {})}, target_tool_use_id, [action.source_event_idx], "at_action", "assistant_declaration"))
    request = observation.attrs.get("request_body", {})
    response = observation.attrs.get("response_body")
    atoms.append(make_atom("dispatch_request", key, request, target_tool_use_id, [observation.source_event_idx], "after", "executed_dispatch"))
    atoms.append(make_atom("tool_response", key, {"status": observation.attrs.get("status"), "body": response}, target_tool_use_id, [observation.source_event_idx], "after", "dispatch_response"))
    effect = next((node for node in nodes if node.type == "ExternalEffect" and node.source_event_idx == observation.source_event_idx and node.attrs.get("tool_name") == action.attrs.get("tool_name")), None)
    if effect:
        atoms.append(make_atom("external_effect", key, {"tool_name": effect.attrs.get("tool_name"), "request": effect.attrs.get("request_body"), "response": effect.attrs.get("response_body")}, target_tool_use_id, [observation.source_event_idx], "after", "external_effect"))
    declared = action.attrs.get("input", {})
    agrees = declared == request
    atoms.append(make_atom("dispatch_correspondence", key, {"declared": declared, "executed": request, "agrees": agrees}, target_tool_use_id, [action.source_event_idx, observation.source_event_idx], "cross_boundary", "identity_check", "support" if agrees else "conflict"))

    source = next((node for node in nodes if node.type == "ToolObservation" and node.attrs.get("tool_use_id") == source_tool_use_id), None)
    if source is not None and source.source_event_idx is not None:
        temporal = "before" if source.source_event_idx < action.source_event_idx else "after"
        atoms.append(make_atom("prior_observation", f"source:{source_tool_use_id}", {"status": source.attrs.get("status"), "body": source.attrs.get("response_body")}, source_tool_use_id or "", [source.source_event_idx], temporal, "independent_tool_call"))
        atoms.append(make_atom("temporal_relation", f"{source_tool_use_id}->{target_tool_use_id}", {"source_index": source.source_event_idx, "action_index": action.source_event_idx, "causal": source.source_event_idx < action.source_event_idx}, f"{source_tool_use_id}->{target_tool_use_id}", [source.source_event_idx, action.source_event_idx], "cross_boundary", "event_order", "support" if source.source_event_idx < action.source_event_idx else "conflict"))
    injected = next((node for node in nodes if node.type == "ToolObservation" and node.attrs.get("tool_use_id") == injected_tool_use_id), None)
    if injected is not None and injected.source_event_idx is not None:
        atoms.append(make_atom("conflict", f"source:{source_tool_use_id}", {"status": injected.attrs.get("status"), "body": injected.attrs.get("response_body")}, injected_tool_use_id or "", [injected.source_event_idx], "before" if injected.source_event_idx < action.source_event_idx else "after", "independent_tool_call", "conflict"))
    if not agrees:
        atoms.append(make_atom("conflict", key, {"declared": declared, "executed": request}, target_tool_use_id, [action.source_event_idx, observation.source_event_idx], "cross_boundary", "dispatch_mismatch", "conflict"))
    reports = [node for node in nodes if node.type == "AssistantMessage" and node.source_event_idx is not None and node.source_event_idx > observation.source_event_idx]
    if reports:
        report = reports[-1]
        atoms.append(make_atom("agent_report", key, report.attrs.get("content", []), f"event:{report.source_event_idx}", [report.source_event_idx], "after", "assistant_report"))

    conflict = any(atom["polarity"] == "conflict" for atom in atoms)
    causal_source = any(atom["atom_type"] == "prior_observation" and atom["temporal_position"] == "before" for atom in atoms)
    status = "conflicting" if conflict else "supported" if causal_source else "missing"
    return atoms, status
