"""Serializable OCESQ contracts and an implementation-independent verifier.

The verifier deliberately depends only on the behavior-graph schema.  It does
not call the production obligation registry, path retriever, selector, or OCEI.
Its correctness claim is therefore limited to the frozen graph/query contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from .schema import BehaviorGraph


CONTRACT_VERSION = "ocesq-contract-v1"

PREDICATE_CLASSES = {
    "parameter_provenance": "bounded_context",
    "target_entity_support": "bounded_context",
    "input_observation": "bounded_context",
    "task_constraint": "typed_path",
    "approval_dependency": "bounded_context",
    "channel_boundary": "node_property",
    "state_freshness": "bounded_context",
    "verification_evidence": "bounded_context",
    "effect_evidence": "typed_path",
    "output_grounding": "bounded_context",
    "entity_state_consistency": "absence_or_conflict",
}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Expected mapping or dataclass, got {type(value)!r}")


def graph_snapshot(graph: BehaviorGraph) -> str:
    payload = json.dumps(graph.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def adapt_ocesq_result(graph: BehaviorGraph, result: Any) -> dict[str, Any]:
    """Normalize a production OCESQ result into the frozen contract schema."""

    row = _mapping(result)
    paths = {_mapping(path)["id"]: _mapping(path) for path in row.get("candidate_paths", [])}
    descriptors = {_mapping(item)["obligation_id"]: _mapping(item) for item in row.get("missing_path_descriptors", [])}
    obligations = []
    snapshot = graph_snapshot(graph)
    all_node_ids = sorted(node.id for node in graph.nodes)
    all_edge_refs = sorted(f"{edge.source}->{edge.type}->{edge.target}" for edge in graph.edges)

    for raw in row.get("obligations", []):
        obligation = _mapping(raw)
        typ = obligation["type"]
        support = [paths[path_id] for path_id in obligation.get("matched_path_ids", []) if path_id in paths]
        conflicts = [paths[path_id] for path_id in obligation.get("conflicting_path_ids", []) if path_id in paths]
        descriptor = descriptors.get(obligation["id"])
        certificate = None
        if obligation.get("required") and not support and not conflicts:
            certificate = {
                "graph_snapshot": snapshot,
                "scope": "full_behavior_graph",
                "searched_node_ids": all_node_ids,
                "searched_edge_refs": all_edge_refs,
                "descriptor": descriptor,
                "complete": descriptor is not None,
            }
        obligations.append({
            "id": obligation["id"],
            "type": typ,
            "predicate_class": PREDICATE_CLASSES.get(typ, "unregistered"),
            "root_action_id": row["action_id"],
            "required": bool(obligation.get("required")),
            "status": obligation.get("status"),
            "required_path_pattern": obligation.get("required_path_pattern"),
            "attrs": obligation.get("attrs", {}),
            "support_facts": support,
            "conflict_facts": conflicts,
            "missing_certificate": certificate,
            "selected_path_ids": list(obligation.get("evidence_contract", {}).get("selected_path_ids", [])),
        })

    compact = row.get("compact_evidence_subgraph", {})
    return {
        "contract_version": CONTRACT_VERSION,
        "trace_id": graph.trace_id,
        "task_id": graph.task_id,
        "graph_snapshot": snapshot,
        "root_action_id": row["action_id"],
        "obligations": obligations,
        "ocres": {
            "node_ids": sorted(item["id"] for item in compact.get("nodes", [])),
            "edge_refs": sorted(
                f"{item['source']}->{item['type']}->{item['target']}"
                for item in compact.get("edges", [])
            ),
        },
    }


def _fact_integrity(
    fact: Mapping[str, Any],
    node_map: Mapping[str, Any],
    edge_refs: set[str],
    action_id: str,
) -> list[str]:
    errors = []
    node_ids = list(fact.get("node_ids", []))
    if action_id not in node_ids:
        errors.append("fact_missing_root")
    if any(node_id not in node_map for node_id in node_ids):
        errors.append("fact_unknown_node")
    if any(ref not in edge_refs for ref in fact.get("edge_refs", [])):
        errors.append("fact_unknown_edge")
    pointers = sorted({node_map[node_id].source_event_idx for node_id in node_ids if node_id in node_map and node_map[node_id].source_event_idx is not None})
    if list(fact.get("source_event_pointers", [])) != pointers:
        errors.append("fact_source_pointer_mismatch")
    return errors


def _satisfies_predicate(
    typ: str,
    kind: str,
    fact: Mapping[str, Any],
    node_map: Mapping[str, Any],
    edge_tuples: set[tuple[str, str, str]],
    action_id: str,
) -> bool:
    ids = list(fact.get("node_ids", []))
    nodes = [node_map[node_id] for node_id in ids if node_id in node_map]
    action = node_map.get(action_id)
    if action is None:
        return False
    if kind == "conflicting":
        if typ == "entity_state_consistency":
            return any(
                (left, "contradicts", right) in edge_tuples or (right, "contradicts", left) in edge_tuples
                for left in ids for right in ids if left != right
            )
        return any(node.type == "StructuralSignals" for node in nodes) or any(
            node.type == "ToolObservation" for node in nodes
        )
    if typ in {"parameter_provenance", "target_entity_support"}:
        has_task = any(node.type == "TaskInstruction" for node in nodes)
        has_prior = any(node.source_event_idx is not None and action.source_event_idx is not None and node.source_event_idx < action.source_event_idx for node in nodes if node.id != action_id)
        has_resolution = any((obs.id, "resolves", entity.id) in edge_tuples for obs in nodes for entity in nodes)
        return has_task or has_prior or has_resolution
    if typ in {"input_observation", "state_freshness", "approval_dependency"}:
        return any(node.id != action_id and node.source_event_idx is not None and action.source_event_idx is not None and node.source_event_idx < action.source_event_idx for node in nodes)
    if typ == "task_constraint":
        return any(
            node.type in {"TaskInstruction", "Requirement"}
            and ((action_id, "depends_on", node.id) in edge_tuples or (action_id, "constrains", node.id) in edge_tuples or (node.id, "constrains", action_id) in edge_tuples)
            for node in nodes
        )
    if typ == "channel_boundary":
        return bool(action.attrs.get("channel")) and ids == [action_id]
    if typ == "verification_evidence":
        return any(node.type in {"VerificationRun", "ToolObservation", "ExternalEffect"} for node in nodes if node.id != action_id)
    if typ == "effect_evidence":
        return any(node.type == "ExternalEffect" and any(edge[0] == action_id and edge[2] == node.id for edge in edge_tuples) for node in nodes)
    if typ == "output_grounding":
        return any(node.type == "OutputAssertion" for node in nodes) and any(node.type in {"ToolObservation", "ExternalEffect", "FileArtifact"} for node in nodes)
    if typ == "entity_state_consistency":
        return any(node.id != action_id for node in nodes) and not any(
            (left, "contradicts", right) in edge_tuples or (right, "contradicts", left) in edge_tuples
            for left in ids for right in ids if left != right
        )
    return False


def verify_ocesq_contract(
    graph: BehaviorGraph,
    contract: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify registry output and OC-RES preservation against hand-written gold."""

    node_map = {node.id: node for node in graph.nodes}
    edge_tuples = {(edge.source, edge.type, edge.target) for edge in graph.edges}
    edge_refs = {f"{source}->{typ}->{target}" for source, typ, target in edge_tuples}
    root = str(contract.get("root_action_id"))
    rows = []
    ocres_nodes = set(contract.get("ocres", {}).get("node_ids", []))
    ocres_edges = set(contract.get("ocres", {}).get("edge_refs", []))

    for obligation in contract.get("obligations", []):
        typ = obligation["type"]
        errors = []
        gold = expected.get(typ)
        if gold is None:
            errors.append("missing_handwritten_gold")
        else:
            if obligation.get("required") != gold.get("required"):
                errors.append("requiredness_mismatch")
            if obligation.get("status") != gold.get("status"):
                errors.append("status_mismatch")
        if obligation.get("predicate_class") != PREDICATE_CLASSES.get(typ):
            errors.append("predicate_class_mismatch")

        support = obligation.get("support_facts", [])
        conflicts = obligation.get("conflict_facts", [])
        for kind, facts in (("supporting", support), ("conflicting", conflicts)):
            for fact in facts:
                errors.extend(_fact_integrity(fact, node_map, edge_refs, root))
                if not _satisfies_predicate(typ, kind, fact, node_map, edge_tuples, root):
                    errors.append(f"invalid_{kind}_fact")

        status = obligation.get("status")
        if status == "supported" and not support:
            errors.append("supported_without_fact")
        if status == "conflicting" and not conflicts:
            errors.append("conflicting_without_fact")
        certificate = obligation.get("missing_certificate")
        if status in {"missing", "unknown"} and obligation.get("required"):
            if not certificate or not certificate.get("complete"):
                errors.append("invalid_missing_certificate")
            elif certificate.get("graph_snapshot") != graph_snapshot(graph):
                errors.append("certificate_snapshot_mismatch")

        selected_ids = set(obligation.get("selected_path_ids", []))
        selected_facts = [fact for fact in [*support, *conflicts] if fact.get("id") in selected_ids]
        if status in {"supported", "conflicting"} and not selected_facts:
            errors.append("ocres_missing_deciding_class")
        for fact in selected_facts:
            if not set(fact.get("node_ids", [])).issubset(ocres_nodes):
                errors.append("ocres_missing_fact_node")
            if not set(fact.get("edge_refs", [])).issubset(ocres_edges):
                errors.append("ocres_missing_fact_edge")
        if status in {"missing", "unknown"} and obligation.get("required") and root not in ocres_nodes:
            errors.append("ocres_missing_anchor")

        rows.append({"type": typ, "passed": not errors, "errors": sorted(set(errors))})

    observed_types = {row["type"] for row in rows}
    missing_types = sorted(set(PREDICATE_CLASSES) - observed_types)
    top_errors = []
    if contract.get("contract_version") != CONTRACT_VERSION:
        top_errors.append("contract_version_mismatch")
    if contract.get("graph_snapshot") != graph_snapshot(graph):
        top_errors.append("graph_snapshot_mismatch")
    if root not in node_map or node_map[root].type != "ToolAction":
        top_errors.append("invalid_root")
    return {
        "passed": not top_errors and not missing_types and all(row["passed"] for row in rows),
        "top_errors": top_errors,
        "missing_obligation_types": missing_types,
        "obligations": rows,
    }
