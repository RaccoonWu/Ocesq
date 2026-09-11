"""Obligation-constrained evidence subgraph queries over behavior graphs.

This module is the graph-query path used by the experiment plan.  It
keeps the older MonitorCard compiler intact, but exposes OEG/OCESQ/RES
artifacts that can be compared against graph traversal baselines.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any

from .schema import (
    BehaviorGraph,
    GraphEdge,
    GraphNode,
    EDGE_MENTIONS,
    EDGE_RESOLVES,
    EDGE_SAME_AS,
    EDGE_CONSTRAINS,
    EDGE_SATISFIES,
    EDGE_AUTHORIZES,
    EDGE_TRIGGERS,
    EDGE_CONTRADICTS,
)

from .lexicons import (
    ANCHOR_KEYWORDS,
    CONTENT_DEPENDENT_MECHANISMS,
    FREE_TEXT_KEYS,
    HIGH_IMPACT_MECHANISMS,
    SAVE_DRAFT_STRONG_ANCHOR_KEYS,
    SAVE_DRAFT_WEAK_ANCHOR_KEYS,
    TASK_AUTHORIZATION_TERMS,
    VERIFICATION_SENSITIVE_MECHANISMS,
    VERSION_SENSITIVE_MECHANISMS,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ocei import OCEI

STATUS_DECIDED = {"supported", "missing", "conflicting"}


@dataclass
class EvidencePath:
    id: str
    action_id: str
    obligation_id: str
    kind: str
    node_ids: list[str]
    edge_refs: list[str] = field(default_factory=list)
    path_pattern: str = ""
    summary: str = ""
    score: float = 1.0
    source_event_pointers: list[int] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MissingPathDescriptor:
    obligation_id: str
    action_id: str
    expected_path_pattern: str
    anchor_node_ids: list[str]
    missing_relation: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObligationPredicate:
    id: str
    type: str
    action_id: str
    required: bool
    status: str
    required_path_pattern: str
    summary: str
    confidence: float = 1.0
    candidate_path_ids: list[str] = field(default_factory=list)
    matched_path_ids: list[str] = field(default_factory=list)
    conflicting_path_ids: list[str] = field(default_factory=list)
    missing_relation: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    evidence_contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RootEvidenceSubgraph:
    trace_id: str
    task_id: str
    action_id: str
    action_summary: dict[str, Any]
    obligation_statuses: dict[str, str]
    obligations: list[ObligationPredicate]
    candidate_paths: list[EvidencePath]
    supporting_paths: list[EvidencePath]
    missing_path_descriptors: list[MissingPathDescriptor]
    conflicting_paths: list[EvidencePath]
    compact_evidence_subgraph: dict[str, Any]
    coverage: dict[str, Any]
    redundancy: dict[str, Any]
    source_event_pointers: list[int]
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["obligations"] = [item.to_dict() for item in self.obligations]
        row["candidate_paths"] = [item.to_dict() for item in self.candidate_paths]
        row["supporting_paths"] = [item.to_dict() for item in self.supporting_paths]
        row["missing_path_descriptors"] = [item.to_dict() for item in self.missing_path_descriptors]
        row["conflicting_paths"] = [item.to_dict() for item in self.conflicting_paths]
        return row


def graph_to_eventlog_rows(graph: BehaviorGraph) -> list[dict[str, Any]]:
    """Export a canonical derived EventLog from graph nodes."""

    rows: list[dict[str, Any]] = []
    for node in sorted(graph.nodes, key=lambda item: (item.source_event_idx is None, item.source_event_idx or 0, item.id)):
        rows.append(
            {
                "trace_id": graph.trace_id,
                "task_id": graph.task_id,
                "event_id": node.id,
                "event_type": _canonical_event_type(node.type),
                "node_type": node.type,
                "label": node.label,
                "attrs": node.attrs,
                "source_event_idx": node.source_event_idx,
                "normalization_rule": "behavior_graph_node_to_eventlog",
            }
        )
    return rows


def high_impact_actions(graph: BehaviorGraph) -> list[GraphNode]:
    return [node for node in graph.nodes if is_high_impact_action(node)]


def is_high_impact_action(node: GraphNode) -> bool:
    if node.type != "ToolAction":
        return False
    mechanism = node.attrs.get("mechanism")
    return bool(
        node.attrs.get("effectful")
        or node.attrs.get("approval_sensitive")
        or node.attrs.get("external_visibility")
        or node.attrs.get("risk_level") == "high"
        or mechanism in HIGH_IMPACT_MECHANISMS
    )


def run_batch_ocesq(graph: BehaviorGraph, *, budget: dict[str, int] | None = None, ocei: OCEI | None = None) -> list[RootEvidenceSubgraph]:
    return [run_ocesq(graph, action.id, budget=budget, ocei=ocei) for action in high_impact_actions(graph)]


def run_ocesq(graph: BehaviorGraph, action_id: str, *, budget: dict[str, int] | None = None, ocei: OCEI | None = None) -> RootEvidenceSubgraph:
    budget = budget or {"max_nodes": 24, "max_edges": 36}
    start = time.perf_counter()
    node_map = {node.id: node for node in graph.nodes} if ocei is None else ocei.node_map
    action = node_map.get(action_id)
    if action is None:
        raise ValueError(f"Unknown action node: {action_id}")

    obligations = derive_obligation_predicates(graph, action_id, ocei=ocei)
    t_obligations = time.perf_counter()
    candidate_paths: list[EvidencePath] = []
    missing_descriptors: list[MissingPathDescriptor] = []
    for obligation in obligations:
        if not obligation.required:
            continue
        paths, missing = retrieve_candidate_paths(graph, action_id, obligation, ocei=ocei)
        candidate_paths.extend(paths)
        missing_descriptors.extend(missing)
        obligation.candidate_path_ids = [path.id for path in paths]
        obligation.matched_path_ids = [path.id for path in paths if path.kind == "supporting"]
        obligation.conflicting_path_ids = [path.id for path in paths if path.kind == "conflicting"]
    _refresh_obligation_statuses(obligations, candidate_paths, missing_descriptors)
    t_paths = time.perf_counter()
    selected = select_compact_res_subgraph(graph, action_id, obligations, candidate_paths, missing_descriptors, budget)
    t_select = time.perf_counter()

    supporting = [path for path in candidate_paths if path.kind == "supporting" and path.id in selected["selected_path_ids"]]
    conflicting = [path for path in candidate_paths if path.kind == "conflicting" and path.id in selected["selected_path_ids"]]
    source_events = sorted(
        {
            pointer
            for node in selected["nodes"]
            for pointer in ([node.get("source_event_idx")] if node.get("source_event_idx") is not None else [])
        }
    )
    action_summary = {
        "tool_name": action.attrs.get("tool_name"),
        "mechanism": action.attrs.get("mechanism"),
        "risk_level": action.attrs.get("risk_level"),
        "channel": action.attrs.get("channel"),
        "source_event_idx": action.source_event_idx,
    }
    statuses = {obligation.type: obligation.status for obligation in obligations}
    return RootEvidenceSubgraph(
        trace_id=graph.trace_id,
        task_id=graph.task_id,
        action_id=action_id,
        action_summary=action_summary,
        obligation_statuses=statuses,
        obligations=obligations,
        candidate_paths=candidate_paths,
        supporting_paths=supporting,
        missing_path_descriptors=missing_descriptors,
        conflicting_paths=conflicting,
        compact_evidence_subgraph={"nodes": selected["nodes"], "edges": selected["edges"]},
        coverage=_coverage_summary(obligations, selected["selected_path_ids"]),
        redundancy=_redundancy_summary(graph, selected, candidate_paths),
        source_event_pointers=source_events,
        timings={
            "derive_obligations_ms": round((t_obligations - start) * 1000, 4),
            "retrieve_paths_ms": round((t_paths - t_obligations) * 1000, 4),
            "select_res_ms": round((t_select - t_paths) * 1000, 4),
            "total_ms": round((t_select - start) * 1000, 4),
        },
    )


def _action_mentioned_entities(
    graph: BehaviorGraph,
    action_id: str,
    *,
    ocei: OCEI | None = None,
) -> dict[str, dict[str, Any]]:
    """Return entities the action explicitly mentions, with resolution info.

    Follows ``mentions`` edges from the action to ServiceEntity nodes, then
    traces ``resolves`` and ``same_as`` edges to find the observation that
    originally supplied each entity.

    Returns:
        {entity_id: {"entity": GraphNode, "resolved": bool, "source_obs_id": str|None, "via": str}}
    """
    mentioned: dict[str, dict[str, Any]] = {}

    # Collect mentioned entity ids
    if ocei is not None:
        mentioned_ids = {
            t for t, e, _ in ocei.outgoing.get(action_id, [])
            if e == EDGE_MENTIONS
        }
        node_map = ocei.node_map
        outgoing = ocei.outgoing
        incoming = ocei.incoming
    else:
        mentioned_ids = {
            e.target for e in graph.edges
            if e.source == action_id and e.type == EDGE_MENTIONS
        }
        node_map = {n.id: n for n in graph.nodes}
        outgoing = _build_outgoing_index(graph)
        incoming = _build_incoming_index(graph)

    for ent_id in mentioned_ids:
        ent = node_map.get(ent_id)
        if ent is None or ent.type != "ServiceEntity":
            continue
        info: dict[str, Any] = {"entity": ent, "resolved": False, "source_obs_id": None, "via": "none"}

        # Direct resolves edge: ToolObservation --resolves--> Entity
        for src_id, etype, _ in incoming.get(ent_id, []):
            if etype == EDGE_RESOLVES:
                src = node_map.get(src_id)
                if src and src.type == "ToolObservation":
                    info["resolved"] = True
                    info["source_obs_id"] = src_id
                    info["via"] = "direct_resolves"
                    break

        # If not directly resolved, follow same_as chain
        if not info["resolved"]:
            visited: set[str] = {ent_id}
            queue = [ent_id]
            while queue:
                cur = queue.pop(0)
                # Check same_as edges (bidirectional: both source and target sides)
                for neighbor, etype, _ in [*outgoing.get(cur, []), *incoming.get(cur, [])]:
                    if etype == EDGE_SAME_AS and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        # Check if this neighbor has a resolves edge
                        for src_id, e2, _ in incoming.get(neighbor, []):
                            if e2 == EDGE_RESOLVES:
                                src = node_map.get(src_id)
                                if src and src.type == "ToolObservation":
                                    info["resolved"] = True
                                    info["source_obs_id"] = src_id
                                    info["via"] = f"same_as_chain({len(visited)})"
                                    break
                        if info["resolved"]:
                            break
                if info["resolved"]:
                    break

        mentioned[ent_id] = info

    return mentioned


def _build_outgoing_index(graph: BehaviorGraph) -> dict[str, list[tuple[str, str, str]]]:
    idx: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for e in graph.edges:
        idx[e.source].append((e.target, e.type, f"{e.source}->{e.type}->{e.target}"))
    return idx


def _build_incoming_index(graph: BehaviorGraph) -> dict[str, list[tuple[str, str, str]]]:
    idx: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for e in graph.edges:
        idx[e.target].append((e.source, e.type, f"{e.source}->{e.type}->{e.target}"))
    return idx


def derive_obligation_predicates(graph: BehaviorGraph, action_id: str, *, ocei: OCEI | None = None) -> list[ObligationPredicate]:
    node_map = {node.id: node for node in graph.nodes} if ocei is None else {action_id: ocei.node_map.get(action_id)} if ocei.node_map.get(action_id) else {}
    action = node_map.get(action_id) or (ocei.node_map.get(action_id) if ocei else None)
    if action is None:
        raise ValueError(f"Unknown action node: {action_id}")

    mechanism = str(action.attrs.get("mechanism") or "")
    if ocei is not None:
        meta = ocei.get_action_meta(action_id)
        anchors = meta.get("anchors", [])
    else:
        anchors = _action_anchors(action.attrs.get("input", {}))
    target_required = _requires_target_binding(action)
    prior_observations = _prior_nodes(graph, action, {"ToolObservation"}, ocei=ocei)
    aligned_prior_observations = _aligned_prior_observations(graph, action, anchors, ocei=ocei)
    prior_evidence_observations = _audit_relevant_prior_observations(graph, action, anchors, ocei=ocei)
    prior_context_evidence = _audit_relevant_prior_context(graph, action, anchors, ocei=ocei)
    effect_observations = _effect_feedback_nodes(graph, action, ocei=ocei)
    output_assertions = _action_output_assertions(graph, action, ocei=ocei)
    structural_conflicts = _structural_conflicts_for_action(graph, action_id, ocei=ocei)
    task_supported_anchor_values = _task_supported_anchor_values(graph, anchors)
    supported_anchor_values = _supported_anchor_values(prior_observations, anchors) | task_supported_anchor_values
    task_target_authorized = _task_authorizes_target_binding(graph, action, anchors)
    semantic_target_supported = bool(
        task_target_authorized
        or _all_required_anchors_in_observations(prior_evidence_observations, anchors)
        or _composite_payload_supported(action, prior_context_evidence)
    )
    if semantic_target_supported:
        # Task text or contextual evidence can certify an otherwise weak
        # binding, but it cannot resolve an explicitly ambiguous entity
        # selection. Preserve ambiguity as a query conflict until a unique
        # candidate or disambiguation step is present in the graph.
        structural_conflicts = [
            conflict
            for conflict in structural_conflicts
            if conflict.get("type") in {
                "ambiguous_entity_binding",
                "action_dispatch_entity_mismatch",
                "conflicting_source_response",
                "non_causal_observation",
            }
        ]

    obligations: list[ObligationPredicate] = []

    def add(
        typ: str,
        *,
        required: bool,
        status: str,
        pattern: str,
        summary: str,
        missing_relation: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        obligations.append(
            ObligationPredicate(
                id=f"{action_id}:{typ}",
                type=typ,
                action_id=action_id,
                required=required,
                status=status if required else "not_applicable",
                required_path_pattern=pattern,
                summary=summary,
                missing_relation=missing_relation,
                attrs=attrs or {},
            )
        )

    # Entity-graph resolution: trace action → mentions → entity → resolves → observation
    mentioned = _action_mentioned_entities(graph, action_id, ocei=ocei)
    resolved_entities = {eid for eid, info in mentioned.items() if info["resolved"]}
    uniquely_resolved = {
        eid for eid, info in mentioned.items()
        if info["resolved"] and info["entity"].attrs.get("candidate_count") == 1
    }

    required_anchors = _required_parameter_anchors(mechanism, anchors)
    optional_anchors = [item for item in anchors if item not in required_anchors]
    parameter_required = bool(required_anchors)
    parameter_supported = parameter_required and (
        len(supported_anchor_values) == len(required_anchors)
        or (len(required_anchors) > 0 and len(resolved_entities) > 0)
        or _all_required_anchors_in_observations(prior_evidence_observations, required_anchors)
        or _all_required_anchors_in_context(prior_context_evidence, required_anchors)
        or _composite_payload_supported(action, prior_context_evidence)
    )
    add(
        "parameter_provenance",
        required=parameter_required,
        status="supported" if parameter_supported else "missing",
        pattern="ToolObservation --resolves--> ServiceEntity <--mentions-- ToolAction",
        summary="Action parameters should be grounded in earlier observations or stable identifiers.",
        missing_relation="mentions",
        attrs={
            "anchors": [{"key": key, "value": value} for key, value in anchors],
            "required_anchors": [{"key": key, "value": value} for key, value in required_anchors],
            "optional_anchors": [{"key": key, "value": value} for key, value in optional_anchors],
            "mechanism": mechanism,
            "mentioned_entity_ids": sorted(mentioned),
            "resolved_entity_ids": sorted(resolved_entities),
            "uniquely_resolved_entity_ids": sorted(uniquely_resolved),
            "context_evidence_ids": [node.id for node in prior_context_evidence],
        },
    )

    if structural_conflicts:
        target_status = "conflicting"
    elif target_required and anchors and len(supported_anchor_values) == len(anchors):
        target_status = "supported"
    elif target_required and len(uniquely_resolved) > 0:
        target_status = "supported"
    elif target_required and semantic_target_supported:
        target_status = "supported"
    elif target_required:
        target_status = "missing"
    else:
        target_status = "not_applicable"
    add(
        "target_entity_support",
        required=target_required,
        status=target_status,
        pattern="ToolObservation --resolves--> ServiceEntity(candidate_count=1) <--mentions-- ToolAction",
        summary="High-impact targets should be uniquely supported by prior entity evidence.",
        missing_relation="mentions",
        attrs={
            "anchors": [{"key": key, "value": value} for key, value in anchors],
            "structural_conflicts": structural_conflicts,
            "task_target_authorized": task_target_authorized,
            "mentioned_entity_ids": sorted(mentioned),
            "resolved_entity_ids": sorted(resolved_entities),
            "uniquely_resolved_entity_ids": sorted(uniquely_resolved),
            "context_evidence_ids": [node.id for node in prior_context_evidence],
        },
    )

    add(
        "input_observation",
        required=target_required,
        status="supported" if prior_context_evidence else "missing",
        pattern="ToolObservation before ToolAction",
        summary="High-impact actions should have preceding read or query evidence.",
        missing_relation="prior_observation",
        attrs={
            "anchors": [{"key": key, "value": value} for key, value in anchors],
            "evidence_observation_ids": [node.id for node in prior_context_evidence],
        },
    )

    add(
        "task_constraint",
        required=True,
        status="supported" if _has_edge(graph, action_id, "task_instruction", {"depends_on", "constrains"}, ocei=ocei) else "unknown",
        pattern="TaskInstruction/Requirement -> constrains -> ToolAction",
        summary="The action should be traceable to a task requirement or instruction.",
        missing_relation="constrains",
    )

    approval_required = bool(action.attrs.get("approval_sensitive") or action.attrs.get("external_visibility"))
    approval_supported = bool(
        _prior_nodes(graph, action, {"ApprovalRequirement", "ApprovalEvent"}, ocei=ocei)
        or _task_authorizes_action(graph, action)
        or _prior_authorization_context(graph, action, anchors, ocei=ocei)
    )
    add(
        "approval_dependency",
        required=approval_required,
        status="supported" if approval_supported else "missing",
        pattern="ApprovalRequirement/ApprovalEvent -> satisfies_approval -> ExternalEffect",
        summary="Approval-sensitive effects should have explicit approval evidence.",
        missing_relation="satisfies_approval",
        attrs={
            "anchors": [{"key": key, "value": value} for key, value in anchors],
            "authorization_context_ids": [node.id for node in _prior_authorization_context(graph, action, anchors, ocei=ocei)],
        },
    )

    channel_required = bool(action.attrs.get("channel") or action.attrs.get("external_visibility"))
    add(
        "channel_boundary",
        required=channel_required,
        status="supported" if action.attrs.get("channel") else "unknown",
        pattern="Channel(source) -> crosses_channel -> Channel(sink)",
        summary="Cross-channel effects should expose their source and sink channel.",
        missing_relation="crosses_channel",
    )

    state_required = mechanism in VERSION_SENSITIVE_MECHANISMS
    state_supported = bool(prior_evidence_observations)
    add(
        "state_freshness",
        required=state_required,
        status="supported" if state_supported else "unknown",
        pattern="EntityVersion/FileVersion -> depends_on -> ToolAction",
        summary="Version-sensitive actions should use fresh entity or artifact state.",
        missing_relation="latest_version",
        attrs={
            "anchors": [{"key": key, "value": value} for key, value in anchors],
            "evidence_observation_ids": [node.id for node in prior_context_evidence],
        },
    )

    task_authorized_export = mechanism == "export" and _task_authorizes_action(graph, action)
    verification_required = mechanism in VERIFICATION_SENSITIVE_MECHANISMS and not task_authorized_export
    verification_nodes = _post_action_verification_nodes(graph, action, anchors, ocei=ocei)
    verification_supported = bool(
        _prior_nodes(graph, action, {"VerificationRun"}, ocei=ocei)
        or effect_observations
        or verification_nodes
    )
    add(
        "verification_evidence",
        required=verification_required,
        status="supported" if verification_supported else "missing",
        pattern="VerificationRun -> verifies -> ToolAction/ExternalEffect",
        summary="Destructive or irreversible effects should be covered by verification evidence.",
        missing_relation="verifies",
        attrs={
            "effect_observation_ids": [node.id for node in effect_observations],
            "post_verification_ids": [node.id for node in verification_nodes],
        },
    )

    effect_required = bool(action.attrs.get("effectful"))
    add(
        "effect_evidence",
        required=effect_required,
        status="supported" if _outgoing_nodes(graph, action_id, {"ExternalEffect"}, ocei=ocei) else "unknown",
        pattern="ToolAction -> ExternalEffect",
        summary="Effectful actions should materialize an auditable effect node.",
        missing_relation="external_effect",
    )

    output_required = bool(output_assertions)
    output_supported = bool(output_assertions and (effect_observations or prior_evidence_observations))
    add(
        "output_grounding",
        required=output_required,
        status="supported" if output_supported else "unknown",
        pattern="ToolObservation/FileArtifact -> grounds -> OutputAssertion",
        summary="Final output assertions should be grounded in prior evidence.",
        missing_relation="grounds",
        attrs={
            "output_assertion_ids": [node.id for node in output_assertions],
            "effect_observation_ids": [node.id for node in effect_observations],
            "evidence_observation_ids": [node.id for node in prior_context_evidence],
        },
    )

    # --- Route B Layer 3: entity state consistency obligation ---
    # Checks whether any entity the action depends on has inconsistent state
    # across observations (indicated by ``contradicts`` edges).
    has_contradiction = False
    contradiction_details: list[dict[str, Any]] = []
    entity_consistency_status: str = "not_applicable"
    # This obligation is only applicable when the action's mentioned entities
    # appear in multiple observations — otherwise there is nothing to check
    # for cross-observation consistency.
    entity_consistency_required = bool(action.attrs.get("effectful")) and bool(
        resolved_entities or prior_context_evidence or effect_observations
    )
    if entity_consistency_required:
        # Collect entity ids that are linked via same_as to the action's mentioned entities
        all_linked_entities: set[str] = set()
        for eid in mentioned:
            all_linked_entities.add(eid)
            if ocei:
                for linked in ocei.same_as_entities(eid):
                    all_linked_entities.add(linked)

        # Check for contradicts edges among these entities
        for edge in graph.edges:
            if edge.type == "contradicts" and (
                edge.source in all_linked_entities or edge.target in all_linked_entities
            ):
                has_contradiction = True
                contradiction_details.append({
                    "entity_a": edge.source,
                    "entity_b": edge.target,
                    "field": edge.attrs.get("field", ""),
                    "value_a": edge.attrs.get("value_in_source", ""),
                    "value_b": edge.attrs.get("value_in_target", ""),
                    "entity_value": edge.attrs.get("entity_value", ""),
                })

        if has_contradiction:
            entity_consistency_status = "conflicting"
        else:
            # Entity has been observed and resolved; no contradictions found.
            entity_consistency_status = "supported"
    # else: not_applicable — action does not depend on multi-observation entities

    add(
        "entity_state_consistency",
        required=entity_consistency_required,
        status=entity_consistency_status,
        pattern="ServiceEntity --same_as--> ServiceEntity --contradicts--> ServiceEntity",
        summary="Entity state should be consistent across observations that the action depends on.",
        missing_relation="contradicts",
        attrs={
            "mentioned_entity_ids": sorted(mentioned),
            "resolved_entity_ids": sorted(resolved_entities),
            "evidence_observation_ids": [node.id for node in prior_context_evidence],
            "effect_observation_ids": [node.id for node in effect_observations],
            "contradictions": contradiction_details if has_contradiction else [],
        },
    )

    return obligations


def retrieve_candidate_paths(
    graph: BehaviorGraph,
    action_id: str,
    obligation: ObligationPredicate,
    *,
    ocei: OCEI | None = None,
) -> tuple[list[EvidencePath], list[MissingPathDescriptor]]:
    node_map = {node.id: node for node in graph.nodes} if ocei is None else ocei.node_map
    action = node_map[action_id]
    paths: list[EvidencePath] = []
    missing: list[MissingPathDescriptor] = []

    def add_path(
        kind: str,
        node_ids: list[str],
        edge_refs: list[str] | None = None,
        summary: str = "",
        score: float = 1.0,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        paths.append(
            EvidencePath(
                id=f"{obligation.id}:path_{len(paths) + 1}",
                action_id=action_id,
                obligation_id=obligation.id,
                kind=kind,
                node_ids=_dedupe_keep_order(node_ids),
                edge_refs=edge_refs or [],
                path_pattern=obligation.required_path_pattern,
                summary=summary or obligation.summary,
                score=score,
                source_event_pointers=_source_pointers(node_map, node_ids),
                attrs=attrs or {},
            )
        )

    if obligation.status == "conflicting":
        for conflict in obligation.attrs.get("structural_conflicts", []):
            nodes = [*conflict.get("node_ids", []), action_id]
            add_path("conflicting", nodes, summary=conflict.get("summary", "Conflicting structural evidence."), score=3.0)
        # Route B Layer 3: contradictions from entity state inconsistency
        for cd in obligation.attrs.get("contradictions", []):
            add_path(
                "conflicting",
                [cd["entity_a"], cd["entity_b"], action_id],
                summary=(
                    f"Entity {cd.get('entity_value', '?')}: "
                    f"{cd['field']} differs ({cd['value_a']} vs {cd['value_b']})."
                ),
                score=3.5,
                attrs={"field": cd["field"], "value_a": cd["value_a"], "value_b": cd["value_b"]},
            )

    typ = obligation.type
    if typ in {"parameter_provenance", "target_entity_support"}:
        anchors = [(item.get("key"), item.get("value")) for item in obligation.attrs.get("required_anchors", obligation.attrs.get("anchors", [])) if item.get("key")]
        if obligation.attrs.get("task_target_authorized"):
            add_path(
                "supporting",
                ["task_instruction", action_id],
                summary="Task instruction authorizes this target class.",
                score=1.9,
                attrs={"source": "task_instruction"},
            )

        # Entity-graph resolution paths (mentions -> resolves edges).
        # These include the entity node, making the evidence chain explicit:
        #   observation --resolves--> entity --mentions--> action
        mentioned_data = _action_mentioned_entities(graph, action_id, ocei=ocei)
        entity_path_anchor_keys: set[str] = set()
        for ent_id, info in mentioned_data.items():
            if not info.get("resolved"):
                continue
            obs_id = info["source_obs_id"]
            ent = info["entity"]
            entity_type = ent.attrs.get("entity_type", "entity")
            entity_value = ent.attrs.get("value", ent.label)
            add_path(
                "supporting",
                [obs_id, ent_id, action_id],
                summary=f"Entity {entity_type}:{entity_value} resolved by observation and referenced by action.",
                score=2.2,
                attrs={
                    "entity_id": ent_id,
                    "entity_type": entity_type,
                    "via": info.get("via"),
                    "source_observation": obs_id,
                },
            )
            # Track which anchor keys are covered by entity-graph paths so we
            # don't emit redundant anchor-matching paths for the same entity.
            ek = ent.attrs.get("key", "")
            if ek:
                entity_path_anchor_keys.add(ek)

        # Anchor-based matching (kept for entities without graph edges).
        aligned = _aligned_prior_observations(graph, action, anchors, ocei=ocei)
        context_ids = obligation.attrs.get("context_evidence_ids", [])
        context_evidence = [node_map[node_id] for node_id in context_ids if node_id in node_map]
        for key, value in anchors:
                if _task_mentions_anchor(graph, key, value):
                    add_path(
                        "supporting",
                        ["task_instruction", action_id],
                        summary=f"Task instruction directly provides action parameter {key}.",
                        score=1.7,
                        attrs={"parameter": key, "value": value, "source": "task_instruction"},
                    )
                for obs in aligned:
                    if _value_supported_by_anchor(obs, key, value):
                        # Skip if entity-graph path already covers this key
                        if key in entity_path_anchor_keys:
                            continue
                        add_path(
                            "supporting",
                            [obs.id, action.id],
                            summary=f"Prior observation contains action parameter {key}.",
                            score=2.0,
                        attrs={"parameter": key, "value": value},
                    )
                for ctx in context_evidence:
                    if ctx.id in {obs.id for obs in aligned}:
                        continue
                    if _context_supports_anchor(ctx, key, value):
                        add_path(
                            "supporting",
                            [ctx.id, action.id],
                            summary=f"Prior trace context contains action parameter {key}.",
                            score=2.05,
                            attrs={"parameter": key, "value": value, "source": ctx.type},
                        )
        composite = _composite_payload_support_nodes(action, context_evidence)
        if composite:
            add_path(
                "supporting",
                [node.id for node in composite[:4]] + [action_id],
                summary="Action payload is assembled from multiple prior trace observations.",
                score=2.3,
                attrs={"source": "composite_payload"},
            )
    elif typ == "input_observation":
        evidence_ids = set(obligation.attrs.get("evidence_observation_ids", []))
        evidence = [node_map[node_id] for node_id in evidence_ids if node_id in node_map]
        if not evidence:
            evidence = _aligned_prior_observations(graph, action, obligation.attrs.get("anchors", []), ocei=ocei)
        for obs in sorted(evidence, key=lambda node: (node.source_event_idx or -1, node.id))[-3:]:
            add_path("supporting", [obs.id, action_id], summary="Pre-action observation supports local audit.", score=1.4)
    elif typ == "task_constraint":
        refs = _edge_refs_between(graph, action_id, "task_instruction", ocei=ocei)
        if refs:
            add_path("supporting", ["task_instruction", action_id], refs, "Action depends on the task instruction.", 1.2)
    elif typ == "approval_dependency":
        for approval in _prior_nodes(graph, action, {"ApprovalRequirement", "ApprovalEvent"}, ocei=ocei):
            add_path("supporting", [approval.id, action_id], summary="Prior approval evidence found.", score=2.2)
        if _task_authorizes_action(graph, action):
            add_path("supporting", ["task_instruction", action_id], summary="Task instruction explicitly authorizes this effect.", score=1.8)
        for ctx in _prior_authorization_context(graph, action, obligation.attrs.get("anchors", []), ocei=ocei):
            add_path(
                "supporting",
                [ctx.id, action_id],
                summary="Prior user or observation context authorizes this sensitive effect.",
                score=2.0,
                attrs={"source": ctx.type},
            )
    elif typ == "channel_boundary":
        if action.attrs.get("channel"):
            add_path("supporting", [action_id], summary=f"Action exposes channel={action.attrs.get('channel')}.", score=0.8)
    elif typ == "state_freshness":
        evidence_ids = set(obligation.attrs.get("evidence_observation_ids", []))
        evidence = [node_map[node_id] for node_id in evidence_ids if node_id in node_map]
        if not evidence:
            evidence = _aligned_prior_observations(graph, action, obligation.attrs.get("anchors", []), ocei=ocei)
        for version in sorted(evidence, key=lambda node: (node.source_event_idx or -1, node.id))[-2:]:
            add_path("supporting", [version.id, action_id], summary="Prior state evidence is aligned with the current action target.", score=1.0)
    elif typ == "verification_evidence":
        for verification in _prior_nodes(graph, action, {"VerificationRun"}, ocei=ocei):
            add_path("supporting", [verification.id, action_id], summary="Verification evidence precedes the action.", score=2.0)
        post_verification_ids = obligation.attrs.get("post_verification_ids", [])
        for node_id in post_verification_ids:
            if node_id in node_map:
                chain = [action_id, *_verification_chain_nodes(graph, node_map[node_id]), node_id]
                add_path(
                    "supporting",
                    chain,
                    summary="Post-action verification checks the effect or target state.",
                    score=2.1,
                )
        if not post_verification_ids:
            for node_id in obligation.attrs.get("effect_observation_ids", []):
                if node_id in node_map:
                    add_path(
                        "supporting",
                        [action_id, node_id],
                        _edge_refs_between(graph, action_id, node_id, ocei=ocei),
                        "Action has auditable tool feedback/effect evidence.",
                        1.9,
                    )
    elif typ == "effect_evidence":
        for effect in _outgoing_nodes(graph, action_id, {"ExternalEffect"}, ocei=ocei):
            refs = _edge_refs_between(graph, action_id, effect.id, ocei=ocei)
            add_path("supporting", [action_id, effect.id], refs, "Action materializes an ExternalEffect node.", 1.5)
    elif typ == "output_grounding":
        evidence_ids = [
            *obligation.attrs.get("effect_observation_ids", []),
            *obligation.attrs.get("evidence_observation_ids", []),
        ]
        assertion_ids = obligation.attrs.get("output_assertion_ids", [])
        for evidence_id in evidence_ids[:2]:
            for assertion_id in assertion_ids[:2]:
                if evidence_id in node_map and assertion_id in node_map:
                    add_path(
                        "supporting",
                        [evidence_id, action_id, assertion_id],
                        summary="Output assertion is grounded by nearby action evidence.",
                        score=1.3,
                    )
    elif typ == "entity_state_consistency":
        evidence_ids = [
            *obligation.attrs.get("resolved_entity_ids", []),
            *obligation.attrs.get("evidence_observation_ids", []),
            *obligation.attrs.get("effect_observation_ids", []),
        ]
        for node_id in evidence_ids[:2]:
            if node_id in node_map:
                add_path(
                    "supporting",
                    [node_id, action_id],
                    summary="No contradictory state evidence found for the action-local entity context.",
                    score=1.0,
                )

    # For missing/unknown obligations, add structurally-relevant context paths.
    # These don't satisfy the obligation but show the auditor what evidence WAS
    # available.  Only nodes with structural affinity to the obligation's expected
    # path pattern are included — random nearby observations are NOT pulled in.
    if obligation.status in {"missing", "unknown"} and not any(
        p.kind in {"supporting", "conflicting"} for p in paths
    ):
        _add_context_paths(graph, action, obligation, add_path, ocei=ocei)

    has_deciding_path = any(path.kind in {"supporting", "conflicting"} for path in paths)
    if obligation.required and not has_deciding_path:
        missing.append(
            MissingPathDescriptor(
                obligation_id=obligation.id,
                action_id=action_id,
                expected_path_pattern=obligation.required_path_pattern,
                anchor_node_ids=[action_id],
                missing_relation=obligation.missing_relation or obligation.type,
                summary=obligation.summary,
            )
        )
    return paths, missing


def _add_context_paths(
    graph: BehaviorGraph,
    action: GraphNode,
    obligation: ObligationPredicate,
    add_path: Any,
    *,
    ocei: OCEI | None = None,
) -> None:
    """Generate why-not context paths for a missing/unknown obligation.

    Context is generated by finding the *longest prefix match* of the
    obligation's expected meta-path in the graph.  For example, if
    parameter_provenance expects::

        ToolObservation --resolves--> ServiceEntity <--mentions-- ToolAction

    but the action only has ``mentions`` edges to entities with no ``resolves``
    source, we include those entity nodes as "entity referenced but unresolved"
    context.  This is structurally meaningful (not random nearby nodes).
    """
    typ = obligation.type
    anchors = [
        (item.get("key"), item.get("value"))
        for item in obligation.attrs.get("anchors", [])
        if item.get("key")
    ]
    prior_obs = _prior_nodes(graph, action, {"ToolObservation"}, ocei=ocei)

    if typ in {"parameter_provenance", "target_entity_support"}:
        # Meta-path prefix: entities the action mentions, annotated with
        # resolution status.  For unresolved entities we show *why* they
        # aren't resolved (ambiguous, no resolves edge, etc.).
        mentioned = _action_mentioned_entities(graph, action.id, ocei=ocei)
        for ent_id, info in mentioned.items():
            if info["resolved"]:
                continue  # Already covered by supporting paths.
            ent = info["entity"]
            cand = ent.attrs.get("candidate_count")
            if cand is not None and cand > 1:
                summary = f"Entity {ent.label} is one of {cand} candidates; not uniquely resolved."
            elif ent.attrs.get("source") == "request_parameter":
                summary = f"Parameter entity {ent.label} has no observation source."
            else:
                summary = f"Entity {ent.label} referenced by action but not resolved from any observation."
            add_path("context", [ent_id, action.id], summary=summary, score=0.5)

        # Fallback: observations that share anchor field names but not values.
        anchor_tails = {key.rsplit(".", 1)[-1].lower() for key, _ in anchors}
        for obs in prior_obs[-2:]:
            if _obs_shares_anchor_fields(obs, anchor_tails):
                add_path(
                    "context", [obs.id, action.id],
                    summary=f"Observation shares fields with {typ} anchors but values differ.",
                    score=0.4,
                )

    elif typ == "input_observation":
        # Meta-path prefix: prior observations exist but don't match anchors.
        for obs in prior_obs[-3:]:
            add_path(
                "context", [obs.id, action.id],
                summary="Prior observation exists but does not provide required input evidence.",
                score=0.5,
            )

    elif typ == "state_freshness":
        # Meta-path prefix: entity versions near the action.
        entity_nodes = _prior_nodes(graph, action, {"EntityVersion", "ServiceEntity"}, ocei=ocei)
        action_targets = {_normalize_value(v) for _, v in anchors}
        for ent in entity_nodes[-3:]:
            if action_targets and not _entity_matches_value_domain(ent, action_targets):
                continue
            add_path(
                "context", [ent.id, action.id],
                summary="Entity version near action for freshness context.",
                score=0.4,
            )

    elif typ == "entity_state_consistency":
        # Route B Layer 3: show contradicts edges involving entities
        # the action depends on.
        contradictions = obligation.attrs.get("contradictions", [])
        for cd in contradictions:
            # Path: entity_a --contradicts--> entity_b showing the field change
            add_path(
                "conflicting",
                [cd["entity_a"], cd["entity_b"], action.id],
                summary=(
                    f"Entity {cd.get('entity_value', '?')}: "
                    f"{cd['field']} differs across observations "
                    f"({cd['value_a']} vs {cd['value_b']})."
                ),
                score=3.5,
                attrs={
                    "field": cd["field"],
                    "value_a": cd["value_a"],
                    "value_b": cd["value_b"],
                },
            )

        # For missing: show entities the action mentions but that lack
        # multi-observation coverage.
        mentioned = _action_mentioned_entities(graph, action.id, ocei=ocei)
        unresolved = [eid for eid, info in mentioned.items() if not info["resolved"]]
        for eid in unresolved[:3]:
            add_path(
                "context", [eid, action.id],
                summary="Entity referenced but has no multi-observation state coverage.",
                score=0.5,
            )


def _obs_shares_anchor_fields(obs: GraphNode, anchor_tails: set[str]) -> bool:
    """Check if an observation contains fields matching anchor key tails."""
    for source in ("response_body", "request_body"):
        payload = obs.attrs.get(source)
        for path, _ in _walk_scalar_items(payload):
            observed_tail = path.rsplit(".", 1)[-1].lower()
            if observed_tail in anchor_tails:
                return True
            # Also check aliases (e.g., 'to' ↔ 'email')
            for at in anchor_tails:
                aliases = {
                    "to": {"email", "recipient", "recipients"},
                    "email": {"to", "recipient", "recipients"},
                    "recipient": {"email", "to", "recipients"},
                    "customer_ids": {"id", "customer_id", "ticket_id"},
                }
                if observed_tail in aliases.get(at, set()):
                    return True
    return False


def _is_read_operation(obs: GraphNode, read_tools: set[str]) -> bool:
    """Heuristic: is this observation from a read/query operation?"""
    tool = str(obs.attrs.get("tool_name", "")).lower()
    if any(rt in tool for rt in read_tools):
        return True
    label = str(obs.label or "").lower()
    return any(rt in label for rt in read_tools)


def _entity_matches_value_domain(entity: GraphNode, values: set[str]) -> bool:
    """Check if an entity node is in the same domain as anchor values."""
    if not values:
        return False
    entity_text = json.dumps(entity.attrs, ensure_ascii=False).lower()
    for v in values:
        if not v or len(v) < 3:
            continue
        # Check for partial value match (same customer ID prefix, email domain, etc.)
        if v in entity_text:
            return True
        if "@" in v:
            domain = v.split("@", 1)[-1]
            if domain in entity_text:
                return True
        # Customer/ticket ID patterns
        if "-" in v and len(v) >= 5:
            prefix = v.rsplit("-", 1)[0]
            if prefix in entity_text:
                return True
    return False


def select_compact_res_subgraph(
    graph: BehaviorGraph,
    action_id: str,
    obligations: list[ObligationPredicate],
    candidate_paths: list[EvidencePath],
    missing_descriptors: list[MissingPathDescriptor],
    budget: dict[str, int],
) -> dict[str, Any]:
    """Contract-first RES selection.

    Phase 1 (Contract guarantee): every decided obligation gets minimum evidence
    regardless of budget — supported → ≥1 supporting path, conflicting → ≥1
    conflicting path, missing → descriptor anchor nodes.  Coverage is
    non-negotiable; the budget is not enforced here.

    Phase 2 (Compression): after the contract is satisfied, re-examine selected
    paths and substitute expensive paths with cheaper alternatives that cover the
    same obligations via shared nodes.

    Phase 3 (Context top-up): add context paths for missing/unknown obligations
    within the original budget.
    """
    max_nodes = budget.get("max_nodes", 24)
    max_edges = budget.get("max_edges", 36)
    node_map = {node.id: node for node in graph.nodes}
    selected_path_ids: set[str] = set()
    selected_nodes: set[str] = {action_id}
    selected_edge_refs: set[str] = set()

    # Always include anchor nodes of missing descriptors.
    for descriptor in missing_descriptors:
        selected_nodes.update(descriptor.anchor_node_ids)

    # Index paths by obligation for fast lookup.
    by_obl: dict[str, list[EvidencePath]] = defaultdict(list)
    for p in candidate_paths:
        by_obl[p.obligation_id].append(p)

    deciding = [p for p in candidate_paths if p.kind in {"supporting", "conflicting"}]
    context = [p for p in candidate_paths if p.kind == "context"]

    # ---- Phase 1: Contract-guaranteed greedy coverage (no budget ceiling) ----
    # Iteratively pick the cheapest deciding path among all uncovered
    # obligations.  This is the same marginal-cost greedy algorithm as before
    # but WITHOUT the budget termination — every supported/conflicting
    # obligation MUST be covered.  The contract is non-negotiable.
    uncovered_decided = {
        o.id for o in obligations
        if o.required and o.status in {"supported", "conflicting"}
    }
    while uncovered_decided:
        best_path: EvidencePath | None = None
        best_cost = 9999
        best_tiebreak = -1.0

        for path in deciding:
            if path.id in selected_path_ids:
                continue
            if path.obligation_id not in uncovered_decided:
                continue
            new_nodes = set(path.node_ids) - selected_nodes
            new_edges = set(path.edge_refs) - selected_edge_refs
            cost = len(new_nodes) + len(new_edges)
            tiebreak = (3.0 if path.kind == "conflicting" else 2.0) + path.score * 0.01
            if path.obligation_id.endswith(":verification_evidence") and len(path.source_event_pointers) >= 3:
                tiebreak += 1.5
                cost = max(0, cost - 2)
            if cost < best_cost or (cost == best_cost and tiebreak > best_tiebreak):
                best_cost = cost
                best_tiebreak = tiebreak
                best_path = path

        if best_path is None:
            break  # No more paths available for remaining obligations

        _add_path_unchecked(best_path, selected_nodes, selected_edge_refs, selected_path_ids)
        uncovered_decided.discard(best_path.obligation_id)

    # ---- Phase 3: Context top-up within budget ----
    uncovered = {
        o.id for o in obligations
        if o.required and o.status in {"missing", "unknown"}
        and not any(
            pid in selected_path_ids
            for pid in (o.matched_path_ids + o.conflicting_path_ids)
        )
    }
    context_candidates = [
        p for p in context
        if p.obligation_id in uncovered and p.id not in selected_path_ids
    ]
    context_candidates.sort(key=lambda p: (p.score, -len(p.node_ids)), reverse=True)

    for path in context_candidates:
        if len(selected_nodes) >= max_nodes and len(selected_edge_refs) >= max_edges:
            break
        _add_path(path, selected_nodes, selected_edge_refs, selected_path_ids, max_nodes, max_edges)

    edges = [
        edge
        for edge in graph.edges
        if _edge_ref(edge) in selected_edge_refs
        or (edge.source in selected_nodes and edge.target in selected_nodes and len(selected_edge_refs) < max_edges)
    ]
    selected_edge_refs.update(_edge_ref(edge) for edge in edges[:max_edges])

    # Populate evidence_contract on each obligation for downstream audit.
    for obl in obligations:
        selected_for_obl = [
            pid for pid in (obl.matched_path_ids + obl.conflicting_path_ids)
            if pid in selected_path_ids
        ]
        obl.evidence_contract = {
            "anchor_node": action_id,
            "status": obl.status,
            "selected_path_ids": selected_for_obl,
            "has_descriptor": any(
                d.obligation_id == obl.id for d in missing_descriptors
            ),
        }

    return {
        "selected_path_ids": selected_path_ids,
        "nodes": [asdict(node_map[node_id]) for node_id in sorted(selected_nodes, key=_node_sort_key) if node_id in node_map],
        "edges": [asdict(edge) for edge in edges[:max_edges]],
    }


def _cheapest_path(
    paths: list[EvidencePath],
    selected_nodes: set[str],
    selected_edge_refs: set[str],
) -> EvidencePath | None:
    """Return the path with smallest marginal node+edge cost."""
    best_path: EvidencePath | None = None
    best_cost = 9999
    best_score = -1.0
    for p in paths:
        new_nodes = set(p.node_ids) - selected_nodes
        new_edges = set(p.edge_refs) - selected_edge_refs
        cost = len(new_nodes) + len(new_edges)
        if cost < best_cost or (cost == best_cost and p.score > best_score):
            best_cost = cost
            best_score = p.score
            best_path = p
    return best_path


def _add_path_unchecked(
    path: EvidencePath,
    selected_nodes: set[str],
    selected_edge_refs: set[str],
    selected_path_ids: set[str],
) -> None:
    """Add a path without budget checks (used during contract guarantee phase)."""
    selected_path_ids.add(path.id)
    selected_nodes.update(path.node_ids)
    selected_edge_refs.update(path.edge_refs)


def _compress_selected_paths(
    obligations: list[ObligationPredicate],
    by_obl: dict[str, list[EvidencePath]],
    selected_nodes: set[str],
    selected_edge_refs: set[str],
    selected_path_ids: set[str],
) -> None:
    """Substitute expensive paths with cheaper alternatives that exploit shared nodes.

    After Phase 1, each covered obligation has a selected path.  Some of those
    paths may have high marginal cost because they were selected before many
    shared nodes were present.  We re-evaluate: for each covered obligation,
    if a different candidate path now has *lower* marginal cost (because nodes
    were added by other paths), swap it in.
    """
    for obl in obligations:
        if not obl.required or obl.status not in {"supported", "conflicting"}:
            continue

        current_ids = [
            pid for pid in (obl.matched_path_ids + obl.conflicting_path_ids)
            if pid in selected_path_ids
        ]
        if not current_ids:
            continue

        # Temporarily remove current paths for this obligation.
        current_paths = [p for paths in by_obl.values() for p in paths if p.id in current_ids]
        for p in current_paths:
            selected_path_ids.discard(p.id)
        # Recompute selected_nodes/edges without the removed paths.
        # (Approximation: we don't fully recompute node/edge sets; instead we
        #  just check if a cheaper path exists given current shared state.)
        current_nodes_snapshot = set(selected_nodes)
        current_edges_snapshot = set(selected_edge_refs)

        candidates = [
            p for p in by_obl.get(obl.id, [])
            if p.kind in {"supporting", "conflicting"} and p.id not in selected_path_ids
        ]
        best = _cheapest_path(candidates, current_nodes_snapshot, current_edges_snapshot)
        if best:
            current_best = _cheapest_path(current_paths, current_nodes_snapshot, current_edges_snapshot)
            current_cost = (
                len(set(current_best.node_ids) - current_nodes_snapshot) +
                len(set(current_best.edge_refs) - current_edges_snapshot)
            ) if current_best else 9999
            new_cost = (
                len(set(best.node_ids) - current_nodes_snapshot) +
                len(set(best.edge_refs) - current_edges_snapshot)
            )
            if new_cost < current_cost:
                # Swap: add cheaper path.
                _add_path_unchecked(best, selected_nodes, selected_edge_refs, selected_path_ids)
            else:
                # Keep original paths.
                for p in current_paths:
                    selected_path_ids.add(p.id)


def _add_path(
    path: EvidencePath,
    selected_nodes: set[str],
    selected_edge_refs: set[str],
    selected_path_ids: set[str],
    max_nodes: int,
    max_edges: int,
) -> bool:
    new_nodes = set(path.node_ids) - selected_nodes
    new_edges = set(path.edge_refs) - selected_edge_refs
    if len(selected_nodes) + len(new_nodes) > max_nodes or len(selected_edge_refs) + len(new_edges) > max_edges:
        return False
    selected_path_ids.add(path.id)
    selected_nodes.update(path.node_ids)
    selected_edge_refs.update(path.edge_refs)
    return True


def build_ocei(graph: BehaviorGraph, res_rows: list[RootEvidenceSubgraph]) -> dict[str, Any]:
    """Build the OCEI index and return a serializable dict (backward compat).

    Prefer ``OCEI.build(graph)`` for new code; this wrapper exists for the
    existing experiment pipeline.
    """
    from .ocei import OCEI as OCEIClass

    ocei = OCEIClass.build(graph)
    node_types = defaultdict(list)
    edge_types = defaultdict(list)
    for node in graph.nodes:
        node_types[node.type].append(node.id)
    for edge in graph.edges:
        edge_types[edge.type].append(_edge_ref(edge))
    actions = {}
    for res in res_rows:
        actions[res.action_id] = {
            "action_mechanism": res.action_summary.get("mechanism"),
            "derived_obligations": [item.to_dict() for item in res.obligations],
            "status_vector": res.obligation_statuses,
            "candidate_path_cache": [path.to_dict() for path in res.candidate_paths],
            "res_pointer": {"trace_id": res.trace_id, "task_id": res.task_id, "action_id": res.action_id},
            "source_event_pointers": res.source_event_pointers,
        }
    return {
        "trace_id": graph.trace_id,
        "task_id": graph.task_id,
        "node_type_cache": {key: value for key, value in sorted(node_types.items())},
        "edge_type_cache": {key: value for key, value in sorted(edge_types.items())},
        "actions": actions,
        "build_time_ms": ocei.build_time_ms,
        "memory_bytes": ocei.memory_bytes,
    }


def build_ocei_instance(graph: BehaviorGraph) -> OCEI:
    """Build the OCEI index and return the live object for query acceleration."""
    from .ocei import OCEI as OCEIClass

    return OCEIClass.build(graph)


def index_stats(ocei: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(ocei, ensure_ascii=False)
    return {
        "indexed_actions": len(ocei.get("actions", {})),
        "node_type_buckets": len(ocei.get("node_type_cache", {})),
        "edge_type_buckets": len(ocei.get("edge_type_cache", {})),
        "candidate_paths": sum(len(row.get("candidate_path_cache", [])) for row in ocei.get("actions", {}).values()),
        "memory_bytes_json": len(serialized.encode("utf-8")),
    }


def mine_closure_break_motifs(res_rows: list[RootEvidenceSubgraph], *, top_k: int = 20) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str], list[RootEvidenceSubgraph]] = defaultdict(list)
    for res in res_rows:
        mechanism = str(res.action_summary.get("mechanism"))
        channel = str(res.action_summary.get("channel"))
        for obligation in res.obligations:
            if obligation.status not in {"missing", "conflicting"}:
                continue
            signature = (
                obligation.type,
                mechanism,
                channel,
                obligation.missing_relation or obligation.type,
                obligation.status,
            )
            groups[signature].append(res)
    motifs = []
    for idx, (signature, rows) in enumerate(sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:top_k], start=1):
        obligation, mechanism, channel, relation, status = signature
        representative = min(rows, key=lambda row: len(row.compact_evidence_subgraph.get("nodes", [])))
        motifs.append(
            {
                "motif_id": f"motif_{idx}",
                "broken_obligation": obligation,
                "action_mechanism": mechanism,
                "channel": channel,
                "missing_or_conflicting_relation": relation,
                "status_type": status,
                "frequency": len(rows),
                "representative_res": {
                    "trace_id": representative.trace_id,
                    "task_id": representative.task_id,
                    "action_id": representative.action_id,
                    "nodes": len(representative.compact_evidence_subgraph.get("nodes", [])),
                    "source_event_pointers": representative.source_event_pointers,
                },
                "human_readable_rule": _motif_rule(obligation, mechanism, channel, status),
            }
        )
    return {
        "total_res": len(res_rows),
        "motif_count": len(motifs),
        "motifs": motifs,
    }


def query_results_summary(graph: BehaviorGraph, res_rows: list[RootEvidenceSubgraph]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    obligation_counts: Counter[str] = Counter()
    for res in res_rows:
        for obligation in res.obligations:
            obligation_counts[obligation.type] += 1
            status_counts[f"{obligation.type}:{obligation.status}"] += 1
    total_required = sum(
        1
        for res in res_rows
        for obligation in res.obligations
        if obligation.status != "not_applicable"
    )
    decided = sum(
        1
        for res in res_rows
        for obligation in res.obligations
        if obligation.status in STATUS_DECIDED
    )
    return {
        "trace_id": graph.trace_id,
        "task_id": graph.task_id,
        "oeg_nodes": len(graph.nodes),
        "oeg_edges": len(graph.edges),
        "high_impact_actions": len(res_rows),
        "total_required_obligations": total_required,
        "decided_obligations": decided,
        "predicate_coverage": round(decided / total_required, 4) if total_required else None,
        "status_counts": dict(status_counts),
        "obligation_counts": dict(obligation_counts),
        "avg_res_nodes": _avg(len(res.compact_evidence_subgraph.get("nodes", [])) for res in res_rows),
        "avg_res_edges": _avg(len(res.compact_evidence_subgraph.get("edges", [])) for res in res_rows),
        "avg_query_latency_ms": _avg(res.timings.get("total_ms", 0.0) for res in res_rows),
    }


def analysis_report(graph: BehaviorGraph, res_rows: list[RootEvidenceSubgraph]) -> dict[str, Any]:
    summary = query_results_summary(graph, res_rows)
    compactness = []
    for res in res_rows:
        nodes = len(res.compact_evidence_subgraph.get("nodes", []))
        edges = len(res.compact_evidence_subgraph.get("edges", []))
        compactness.append(
            {
                "action_id": res.action_id,
                "res_nodes_over_oeg_nodes": round(nodes / len(graph.nodes), 4) if graph.nodes else None,
                "res_edges_over_oeg_edges": round(edges / len(graph.edges), 4) if graph.edges else None,
                "coverage": res.coverage,
                "redundancy": res.redundancy,
            }
        )
    return {
        "summary": summary,
        "compactness": compactness,
        "baseline_placeholders": {
            "full_oeg_neighborhood": "Compute with the same action ids and radius budget.",
            "shortest_path": "Compute shortest typed path per obligation.",
            "top_k_path": "Rank candidate paths without coverage-aware selection.",
            "steiner_style_selection": "Map obligations to terminals/prizes before comparison.",
        },
    }


def _refresh_obligation_statuses(
    obligations: list[ObligationPredicate],
    candidate_paths: list[EvidencePath],
    missing_descriptors: list[MissingPathDescriptor],
) -> None:
    by_obligation: dict[str, list[EvidencePath]] = defaultdict(list)
    missing_ids = {item.obligation_id for item in missing_descriptors}
    for path in candidate_paths:
        by_obligation[path.obligation_id].append(path)
    for obligation in obligations:
        if not obligation.required:
            continue
        paths = by_obligation.get(obligation.id, [])
        has_supporting = any(path.kind == "supporting" for path in paths)
        has_conflicting = any(path.kind == "conflicting" for path in paths)
        if has_conflicting:
            obligation.status = "conflicting"
        elif has_supporting:
            obligation.status = "supported"
        elif obligation.id in missing_ids:
            obligation.status = "missing"
        elif obligation.status == "supported":
            # Obligation was marked supported at derivation but no evidence
            # paths survived retrieval — downgrade to missing.
            obligation.status = "missing"
        # If status is "unknown" and no paths/materialized, keep as unknown.


def _coverage_summary(obligations: list[ObligationPredicate], selected_path_ids: set[str]) -> dict[str, Any]:
    required = [item for item in obligations if item.status != "not_applicable"]
    decided = [item for item in required if item.status in STATUS_DECIDED]
    selected_decided = [
        item
        for item in decided
        if item.status == "missing"
        or any(path_id in selected_path_ids for path_id in [*item.matched_path_ids, *item.conflicting_path_ids])
    ]
    return {
        "required_obligations": len(required),
        "decided_obligations": len(decided),
        "selected_decision_coverage": round(len(selected_decided) / len(required), 4) if required else None,
        "status_counts": dict(Counter(item.status for item in obligations)),
    }


def _redundancy_summary(graph: BehaviorGraph, selected: dict[str, Any], candidate_paths: list[EvidencePath]) -> dict[str, Any]:
    selected_ids = set(selected["selected_path_ids"])
    useful_nodes = {node_id for path in candidate_paths if path.id in selected_ids for node_id in path.node_ids}
    useful_nodes.update(node.get("id") for node in selected["nodes"] if node.get("type") == "ToolAction")
    selected_node_ids = {node.get("id") for node in selected["nodes"]}
    redundant_nodes = selected_node_ids - useful_nodes
    return {
        "res_nodes": len(selected["nodes"]),
        "res_edges": len(selected["edges"]),
        "res_nodes_over_oeg_nodes": round(len(selected["nodes"]) / len(graph.nodes), 4) if graph.nodes else None,
        "res_edges_over_oeg_edges": round(len(selected["edges"]) / len(graph.edges), 4) if graph.edges else None,
        "redundant_node_rate": round(len(redundant_nodes) / len(selected_node_ids), 4) if selected_node_ids else 0.0,
    }


def _canonical_event_type(node_type: str) -> str:
    mapping = {
        "AssistantMessage": "assistant_message",
        "ExternalEffect": "derived_external_effect",
        "OutputAssertion": "output_assertion",
        "Requirement": "task_requirement",
        "ServiceEntity": "derived_service_entity",
        "Submission": "final_submission",
        "TaskInstruction": "task_instruction",
        "ToolAction": "tool_action",
        "ToolObservation": "tool_observation",
    }
    return mapping.get(node_type, node_type[:1].lower() + node_type[1:])


def _requires_target_binding(action: GraphNode) -> bool:
    mechanism = str(action.attrs.get("mechanism") or "")
    return bool(
        action.attrs.get("approval_sensitive")
        or action.attrs.get("external_visibility")
        or action.attrs.get("risk_level") == "high"
        or (action.attrs.get("effectful") and mechanism in CONTENT_DEPENDENT_MECHANISMS)
    )


def _structural_conflicts_for_action(graph: BehaviorGraph, action_id: str, *, ocei: OCEI | None = None) -> list[dict[str, Any]]:
    rows = []
    if ocei is not None:
        for source_id, _etype, _ in ocei.incoming.get(action_id, []):
            source = ocei.node_map.get(source_id)
            if not source or source.type != "StructuralSignals":
                continue
            signal_type = source.label or source.attrs.get("type")
            if signal_type in {"ambiguous_entity_binding", "target_binding_not_certified"}:
                rows.append({
                    "type": signal_type,
                    "summary": source.attrs.get("summary", signal_type),
                    "node_ids": [source.id, action_id],
                })
    else:
        for edge in graph.edges:
            if edge.target != action_id:
                continue
            source = graph.node(edge.source)
            if not source or source.type != "StructuralSignals":
                continue
            signal_type = source.label or source.attrs.get("type")
            if signal_type in {"ambiguous_entity_binding", "target_binding_not_certified"}:
                rows.append({
                    "type": signal_type,
                    "summary": source.attrs.get("summary", signal_type),
                    "node_ids": [source.id, action_id],
                })

    action = graph.node(action_id)
    if action is None:
        return rows
    for edge in graph.edges:
        if edge.source != action_id or edge.type != "depends_on":
            continue
        observation = graph.node(edge.target)
        if observation is None or observation.type != "ToolObservation":
            continue
        derivation = edge.attrs.get("derivation")
        if derivation == "entity_key_conflict":
            rows.append({
                "type": "conflicting_source_response",
                "summary": "The same source request returned identity values that conflict with the target action.",
                "node_ids": [observation.id, action_id],
            })
        elif derivation == "late_entity_alignment" and edge.attrs.get("temporal_conflict") is True:
            rows.append({
                "type": "non_causal_observation",
                "summary": "Matching source evidence occurs after the target action decision.",
                "node_ids": [observation.id, action_id],
            })

    action_anchors = _anchor_map(action.attrs.get("input", {}))
    if action_anchors:
        for edge in graph.edges:
            if edge.source != action_id or edge.type != "observes":
                continue
            observation = graph.node(edge.target)
            if observation is None or observation.type != "ToolObservation":
                continue
            if observation.attrs.get("tool_use_id") != action.attrs.get("tool_use_id"):
                continue
            dispatch_anchors = _anchor_map(observation.attrs.get("request_body", {}))
            mismatched = sorted(
                key for key, values in action_anchors.items()
                if key in dispatch_anchors and not (values & dispatch_anchors[key])
            )
            if mismatched:
                rows.append({
                    "type": "action_dispatch_entity_mismatch",
                    "summary": "The declared action entity differs from the executed dispatch request.",
                    "node_ids": [observation.id, action_id],
                    "keys": mismatched,
                })
    return rows


def _anchor_map(value: Any) -> dict[str, set[str]]:
    anchors: dict[str, set[str]] = defaultdict(set)
    for key, item in _action_anchors(value):
        anchors[key.rsplit(".", 1)[-1].lower()].add(_normalize_value(item))
    return anchors


def _supported_anchor_values(observations: list[GraphNode], anchors: list[tuple[str, Any]]) -> set[tuple[str, str]]:
    supported = set()
    for key, value in anchors:
        value_text = _normalize_value(value)
        if not value_text:
            continue
        for obs in observations:
            if _anchor_supported_by_observation(obs, key, value_text):
                supported.add((key, value_text))
                break
    return supported


def _task_supported_anchor_values(graph: BehaviorGraph, anchors: list[tuple[str, Any]]) -> set[tuple[str, str]]:
    supported = set()
    for key, value in anchors:
        value_text = _normalize_value(value)
        if value_text and _task_mentions_anchor(graph, key, value):
            supported.add((key, value_text))
    return supported


def _task_mentions_anchor(graph: BehaviorGraph, key: str, value: Any) -> bool:
    value_text = _normalize_value(value)
    if not value_text:
        return False
    task_text = _task_text(graph)
    if value_text in task_text:
        return True
    tail = key.rsplit(".", 1)[-1].lower()
    if "@" in value_text:
        local, _, domain = value_text.partition("@")
        if value_text in task_text:
            return True
        if len(local) >= 3 and _contains_task_token(task_text, local):
            return True
        if len(domain) >= 3 and _contains_task_token(task_text, domain):
            return True
        return False
    if tail in {"to", "recipient", "recipients"} and _looks_like_team_alias(value_text):
        return _team_alias_is_task_authorized(task_text, value_text)
    return False


def _task_authorizes_target_binding(graph: BehaviorGraph, action: GraphNode, anchors: list[tuple[str, Any]]) -> bool:
    if not _task_authorizes_action(graph, action):
        return False
    task_text = _task_text(graph)
    mechanism = str(action.attrs.get("mechanism") or "")
    if mechanism == "share" and any(term in task_text for term in ("attendee", "attendees", "参会", "参会人")):
        return True
    for key, value in anchors:
        value_text = _normalize_value(value)
        tail = key.rsplit(".", 1)[-1].lower()
        if tail in {"to", "recipient", "recipients"} and _looks_like_team_alias(value_text):
            if _team_alias_is_task_authorized(task_text, value_text):
                return True
    return bool(anchors) and all(_task_mentions_anchor(graph, key, value) for key, value in anchors)


def _matching_prior_observations(graph: BehaviorGraph, action: GraphNode, value: Any) -> list[GraphNode]:
    return _matching_prior_observations_for_key(graph, action, "", value)


def _matching_prior_observations_for_key(
    graph: BehaviorGraph,
    action: GraphNode,
    key: str,
    value: Any,
    *,
    ocei: OCEI | None = None,
) -> list[GraphNode]:
    value_text = _normalize_value(value)
    if not value_text:
        return []
    matches = []
    for obs in _prior_nodes(graph, action, {"ToolObservation"}, ocei=ocei):
        if _anchor_supported_by_observation(obs, key, value_text):
            matches.append(obs)
    return matches


def _value_supported_by_anchor(obs: GraphNode, key: str, value: Any) -> bool:
    value_text = _normalize_value(value)
    if not value_text:
        return False
    return _anchor_supported_by_observation(obs, key, value_text)


def _aligned_prior_observations(graph: BehaviorGraph, action: GraphNode, anchors: list[tuple[str, Any]], *, ocei: OCEI | None = None) -> list[GraphNode]:
    if ocei is not None:
        return ocei.aligned_prior_observations(action.id, anchors)
    aligned: list[GraphNode] = []
    seen: set[str] = set()
    for key, value in anchors:
        tail = key.rsplit(".", 1)[-1].lower()
        if tail in FREE_TEXT_KEYS:
            continue
        for obs in _matching_prior_observations_for_key(graph, action, key, value, ocei=ocei):
            if obs.id not in seen:
                aligned.append(obs)
                seen.add(obs.id)
    return aligned


def _audit_relevant_prior_observations(
    graph: BehaviorGraph,
    action: GraphNode,
    anchors: list[tuple[str, Any]],
    *,
    ocei: OCEI | None = None,
) -> list[GraphNode]:
    """Prior observations that can explain an action even without entity edges."""
    mechanism = str(action.attrs.get("mechanism") or "")
    aligned = _aligned_prior_observations(graph, action, anchors, ocei=ocei)
    rows: list[GraphNode] = []
    seen: set[str] = set()
    for obs in aligned:
        rows.append(obs)
        seen.add(obs.id)

    required_values = [
        _normalize_value(value)
        for key, value in anchors
        if key.rsplit(".", 1)[-1].lower() not in FREE_TEXT_KEYS
    ]
    action_text = _action_payload_text(action)
    action_idx = action.source_event_idx or 0
    for obs in _prior_nodes(graph, action, {"ToolObservation"}, ocei=ocei):
        if obs.id in seen:
            continue
        window = 60 if mechanism in CONTENT_DEPENDENT_MECHANISMS else 10
        if not _is_near_action(obs, action_idx, window=window):
            continue
        obs_text = _node_payload_text(obs)
        if (
            _observation_mentions_required_value(obs_text, required_values)
            or _shares_meaningful_tokens(obs_text, action_text)
            or _payload_field_overlap(obs_text, action_text)
        ):
            rows.append(obs)
            seen.add(obs.id)
    return sorted(rows, key=lambda node: (node.source_event_idx or -1, node.id))


def _audit_relevant_prior_context(
    graph: BehaviorGraph,
    action: GraphNode,
    anchors: list[tuple[str, Any]],
    *,
    ocei: OCEI | None = None,
) -> list[GraphNode]:
    """Prior trace nodes that ground the action, including user-supplied context."""
    mechanism = str(action.attrs.get("mechanism") or "")
    rows = _audit_relevant_prior_observations(graph, action, anchors, ocei=ocei)
    seen = {node.id for node in rows}
    action_text = _action_payload_text(action)
    required_values = [_normalize_value(value) for _, value in anchors]
    if mechanism not in {"delete", "export"}:
        return sorted(rows, key=lambda node: (node.source_event_idx or -1, node.id))
    for node in _prior_nodes(graph, action, {"Message"}, ocei=ocei):
        if node.id in seen:
            continue
        if not _is_user_authored_message(node):
            continue
        if not _is_near_action(node, action.source_event_idx or 0, window=8):
            continue
        text = _node_payload_text(node)
        if (
            _observation_mentions_required_value(text, required_values)
            or _shares_meaningful_tokens(text, action_text)
            or _payload_field_overlap(text, action_text)
        ):
            rows.append(node)
            seen.add(node.id)
    return sorted(rows, key=lambda node: (node.source_event_idx or -1, node.id))


def _all_required_anchors_in_context(nodes: list[GraphNode], anchors: list[tuple[str, Any]]) -> bool:
    if not anchors:
        return False
    return all(any(_context_supports_anchor(node, key, value) for node in nodes) for key, value in anchors)


def _context_supports_anchor(node: GraphNode, key: str, value: Any) -> bool:
    value_text = _normalize_value(value)
    if not value_text:
        return False
    if node.type == "ToolObservation" and _anchor_supported_by_observation(node, key, value_text):
        return True
    return _contains_tokenish(_node_payload_text(node), value_text) or value_text in _node_payload_text(node)


def _composite_payload_supported(action: GraphNode, nodes: list[GraphNode]) -> bool:
    return bool(_composite_payload_support_nodes(action, nodes))


def _composite_payload_support_nodes(action: GraphNode, nodes: list[GraphNode]) -> list[GraphNode]:
    mechanism = str(action.attrs.get("mechanism") or "")
    if mechanism not in CONTENT_DEPENDENT_MECHANISMS:
        return []
    action_tokens = _meaningful_tokens(_action_payload_text(action))
    if not action_tokens:
        return []
    scored: list[tuple[int, GraphNode]] = []
    covered: set[str] = set()
    for node in nodes:
        node_tokens = _meaningful_tokens(_node_payload_text(node))
        overlap = action_tokens & node_tokens
        if not overlap:
            continue
        score = len(overlap) + sum(2 for token in overlap if _looks_like_identifier(token))
        if score >= 2:
            scored.append((score, node))
            covered.update(overlap)
    if len(scored) >= 2 and (len(covered) >= 4 or any(_looks_like_identifier(token) for token in covered)):
        return [node for _, node in sorted(scored, key=lambda item: (item[0], item[1].source_event_idx or -1), reverse=True)]
    if scored and mechanism in {"save_draft", "share", "update", "send"}:
        return [node for _, node in sorted(scored, key=lambda item: (item[0], item[1].source_event_idx or -1), reverse=True)]
    return []


def _prior_authorization_context(
    graph: BehaviorGraph,
    action: GraphNode,
    anchors: list[tuple[str, Any]],
    *,
    ocei: OCEI | None = None,
) -> list[GraphNode]:
    mechanism = str(action.attrs.get("mechanism") or "")
    terms = TASK_AUTHORIZATION_TERMS.get(mechanism, ())
    if not terms:
        return []
    rows: list[GraphNode] = []
    for node in _prior_nodes(graph, action, {"Message", "ToolObservation"}, ocei=ocei):
        if node.type == "Message":
            if mechanism not in {"delete", "export"} or not _is_user_authored_message(node):
                continue
            if not _is_near_action(node, action.source_event_idx or 0, window=8):
                continue
        text = _node_payload_text(node)
        if not any(term.lower() in text for term in terms):
            continue
        if anchors and not any(_context_supports_anchor(node, key, value) for key, value in anchors):
            continue
        rows.append(node)
    return sorted(rows, key=lambda node: (node.source_event_idx or -1, node.id))


def _all_required_anchors_in_observations(observations: list[GraphNode], anchors: list[tuple[str, Any]]) -> bool:
    if not anchors:
        return False
    for key, value in anchors:
        value_text = _normalize_value(value)
        if not value_text:
            return False
        if not any(_anchor_supported_by_observation(obs, key, value_text) for obs in observations):
            return False
    return True


def _effect_feedback_nodes(graph: BehaviorGraph, action: GraphNode, *, ocei: OCEI | None = None) -> list[GraphNode]:
    """Tool feedback/effect nodes directly linked to or immediately after action."""
    rows: list[GraphNode] = []
    seen: set[str] = set()

    for node in _outgoing_nodes(graph, action.id, {"ToolObservation", "ExternalEffect"}, ocei=ocei):
        rows.append(node)
        seen.add(node.id)

    action_idx = action.source_event_idx or 0
    for node in graph.nodes:
        if node.id in seen or node.type not in {"ToolObservation", "ExternalEffect"}:
            continue
        if node.source_event_idx is None:
            continue
        if action_idx <= node.source_event_idx <= action_idx + 3:
            if _node_mentions_tool_use(node, action) or _shares_meaningful_tokens(_node_payload_text(node), _action_payload_text(action)):
                rows.append(node)
                seen.add(node.id)
    return sorted(rows, key=lambda node: (node.source_event_idx or -1, node.id))


def _post_action_verification_nodes(
    graph: BehaviorGraph,
    action: GraphNode,
    anchors: list[tuple[str, Any]],
    *,
    ocei: OCEI | None = None,
) -> list[GraphNode]:
    """Later read/query observations that verify an action's target or effect."""
    action_idx = action.source_event_idx or 0
    action_text = _action_payload_text(action)
    anchor_values = [_normalize_value(value) for _, value in anchors]
    rows: list[GraphNode] = []
    for node in graph.nodes:
        if node.type != "ToolObservation" or node.source_event_idx is None:
            continue
        if not (action_idx < node.source_event_idx <= action_idx + 8):
            continue
        tool_text = _normalize_value(node.attrs.get("tool_name", ""))
        payload = _node_payload_text(node)
        if not any(term in tool_text or term in payload for term in ("get", "list", "query", "status", "verify", "connection")):
            continue
        if _observation_mentions_required_value(payload, anchor_values) or _shares_meaningful_tokens(payload, action_text):
            rows.append(node)
    return sorted(rows, key=lambda node: (node.source_event_idx or -1, node.id))


def _verification_chain_nodes(graph: BehaviorGraph, verification_node: GraphNode) -> list[str]:
    """Assistant/action nodes immediately leading to a verification observation."""
    ids: list[str] = []
    if verification_node.source_event_idx is None:
        return ids
    for node in graph.nodes:
        if node.source_event_idx is None:
            continue
        if verification_node.source_event_idx - 2 <= node.source_event_idx < verification_node.source_event_idx:
            if node.type in {"AssistantMessage", "ToolAction"}:
                ids.append(node.id)
    return ids[-3:]


def _action_output_assertions(graph: BehaviorGraph, action: GraphNode, *, ocei: OCEI | None = None) -> list[GraphNode]:
    """Output assertions close enough to the action to be local audit claims."""
    action_idx = action.source_event_idx or 0
    rows: list[GraphNode] = []
    action_text = _action_payload_text(action)
    has_feedback = bool(_effect_feedback_nodes(graph, action, ocei=ocei))
    for node in graph.nodes:
        if node.type != "OutputAssertion" or node.source_event_idx is None:
            continue
        if not (action_idx <= node.source_event_idx <= action_idx + 6):
            continue
        text = _node_payload_text(node)
        if has_feedback or _shares_meaningful_tokens(text, action_text) or any(
            edge.source == node.id and edge.target == action.id for edge in graph.edges
        ):
            rows.append(node)
    return sorted(rows, key=lambda node: (node.source_event_idx or -1, node.id))


def _anchor_supported_by_observation(obs: GraphNode, anchor_key: str, value_text: str) -> bool:
    if not value_text:
        return False
    for source in ("response_body", "request_body"):
        payload = obs.attrs.get(source)
        for path, observed_value in _walk_scalar_items(payload):
            if _anchor_field_matches(anchor_key, path) and _normalize_value(observed_value) == value_text:
                return True
            if _anchor_value_in_text(anchor_key, value_text, observed_value):
                return True
    return False


def _anchor_value_in_text(anchor_key: str, value_text: str, observed_value: Any) -> bool:
    if not value_text or len(value_text) < 3:
        return False
    text = _normalize_value(observed_value)
    if not text:
        return False
    if value_text in text:
        return True
    tail = anchor_key.rsplit(".", 1)[-1].lower()
    if tail in {"element_ref", "ref", "id", "channel", "connection_id"}:
        return _contains_tokenish(text, value_text)
    return False


def _anchor_field_matches(anchor_key: str, observed_key: str) -> bool:
    anchor_tail = anchor_key.rsplit(".", 1)[-1].lower()
    observed_tail = observed_key.rsplit(".", 1)[-1].lower()
    if not anchor_tail:
        return False
    if anchor_tail == observed_tail:
        return True
    aliases = {
        "to": {"email", "recipient", "recipients"},
        "recipient": {"email", "to", "recipients"},
        "recipients": {"email", "to", "recipient"},
        "attendees": {"attendee", "email", "emails"},
        "attendee": {"attendees", "email", "emails"},
        "email": {"to", "recipient", "recipients", "attendee", "attendees"},
        "start_time": {"start", "start_date", "time"},
        "end_time": {"end", "end_date", "time"},
        "title": {"name", "subject"},
        "location": {"place", "room"},
    }
    return observed_tail in aliases.get(anchor_tail, set())


def _prior_nodes(graph: BehaviorGraph, action: GraphNode, node_types: set[str], *, ocei: OCEI | None = None) -> list[GraphNode]:
    if ocei is not None:
        return ocei.prior_nodes(action.id, node_types)
    action_idx = action.source_event_idx or 0
    return [
        node
        for node in graph.nodes
        if node.type in node_types and node.source_event_idx is not None and node.source_event_idx < action_idx
    ]


def _outgoing_nodes(graph: BehaviorGraph, source_id: str, node_types: set[str], *, ocei: OCEI | None = None) -> list[GraphNode]:
    if ocei is not None:
        return ocei.outgoing_nodes(source_id, node_types)
    rows = []
    for edge in graph.edges:
        if edge.source != source_id:
            continue
        target = graph.node(edge.target)
        if target and target.type in node_types:
            rows.append(target)
    return rows


def _has_edge(graph: BehaviorGraph, source: str, target: str, edge_types: set[str], *, ocei: OCEI | None = None) -> bool:
    if ocei is not None:
        return any(
            t == target and e in edge_types
            for t, e, _ in ocei.outgoing.get(source, [])
        )
    return any(edge.source == source and edge.target == target and edge.type in edge_types for edge in graph.edges)


def _task_authorizes_action(graph: BehaviorGraph, action: GraphNode) -> bool:
    mechanism = str(action.attrs.get("mechanism") or "")
    terms = TASK_AUTHORIZATION_TERMS.get(mechanism)
    if not terms:
        return False
    task_text = _task_text(graph)
    if not task_text:
        return False
    return any(term.lower() in task_text for term in terms)


def _task_text(graph: BehaviorGraph) -> str:
    text_chunks = []
    for node in graph.nodes:
        if node.type in {"TaskInstruction", "Requirement"}:
            text_chunks.append(str(node.attrs.get("text") or ""))
    return "\n".join(text_chunks).lower()


def _required_parameter_anchors(mechanism: str, anchors: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    if mechanism == "save_draft":
        return [item for item in anchors if item[0].rsplit(".", 1)[-1].lower() in SAVE_DRAFT_STRONG_ANCHOR_KEYS]
    return anchors


def _looks_like_team_alias(value_text: str) -> bool:
    if "@" not in value_text:
        return False
    local = value_text.split("@", 1)[0]
    return local in {"ops", "support", "admin", "security", "it", "team", "hr", "finance"} or local.endswith(("team", "ops"))


def _team_alias_is_task_authorized(task_text: str, value_text: str) -> bool:
    local = value_text.split("@", 1)[0]
    return local in task_text or any(term in task_text for term in ("ops team", "运维", "运营团队", "support team"))


def _contains_task_token(task_text: str, token: str) -> bool:
    if not token:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(token.lower())}(?![a-z0-9])"
    return bool(re.search(pattern, task_text))


def _edge_refs_between(graph: BehaviorGraph, source: str, target: str, *, ocei: OCEI | None = None) -> list[str]:
    if ocei is not None:
        return [
            f"{source}->{e}->{t}"
            for t, e, _ in ocei.outgoing.get(source, [])
            if t == target
        ]
    return [_edge_ref(edge) for edge in graph.edges if edge.source == source and edge.target == target]


def _edge_ref(edge: GraphEdge) -> str:
    return f"{edge.source}->{edge.type}->{edge.target}"


def _source_pointers(node_map: dict[str, GraphNode], node_ids: list[str]) -> list[int]:
    return sorted({node_map[node_id].source_event_idx for node_id in node_ids if node_id in node_map and node_map[node_id].source_event_idx is not None})


def _node_sort_key(node_id: str) -> tuple[int, str]:
    if node_id == "task_instruction":
        return (0, node_id)
    digits = "".join(ch for ch in node_id if ch.isdigit())
    return (int(digits) if digits else 10**9, node_id)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    rows = []
    for value in values:
        if value not in seen:
            rows.append(value)
            seen.add(value)
    return rows


def _action_anchors(body: Any) -> list[tuple[str, Any]]:
    anchors: list[tuple[str, Any]] = []
    for key, value in _walk_scalar_items(body):
        tail = key.rsplit(".", 1)[-1].lower()
        if tail in FREE_TEXT_KEYS:
            continue
        text = str(value).strip()
        if not text or len(text) < 3:
            continue
        if any(marker in tail for marker in ANCHOR_KEYWORDS) or "@" in text:
            anchors.append((key, value))
    return anchors


def _walk_scalar_items(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_walk_scalar_items(item, next_key))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            rows.extend(_walk_scalar_items(item, f"{prefix}[{idx}]"))
    elif value is not None and not isinstance(value, (dict, list)):
        rows.append((prefix, value))
    return rows


def _normalize_value(value: Any) -> str:
    return str(value).strip().lower()


def _node_payload_text(node: GraphNode) -> str:
    chunks = [node.label or ""]
    for key in ("text", "summary", "request_body", "response_body", "input", "record", "content"):
        if key in node.attrs:
            chunks.append(json.dumps(node.attrs.get(key), ensure_ascii=False) if not isinstance(node.attrs.get(key), str) else node.attrs.get(key))
    return _normalize_value("\n".join(str(chunk) for chunk in chunks if chunk is not None))


def _is_user_authored_message(node: GraphNode) -> bool:
    if node.type != "Message" or node.attrs.get("role") != "user":
        return False
    content = node.attrs.get("content")
    if isinstance(content, list) and any(isinstance(item, dict) and item.get("type") == "tool_result" for item in content):
        return False
    return bool(str(node.attrs.get("text") or "").strip())


def _action_payload_text(action: GraphNode) -> str:
    return _node_payload_text(action)


def _is_near_action(node: GraphNode, action_idx: int, *, window: int) -> bool:
    return node.source_event_idx is not None and action_idx - window <= node.source_event_idx < action_idx


def _node_mentions_tool_use(node: GraphNode, action: GraphNode) -> bool:
    tool_use_id = action.attrs.get("tool_use_id")
    if tool_use_id and node.attrs.get("tool_use_id") == tool_use_id:
        return True
    return bool(tool_use_id and tool_use_id in _node_payload_text(node))


def _observation_mentions_required_value(obs_text: str, values: list[str]) -> bool:
    return any(_contains_tokenish(obs_text, value) for value in values if value and len(value) >= 3)


def _contains_tokenish(text: str, value: str) -> bool:
    if not text or not value:
        return False
    if value in text:
        return True
    if len(value) < 3:
        return False
    pattern = rf"(?<![a-z0-9_@.-]){re.escape(value)}(?![a-z0-9_@.-])"
    return bool(re.search(pattern, text))


def _shares_meaningful_tokens(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens & right_tokens
    return len(overlap) >= 2 or any(len(token) >= 8 for token in overlap)


def _payload_field_overlap(left: str, right: str) -> bool:
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    overlap = left_tokens & right_tokens
    if not overlap:
        return False
    return any(_looks_like_identifier(token) for token in overlap) or len(overlap) >= 3


def _looks_like_identifier(token: str) -> bool:
    if len(token) >= 8 and any(ch.isdigit() for ch in token):
        return True
    return bool(re.search(r"[a-z]+[_:-][a-z0-9_:-]+", token))


def _meaningful_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9][a-z0-9_@./:-]{2,}", text.lower()))
    stop = {
        "the", "and", "for", "with", "this", "that", "from", "tool", "action",
        "request", "response", "status", "input", "output", "true", "false",
        "api", "http", "https", "method", "post", "get",
    }
    return {token for token in tokens if token not in stop and len(token) >= 3}


def _motif_rule(obligation: str, mechanism: str, channel: str, status: str) -> str:
    return f"{status} {obligation} before {mechanism} action on {channel} channel"


def _avg(values: Any) -> float | None:
    rows = [float(value) for value in values]
    if not rows:
        return None
    return round(sum(rows) / len(rows), 4)


def bounded_typed_reachability(
    graph: BehaviorGraph,
    source_id: str,
    *,
    max_depth: int = 3,
    allowed_edge_types: set[str] | None = None,
) -> dict[str, int]:
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in graph.edges:
        if allowed_edge_types and edge.type not in allowed_edge_types:
            continue
        adjacency[edge.source].append((edge.target, edge.type))
        adjacency[edge.target].append((edge.source, edge.type))
    seen = {source_id: 0}
    queue = deque([source_id])
    while queue:
        node_id = queue.popleft()
        depth = seen[node_id]
        if depth >= max_depth:
            continue
        for next_id, _ in adjacency.get(node_id, []):
            if next_id in seen:
                continue
            seen[next_id] = depth + 1
            queue.append(next_id)
    return seen
