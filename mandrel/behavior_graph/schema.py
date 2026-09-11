"""Minimal schema for passive tool-service behavior graphs.

Edge type taxonomy (three-layer semantics):
  Entity Layer: resolves, mentions, same_as, contradicts
  Control Layer: depends_on, requires, constrains, satisfies, authorizes
  Effect Layer:  triggers, observes, verifies, grounds, invalidates
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .lexicons import ENTITY_TYPE_MAP

# ---------------------------------------------------------------------------
# Edge type constants
# ---------------------------------------------------------------------------

# Entity Layer — entity resolution, parameter-to-entity binding, cross-action
# entity alignment and contradiction detection.
EDGE_RESOLVES = "resolves"  # ToolObservation uniquely identifies a ServiceEntity
EDGE_MENTIONS = "mentions"  # ToolAction parameter references a ServiceEntity
EDGE_SAME_AS = "same_as"  # Two ServiceEntity nodes refer to the same real-world object
EDGE_CONTRADICTS = "contradicts"  # Two observations give conflicting values for the same entity

# Control Layer — task-to-action constraints, approval evidence, task-level
# authorization overrides.
EDGE_CONSTRAINS = "constrains"  # Requirement constrains an action
EDGE_SATISFIES = "satisfies"  # ApprovalEvent satisfies an approval requirement
EDGE_AUTHORIZES = "authorizes"  # TaskInstruction explicitly authorizes an action class

# Effect Layer — action-to-effect materialization, verification coverage,
# and effect invalidation.
EDGE_TRIGGERS = "triggers"  # ToolAction triggers an ExternalEffect
EDGE_INVALIDATES = "invalidates"  # A later effect invalidates an earlier effect

# Legacy / general-purpose (kept for backward compatibility)
EDGE_DEPENDS_ON = "depends_on"
EDGE_REQUIRES = "requires"
EDGE_OBSERVES = "observes"
EDGE_VERIFIES = "verifies"
EDGE_GROUNDS = "grounds"
EDGE_SUBMITS = "submits"

@dataclass
class GraphNode:
    id: str
    type: str
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)
    source_event_idx: int | None = None


@dataclass
class GraphEdge:
    source: str
    target: str
    type: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralSignal:
    id: str
    type: str
    severity: str
    summary: str
    node_ids: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateBehavior:
    id: str
    kind: str
    risk_level: str
    action_node_id: str | None
    summary: str
    signal_ids: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorCard:
    card_id: str
    trace_id: str
    task_id: str
    candidate: CandidateBehavior
    current_action: dict[str, Any]
    relevant_history: list[dict[str, Any]]
    graph_context: dict[str, Any]
    structural_signals: list[StructuralSignal]
    behavior_context: dict[str, Any] = field(default_factory=dict)
    official_result: dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorGraph:
    trace_id: str
    task_id: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(
        self,
        node_type: str,
        label: str,
        attrs: dict[str, Any] | None = None,
        *,
        node_id: str | None = None,
        source_event_idx: int | None = None,
    ) -> str:
        node_id = node_id or f"n{len(self.nodes) + 1}"
        self.nodes.append(
            GraphNode(
                id=node_id,
                type=node_type,
                label=label,
                attrs=attrs or {},
                source_event_idx=source_event_idx,
            )
        )
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.edges.append(GraphEdge(source=source, target=target, type=edge_type, attrs=attrs or {}))

    def node(self, node_id: str) -> GraphNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompilationResult:
    graph: BehaviorGraph
    signals: list[StructuralSignal]
    candidates: list[CandidateBehavior]
    cards: list[MonitorCard]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
