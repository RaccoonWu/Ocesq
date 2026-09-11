"""Obligation-Constrained Evidence Index (OCEI).

Accelerates repeated OCESQ queries over the same BehaviorGraph by replacing
linear scans with pre-built hash indices.  Built once per graph; supports
efficient prior-node lookups, anchor-to-observation matching, and typed edge
traversal.
"""

from __future__ import annotations

import bisect
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
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
    ANCHOR_ALIASES as _ANCHOR_ALIASES,
    ANCHOR_KEYWORDS,
    FREE_TEXT_KEYS,
)


# Reused stateless helpers from ocesq.py (kept local to avoid circular imports)
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


@dataclass
class OCEI:
    """Index built once per BehaviorGraph for efficient repeated OCESQ queries."""

    graph: BehaviorGraph
    trace_id: str = ""
    task_id: str = ""

    # Node indices
    node_map: dict[str, GraphNode] = field(default_factory=dict)
    nodes_by_type: dict[str, list[GraphNode]] = field(default_factory=lambda: defaultdict(list))
    # Sorted source_event_idx per type for binary search
    _type_event_idx: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    # Edge indices
    adjacency: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))
    outgoing: dict[str, list[tuple[str, str, str]]] = field(default_factory=lambda: defaultdict(list))
    incoming: dict[str, list[tuple[str, str, str]]] = field(default_factory=lambda: defaultdict(list))

    # Anchor index: (anchor_key_tail, normalized_value) -> [obs_node_ids]
    obs_anchor_index: dict[tuple[str, str], list[str]] = field(default_factory=lambda: defaultdict(list))

    # Pre-computed per-action metadata
    action_meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Build stats
    build_time_ms: float = 0.0
    memory_bytes: int = 0

    @classmethod
    def build(cls, graph: BehaviorGraph) -> OCEI:
        start = time.perf_counter()
        ocei = cls(graph=graph, trace_id=graph.trace_id, task_id=graph.task_id)
        ocei._build()
        ocei.build_time_ms = round((time.perf_counter() - start) * 1000, 4)
        ocei.memory_bytes = len(json.dumps(ocei._serializable(), ensure_ascii=False).encode("utf-8"))
        return ocei

    def _build(self) -> None:
        # --- Node indices ---
        for node in self.graph.nodes:
            self.node_map[node.id] = node
            self.nodes_by_type[node.type].append(node)

        for ntype, nodes in self.nodes_by_type.items():
            nodes.sort(key=lambda n: (n.source_event_idx or 0, n.id))
            self._type_event_idx[ntype] = [n.source_event_idx or 0 for n in nodes]

        # --- Edge indices ---
        for edge in self.graph.edges:
            ekey = self._edge_key(edge)
            self.adjacency[edge.source].append((edge.target, ekey))
            self.adjacency[edge.target].append((edge.source, ekey))
            self.outgoing[edge.source].append((edge.target, edge.type, ekey))
            self.incoming[edge.target].append((edge.source, edge.type, ekey))

        # --- Observation anchor value index ---
        for node in self.graph.nodes:
            if node.type != "ToolObservation":
                continue
            for source in ("response_body", "request_body"):
                payload = node.attrs.get(source)
                if payload is None:
                    continue
                for path, value in _walk_scalar_items(payload):
                    tail = path.rsplit(".", 1)[-1].lower()
                    if tail in FREE_TEXT_KEYS:
                        continue
                    vtext = _normalize_value(value)
                    if len(vtext) < 2:
                        continue
                    self.obs_anchor_index[(tail, vtext)].append(node.id)

        # --- Action metadata cache ---
        for node in self.graph.nodes:
            if node.type != "ToolAction":
                continue
            self.action_meta[node.id] = {
                "mechanism": node.attrs.get("mechanism", ""),
                "tool_name": node.attrs.get("tool_name", ""),
                "channel": node.attrs.get("channel"),
                "risk_level": node.attrs.get("risk_level"),
                "effectful": node.attrs.get("effectful"),
                "approval_sensitive": node.attrs.get("approval_sensitive"),
                "external_visibility": node.attrs.get("external_visibility"),
                "source_event_idx": node.source_event_idx,
                "anchors": _extract_action_anchors(node.attrs.get("input", {})),
            }

    # ---- Public query API ----

    def prior_nodes(self, action_id: str, node_types: set[str]) -> list[GraphNode]:
        """Indexed version of _prior_nodes — O(log n) per type via binary search."""
        action = self.node_map.get(action_id)
        if action is None:
            return []
        max_idx = action.source_event_idx or 0
        result: list[GraphNode] = []
        for ntype in node_types:
            nodes = self.nodes_by_type.get(ntype, [])
            idxs = self._type_event_idx.get(ntype, [])
            if not nodes:
                continue
            pos = bisect.bisect_left(idxs, max_idx)
            result.extend(nodes[:pos])
        return result

    def aligned_prior_observations(self, action_id: str, anchors: list[tuple[str, Any]]) -> list[GraphNode]:
        """Index-accelerated aligned observation matching."""
        action = self.node_map.get(action_id)
        if action is None or not anchors:
            return []
        max_idx = action.source_event_idx or 0
        aligned: list[GraphNode] = []
        seen: set[str] = set()
        for key, value in anchors:
            tail = key.rsplit(".", 1)[-1].lower()
            if tail in FREE_TEXT_KEYS:
                continue
            vtext = _normalize_value(value)
            if not vtext:
                continue
            # Direct lookup + alias expansion
            candidate_ids = self._lookup_anchor(tail, vtext)
            for obs_id in candidate_ids:
                if obs_id in seen:
                    continue
                obs = self.node_map.get(obs_id)
                if obs is None:
                    continue
                if (obs.source_event_idx or 0) >= max_idx:
                    continue
                aligned.append(obs)
                seen.add(obs_id)
        return aligned

    def has_edge(self, source: str, target: str, edge_types: set[str]) -> bool:
        """Check if a typed edge exists between two nodes."""
        for _, etype, _ in self.outgoing.get(source, []):
            if etype in edge_types:
                # Need to check target
                return True
        # More precise: check outgoing with target match
        return any(
            t == target and e in edge_types
            for t, e, _ in self.outgoing.get(source, [])
        )

    def mentioned_entities(self, action_id: str) -> list[tuple[str, GraphNode]]:
        """Return entities the action explicitly mentions via ``mentions`` edges.

        Returns list of (entity_id, entity_node) tuples.
        """
        result: list[tuple[str, GraphNode]] = []
        for target_id, etype, _ in self.outgoing.get(action_id, []):
            if etype == EDGE_MENTIONS:
                entity = self.node_map.get(target_id)
                if entity and entity.type == "ServiceEntity":
                    result.append((target_id, entity))
        return result

    def resolving_observations(self, entity_id: str) -> list[tuple[str, GraphNode]]:
        """Return observations that uniquely resolve this entity via ``resolves`` edges.

        Returns list of (obs_id, obs_node) tuples.
        """
        result: list[tuple[str, GraphNode]] = []
        for src_id, etype, _ in self.incoming.get(entity_id, []):
            if etype == EDGE_RESOLVES:
                obs = self.node_map.get(src_id)
                if obs and obs.type == "ToolObservation":
                    result.append((src_id, obs))
        return result

    def same_as_entities(self, entity_id: str) -> list[str]:
        """Return entity ids linked to the given entity via ``same_as`` edges (BFS)."""
        seen: set[str] = {entity_id}
        queue = [entity_id]
        while queue:
            cur = queue.pop(0)
            for neighbor, etype, _ in [*self.outgoing.get(cur, []), *self.incoming.get(cur, [])]:
                if etype == EDGE_SAME_AS and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        seen.discard(entity_id)
        return list(seen)

    def outgoing_nodes(self, source_id: str, node_types: set[str]) -> list[GraphNode]:
        """Find target nodes of specific types reachable from source."""
        result: list[GraphNode] = []
        seen: set[str] = set()
        for target_id, _, _ in self.outgoing.get(source_id, []):
            if target_id in seen:
                continue
            seen.add(target_id)
            target = self.node_map.get(target_id)
            if target and target.type in node_types:
                result.append(target)
        return result

    def edge_refs_between(self, source: str, target: str) -> list[str]:
        """Typed edge references between two nodes."""
        return [
            f"{s}->{e}->{t}"
            for s, e, (t, _, _) in [
                (source, etype, (target, "", ""))
                for _, (t2, etype, _) in [
                    (None, item) for item in self.outgoing.get(source, [])
                ]
            ]
            if t == target
        ]

    def get_action_meta(self, action_id: str) -> dict[str, Any]:
        """Cached action metadata."""
        return self.action_meta.get(action_id, {})

    def get_action_anchors(self, action_id: str) -> list[tuple[str, Any]]:
        """Cached action anchors."""
        return self.action_meta.get(action_id, {}).get("anchors", [])

    def graph_size(self) -> tuple[int, int]:
        return len(self.graph.nodes), len(self.graph.edges)

    # ---- Internal helpers ----

    @staticmethod
    def _edge_key(edge: GraphEdge) -> str:
        return f"{edge.source}->{edge.type}->{edge.target}"

    def _lookup_anchor(self, tail: str, vtext: str) -> list[str]:
        """Look up observations matching an anchor (tail, value) with alias expansion."""
        # Direct match
        ids = list(self.obs_anchor_index.get((tail, vtext), []))
        # Alias expansion
        for alias in _ANCHOR_ALIASES.get(tail, ()):
            ids.extend(self.obs_anchor_index.get((alias, vtext), []))
        return ids

    def _serializable(self) -> dict[str, Any]:
        """Lightweight serializable summary (for memory estimation)."""
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "node_count": len(self.node_map),
            "edge_count": len(self.graph.edges),
            "nodes_by_type": {k: len(v) for k, v in self.nodes_by_type.items()},
            "obs_anchor_index_entries": len(self.obs_anchor_index),
            "action_meta_count": len(self.action_meta),
        }

    def to_dict(self) -> dict[str, Any]:
        """Full index serialization for storage."""
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "node_type_cache": {k: [n.id for n in v] for k, v in sorted(self.nodes_by_type.items())},
            "edge_type_cache": {
                etype: [self._edge_key(e) for e in edges]
                for etype, edges in self._group_edges_by_type().items()
            },
            "action_meta": self.action_meta,
            "build_time_ms": self.build_time_ms,
            "memory_bytes": self.memory_bytes,
        }

    def _group_edges_by_type(self) -> dict[str, list[GraphEdge]]:
        groups: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in self.graph.edges:
            groups[edge.type].append(edge)
        return groups


def _extract_action_anchors(body: Any) -> list[tuple[str, Any]]:
    """Extract anchor (key, value) pairs from action input."""
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


def index_stats(ocei: OCEI) -> dict[str, Any]:
    """Compute index statistics for reporting."""
    return {
        "trace_id": ocei.trace_id,
        "task_id": ocei.task_id,
        "oeg_nodes": len(ocei.node_map),
        "oeg_edges": len(ocei.graph.edges),
        "indexed_actions": len(ocei.action_meta),
        "obs_anchor_index_entries": len(ocei.obs_anchor_index),
        "node_type_buckets": len(ocei.nodes_by_type),
        "edge_type_buckets": len(
            set(
                edge.type
                for edge in ocei.graph.edges
            )
        ),
        "build_time_ms": ocei.build_time_ms,
        "memory_bytes": ocei.memory_bytes,
    }
