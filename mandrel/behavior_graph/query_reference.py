"""Independent exhaustive reference semantics for deterministic graph queries.

This module intentionally does not import the production query evaluator,
candidate retrieval, index, or materializer. It scans the edge list directly
so differential tests do not compare two entry points backed by one traversal.
"""

from __future__ import annotations

from typing import Any


def evaluate_reference(graph: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    if query.get("applicable") is False:
        return _answer("not_applicable", [], [], None)

    nodes = {str(node["id"]): node for node in graph.get("nodes", [])}
    edges = list(graph.get("edges", []))
    root = str(query.get("root_selector", {}).get("action_id"))
    pattern = list(query.get("path_pattern", []))
    if not pattern or root not in nodes or nodes[root].get("type") != pattern[0]:
        return _answer("missing", [], [], _certificate(graph, query, [], edges))

    paths = _enumerate_paths(nodes, edges, root, pattern, list(query.get("edge_constraints", [])))
    witnesses: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for path in paths:
        violations = _violations(path, nodes, query)
        row = {"node_ids": path, "violations": violations}
        (conflicts if violations else witnesses).append(row)

    if conflicts:
        return _answer("conflicting", witnesses, conflicts, None)
    if witnesses:
        return _answer("supported", witnesses, conflicts, None)
    return _answer("missing", [], [], _certificate(graph, query, list(nodes), edges))


def _enumerate_paths(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    root: str,
    pattern: list[str],
    edge_constraints: list[Any],
) -> list[list[str]]:
    complete: list[list[str]] = []

    def visit(path: list[str], position: int) -> None:
        if position == len(pattern):
            complete.append(path)
            return
        raw_constraint = edge_constraints[position - 1] if position - 1 < len(edge_constraints) else None
        edge_type = raw_constraint.get("type") if isinstance(raw_constraint, dict) else raw_constraint
        direction = raw_constraint.get("direction", "out") if isinstance(raw_constraint, dict) else "out"
        current = path[-1]
        for edge in edges:
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            if direction == "in":
                if target != current:
                    continue
                candidate = source
            else:
                if source != current:
                    continue
                candidate = target
            if edge_type is not None and edge.get("type") != edge_type:
                continue
            if candidate in path:
                continue
            node = nodes.get(candidate)
            if node is None or node.get("type") != pattern[position]:
                continue
            visit([*path, candidate], position + 1)

    visit([root], 1)
    return complete


def _violations(path: list[str], nodes: dict[str, dict[str, Any]], query: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for constraint in [*query.get("state_constraints", []), *query.get("entity_constraints", [])]:
        node_id = constraint.get("node_id")
        if node_id not in path:
            expected_type = constraint.get("node_type")
            node_id = next(
                (candidate for candidate in reversed(path) if nodes.get(candidate, {}).get("type") == expected_type),
                path[-1],
            )
        node = nodes.get(str(node_id))
        actual = (node or {}).get("attrs", {}).get(constraint.get("field"))
        if node is None or actual != constraint.get("equals"):
            failures.append(str(constraint.get("id", "state_constraint")))

    for constraint in query.get("temporal_constraints", []):
        left = nodes.get(str(constraint.get("left_node_id")))
        right = nodes.get(str(constraint.get("right_node_id")))
        left_index = (left or {}).get("attrs", {}).get("event_index")
        right_index = (right or {}).get("attrs", {}).get("event_index")
        order = constraint.get("order")
        valid = left_index is not None and right_index is not None
        if order == "before":
            valid = valid and left_index < right_index
        elif order == "after":
            valid = valid and left_index > right_index
        else:
            valid = False
        if not valid:
            failures.append(str(constraint.get("id", "temporal_constraint")))
    return failures


def _certificate(
    graph: dict[str, Any],
    query: dict[str, Any],
    searched_nodes: list[str],
    searched_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "scope": {
            "trajectory_id": graph.get("trajectory_id"),
            "graph_snapshot": graph.get("graph_snapshot"),
            "schema_version": graph.get("schema_version"),
            "root_action_id": query.get("root_selector", {}).get("action_id"),
            "path_depth_limit": len(query.get("path_pattern", [])),
        },
        "candidate_count": 0,
        "searched_node_ids": searched_nodes,
        "searched_edge_ids": [f"{edge.get('source')}->{edge.get('target')}" for edge in searched_edges],
        "failed_constraints": [
            str(item.get("id", "constraint"))
            for item in [
                *query.get("state_constraints", []),
                *query.get("entity_constraints", []),
                *query.get("temporal_constraints", []),
            ]
        ],
        "complete": True,
    }


def _answer(
    status: str,
    witnesses: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    certificate: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "contract_version": "1.0",
        "status": status,
        "witnesses": witnesses,
        "conflicts": conflicts,
        "certificate": certificate,
    }
