"""Executable evaluator for the deterministic v1 query contract.

This remains separate from OCESQ, but oracle workload generators historically
used it to write answer files. Independent differential verification therefore
uses :mod:`query_reference`, which shares no traversal or classification code.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import hashlib
import json
from typing import Any


def evaluate(graph: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a typed path query over a JSON graph fixture."""

    index = build_query_index(graph)
    return evaluate_indexed(graph, query, index)


def build_query_index(graph: dict[str, Any]) -> dict[str, Any]:
    """Build typed outgoing/incoming adjacency for repeated reference queries."""

    nodes = {str(node["id"]): node for node in graph.get("nodes", [])}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph.get("edges", []):
        outgoing[str(edge["source"])].append(edge)
        incoming[str(edge["target"])].append(edge)
    return {"nodes": nodes, "outgoing": dict(outgoing), "incoming": dict(incoming)}


def evaluate_indexed(graph: dict[str, Any], query: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    """Evaluate using a prebuilt index without rebuilding adjacency."""

    if query.get("applicable") is False:
        return _answer("not_applicable", [], [], None)

    nodes = index["nodes"]
    outgoing = index["outgoing"]
    incoming = index["incoming"]
    root = str(query["root_selector"]["action_id"])
    pattern = list(query.get("path_pattern", []))
    if not pattern or root not in nodes or nodes[root].get("type") != pattern[0]:
        return _answer("missing", [], [], _certificate(graph, query, [], graph.get("edges", [])))

    paths = _typed_paths(nodes, outgoing, root, pattern, query.get("edge_constraints", []), incoming=incoming)
    valid, conflicting = [], []
    for path in paths:
        violations = _constraint_violations(path, nodes, query)
        (conflicting if violations else valid).append({"node_ids": path, "violations": violations})
    if conflicting:
        return _answer("conflicting", valid, conflicting, None)
    if valid:
        return _answer("supported", valid, conflicting, None)
    return _answer("missing", [], [], _certificate(graph, query, list(nodes), graph.get("edges", [])))


def compare_answers(full: dict[str, Any], materialized: dict[str, Any]) -> bool:
    """Check the contract equality used by answer-preserving materialization."""

    if full.get("status") != materialized.get("status"):
        return False
    if full.get("status") == "missing":
        full_certificate = full.get("certificate", {})
        materialized_certificate = materialized.get("certificate", {})
        return (
            full_certificate.get("complete") is True
            and materialized_certificate.get("complete") is True
            and full_certificate.get("scope") == materialized_certificate.get("scope")
            and full_certificate.get("searched_node_ids") == materialized_certificate.get("searched_node_ids")
        )
    key = lambda item: (tuple(item.get("node_ids", [])), tuple(sorted(item.get("violations", []))))
    return (
        sorted(map(key, full.get("witnesses", []))) == sorted(map(key, materialized.get("witnesses", [])))
        and sorted(map(key, full.get("conflicts", []))) == sorted(map(key, materialized.get("conflicts", [])))
    )


def witness_union_materialize(graph: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    """Retain the root and all nodes/edges used by the full answer paths."""

    answer = evaluate(graph, query)
    paths = [*answer.get("witnesses", []), *answer.get("conflicts", [])]
    keep = {str(query["root_selector"]["action_id"])}
    for path in paths:
        keep.update(map(str, path.get("node_ids", [])))
    return _induced_subgraph(graph, keep)


def exact_minimum_answer_preserving_subgraph(graph: dict[str, Any], query: dict[str, Any]) -> dict[str, Any] | None:
    """Find a minimum-node induced subgraph by exhaustive enumeration."""

    full_answer = evaluate(graph, query)
    root = str(query["root_selector"]["action_id"])
    node_ids = [str(node["id"]) for node in graph.get("nodes", [])]
    optional = [node_id for node_id in node_ids if node_id != root]
    for size in range(len(optional) + 1):
        for selected in combinations(optional, size):
            subgraph = _induced_subgraph(graph, {root, *selected})
            if compare_answers(full_answer, evaluate(subgraph, query)):
                return subgraph
    return None


def certificate_aware_materialize(graph: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    """Materialize a compact graph together with a full-scope answer object."""

    answer = evaluate(graph, query)
    root = str(query["root_selector"]["action_id"])
    if answer["status"] in {"missing", "not_applicable"}:
        subgraph = _induced_subgraph(graph, {root})
    else:
        subgraph = witness_union_materialize(graph, query)
    certificate = answer.get("certificate")
    return {
        "graph": subgraph,
        "answer": answer,
        "certificate_digest": _digest(certificate) if certificate is not None else None,
        "certificate_source_snapshot": graph.get("graph_snapshot"),
    }


def validate_materialized_result(full_graph: dict[str, Any], query: dict[str, Any], result: dict[str, Any]) -> bool:
    """Validate a materialized graph/answer pair against a fresh full query."""

    expected = evaluate(full_graph, query)
    supplied = result.get("answer", {})
    if not compare_answers(expected, supplied):
        return False
    if expected["status"] == "missing":
        certificate = supplied.get("certificate")
        return (
            result.get("certificate_source_snapshot") == full_graph.get("graph_snapshot")
            and result.get("certificate_digest") == _digest(certificate)
            and certificate == expected.get("certificate")
        )
    return compare_answers(expected, evaluate(result.get("graph", {}), query))


def _induced_subgraph(graph: dict[str, Any], keep: set[str]) -> dict[str, Any]:
    return {
        **graph,
        "nodes": [node for node in graph.get("nodes", []) if str(node["id"]) in keep],
        "edges": [edge for edge in graph.get("edges", []) if str(edge["source"]) in keep and str(edge["target"]) in keep],
    }


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _typed_paths(nodes: dict[str, Any], outgoing: dict[str, list[dict[str, Any]]], root: str, pattern: list[str], edge_constraints: list[Any], *, incoming: dict[str, list[dict[str, Any]]] | None = None) -> list[list[str]]:
    if incoming is None:
        incoming = defaultdict(list)
        for edges in outgoing.values():
            for edge in edges:
                incoming[str(edge["target"])].append(edge)
    paths = [[root]]
    for position, expected_type in enumerate(pattern[1:], start=1):
        next_paths: list[list[str]] = []
        constraint = edge_constraints[position - 1] if position - 1 < len(edge_constraints) else None
        edge_type = constraint.get("type") if isinstance(constraint, dict) else constraint
        direction = constraint.get("direction", "out") if isinstance(constraint, dict) else "out"
        for path in paths:
            candidate_edges = incoming.get(path[-1], []) if direction == "in" else outgoing.get(path[-1], [])
            for edge in candidate_edges:
                if edge_type is not None and edge.get("type") != edge_type:
                    continue
                target = str(edge["source"] if direction == "in" else edge["target"])
                if target in path or nodes.get(target, {}).get("type") != expected_type:
                    continue
                next_paths.append([*path, target])
        paths = next_paths
    return paths


def _constraint_violations(path: list[str], nodes: dict[str, Any], query: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for constraint in [*query.get("state_constraints", []), *query.get("entity_constraints", [])]:
        node_id = constraint.get("node_id")
        if node_id not in path:
            expected_type = constraint.get("node_type")
            node_id = next((candidate for candidate in reversed(path) if nodes.get(candidate, {}).get("type") == expected_type), path[-1])
        node_id = str(node_id)
        node = nodes.get(node_id)
        if node is None or node.get("attrs", {}).get(constraint.get("field")) != constraint.get("equals"):
            violations.append(str(constraint.get("id", "state_constraint")))
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
            violations.append(str(constraint.get("id", "temporal_constraint")))
    return violations


def _certificate(graph: dict[str, Any], query: dict[str, Any], searched_nodes: list[str], searched_edges: list[Any]) -> dict[str, Any]:
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
        "searched_edge_ids": [f"{edge.get('source')}->{edge.get('target')}" for edge in searched_edges if isinstance(edge, dict)],
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


def _answer(status: str, witnesses: list[dict[str, Any]], conflicts: list[dict[str, Any]], certificate: dict[str, Any] | None) -> dict[str, Any]:
    return {"contract_version": "1.0", "status": status, "witnesses": witnesses, "conflicts": conflicts, "certificate": certificate}
