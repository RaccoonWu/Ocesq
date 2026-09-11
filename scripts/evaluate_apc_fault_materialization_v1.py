#!/usr/bin/env python3
"""Compare APC materializers on frozen clean/perturbed fault pairs.

The complete normalized OCESQ contract is the reference answer. Graph-only
baselines select nodes without consulting OCESQ paths; OC-RES and complete
packages are evaluated as distinct materialization strategies.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Any

from ocesq.behavior_graph.ocesq import build_ocei_instance, run_ocesq
from ocesq.behavior_graph.ocesq_contract import adapt_ocesq_result, graph_snapshot
from ocesq.behavior_graph.trace_compiler import TraceCompiler, load_jsonl


METHODS = (
    "two_hop_neighborhood",
    "rooted_steiner",
    "why_not_closure",
    "ocres_selected",
    "complete_normalized",
    "full_graph",
)
GRAPH_ONLY = {"two_hop_neighborhood", "rooted_steiner", "why_not_closure"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sources(specs: list[str]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        name, sep, rest = spec.partition("=")
        path_text, sep2, kind = rest.rpartition(":")
        if not sep or not sep2 or kind not in {"v2", "expanded"}:
            raise ValueError(f"invalid source spec: {spec}; use NAME=PATH:v2|expanded")
        for item in read_jsonl(Path(path_text)):
            if kind == "v2":
                baseline, variant = item["clean_trace"], item["perturbed_trace"]
            else:
                baseline, variant = item["baseline_trace"], item["variant_trace"]
            rows.append({
                "dataset": name,
                "case_id": item["case_id"],
                "family": item.get("family") or item.get("intervention", {}).get("type"),
                "baseline_trace": baseline,
                "variant_trace": variant,
                "target_tool_use_id": item["target_action"]["tool_use_id"],
            })
    return rows


def edge_ref(edge: Any) -> str:
    return f"{edge.source}->{edge.type}->{edge.target}"


def compile_query(path: str, target_id: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    graph = TraceCompiler(load_jsonl(Path(path))).compile().graph
    action = next(node for node in graph.nodes if node.type == "ToolAction" and node.attrs.get("tool_use_id") == target_id)
    result = run_ocesq(graph, action.id, budget={"max_nodes": 24, "max_edges": 36}, ocei=build_ocei_instance(graph))
    return graph, action, result, adapt_ocesq_result(graph, result)


def graph_index(graph: Any) -> tuple[dict[str, Any], dict[str, list[tuple[str, str]]]]:
    nodes = {node.id: node for node in graph.nodes}
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in graph.edges:
        ref = edge_ref(edge)
        adjacency[edge.source].append((edge.target, ref))
        adjacency[edge.target].append((edge.source, ref))
    return nodes, adjacency


def two_hop(graph: Any, root: str) -> tuple[set[str], set[str]]:
    _, adjacency = graph_index(graph)
    nodes = {root}
    edges: set[str] = set()
    queue = deque([(root, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth == 2:
            continue
        for neighbor, ref in adjacency.get(current, []):
            edges.add(ref)
            if neighbor not in nodes:
                nodes.add(neighbor)
                queue.append((neighbor, depth + 1))
    return nodes, edges


def rooted_steiner(graph: Any, root: str, action: Any) -> tuple[set[str], set[str]]:
    nodes, adjacency = graph_index(graph)
    distance = {root: 0}
    parent: dict[str, tuple[str, str]] = {}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor, ref in adjacency.get(current, []):
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                parent[neighbor] = (current, ref)
                queue.append(neighbor)
    families = [
        ("task", {"TaskInstruction", "Requirement"}),
        ("input", {"ToolObservation", "FileArtifact"}),
        ("entity", {"ServiceEntity", "EntityVersion"}),
        ("effect", {"ExternalEffect"}),
        ("output", {"OutputAssertion"}),
    ]
    terminals = []
    for _, types in families:
        candidates = [node for node in nodes.values() if node.type in types and node.id in distance]
        if candidates:
            terminals.append(min(candidates, key=lambda item: (distance[item.id], item.source_event_idx is None, item.source_event_idx or 0, item.id)).id)
    selected = {root}
    selected_edges: set[str] = set()
    for terminal in terminals:
        current = terminal
        while current != root and current in parent:
            previous, ref = parent[current]
            selected.update({previous, current})
            selected_edges.add(ref)
            current = previous
    return selected, selected_edges


def why_not_closure(graph: Any, root: str) -> tuple[set[str], set[str]]:
    _, adjacency = graph_index(graph)
    nodes = {root}
    edges: set[str] = set()
    queue = deque([(root, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth == 3:
            continue
        for neighbor, ref in adjacency.get(current, []):
            edges.add(ref)
            if neighbor not in nodes:
                nodes.add(neighbor)
                queue.append((neighbor, depth + 1))
    return nodes, edges


def selected_ocres(result: Any) -> tuple[set[str], set[str], set[str], bool]:
    selected_ids = {
        path_id for obligation in result.obligations
        for path_id in obligation.evidence_contract.get("selected_path_ids", [])
    }
    nodes = {node["id"] for node in result.compact_evidence_subgraph.get("nodes", [])}
    edges = {f"{edge['source']}->{edge['type']}->{edge['target']}" for edge in result.compact_evidence_subgraph.get("edges", [])}
    return nodes, edges, selected_ids, True


def all_normalized(result: Any) -> tuple[set[str], set[str], set[str], bool]:
    ids = {path.id for path in result.candidate_paths if path.kind in {"supporting", "conflicting"}}
    paths = [path for path in result.candidate_paths if path.id in ids]
    nodes = {result.action_id}
    edges = set()
    for path in paths:
        nodes.update(path.node_ids)
        edges.update(path.edge_refs)
    for descriptor in result.missing_path_descriptors:
        nodes.update(descriptor.anchor_node_ids)
    return nodes, edges, ids, True


def full_graph(graph: Any, result: Any) -> tuple[set[str], set[str], set[str], bool]:
    return {node.id for node in graph.nodes}, {edge_ref(edge) for edge in graph.edges}, {path.id for path in result.candidate_paths}, True


def path_signature(path: Any) -> tuple[str, str, str, str, str]:
    return (
        path.obligation_id.rsplit(":", 1)[-1], path.kind, path.summary, path.path_pattern,
        json.dumps(path.attrs, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def retained_paths(result: Any, node_ids: set[str], edge_refs: set[str], selected_ids: set[str] | None) -> list[Any]:
    rows = []
    for path in result.candidate_paths:
        if path.kind not in {"supporting", "conflicting"}:
            continue
        selected = path.id in selected_ids if selected_ids is not None else (
            set(path.node_ids).issubset(node_ids) and set(path.edge_refs).issubset(edge_refs)
        )
        if selected:
            rows.append(path)
    return rows


def certificate_valid(graph: Any, result: Any, contract: dict[str, Any], include_certificate: bool) -> bool:
    missing = [item for item in contract["obligations"] if item["required"] and item["status"] == "missing"]
    if not missing:
        return True
    if not include_certificate:
        return False
    refs = sorted(edge_ref(edge) for edge in graph.edges)
    node_ids = sorted(node.id for node in graph.nodes)
    for item in missing:
        cert = item.get("missing_certificate") or {}
        if not (
            cert.get("complete") is True and cert.get("scope") == "full_behavior_graph"
            and cert.get("graph_snapshot") == graph_snapshot(graph)
            and cert.get("searched_node_ids") == node_ids and cert.get("searched_edge_refs") == refs
        ):
            return False
    return True


def evaluate_materializer(graph: Any, action: Any, result: Any, contract: dict[str, Any], method: str) -> dict[str, Any]:
    if method == "two_hop_neighborhood":
        nodes, edges = two_hop(graph, action.id); selected_ids = None; cert = False
    elif method == "rooted_steiner":
        nodes, edges = rooted_steiner(graph, action.id, action); selected_ids = None; cert = False
    elif method == "why_not_closure":
        nodes, edges = why_not_closure(graph, action.id); selected_ids = None; cert = False
    elif method == "ocres_selected":
        nodes, edges, selected_ids, cert = selected_ocres(result)
    elif method == "complete_normalized":
        nodes, edges, selected_ids, cert = all_normalized(result)
    else:
        nodes, edges, selected_ids, cert = full_graph(graph, result)
    retained = retained_paths(result, nodes, edges, selected_ids)
    retained_ids = {path.id for path in retained}
    full_deciding = [path for path in result.candidate_paths if path.kind in {"supporting", "conflicting"}]
    decision_ok = True
    for obligation in result.obligations:
        if not obligation.required:
            continue
        if obligation.status == "supported":
            decision_ok &= any(path.id in retained_ids and path.kind == "supporting" for path in full_deciding if path.obligation_id == obligation.id)
        elif obligation.status == "conflicting":
            decision_ok &= any(path.id in retained_ids and path.kind == "conflicting" for path in full_deciding if path.obligation_id == obligation.id)
        elif obligation.status == "missing":
            decision_ok &= cert and action.id in nodes
    full_answer = len(retained_ids) == len(full_deciding) and certificate_valid(graph, result, contract, cert)
    package = {
        "nodes": sorted(nodes), "edges": sorted(edges), "retained_path_ids": sorted(retained_ids),
        "certificate": contract if cert else None,
    }
    return {
        "node_ids": nodes, "edge_refs": edges, "retained_paths": retained,
        "decision_preserved": bool(decision_ok), "full_answer_preserved": full_answer,
        "certificate_valid": certificate_valid(graph, result, contract, cert),
        "nodes": len(nodes), "edges": len(edges),
        "bytes": len(json.dumps(package, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")),
    }


def evaluate_pair(item: dict[str, Any]) -> dict[str, Any]:
    bg, ba, br, bc = compile_query(item["baseline_trace"], item["target_tool_use_id"])
    vg, va, vr, vc = compile_query(item["variant_trace"], item["target_tool_use_id"])
    methods = {}
    for method in METHODS:
        left = evaluate_materializer(bg, ba, br, bc, method)
        right = evaluate_materializer(vg, va, vr, vc, method)
        reference_added = {path_signature(p) for p in vr.candidate_paths if p.kind in {"supporting", "conflicting"}} - {path_signature(p) for p in br.candidate_paths if p.kind in {"supporting", "conflicting"}}
        reference_removed = {path_signature(p) for p in br.candidate_paths if p.kind in {"supporting", "conflicting"}} - {path_signature(p) for p in vr.candidate_paths if p.kind in {"supporting", "conflicting"}}
        retained_added = {path_signature(p) for p in right["retained_paths"]} - {path_signature(p) for p in left["retained_paths"]}
        retained_removed = {path_signature(p) for p in left["retained_paths"]} - {path_signature(p) for p in right["retained_paths"]}
        ref_conflict = {sig for sig in reference_added if sig[1] == "conflicting"}
        got_conflict = {sig for sig in retained_added if sig[1] == "conflicting"}
        conflict_recall = len(ref_conflict & got_conflict) / len(ref_conflict) if ref_conflict else 1.0
        methods[method] = {
            "decision_preserved_both": left["decision_preserved"] and right["decision_preserved"],
            "full_answer_preserved_both": left["full_answer_preserved"] and right["full_answer_preserved"],
            "change_preserved": reference_added == retained_added and reference_removed == retained_removed,
            "conflict_delta_recall": conflict_recall,
            "certificate_valid_both": left["certificate_valid"] and right["certificate_valid"],
            "mean_nodes": mean((left["nodes"], right["nodes"])), "mean_edges": mean((left["edges"], right["edges"])),
            "mean_bytes": mean((left["bytes"], right["bytes"])),
        }
    return {"dataset": item["dataset"], "case_id": item["case_id"], "family": item["family"], "methods": methods}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {"all": rows}
    for dataset in sorted({row["dataset"] for row in rows}):
        grouped[dataset] = [row for row in rows if row["dataset"] == dataset]
    output = {"protocol": "apc-fault-materialization-v1", "pairs": len(rows), "methods": {}, "by_dataset": {}}
    for group, selected in grouped.items():
        target = output["methods"] if group == "all" else output["by_dataset"].setdefault(group, {})
        for method in METHODS:
            values = [row["methods"][method] for row in selected]
            target[method] = {
                "pairs": len(values),
                **{key: mean(value[key] for value in values) for key in (
                    "decision_preserved_both", "full_answer_preserved_both", "change_preserved",
                    "conflict_delta_recall", "certificate_valid_both", "mean_nodes", "mean_edges", "mean_bytes",
                )},
            }
    output["reference_boundary"] = "complete normalized OCESQ contract; this evaluates materialization fidelity, not external semantic correctness"
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="NAME=MANIFEST:v2|expanded")
    parser.add_argument("--output-dir", type=Path, default=Path("results/apc_fault_materialization_v1"))
    args = parser.parse_args()
    rows = [evaluate_pair(item) for item in sources(args.source)]
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
