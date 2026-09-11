#!/usr/bin/env python3
"""Build blinded downstream fact-recovery inputs from frozen raw interventions.

Ground-truth labels are derived only from raw-event mutation manifests. OEG and
APC artifacts are presentation conditions and never contribute scoring labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluate_apc_fault_injection_v1 import build_raw_oracle
from evaluate_apc_fault_materialization_v1 import all_normalized, selected_ocres, two_hop
from evaluate_apc_expanded_faults_v1 import raw_dispatch, raw_tool_use
from mandrel.behavior_graph.ocesq import build_ocei_instance, run_ocesq
from mandrel.behavior_graph.ocesq_contract import adapt_ocesq_result
from mandrel.behavior_graph.trace_compiler import TraceCompiler, load_jsonl


CONDITIONS = (
    "raw_trace",
    "action_window",
    "oeg_full",
    "generic_two_hop",
    "complete_audit_package",
    "ocres_diagnostic",
    "apc_without_status",
    "status_only",
    "apc_without_source_pointers",
    "apc_without_conflicts",
    "apc_without_absence_scope",
)

FAMILY_GOLD = {
    "delete_unique_entity_observation": ("provenance", "missing", "parameter_provenance"),
    "replace_action_entity_only": ("entity", "conflicting", "target_entity_support"),
    "move_observation_after_action": ("time", "conflicting", "target_entity_support"),
    "inject_conflicting_response": ("entity", "conflicting", "target_entity_support"),
    "remove_unique_effect_result": ("effect", "missing", "effect_evidence"),
    "inject_stale_entity_version": ("state", "conflicting", "entity_state_consistency"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def normalize_source(
    dataset: str,
    manifest: Path,
    kind: str,
) -> list[dict[str, Any]]:
    rows = []
    for raw in read_jsonl(manifest):
        if kind == "pilot":
            family = raw["intervention"]["type"]
            baseline = raw["clean_trace"]
            variant = raw["perturbed_trace"]
        elif kind == "expanded":
            family = raw["family"]
            baseline = raw["baseline_trace"]
            variant = raw["variant_trace"]
        elif kind == "benign":
            family = raw["family"]
            baseline = raw["baseline_trace"]
            variant = raw["variant_trace"]
        else:
            raise ValueError(f"unsupported manifest kind: {kind}")
        rows.append({
            "dataset": dataset,
            "manifest_kind": kind,
            "manifest_path": str(manifest),
            "raw": raw,
            "case_id": raw["case_id"],
            "family": family,
            "baseline_trace": baseline,
            "variant_trace": variant,
            "target_tool_use_id": raw["target_action"]["tool_use_id"],
        })
    return rows


def expanded_gold(item: dict[str, Any]) -> dict[str, Any]:
    raw = item["raw"]
    family = item["family"]
    target = item["target_tool_use_id"]
    baseline = load_jsonl(Path(item["baseline_trace"]))
    variant = load_jsonl(Path(item["variant_trace"]))
    if family == "remove_unique_effect_result":
        pointers = sorted({raw_tool_use(baseline, target), raw_dispatch(baseline, target)})
        allowed = []
    elif family == "inject_stale_entity_version":
        source_id = raw["source"]["source_tool_use_id"]
        injected_id = raw["transform"]["injected_tool_use_id"]
        source_idx = raw_dispatch(variant, source_id)
        injected_idx = raw_dispatch(variant, injected_id)
        action_idx = raw_tool_use(variant, target)
        if source_idx is None or injected_idx is None:
            raise ValueError(f"{item['case_id']}: stale-state source events missing")
        pointers = sorted({source_idx, injected_idx, action_idx})
        allowed = sorted(set(raw["source"].get("equivalent_source_event_indices", [source_idx])))
    else:
        raise ValueError(f"unsupported expanded family: {family}")
    component, status, relation = FAMILY_GOLD[family]
    return {
        "change_present": True,
        "changed_component": component,
        "expected_relation_status": status,
        "expected_relation": relation,
        "source_event_ids": pointers,
        "allowed_equivalent_source_ids": allowed,
        "scope_bound_absence_required": status == "missing",
        "oracle_boundary": "raw mutation manifest and raw event identity only",
    }


def build_gold(item: dict[str, Any]) -> dict[str, Any]:
    kind = item["manifest_kind"]
    family = item["family"]
    if kind == "benign":
        return {
            "change_present": False,
            "changed_component": "none",
            "expected_relation_status": "unchanged",
            "expected_relation": "none",
            "source_event_ids": [],
            "allowed_equivalent_source_ids": [],
            "scope_bound_absence_required": False,
            "oracle_boundary": "raw benign transformation manifest only",
        }
    if kind == "pilot":
        oracle = build_raw_oracle(item["raw"])
        component, status, relation = FAMILY_GOLD[family]
        return {
            "change_present": True,
            "changed_component": component,
            "expected_relation_status": status,
            "expected_relation": relation,
            "source_event_ids": oracle["expected_localization_pointers"],
            "allowed_equivalent_source_ids": [],
            "scope_bound_absence_required": status == "missing",
            "oracle_boundary": oracle["oracle_boundary"],
        }
    return expanded_gold(item)


def graph_and_contract(path: str, target_id: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    graph = TraceCompiler(load_jsonl(Path(path))).compile().graph
    action = next(
        node for node in graph.nodes
        if node.type == "ToolAction" and node.attrs.get("tool_use_id") == target_id
    )
    result = run_ocesq(
        graph,
        action.id,
        budget={"max_nodes": 24, "max_edges": 36},
        ocei=build_ocei_instance(graph),
    )
    return graph, action, result, adapt_ocesq_result(graph, result)


def node_rows(graph: Any, selected: set[str] | None = None) -> list[dict[str, Any]]:
    return [
        asdict(node) for node in graph.nodes
        if selected is None or node.id in selected
    ]


def edge_ref(edge: Any) -> str:
    return f"{edge.source}->{edge.type}->{edge.target}"


def edge_rows(graph: Any, selected: set[str] | None = None) -> list[dict[str, Any]]:
    return [
        asdict(edge) for edge in graph.edges
        if selected is None or edge_ref(edge) in selected
    ]


def reduce_contract(contract: dict[str, Any], selected_paths: set[str]) -> dict[str, Any]:
    reduced = copy.deepcopy(contract)
    for obligation in reduced["obligations"]:
        obligation["support_facts"] = [
            fact for fact in obligation["support_facts"] if fact["id"] in selected_paths
        ]
        obligation["conflict_facts"] = [
            fact for fact in obligation["conflict_facts"] if fact["id"] in selected_paths
        ]
    return reduced


def without_status(package: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(package)
    for obligation in value["contract"]["obligations"]:
        obligation.pop("status", None)
    return value


def status_only(package: dict[str, Any]) -> dict[str, Any]:
    contract = package["contract"]
    return {
        "root_action_id": package["root_action_id"],
        "obligations": [
            {
                "type": obligation["type"],
                "required": obligation["required"],
                "status": obligation.get("status"),
            }
            for obligation in contract["obligations"]
        ],
    }


def without_source_pointers(package: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(package)
    all_ids = {node["id"] for node in value["nodes"]}
    all_ids.add(value["root_action_id"])
    for obligation in value["contract"]["obligations"]:
        for fact in [*obligation["support_facts"], *obligation["conflict_facts"]]:
            all_ids.update(fact.get("node_ids", []))
        certificate = obligation.get("missing_certificate") or {}
        all_ids.update(certificate.get("searched_node_ids", []))
        all_ids.update((certificate.get("descriptor") or {}).get("anchor_node_ids", []))
    node_map = {node_id: f"v{index + 1}" for index, node_id in enumerate(sorted(all_ids))}

    def map_ref(ref: str) -> str:
        parts = ref.split("->", 2)
        if len(parts) != 3:
            return ref
        return f"{node_map.get(parts[0], parts[0])}->{parts[1]}->{node_map.get(parts[2], parts[2])}"

    for node in value["nodes"]:
        node["id"] = node_map[node["id"]]
        node.pop("source_event_idx", None)
    edge_map = {}
    for edge in value["edges"]:
        old_ref = f"{edge['source']}->{edge['type']}->{edge['target']}"
        edge["source"] = node_map.get(edge["source"], edge["source"])
        edge["target"] = node_map.get(edge["target"], edge["target"])
        edge_map[old_ref] = f"{edge['source']}->{edge['type']}->{edge['target']}"
    value["root_action_id"] = node_map.get(value["root_action_id"], value["root_action_id"])
    contract = value["contract"]
    contract["root_action_id"] = node_map.get(contract["root_action_id"], contract["root_action_id"])
    for obligation in contract["obligations"]:
        obligation["root_action_id"] = node_map.get(
            obligation["root_action_id"], obligation["root_action_id"]
        )
        for fact in [*obligation["support_facts"], *obligation["conflict_facts"]]:
            fact.pop("source_event_pointers", None)
            fact["node_ids"] = [node_map.get(item, item) for item in fact.get("node_ids", [])]
            fact["edge_refs"] = [edge_map.get(item, map_ref(item)) for item in fact.get("edge_refs", [])]
        certificate = obligation.get("missing_certificate")
        if certificate:
            certificate["searched_node_ids"] = [
                node_map.get(item, item) for item in certificate.get("searched_node_ids", [])
            ]
            certificate["searched_edge_refs"] = [
                edge_map.get(item, map_ref(item)) for item in certificate.get("searched_edge_refs", [])
            ]
            descriptor = certificate.get("descriptor") or {}
            descriptor["anchor_node_ids"] = [
                node_map.get(item, item) for item in descriptor.get("anchor_node_ids", [])
            ]
    return value


def without_conflicts(package: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(package)
    for obligation in value["contract"]["obligations"]:
        obligation["conflict_facts"] = []
    return value


def without_absence_scope(package: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(package)
    for obligation in value["contract"]["obligations"]:
        obligation["missing_certificate"] = None
    return value


def action_index(events: list[dict[str, Any]], target_id: str) -> int:
    return raw_tool_use(events, target_id)


def action_window(events: list[dict[str, Any]], target_id: str, radius: int) -> dict[str, Any]:
    center = action_index(events, target_id)
    start = max(0, center - radius)
    stop = min(len(events), center + radius + 1)
    return {"start_event_id": start, "end_event_id": stop - 1, "events": events[start:stop]}


def presentation(path: str, target_id: str, radius: int) -> dict[str, Any]:
    events = load_jsonl(Path(path))
    graph, action, result, contract = graph_and_contract(path, target_id)
    two_nodes, two_edges = two_hop(graph, action.id)
    complete_nodes, complete_edges, complete_paths, _ = all_normalized(result)
    ocres_nodes, ocres_edges, ocres_paths, _ = selected_ocres(result)
    complete_package = {
        "root_action_id": action.id,
        "contract": contract,
        "nodes": node_rows(graph, complete_nodes),
        "edges": edge_rows(graph, complete_edges),
        "retained_fact_ids": sorted(complete_paths),
    }
    return {
        "raw_trace": {"events": events},
        "action_window": action_window(events, target_id, radius),
        "oeg_full": {"root_action_id": action.id, "nodes": node_rows(graph), "edges": edge_rows(graph)},
        "generic_two_hop": {
            "root_action_id": action.id,
            "nodes": node_rows(graph, two_nodes),
            "edges": edge_rows(graph, two_edges),
        },
        "complete_audit_package": complete_package,
        "ocres_diagnostic": {
            "root_action_id": action.id,
            "contract": reduce_contract(contract, ocres_paths),
            "nodes": node_rows(graph, ocres_nodes),
            "edges": edge_rows(graph, ocres_edges),
            "retained_fact_ids": sorted(ocres_paths),
        },
        "apc_without_status": without_status(complete_package),
        "status_only": status_only(complete_package),
        "apc_without_source_pointers": without_source_pointers(complete_package),
        "apc_without_conflicts": without_conflicts(complete_package),
        "apc_without_absence_scope": without_absence_scope(complete_package),
    }


def auditor_prompt(case_token: str, left: Any, right: Any) -> dict[str, Any]:
    return {
        "case_token": case_token,
        "task": (
            "Compare version A with version B for the target action. Report only recorded-fact changes; "
            "do not judge whether the action is safe, correct, or compliant."
        ),
        "version_a": left,
        "version_b": right,
        "output_schema": {
            "change_present": "boolean",
            "relation_status": "supported|conflicting|missing|not_applicable|unchanged|unknown",
            "changed_component": "provenance|entity|time|effect|state|none",
            "source_event_ids": "integer array",
            "absence_is_scope_bound": "boolean",
            "confidence": "number from 0 to 1",
            "rationale": "empty string; not evaluated",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="NAME=MANIFEST:pilot|expanded|benign")
    parser.add_argument("--window-radius", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-dir", type=Path, default=Path("results/current_downstream_fact_recovery_inputs"))
    args = parser.parse_args()
    if args.window_radius < 0:
        raise ValueError("window radius must be non-negative")

    items: list[dict[str, Any]] = []
    for source in args.source:
        name, sep, rest = source.partition("=")
        path_text, sep2, kind = rest.rpartition(":")
        if not sep or not sep2 or kind not in {"pilot", "expanded", "benign"}:
            raise ValueError(f"invalid source: {source}")
        items.extend(normalize_source(name, Path(path_text), kind))

    rng = random.Random(args.seed)
    aliases = [f"R{index + 1}" for index in range(len(CONDITIONS))]
    rng.shuffle(aliases)
    condition_alias = dict(zip(CONDITIONS, aliases))
    condition_rows: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    gold_rows = []

    for item in items:
        token = stable_id(item["dataset"], item["case_id"], str(args.seed))
        left = presentation(item["baseline_trace"], item["target_tool_use_id"], args.window_radius)
        right = presentation(item["variant_trace"], item["target_tool_use_id"], args.window_radius)
        gold_rows.append({
            "case_token": token,
            "dataset": item["dataset"],
            "source_case_id": item["case_id"],
            "family": item["family"],
            "valid_source_event_ids": list(range(max(
                len(load_jsonl(Path(item["baseline_trace"]))),
                len(load_jsonl(Path(item["variant_trace"]))),
            ))),
            **build_gold(item),
        })
        for condition in CONDITIONS:
            condition_rows[condition].append({
                "representation_id": condition_alias[condition],
                **auditor_prompt(token, left[condition], right[condition]),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for condition, rows in condition_rows.items():
        write_jsonl(args.output_dir / f"{condition_alias[condition]}.jsonl", rows)
    write_jsonl(args.output_dir / "ground_truth.jsonl", gold_rows)
    (args.output_dir / "private_condition_map.json").write_text(
        json.dumps({"seed": args.seed, "condition_alias": condition_alias}, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "protocol": "downstream-fact-recovery-inputs-v1",
        "cases": len(items),
        "fault_cases": sum(row["change_present"] for row in gold_rows),
        "benign_cases": sum(not row["change_present"] for row in gold_rows),
        "by_family": dict(sorted(Counter(row["family"] for row in gold_rows).items())),
        "representations": len(CONDITIONS),
        "window_radius": args.window_radius,
        "seed": args.seed,
        "ground_truth_boundary": "raw mutation and benign-rewrite manifests only",
        "model_outputs_are_ground_truth": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
