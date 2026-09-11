#!/usr/bin/env python3
"""Run the deterministic production-OCESQ to OC-RES integration gate."""

from __future__ import annotations

import argparse
import ast
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean

from mandrel.behavior_graph.ocesq import run_ocesq
from mandrel.behavior_graph.ocesq_contract import adapt_ocesq_result, verify_ocesq_contract
from mandrel.behavior_graph.schema import BehaviorGraph


ALL_TYPES = (
    "parameter_provenance",
    "target_entity_support",
    "input_observation",
    "task_constraint",
    "approval_dependency",
    "channel_boundary",
    "state_freshness",
    "verification_evidence",
    "effect_evidence",
    "output_grounding",
    "entity_state_consistency",
)


def _gold(statuses: dict[str, str], required: set[str]) -> dict[str, dict[str, object]]:
    return {
        typ: {"required": typ in required, "status": statuses.get(typ, "not_applicable")}
        for typ in ALL_TYPES
    }


def _supported_graph(case_id: str, *, conflicts: bool) -> tuple[BehaviorGraph, dict[str, dict[str, object]]]:
    graph = BehaviorGraph(trace_id=case_id, task_id=f"task_{case_id}")
    graph.add_node(
        "TaskInstruction",
        "Submit status",
        {"text": "Submit the current status for alice@example.com to the external API."},
        node_id="task_instruction",
        source_event_idx=0,
    )
    graph.add_node(
        "ToolObservation",
        "contact lookup",
        {"tool_name": "contacts_get", "response_body": {"email": "alice@example.com", "status": "active"}},
        node_id="obs_before",
        source_event_idx=1,
    )
    graph.add_node(
        "ServiceEntity",
        "email:alice@example.com",
        {"key": "to", "value": "alice@example.com", "entity_type": "email", "candidate_count": 1},
        node_id="entity_alice",
        source_event_idx=1,
    )
    graph.add_node(
        "ApprovalEvent",
        "approved external submission",
        {"approved": True},
        node_id="approval_event",
        source_event_idx=2,
    )
    graph.add_node(
        "ToolAction",
        "external_api_call",
        {
            "tool_name": "external_api_call",
            "tool_use_id": "send_1",
            "mechanism": "api_call",
            "input": {"to": "alice@example.com", "subject": "Status", "body": "Current status is active."},
            "effectful": True,
            "approval_sensitive": True,
            "external_visibility": True,
            "risk_level": "high",
            "channel": "email",
        },
        node_id="action_send",
        source_event_idx=3,
    )
    graph.add_node(
        "ExternalEffect",
        "message sent",
        {"tool_use_id": "send_1", "mechanism": "api_call"},
        node_id="effect_send",
        source_event_idx=4,
    )
    graph.add_node(
        "ToolObservation",
        "send result",
        {"tool_name": "external_api_call", "tool_use_id": "send_1", "request_body": {"to": "alice@example.com"}, "response_body": {"status": "submitted"}},
        node_id="obs_after",
        source_event_idx=4,
    )
    graph.add_node(
        "OutputAssertion",
        "message sent",
        {"text": "The status was sent to alice@example.com."},
        node_id="assertion_sent",
        source_event_idx=5,
    )
    graph.add_edge("action_send", "task_instruction", "depends_on")
    graph.add_edge("obs_before", "entity_alice", "resolves")
    graph.add_edge("action_send", "entity_alice", "mentions")
    graph.add_edge("action_send", "effect_send", "triggers")
    graph.add_edge("action_send", "obs_after", "observes")
    graph.add_edge("assertion_sent", "action_send", "grounds")

    statuses = {typ: "supported" for typ in ALL_TYPES}
    if conflicts:
        graph.add_node(
            "StructuralSignals",
            "ambiguous_entity_binding",
            {"summary": "Two candidate identities remain."},
            node_id="signal_ambiguous",
            source_event_idx=2,
        )
        graph.add_node(
            "ServiceEntity",
            "email:alice@example.com stale",
            {"key": "to", "value": "alice@example.com", "entity_type": "email", "candidate_count": 1},
            node_id="entity_alice_stale",
            source_event_idx=2,
        )
        graph.add_edge("signal_ambiguous", "action_send", "flags")
        graph.add_edge("entity_alice", "entity_alice_stale", "same_as")
        graph.add_edge(
            "entity_alice",
            "entity_alice_stale",
            "contradicts",
            {"field": "status", "value_in_source": "active", "value_in_target": "disabled", "entity_value": "alice@example.com"},
        )
        statuses["target_entity_support"] = "conflicting"
        statuses["entity_state_consistency"] = "conflicting"
    return graph, _gold(statuses, set(ALL_TYPES))


def _missing_graph() -> tuple[BehaviorGraph, dict[str, dict[str, object]]]:
    graph = BehaviorGraph(trace_id="missing_boundaries", task_id="task_missing_boundaries")
    graph.add_node(
        "TaskInstruction",
        "Process request",
        {"text": "Process the request."},
        node_id="task_instruction",
        source_event_idx=0,
    )
    graph.add_node(
        "ToolAction",
        "external_api_call",
        {
            "tool_name": "external_api_call",
            "tool_use_id": "api_1",
            "mechanism": "api_call",
            "input": {"customer_id": "CUS-404"},
            "effectful": False,
            "approval_sensitive": True,
            "external_visibility": True,
            "risk_level": "high",
            "channel": None,
        },
        node_id="action_api",
        source_event_idx=3,
    )
    graph.add_edge("action_api", "task_instruction", "depends_on")
    statuses = {
        "parameter_provenance": "missing",
        "target_entity_support": "missing",
        "input_observation": "missing",
        "task_constraint": "supported",
        "approval_dependency": "missing",
        "channel_boundary": "missing",
        "state_freshness": "missing",
        "verification_evidence": "missing",
    }
    return graph, _gold(statuses, set(statuses))


def _dependency_audit(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = [name for name in imports if name.endswith((".ocesq", ".ocei"))]
    return {"file": str(path), "forbidden_imports": forbidden, "passed": not forbidden}


def _negative_control(contract: dict[str, object], graph: BehaviorGraph, gold: dict[str, dict[str, object]]) -> bool:
    corrupted = deepcopy(contract)
    target = next(item for item in corrupted["obligations"] if item["status"] in {"supported", "conflicting"})
    target["selected_path_ids"] = []
    return not verify_ocesq_contract(graph, corrupted, gold)["passed"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ocesq_contract_integration_v1"))
    args = parser.parse_args()
    cases = [
        ("all_supported", *_supported_graph("all_supported", conflicts=False)),
        ("support_and_conflict", *_supported_graph("support_and_conflict", conflicts=True)),
        ("missing_boundaries", *_missing_graph()),
    ]
    rows = []
    for case_id, graph, gold in cases:
        result = run_ocesq(graph, "action_send" if case_id != "missing_boundaries" else "action_api", budget={"max_nodes": 64, "max_edges": 96})
        contract = adapt_ocesq_result(graph, result)
        verification = verify_ocesq_contract(graph, contract, gold)
        rows.append({
            "case_id": case_id,
            "gold": gold,
            "contract": contract,
            "verification": verification,
            "negative_control_detected": _negative_control(contract, graph, gold),
        })

    obligation_rows = [item for row in rows for item in row["verification"]["obligations"]]
    total_obligations = len(obligation_rows)
    status_checks = []
    required_checks = []
    certificate_checks = []
    witness_checks = []
    preservation_checks = []
    for row in rows:
        contract_by_type = {item["type"]: item for item in row["contract"]["obligations"]}
        for typ, gold in row["gold"].items():
            item = contract_by_type[typ]
            status_checks.append(item["status"] == gold["status"])
            required_checks.append(item["required"] == gold["required"])
            if item["status"] in {"missing", "unknown"} and item["required"]:
                certificate_checks.append(bool(item["missing_certificate"] and item["missing_certificate"]["complete"]))
            for fact in [*item["support_facts"], *item["conflict_facts"]]:
                errors = next(check["errors"] for check in row["verification"]["obligations"] if check["type"] == typ)
                witness_checks.append(not any(error.startswith("invalid_") or error.startswith("fact_") for error in errors))
            if item["status"] in {"supported", "conflicting", "missing", "unknown"} and item["required"]:
                errors = next(check["errors"] for check in row["verification"]["obligations"] if check["type"] == typ)
                preservation_checks.append(not any(error.startswith("ocres_") for error in errors))

    verifier_path = Path(__file__).resolve().parents[1] / "mandrel" / "behavior_graph" / "ocesq_contract.py"
    audit = _dependency_audit(verifier_path)
    summary = {
        "experiment_id": "ocesq-contract-integration-v1",
        "cases": len(rows),
        "obligation_instances": total_obligations,
        "obligation_types_covered": len({item["type"] for row in rows for item in row["contract"]["obligations"]}),
        "obligation_instantiation_exactness": mean(required_checks),
        "normalized_status_exactness": mean(status_checks),
        "witness_conflict_validity": mean(witness_checks) if witness_checks else None,
        "certificate_validity": mean(certificate_checks) if certificate_checks else None,
        "ocres_preservation_rate": mean(preservation_checks),
        "case_pass_rate": mean(row["verification"]["passed"] for row in rows),
        "negative_control_detection_rate": mean(row["negative_control_detected"] for row in rows),
        "dependency_audit": audit,
        "failures_by_obligation": {
            item["type"]: item["errors"]
            for row in rows for item in row["verification"]["obligations"] if not item["passed"]
        },
        "interpretation": "Registry-to-contract correctness and OC-RES answer preservation relative to the frozen OEG/OCESQ semantics; not external semantic completeness.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    passed = all(row["verification"]["passed"] and row["negative_control_detected"] for row in rows) and audit["passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
