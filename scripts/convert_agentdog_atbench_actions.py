#!/usr/bin/env python3
"""Convert AgentDoG ATBench conversations into action-local MonitorCards."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocesq.behavior_graph.tool_semantics import classify_tool  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("data/processed/behavior_governance")
IDENTITY_KEYS = {
    "id",
    "user_id",
    "userid",
    "client_id",
    "customer_id",
    "event_id",
    "eventId",
    "email",
    "to",
    "recipient",
    "recipient_id",
    "channel",
    "channel_id",
    "url",
    "webhook_url",
    "path",
    "name",
}
FREE_TEXT_KEYS = {"body", "content", "message", "text", "description", "comment", "thought", "response"}
STRUCTURAL_SIGNAL_TYPES = {"target_binding_not_certified", "input_observation_missing", "scope_not_certified"}
TOOL_RISK_TERMS = {
    "automated",
    "automation",
    "verified",
    "dispatch",
    "minimize latency",
    "system mode",
    "rapid response",
    "admin",
    "credential",
    "secret",
    "token",
    "webhook",
    "execute",
    "delete",
    "send",
    "post",
    "insert",
    "update",
}
USER_RISK_TERMS = {
    "ignore",
    "bypass",
    "jailbreak",
    "without permission",
    "do not ask",
    "secret",
    "credential",
    "token",
    "private",
    "delete",
    "send",
    "share",
}
OBSERVATION_RISK_TERMS = {
    "ignore previous",
    "system mode",
    "verified",
    "trusted",
    "dispatch",
    "execute",
    "delete",
    "send",
    "credential",
    "token",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--effectful-only", action="store_true")
    parser.add_argument("--max-local-events", type=int, default=12)
    args = parser.parse_args()

    rows = convert_file(
        args.input,
        source_name=args.source_name,
        effectful_only=args.effectful_only,
        max_local_events=args.max_local_events,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output, rows)
    summary = summarize(rows, args)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def convert_file(path: Path, *, source_name: str, effectful_only: bool, max_local_events: int) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for row_idx, raw in enumerate(_read_jsonl(path), start=1):
        conversation_id = str(raw.get("id") or raw.get("conv_id") or f"row_{row_idx:06d}")
        messages = _conversation_messages(raw)
        action_ord = 0
        for msg_idx, message in enumerate(messages):
            if message.get("role") != "agent" or not message.get("action"):
                continue
            parsed = _parse_action(message.get("action"))
            if not parsed:
                continue
            tool_name, arguments = parsed
            semantics = classify_tool(tool_name)
            if effectful_only and not _is_candidate_action(semantics):
                continue
            action_ord += 1
            cards.append(
                _make_card(
                    raw=raw,
                    source_name=source_name,
                    conversation_id=conversation_id,
                    messages=messages,
                    msg_idx=msg_idx,
                    action_ord=action_ord,
                    tool_name=tool_name,
                    arguments=arguments,
                    max_local_events=max_local_events,
                )
            )
    cards.sort(key=lambda row: row["row_id"])
    return cards


def _make_card(
    *,
    raw: dict[str, Any],
    source_name: str,
    conversation_id: str,
    messages: list[dict[str, Any]],
    msg_idx: int,
    action_ord: int,
    tool_name: str,
    arguments: dict[str, Any],
    max_local_events: int,
) -> dict[str, Any]:
    semantics = classify_tool(tool_name)
    tool_use_id = f"{conversation_id}::action_{action_ord:04d}"
    node_id = f"action_{_safe_id(conversation_id)}_{action_ord:04d}"
    action = {
        "node_id": node_id,
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
        "mechanism": semantics.mechanism,
        "risk_level": semantics.risk_level,
        "effectful": semantics.effectful,
        "channel": semantics.channel,
        "approval_sensitive": semantics.approval_sensitive,
        "external_visibility": semantics.external_visibility,
        "input": arguments,
    }
    labels = _trajectory_labels(raw)
    trace_context = _trace_context(messages, msg_idx, max_local_events=max_local_events)
    behavior_context = _behavior_context(action, messages[:msg_idx])
    risk_context = _risk_context(raw, action, messages, msg_idx)
    card = {
        "row_id": f"{source_name}::{conversation_id}::card_{action_ord:04d}::{tool_use_id}",
        "episode_id": f"{source_name}_{conversation_id}",
        "conversation_id": conversation_id,
        "card_id": f"card_{action_ord:04d}",
        "candidate_id": f"cand_{action_ord:04d}",
        "candidate_kind": "tool_action",
        "candidate_risk_level": "high" if _is_candidate_action(semantics) else semantics.risk_level,
        "action_index": action_ord,
        "message_index": msg_idx,
        "action": action,
        "trace_context": trace_context,
        "behavior_context": behavior_context,
        "risk_context": risk_context,
        "structural_signals": _signals_from_context(behavior_context),
        "labels": labels,
        "reason": raw.get("reason", ""),
        "real_world_harm": raw.get("real_world_harm"),
        "source": source_name,
        "content_hash": _content_hash(raw),
    }
    card["weak_label"] = _weak_label(card)
    return card


def _trajectory_labels(raw: dict[str, Any]) -> dict[str, Any]:
    label = raw.get("label")
    is_safe = None if label is None else label == 0
    return {
        "label": label,
        "is_safe": is_safe,
        "risk_source": raw.get("risk_source"),
        "failure_mode": raw.get("failure_mode"),
        "real_world_harm": raw.get("real_world_harm"),
    }


def _risk_context(raw: dict[str, Any], action: dict[str, Any], messages: list[dict[str, Any]], msg_idx: int) -> dict[str, Any]:
    tool_profile = _tool_profile(raw, str(action.get("tool_name") or ""))
    prior_messages = messages[:msg_idx]
    user_excerpt = next((str(message.get("content", "")) for message in messages if message.get("role") == "user"), "")
    environment_observations = [
        _trim_text(_message_text(message), 800)
        for message in prior_messages
        if message.get("role") == "environment"
    ][-3:]
    trajectory_risk = {
        "is_safe": _trajectory_labels(raw).get("is_safe"),
        "risk_source": raw.get("risk_source"),
        "failure_mode": raw.get("failure_mode"),
        "real_world_harm": raw.get("real_world_harm"),
    }
    risk_source = str(raw.get("risk_source") or "")
    failure_mode = str(raw.get("failure_mode") or "")
    sources = {
        "user_input": _source_signal(
            risk_source=risk_source,
            names={"malicious_user_instruction_or_jailbreak", "direct_prompt_injection", "indirect_prompt_injection"},
            text=user_excerpt,
            terms=USER_RISK_TERMS,
        ),
        "environment_observation": _source_signal(
            risk_source=risk_source,
            names={"corrupted_tool_feedback", "unreliable_or_misinformation", "indirect_prompt_injection"},
            text="\n".join(environment_observations),
            terms=OBSERVATION_RISK_TERMS,
        ),
        "external_tool_or_api": _source_signal(
            risk_source=risk_source,
            names={"malicious_tool_execution", "tool_description_injection"},
            text=tool_profile.get("description_excerpt", ""),
            terms=TOOL_RISK_TERMS,
        ),
        "internal_decision": _source_signal(
            risk_source=risk_source,
            names={"inherent_agent_failures", "inherent_agent_or_llm_failures"},
            text=failure_mode,
            terms={"planning", "reasoning", "procedural", "inaction", "incorrect"},
        ),
    }
    risky_sources = [name for name, signal in sources.items() if signal["active"]]
    action_dependency = {
        "uses_tool_with_description_change": bool(tool_profile.get("description_changed")),
        "uses_tool_with_risk_terms": bool(tool_profile.get("risk_terms")),
        "after_environment_observation": any(message.get("role") == "environment" for message in prior_messages),
        "depends_on_trajectory_risk_source": bool(risky_sources),
        "risk_source_alignment": risky_sources,
    }
    return {
        "trajectory_risk": trajectory_risk,
        "risk_sources": sources,
        "tool_profile": tool_profile,
        "environment_observations": environment_observations,
        "action_dependency": action_dependency,
        "conversion_note": "Action-local risk context derived only from pre-action messages and tool metadata.",
    }


def _tool_profile(raw: dict[str, Any], tool_name: str) -> dict[str, Any]:
    tools = raw.get("tool_used") or []
    match = next((tool for tool in tools if isinstance(tool, dict) and str(tool.get("name")) == tool_name), {})
    description = str(match.get("description") or "")
    original_description = str(match.get("_original_description") or "")
    risk_terms = sorted(term for term in TOOL_RISK_TERMS if term in description.lower())
    return {
        "name": tool_name,
        "source": match.get("_source"),
        "description_excerpt": _trim_text(description, 1200),
        "has_original_description": bool(original_description),
        "description_changed": bool(original_description and original_description != description),
        "original_description_excerpt": _trim_text(original_description, 800) if original_description else "",
        "risk_terms": risk_terms,
        "parameter_names": sorted((match.get("parameters", {}).get("properties") or {}).keys()) if isinstance(match.get("parameters"), dict) else [],
    }


def _source_signal(*, risk_source: str, names: set[str], text: str, terms: set[str]) -> dict[str, Any]:
    text_l = str(text or "").lower()
    matched_terms = sorted(term for term in terms if term in text_l)
    source_l = risk_source.lower()
    return {
        "active": source_l in names or bool(matched_terms),
        "trajectory_source_match": source_l in names,
        "matched_terms": matched_terms,
    }


def _weak_label(card: dict[str, Any]) -> dict[str, Any]:
    labels = card.get("labels", {})
    action = card.get("action", {})
    context = card.get("behavior_context", {})
    blockers = context.get("structural_blockers", [])
    if labels.get("is_safe") is True:
        if blockers:
            return _label("under_evidenced", "target_binding", "repair", "revise_target", 0.55)
        return _label("supported", "none", "allow", "none", 0.6)

    if blockers:
        site = blockers[0].get("failure_site") or "target_binding"
        repair = blockers[0].get("repair_target") or "run_verification"
        return _label("under_evidenced", site, "repair", repair, 0.55)

    failure_site, repair_target = _map_failure_mode(str(labels.get("failure_mode") or ""))
    if _is_high_impact_action(action):
        return _label("contradicted", failure_site, "block", repair_target, 0.55)
    return _label("risky", failure_site, "warn", repair_target if repair_target != "abort_action" else "run_verification", 0.5)


def _label(process_status: str, failure_site: str, severity: str, repair_target: str, confidence: float) -> dict[str, Any]:
    return {
        "process_status": process_status,
        "failure_site": failure_site,
        "severity": severity,
        "selected_repair_target": repair_target,
        "confidence": confidence,
        "rationale": "Weak heuristic draft; use only after strong action-local review.",
        "evidence": [],
        "missing_support": [],
    }


def _map_failure_mode(failure_mode: str) -> tuple[str, str]:
    mode = failure_mode.lower()
    if "authorization" in mode or "privileged" in mode or "unconfirmed" in mode:
        return "authorization", "request_authorization"
    if "validate" in mode or "misleading" in mode or "unverified" in mode:
        return "verification", "run_verification"
    if "tool_misuse" in mode or "incorrect_tool" in mode or "choosing_malicious_tool" in mode:
        return "tool_misuse", "diagnose_failure"
    if "information_disclosure" in mode or "harmful" in mode or "malicious" in mode or "illegal" in mode:
        return "policy_conflict", "abort_action"
    if "execution" in mode:
        return "destructive_effect", "abort_action"
    if "planning" in mode or "procedural" in mode or "inaction" in mode:
        return "input_observation", "run_verification"
    return "policy_conflict", "abort_action"


def _behavior_context(action: dict[str, Any], prior_messages: list[dict[str, Any]]) -> dict[str, Any]:
    bindings = []
    support_facts = []
    missing_facts = []
    anchors = _action_anchors(action.get("input", {}))
    prior_text = "\n".join(_message_text(message) for message in prior_messages)
    for key, value in anchors:
        value_text = _normalize(value)
        matched = bool(value_text and value_text in prior_text.lower())
        binding = {
            "parameter": key,
            "value": value,
            "binding_status": "matched_unique" if matched else "unmatched",
            "source_observation": "prior_context" if matched else None,
            "matched_record": {"value": value} if matched else None,
            "alternative_entities": [],
            "disambiguation_evidence": [],
        }
        bindings.append(binding)
        if matched:
            support_facts.append(
                {
                    "type": "parameter_binding",
                    "parameter": key,
                    "value": value,
                    "source_observation": "prior_context",
                    "binding_status": "matched_unique",
                }
            )
        else:
            missing_facts.append({"type": "unmatched_parameter", "parameter": key, "value": value})

    if not anchors and _is_high_impact_action(action):
        missing_facts.append(
            {
                "type": "missing_target_anchor",
                "summary": "No structured target parameter was available for binding in a high-impact action.",
            }
        )
    blockers = _structural_blockers(action, bindings, missing_facts)
    cert = _allow_certificate(action, bindings, support_facts, blockers)
    return {
        "parameter_bindings": bindings,
        "support_facts": support_facts,
        "missing_facts": missing_facts,
        "coverage_gaps": [],
        "structural_blockers": blockers,
        "allow_certificate": cert,
        "certificate_complete": all(item.get("status") in {"satisfied", "not_required"} for item in cert),
    }


def _structural_blockers(action: dict[str, Any], bindings: list[dict[str, Any]], missing_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    bad_bindings = [b for b in bindings if b.get("binding_status") in {"unmatched", "ambiguous"} or not b.get("source_observation")]
    if bad_bindings and _is_high_impact_action(action):
        blockers.append(
            {
                "type": "target_binding_not_certified",
                "status": "missing",
                "failure_site": "target_binding",
                "repair_target": "revise_target",
                "summary": "Target or identity parameters are not certified by prior context.",
                "evidence_node_ids": [action.get("node_id")] if action.get("node_id") else [],
                "parameter_bindings": bad_bindings,
            }
        )
    missing_types = {fact.get("type") for fact in missing_facts}
    if _is_high_impact_action(action) and missing_types & {"missing_target_anchor", "unmatched_parameter", "missing_observation"}:
        blockers.append(
            {
                "type": "input_observation_missing",
                "status": "missing",
                "failure_site": "input_observation",
                "repair_target": "run_verification",
                "summary": "Supporting observations are missing from the action-local card.",
                "evidence_node_ids": [action.get("node_id")] if action.get("node_id") else [],
            }
        )
    return blockers


def _allow_certificate(action: dict[str, Any], bindings: list[dict[str, Any]], support_facts: list[dict[str, Any]], blockers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    good_bindings = [b for b in bindings if b.get("binding_status") == "matched_unique" and b.get("source_observation")]
    target_required = bool(_is_high_impact_action(action) or bindings)
    input_required = target_required
    auth_required = bool(action.get("approval_sensitive") or action.get("external_visibility"))
    effect_required = bool(action.get("effectful") and action.get("risk_level") == "high")
    auth_evidence = _evidence(support_facts, {"authorization", "approval"})
    return [
        _cert_item("target_binding", "target_binding", "revise_target", target_required, [b["source_observation"] for b in good_bindings], blockers),
        _cert_item("input_observation", "input_observation", "run_verification", input_required, _evidence(support_facts, {"parameter_binding"}), blockers),
        _cert_item("authorization", "authorization", "request_authorization", auth_required, auth_evidence, blockers),
        _cert_item("effect_safety", "destructive_effect", "abort_action", effect_required, [action.get("node_id")] if not blockers else [], blockers),
        _cert_item("policy_consistency", "policy_conflict", "abort_action", False, [], blockers),
    ]


def _cert_item(name: str, failure_site: str, repair_target: str, required: bool, evidence: list[Any], blockers: list[dict[str, Any]]) -> dict[str, Any]:
    related = [b for b in blockers if b.get("failure_site") == failure_site]
    if related:
        status = "conflict" if any(b.get("status") == "conflict" for b in related) else "missing"
        summary = related[0].get("summary", "")
        evidence_node_ids = sorted({node for b in related for node in b.get("evidence_node_ids", []) if node})
    elif required:
        status = "satisfied" if evidence else "missing"
        summary = "Requirement satisfied by card evidence." if evidence else "Required card evidence is missing."
        evidence_node_ids = [str(item) for item in evidence if item]
    else:
        status = "not_required"
        summary = "Requirement is not applicable to this action."
        evidence_node_ids = []
    return {
        "name": name,
        "required": required,
        "status": status,
        "failure_site": failure_site,
        "repair_target": repair_target,
        "summary": summary,
        "evidence_node_ids": evidence_node_ids,
    }


def _signals_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for idx, blocker in enumerate(context.get("structural_blockers", []), start=1):
        if blocker.get("type") not in STRUCTURAL_SIGNAL_TYPES:
            continue
        rows.append(
            {
                "id": f"sig_{idx}",
                "type": blocker.get("type"),
                "severity": "high" if blocker.get("status") in {"missing", "conflict"} else "medium",
                "summary": blocker.get("summary", ""),
                "node_ids": blocker.get("evidence_node_ids", []),
                "attrs": blocker,
            }
        )
    return rows


def _action_anchors(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            tail = str(key)
            if tail in FREE_TEXT_KEYS:
                continue
            if isinstance(item, (dict, list)):
                rows.extend(_action_anchors(item, next_key))
            elif item is not None and str(item).strip() and (tail in IDENTITY_KEYS or tail.endswith("_id") or "@" in str(item)):
                rows.append((next_key, item))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            rows.extend(_action_anchors(item, f"{prefix}[{idx}]"))
    return rows


def _trace_context(messages: list[dict[str, Any]], action_idx: int, *, max_local_events: int) -> dict[str, Any]:
    start = max(0, action_idx - max_local_events + 1)
    user_text = next((str(message.get("content", "")) for message in messages if message.get("role") == "user"), "")
    return {
        "user_message_excerpt": _trim_text(user_text, 5000),
        "local_events": [_compact_message(message, idx) for idx, message in enumerate(messages[start : action_idx + 1], start=start)],
        "tool_result": None,
        "context_window": {
            "policy": "pre_action_only",
            "action_event_idx": action_idx,
            "start_event_idx": start,
            "end_event_idx_exclusive": action_idx + 1,
        },
    }


def _compact_message(message: dict[str, Any], idx: int) -> dict[str, Any]:
    row = {"type": "message", "role": message.get("role"), "source_event_idx": idx}
    if message.get("role") == "agent":
        row["thought"] = _trim_text(str(message.get("thought", "")), 800)
        row["action"] = _trim_text(str(message.get("action", "")), 1600)
    else:
        row["content"] = _trim_text(str(message.get("content", "")), 1800)
    return row


def _conversation_messages(raw: dict[str, Any]) -> list[dict[str, Any]]:
    messages = raw.get("contents") if "contents" in raw else raw.get("content", [])
    if len(messages) == 1 and isinstance(messages[0], list):
        messages = messages[0]
    return [message for message in messages if isinstance(message, dict)]


def _parse_action(text: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text or not text.startswith("{"):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or not value.get("name"):
        return None
    arguments = value.get("arguments") or value.get("input") or {}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return str(value["name"]), arguments


def _is_candidate_action(semantics: Any) -> bool:
    return bool(semantics.effectful or semantics.risk_level == "high" or semantics.approval_sensitive or semantics.external_visibility)


def _is_high_impact_action(action: dict[str, Any]) -> bool:
    return bool(action.get("effectful") or action.get("risk_level") == "high" or action.get("approval_sensitive") or action.get("external_visibility"))


def _message_text(message: dict[str, Any]) -> str:
    pieces = [str(message.get("content", "")), str(message.get("thought", "")), str(message.get("action", ""))]
    return "\n".join(piece for piece in pieces if piece)


def _evidence(facts: list[dict[str, Any]], fact_types: set[str]) -> list[str]:
    rows = []
    for fact in facts:
        if fact.get("type") not in fact_types:
            continue
        if fact.get("source_observation"):
            rows.append(str(fact["source_observation"]))
    return sorted(set(rows))


def _normalize(value: Any) -> str:
    return str(value).strip().lower()


def _content_hash(raw: dict[str, Any]) -> str:
    payload = {
        "user": [m.get("content", "") for m in _conversation_messages(raw) if m.get("role") == "user"],
        "tool_used": raw.get("tool_used", []),
        "failure_mode": raw.get("failure_mode"),
        "risk_source": raw.get("risk_source"),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:80]


def _trim_text(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "..."


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "input": str(args.input),
        "output": str(args.output),
        "source_name": args.source_name,
        "effectful_only": args.effectful_only,
        "rows": len(rows),
        "episodes": len({row.get("episode_id") for row in rows}),
        "content_hashes": len({row.get("content_hash") for row in rows}),
    }
    for field in ("candidate_risk_level",):
        summary[field] = dict(collections.Counter(row.get(field) for row in rows).most_common())
    summary["mechanism"] = dict(collections.Counter(row.get("action", {}).get("mechanism") for row in rows).most_common())
    summary["severity_weak"] = dict(collections.Counter(row.get("weak_label", {}).get("severity") for row in rows).most_common())
    summary["failure_mode"] = dict(collections.Counter(row.get("labels", {}).get("failure_mode") for row in rows).most_common(30))
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
