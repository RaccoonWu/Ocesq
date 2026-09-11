#!/usr/bin/env python3
"""Scan raw traces for expanded APC fault-injection preconditions.

This scanner does not compile graphs or call OCESQ. Tool semantics are used as
the frozen contract-applicability specification, while all evidence and
identity/state checks operate directly on raw events.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from ocesq.behavior_graph.tool_semantics import classify_tool


PROTOCOL = "apc-fault-eligibility-v1"
STRUCTURED_APPROVAL_TYPES = {"approval", "approval_event", "authorization", "permission"}
IDENTITY_KEYS = {"id", "email", "to", "from", "recipient", "path", "file"}
# Frozen fields implemented by the OEG entity-state predicate. Other raw state
# fields are counted as schema-unsupported rather than injected as query cases.
STATE_KEYS = ("status", "priority", "resolution", "stage", "state", "current_stock", "is_read")
VERSION_SENSITIVE = {
    "api_call", "browser_interaction", "code_execution", "delete", "export", "schedule",
    "save_draft", "send", "share", "session_write", "shell_input", "update",
}


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def declarations(events: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    for idx, event in enumerate(events):
        if event.get("type") != "message":
            continue
        for part in event.get("message", {}).get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                rows.append((idx, part))
    return rows


def dispatches(events: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, event in enumerate(events):
        if event.get("type") == "tool_dispatch" and event.get("tool_use_id"):
            rows.setdefault(str(event["tool_use_id"]), []).append((idx, event))
    return rows


def direct_identities(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(child)
        for key, child in value.items()
        if isinstance(child, (str, int, float, bool))
        and (str(key).lower() in IDENTITY_KEYS or str(key).lower().endswith("_id"))
    }


def nested_identities(value: Any) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        found.update(direct_identities(value).items())
        for child in value.values():
            found.update(nested_identities(child))
    elif isinstance(value, list):
        for child in value:
            found.update(nested_identities(child))
    return found


def record_rows(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from record_rows(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from record_rows(child, (*path, index))


def direct_states(record: dict[str, Any]) -> list[tuple[str, Any]]:
    by_lower = {str(key).lower(): (str(key), value) for key, value in record.items()}
    return [by_lower[key] for key in STATE_KEYS if key in by_lower and isinstance(by_lower[key][1], (str, int, float, bool))]


def canonical_record_identity(record: dict[str, Any]) -> tuple[str, str] | None:
    # Mirrors the frozen OEG schema policy as an explicit raw-data rule: one
    # canonical stable identity is materialized per response record.
    for key, value in record.items():
        tail = str(key).lower()
        if value and (tail == "id" or tail.endswith("_id") or tail.endswith("_ids")):
            return str(key), str(value)
    for key in ("id", "entity_id", "message_id", "contact_id", "customer_id", "ticket_id", "integration_id", "email", "to", "from", "name", "subject"):
        if record.get(key):
            return key, str(record[key])
    return None


def stale_source(
    events: list[dict[str, Any]],
    action_idx: int,
    action: dict[str, Any],
    dispatch_map: dict[str, list[tuple[int, dict[str, Any]]]],
) -> dict[str, Any] | None:
    target_ids = nested_identities(action.get("input", {}))
    if not target_ids:
        return None
    candidates = []
    for source_id, source_rows in dispatch_map.items():
        if source_id == action.get("id"):
            continue
        for source_idx, source in source_rows:
            if source_idx >= action_idx:
                continue
            for record_path, record in record_rows(source.get("response_body")):
                primary = canonical_record_identity(record)
                shared = [primary] if primary is not None and primary in target_ids else []
                states = direct_states(record)
                if shared and states:
                    candidates.append((source_idx, source_id, source, record_path, shared, states))
    if not candidates:
        return None
    source_idx, source_id, source, record_path, shared, states = sorted(
        candidates,
        key=lambda row: (-row[0], row[1], tuple(map(str, row[3])), row[5][0][0]),
    )[0]
    state_key, state_value = states[0]
    identity_key, identity_value = shared[0]
    equivalent_indices = sorted({
        candidate[0]
        for candidate in candidates
        if (identity_key, identity_value) in candidate[4]
        and any(key == state_key and value == state_value for key, value in candidate[5])
    })
    return {
        "source_tool_use_id": source_id,
        "source_event_idx": source_idx,
        "source_tool_name": source.get("tool_name"),
        "record_path": list(record_path),
        "identity_key": identity_key,
        "identity_value_sha256": digest(identity_value),
        "state_key": state_key,
        "state_value_sha256": digest(state_value),
        "equivalent_source_event_indices": equivalent_indices,
    }


def scan(pattern: str, output_dir: Path) -> dict[str, Any]:
    paths = [Path(path) for path in sorted(glob.glob(pattern, recursive=True))]
    rows = []
    structured_approvals = []
    effect_tools: Counter[str] = Counter()
    stale_tools: Counter[str] = Counter()
    effect_traces: set[str] = set()
    stale_traces: set[str] = set()

    for trace_path in paths:
        events = read_events(trace_path)
        dispatch_map = dispatches(events)
        trace_id = str(events[0].get("trace_id") or trace_path.stem)
        for idx, event in enumerate(events):
            if str(event.get("type", "")).lower() in STRUCTURED_APPROVAL_TYPES:
                structured_approvals.append({"trace_path": str(trace_path), "event_idx": idx, "event_type": event.get("type")})
        for action_idx, action in declarations(events):
            tool_use_id = str(action.get("id"))
            tool_name = str(action.get("name") or "")
            semantics = classify_tool(tool_name)
            target_dispatches = dispatch_map.get(tool_use_id, [])
            if semantics.effectful and len(target_dispatches) == 1:
                dispatch_idx, _ = target_dispatches[0]
                rows.append({
                    "family": "remove_unique_effect_result",
                    "trace_id": trace_id,
                    "trace_path": str(trace_path),
                    "target_action": {
                        "tool_use_id": tool_use_id,
                        "tool_name": tool_name,
                        "declaration_event_idx": action_idx,
                        "dispatch_event_idx": dispatch_idx,
                    },
                    "precondition": {"effectful_by_frozen_tool_semantics": True, "unique_target_dispatch": True},
                })
                effect_tools[tool_name] += 1
                effect_traces.add(str(trace_path))
            if semantics.mechanism in VERSION_SENSITIVE:
                source = stale_source(events, action_idx, action, dispatch_map)
                if source:
                    rows.append({
                        "family": "inject_stale_entity_version",
                        "trace_id": trace_id,
                        "trace_path": str(trace_path),
                        "target_action": {
                            "tool_use_id": tool_use_id,
                            "tool_name": tool_name,
                            "declaration_event_idx": action_idx,
                        },
                        "source": source,
                        "precondition": {
                            "version_sensitive_by_frozen_tool_semantics": True,
                            "prior_same_identity_record": True,
                            "mutable_state_field_present": True,
                        },
                    })
                    stale_tools[tool_name] += 1
                    stale_traces.add(str(trace_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "structured_approvals.json").write_text(
        json.dumps(structured_approvals, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "protocol": PROTOCOL,
        "trace_pattern": pattern,
        "traces": len(paths),
        "structured_approval_events": len(structured_approvals),
        "approval_fault_eligible": 0,
        "approval_exclusion_reason": "no structured approval/authorization event; natural-language authorization is not mutated without labels",
        "remove_unique_effect_result": {
            "eligible_actions": sum(row["family"] == "remove_unique_effect_result" for row in rows),
            "eligible_traces": len(effect_traces),
            "tool_counts": dict(effect_tools),
        },
        "inject_stale_entity_version": {
            "eligible_actions": sum(row["family"] == "inject_stale_entity_version" for row in rows),
            "eligible_traces": len(stale_traces),
            "tool_counts": dict(stale_tools),
        },
        "graph_or_query_output_used": False,
        "human_labels_used": False,
        "model_labels_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-glob", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/apc_fault_eligibility_v1"))
    args = parser.parse_args()
    print(json.dumps(scan(args.trace_glob, args.output_dir), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
