#!/usr/bin/env python3
"""Build source-trace-disjoint effect-removal and stale-state APC faults."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


PROTOCOL = "apc-expanded-faults-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_bytes(events: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for event in events
    )


def serialization_bytes(events: list[dict[str, Any]]) -> bytes:
    # Preserve raw dictionary order. The frozen OEG schema's canonical-record
    # identity policy is order-sensitive, so checksum canonicalization must not
    # be reused as the executable trace serialization.
    return b"".join(
        (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for event in events
    )


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> str:
    payload = serialization_bytes(events)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def tool_part(events: list[dict[str, Any]], tool_use_id: str) -> tuple[int, dict[str, Any]]:
    matches = []
    for idx, event in enumerate(events):
        if event.get("type") != "message":
            continue
        for part in event.get("message", {}).get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "tool_use" and part.get("id") == tool_use_id:
                matches.append((idx, part))
    if len(matches) != 1:
        raise ValueError(f"expected one tool declaration {tool_use_id}, found {len(matches)}")
    return matches[0]


def dispatch(events: list[dict[str, Any]], tool_use_id: str) -> tuple[int, dict[str, Any]] | None:
    matches = [
        (idx, event) for idx, event in enumerate(events)
        if event.get("type") == "tool_dispatch" and event.get("tool_use_id") == tool_use_id
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple dispatches for {tool_use_id}")
    return matches[0] if matches else None


def result_part(events: list[dict[str, Any]], tool_use_id: str) -> tuple[int, dict[str, Any]] | None:
    matches = []
    for idx, event in enumerate(events):
        if event.get("type") != "message":
            continue
        for part in event.get("message", {}).get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "tool_result" and part.get("tool_use_id") == tool_use_id:
                matches.append((idx, part))
    if len(matches) > 1:
        raise ValueError(f"expected at most one tool result {tool_use_id}, found {len(matches)}")
    return matches[0] if matches else None


def transaction(events: list[dict[str, Any]], tool_use_id: str) -> dict[str, Any]:
    _, declaration = tool_part(events, tool_use_id)
    dispatched = dispatch(events, tool_use_id)
    return {"declaration": declaration, "dispatch": dispatched[1] if dispatched else None}


def single_part_message(event: dict[str, Any], part: dict[str, Any]) -> dict[str, Any]:
    cloned = deepcopy(event)
    cloned["message"]["content"] = [deepcopy(part)]
    cloned["message"]["reasoning_content"] = None
    return cloned


def remove_effect(events: list[dict[str, Any]], target_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variant = deepcopy(events)
    dispatch_rows = [i for i, event in enumerate(variant) if event.get("type") == "tool_dispatch" and event.get("tool_use_id") == target_id]
    if len(dispatch_rows) != 1:
        raise ValueError(f"effect removal requires one target dispatch, found {len(dispatch_rows)}")
    removed_dispatch_idx = dispatch_rows[0]
    del variant[removed_dispatch_idx]
    removed_results = 0
    empty_messages = []
    for idx, event in enumerate(variant):
        if event.get("type") != "message":
            continue
        content = event.get("message", {}).get("content", []) or []
        kept = [
            part for part in content
            if not (
                isinstance(part, dict)
                and part.get("type") == "tool_result"
                and part.get("tool_use_id") == target_id
            )
        ]
        removed_results += len(content) - len(kept)
        event["message"]["content"] = kept
        if content and not kept:
            empty_messages.append(idx)
    for idx in reversed(empty_messages):
        del variant[idx]
    if removed_results > 1:
        raise ValueError(f"effect removal supports at most one target tool_result, found {removed_results}")
    return variant, {"removed_dispatch_event_idx": removed_dispatch_idx, "removed_tool_results": removed_results}


def at_path(value: Any, path: list[Any]) -> Any:
    current = value
    for item in path:
        current = current[item]
    return current


def inject_stale(events: list[dict[str, Any]], case: dict[str, Any], case_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variant = deepcopy(events)
    source_id = case["source"]["source_tool_use_id"]
    target_id = case["target_action"]["tool_use_id"]
    source_message_idx, source_tool = tool_part(variant, source_id)
    source_dispatch = dispatch(variant, source_id)
    source_result_row = result_part(variant, source_id)
    target_message_idx, _ = tool_part(variant, target_id)
    if source_dispatch is None:
        raise ValueError(f"source dispatch missing for {source_id}")

    injected_id = f"apc-stale-{hashlib.sha256(case_id.encode()).hexdigest()[:16]}"
    cloned_tool = deepcopy(source_tool)
    cloned_tool["id"] = injected_id
    cloned_message = single_part_message(variant[source_message_idx], cloned_tool)
    cloned_dispatch = deepcopy(source_dispatch[1])
    cloned_dispatch["tool_use_id"] = injected_id

    record = at_path(cloned_dispatch["response_body"], case["source"]["record_path"])
    state_key = case["source"]["state_key"]
    old_state = record[state_key]
    new_state = f"APC-STALE-{hashlib.sha256((case_id + state_key).encode()).hexdigest()[:12]}"
    record[state_key] = new_state

    # The injected response is the latest same-identity observation before the
    # target, while the original observation remains available for conflict.
    target_message_idx, _ = tool_part(variant, target_id)
    injected_events = [cloned_message, cloned_dispatch]
    if source_result_row is not None:
        source_result_idx, source_result = source_result_row
        cloned_result = deepcopy(source_result)
        cloned_result["tool_use_id"] = injected_id
        cloned_result["content"] = [{
            "type": "text",
            "text": json.dumps(cloned_dispatch.get("response_body"), ensure_ascii=False, sort_keys=True),
        }]
        injected_events.append(single_part_message(variant[source_result_idx], cloned_result))
    variant[target_message_idx:target_message_idx] = injected_events
    return variant, {
        "injected_tool_use_id": injected_id,
        "inserted_before_target_event_idx": target_message_idx,
        "state_key": state_key,
        "old_state_sha256": digest(old_state),
        "new_state_sha256": digest(new_state),
        "identity_value_sha256": case["source"]["identity_value_sha256"],
        "inserted_event_count": len(injected_events),
    }


def select(rows: list[dict[str, Any]], family: str, cap: int, seed: int) -> list[dict[str, Any]]:
    if cap <= 0:
        return []
    candidates = [row for row in rows if row["family"] == family]
    rng = random.Random(f"{seed}:{family}")
    rng.shuffle(candidates)
    selected = []
    used_traces = set()
    for row in candidates:
        if row["trace_path"] in used_traces:
            continue
        selected.append(row)
        used_traces.add(row["trace_path"])
        if len(selected) == cap:
            break
    return selected


def build(eligibility_manifest: Path, output_dir: Path, effect_cap: int, stale_cap: int, seed: int) -> dict[str, Any]:
    eligibility = read_jsonl(eligibility_manifest)
    selected = [
        *select(eligibility, "remove_unique_effect_result", effect_cap, seed),
        *select(eligibility, "inject_stale_entity_version", stale_cap, seed),
    ]
    variants_dir = output_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, source in enumerate(selected):
        family = source["family"]
        case_id = f"apcx_{number:04d}"
        baseline_path = Path(source["trace_path"])
        baseline = read_jsonl(baseline_path)
        target_id = source["target_action"]["tool_use_id"]
        baseline_transaction = transaction(baseline, target_id)
        if family == "remove_unique_effect_result":
            variant, details = remove_effect(baseline, target_id)
            integrity = {
                "target_declaration_preserved": transaction(variant, target_id)["declaration"] == baseline_transaction["declaration"],
                "target_dispatch_removed": dispatch(variant, target_id) is None,
                "target_result_removed": not any(
                    isinstance(part, dict) and part.get("type") == "tool_result" and part.get("tool_use_id") == target_id
                    for event in variant if event.get("type") == "message"
                    for part in event.get("message", {}).get("content", []) or []
                ),
            }
        else:
            variant, details = inject_stale(baseline, source, case_id)
            injected = dispatch(variant, details["injected_tool_use_id"])
            integrity = {
                "target_transaction_preserved": transaction(variant, target_id) == baseline_transaction,
                "original_source_preserved": dispatch(variant, source["source"]["source_tool_use_id"]) is not None,
                "injected_response_present": injected is not None,
                "state_value_changed": details["old_state_sha256"] != details["new_state_sha256"],
                "identity_value_preserved": details["identity_value_sha256"] == source["source"]["identity_value_sha256"],
            }
        if not all(integrity.values()):
            raise RuntimeError(f"integrity failure for {case_id}: {integrity}")
        family_dir = variants_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        variant_path = family_dir / f"{case_id}.jsonl"
        rows.append({
            "case_id": case_id,
            "protocol": PROTOCOL,
            "family": family,
            "baseline_trace": str(baseline_path),
            "variant_trace": str(variant_path),
            "baseline_sha256": hashlib.sha256(canonical_bytes(baseline)).hexdigest(),
            "variant_sha256": write_jsonl(variant_path, variant),
            "trace_id": source["trace_id"],
            "target_action": source["target_action"],
            "source": source.get("source"),
            "transform": details,
            "integrity": integrity,
            "oracle_boundary": "raw-event transaction identity, deletion, ordering, and state-field equality only",
        })

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "protocol": PROTOCOL,
        "seed": seed,
        "pairs": len(rows),
        "by_family": {
            family: {
                "pairs": sum(row["family"] == family for row in rows),
                "source_traces": len({row["baseline_trace"] for row in rows if row["family"] == family}),
                "tool_counts": dict(Counter(
                    row["target_action"]["tool_name"] for row in rows if row["family"] == family
                )),
            }
            for family in ("remove_unique_effect_result", "inject_stale_entity_version")
        },
        "source_trace_disjoint_within_family": all(
            len([row for row in rows if row["family"] == family])
            == len({row["baseline_trace"] for row in rows if row["family"] == family})
            for family in ("remove_unique_effect_result", "inject_stale_entity_version")
        ),
        "integrity_passed": all(all(row["integrity"].values()) for row in rows),
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
    parser.add_argument("--eligibility-manifest", type=Path, default=Path("data/apc_fault_eligibility_v1/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/apc_expanded_faults_v1"))
    parser.add_argument("--effect-cap", type=int, default=24)
    parser.add_argument("--stale-cap", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    print(json.dumps(
        build(args.eligibility_manifest, args.output_dir, args.effect_cap, args.stale_cap, args.seed),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
