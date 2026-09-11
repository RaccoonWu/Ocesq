#!/usr/bin/env python3
"""Build raw-trace-only benign rewrites for APC invariance evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ocesq.behavior_graph.trace_compiler import load_jsonl


PROTOCOL = "apc-benign-invariance-v1"
FAMILIES = ("disconnected_noise", "transparent_wrapper", "unrelated_event_reorder")
WRAPPER_KEY = "apc_transport_metadata"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_bytes(events: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for event in events
    )


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> str:
    payload = canonical_bytes(events)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def tool_transaction(events: list[dict[str, Any]], tool_use_id: str) -> dict[str, Any]:
    declarations = []
    for event in events:
        if event.get("type") != "message":
            continue
        declarations.extend(
            part for part in event.get("message", {}).get("content", []) or []
            if isinstance(part, dict) and part.get("type") == "tool_use" and part.get("id") == tool_use_id
        )
    dispatches = [
        event for event in events
        if event.get("type") == "tool_dispatch" and event.get("tool_use_id") == tool_use_id
    ]
    if len(declarations) != 1 or len(dispatches) != 1:
        raise ValueError(
            f"expected one declaration and dispatch for {tool_use_id}, "
            f"found {len(declarations)} and {len(dispatches)}"
        )
    dispatch = {
        key: value for key, value in dispatches[0].items()
        if key != WRAPPER_KEY
    }
    return {"declaration": declarations[0], "dispatch": dispatch}


def transform(events: list[dict[str, Any]], family: str, case_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variant = deepcopy(events)
    if family == "disconnected_noise":
        trace_id = events[0].get("trace_id")
        noise_id = f"benign-noise-{case_id}"
        token = f"noise-{hashlib.sha256(case_id.encode()).hexdigest()[:12]}"
        variant.extend([
            {
                "type": "message",
                "trace_id": trace_id,
                "message": {
                    "role": "assistant",
                    "reasoning_content": None,
                    "content": [{
                        "type": "tool_use",
                        "id": noise_id,
                        "name": "apc_diagnostic_ping",
                        "input": {"diagnostic_token": token},
                    }],
                },
            },
            {
                "type": "tool_dispatch",
                "trace_id": trace_id,
                "tool_use_id": noise_id,
                "tool_name": "apc_diagnostic_ping",
                "request_body": {"diagnostic_token": token},
                "response_status": 200,
                "response_body": {"diagnostic_token": token, "status": "ok"},
            },
            {
                "type": "message",
                "trace_id": trace_id,
                "message": {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": noise_id,
                        "content": [{
                            "type": "text",
                            "text": json.dumps(
                                {"diagnostic_token": token, "status": "ok"},
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        }],
                    }],
                },
            },
        ])
        return variant, {
            "inserted_indices": list(range(len(events), len(events) + 3)),
            "inserted_transaction": noise_id,
            "shared_target_identity": False,
        }
    if family == "transparent_wrapper":
        for index, event in enumerate(variant):
            event[WRAPPER_KEY] = {"transparent": True, "version": 1, "event_ordinal": index}
        return variant, {"wrapped_events": len(variant), "wrapper_key": WRAPPER_KEY}
    if family == "unrelated_event_reorder":
        trace_end = [i for i, event in enumerate(variant) if event.get("type") == "trace_end"]
        grading = [i for i, event in enumerate(variant) if event.get("type") == "grading_result"]
        if len(trace_end) != 1 or len(grading) != 1 or trace_end[0] >= grading[0]:
            raise ValueError("reorder requires one trailing trace_end before one grading_result")
        left, right = trace_end[0], grading[0]
        variant[left], variant[right] = variant[right], variant[left]
        return variant, {"swapped_indices": [left, right], "swapped_types": ["trace_end", "grading_result"]}
    raise ValueError(f"unknown benign family: {family}")


def validate_transform(
    baseline: list[dict[str, Any]],
    variant: list[dict[str, Any]],
    family: str,
    target_id: str,
) -> dict[str, bool]:
    baseline_transaction = tool_transaction(baseline, target_id)
    variant_transaction = tool_transaction(variant, target_id)
    integrity = {
        "target_transaction_preserved": digest(baseline_transaction) == digest(variant_transaction),
    }
    if family == "disconnected_noise":
        noise_id = variant[-2].get("tool_use_id")
        integrity.update({
            "baseline_is_exact_prefix": variant[:-3] == baseline,
            "noise_transaction_complete": bool(
                noise_id
                and tool_transaction(variant, noise_id)
                and noise_id != target_id
            ),
            "noise_appended_after_baseline": len(variant) == len(baseline) + 3,
        })
    elif family == "transparent_wrapper":
        stripped = [{key: value for key, value in event.items() if key != WRAPPER_KEY} for event in variant]
        integrity.update({
            "all_events_wrapped": all(WRAPPER_KEY in event for event in variant),
            "core_events_unchanged": stripped == baseline,
        })
    else:
        integrity.update({
            "event_multiset_preserved": sorted(map(digest, baseline)) == sorted(map(digest, variant)),
            "only_evaluation_events_moved": all(
                baseline[index].get("type") in {"trace_end", "grading_result"}
                for index, (left, right) in enumerate(zip(baseline, variant)) if left != right
            ),
        })
    return integrity


def build(source_manifest: Path, output_dir: Path) -> dict[str, Any]:
    source_rows = read_jsonl(source_manifest)
    variant_dir = output_dir / "variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in source_rows:
        baseline_path = Path(source["clean_trace"])
        baseline = load_jsonl(baseline_path)
        target_id = source["target_action"]["tool_use_id"]
        baseline_sha = hashlib.sha256(canonical_bytes(baseline)).hexdigest()
        for family in FAMILIES:
            case_id = f"{source['case_id']}:{family}"
            variant, details = transform(baseline, family, source["case_id"])
            integrity = validate_transform(baseline, variant, family, target_id)
            if not all(integrity.values()):
                raise RuntimeError(f"integrity failure for {case_id}: {integrity}")
            family_dir = variant_dir / family
            family_dir.mkdir(parents=True, exist_ok=True)
            variant_path = family_dir / f"{source['case_id']}.jsonl"
            variant_sha = write_jsonl(variant_path, variant)
            rows.append({
                "case_id": case_id,
                "source_case_id": source["case_id"],
                "protocol": PROTOCOL,
                "family": family,
                "baseline_trace": str(baseline_path),
                "variant_trace": str(variant_path),
                "baseline_sha256": baseline_sha,
                "variant_sha256": variant_sha,
                "target_action": source["target_action"],
                "transform": details,
                "integrity": integrity,
                "expected": {
                    "normalized_apc_answer_invariant": True,
                    "raw_snapshot_changes": True,
                    "certificate_must_validate_against_own_snapshot": True,
                },
                "oracle_boundary": "raw-event transformation and target-transaction equality only; no graph or query output",
            })

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "protocol": PROTOCOL,
        "source_cases": len(source_rows),
        "pairs": len(rows),
        "pairs_per_family": {family: sum(row["family"] == family for row in rows) for family in FAMILIES},
        "integrity_passed": all(all(row["integrity"].values()) for row in rows),
        "query_output_used": False,
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
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/audit_intervention_v2_pilot/manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/apc_benign_invariance_v1"))
    args = parser.parse_args()
    print(json.dumps(build(args.source_manifest, args.output_dir), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
