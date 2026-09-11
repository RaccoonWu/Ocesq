#!/usr/bin/env python3
"""Build additional downstream fact-recovery baselines.

Generates three new evaluation conditions beyond the existing 11 representations:

1. **deterministic_diff** — rule-based field comparison (no ML).
   Directly scored against ground truth. Proves whether the task is trivially
   solvable without any model.

2. **structured_json** — flat JSON table of structured event fields with
   source pointers, no contract statuses. Same information as APC but
   without the contract abstraction. Isolates "formatting" from "semantics".

3. **equal_token_raw** — raw trace truncated to match APC token count.
   Isolates "information density" from "information quality".

The deterministic diff can be evaluated immediately.  The model baselines
(structured_json, equal_token_raw) produce representation JSONL files that
feed into ``run_downstream_fact_recovery_models_v1.py``.

Usage::

    cd /path/to/repo
    source ../.venv/bin/activate

    python scripts/build_downstream_baselines_v2.py \
      --ground-truth results/current_downstream_fact_recovery_inputs/ground_truth.jsonl \
      --output-dir results/baselines_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Structured field extraction
# ---------------------------------------------------------------------------

STRUCTURED_EVENT_KEYS = {
    "type", "event_index", "source_event_idx", "timestamp",
    "tool_name", "mechanism", "channel",
    "parameters", "params", "arguments", "args",
    "entity_id", "entity_name", "target_id", "target_name",
    "state", "status", "result", "effect",
    "message_id", "customer_id", "ticket_id", "contact_id",
    "observation_id", "observation_value",
}

# Keys to skip because they contain free text
FREE_TEXT_KEYS = {"content", "text", "description", "notes", "summary", "body",
                  "message", "_raw", "_text", "rationale", "reasoning"}


def _extract_structured_fields(event: dict[str, Any]) -> dict[str, Any]:
    """Extract only structured (non-free-text) fields from one event."""
    result: dict[str, Any] = {}
    for key, value in event.items():
        if key in FREE_TEXT_KEYS:
            continue
        if isinstance(value, str) and len(value) > 200:
            continue  # long strings are likely free text
        if isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        elif isinstance(value, (list, dict)):
            result[key] = _extract_structured_value(value)
    return result


def _extract_structured_value(value: Any) -> Any:
    """Recursively extract structured values, truncating long strings."""
    if isinstance(value, dict):
        return {k: _extract_structured_value(v) for k, v in value.items()
                if k not in FREE_TEXT_KEYS}
    if isinstance(value, list):
        if len(value) > 10:
            return [_extract_structured_value(v) for v in value[:10]] + ["..."]
        return [_extract_structured_value(v) for v in value]
    if isinstance(value, str) and len(value) > 200:
        return value[:200] + "..."
    return value


def _event_key(event: dict[str, Any], fallback: int = -1) -> int:
    """Stable event ordering key. Uses explicit field or falls back to list position."""
    for k in ("event_index", "source_event_idx", "source_index", "index"):
        if k in event and isinstance(event[k], int):
            return event[k]
    return fallback


# ---------------------------------------------------------------------------
# Deterministic structured diff
# ---------------------------------------------------------------------------

def deterministic_diff(
    clean_events: list[dict[str, Any]],
    variant_events: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[int]]:
    """Compare clean vs variant events field-by-field.

    Uses list position as the event index (matching ground truth source_event_ids).
    Returns (diff_summary, changed_event_indices).
    """
    # Assign list-position indices
    clean_by_idx: dict[int, dict[str, Any]] = {}
    for i, ev in enumerate(clean_events):
        idx = _event_key(ev, fallback=i)
        clean_by_idx[idx] = _extract_structured_fields(ev)

    variant_by_idx: dict[int, dict[str, Any]] = {}
    for i, ev in enumerate(variant_events):
        idx = _event_key(ev, fallback=i)
        variant_by_idx[idx] = _extract_structured_fields(ev)

    # Find changes
    changed_fields: list[dict[str, Any]] = []
    changed_indices: set[int] = set()
    all_indices = sorted(set(list(clean_by_idx.keys()) + list(variant_by_idx.keys())))

    for idx in all_indices:
        clean = clean_by_idx.get(idx)
        variant = variant_by_idx.get(idx)

        if clean is None:
            changed_fields.append({"event_index": idx, "change": "added"})
            changed_indices.add(idx)
        elif variant is None:
            changed_fields.append({"event_index": idx, "change": "removed"})
            changed_indices.add(idx)
        else:
            # Compare field-by-field
            if json.dumps(clean, sort_keys=True, default=str) != json.dumps(variant, sort_keys=True, default=str):
                changed_fields.append({"event_index": idx, "change": "modified"})
                changed_indices.add(idx)

    return {
        "changed_fields": changed_fields,
        "changed_count": len(changed_fields),
        "changed_event_indices": sorted(changed_indices),
    }, sorted(changed_indices)


def score_deterministic_diff(
    predicted_indices: list[int],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Score the deterministic diff against ground truth source event IDs."""
    gt_ids = set(ground_truth.get("source_event_ids", []))
    pred_ids = set(predicted_indices)

    tp = len(gt_ids & pred_ids)
    fp = len(pred_ids - gt_ids)
    fn = len(gt_ids - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    comp_match = ground_truth.get("changed_component", "none") != "none"
    return {
        "case_token": ground_truth.get("case_token", ""),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "source_exact": int(tp == len(gt_ids) and fp == 0),
        "tp": tp, "fp": fp, "fn": fn,
        "gt_ids": sorted(gt_ids),
        "pred_ids": sorted(pred_ids),
    }


# ---------------------------------------------------------------------------
# Equal-token raw trace
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English text."""
    return len(text) // 4


def build_equal_token_raw(
    baseline_events: list[dict[str, Any]],
    variant_events: list[dict[str, Any]],
    target_action_id: str,
    apc_token_budget: int,
) -> tuple[str, str]:
    """Truncate raw traces to match APC token budget, keeping events near target."""
    def _find_target(events: list[dict[str, Any]]) -> int:
        for i, ev in enumerate(events):
            eid = ev.get("id") or ev.get("event_id") or ""
            if eid == target_action_id:
                return i
        # Try tool_dispatch
        for i, ev in enumerate(events):
            if ev.get("type") == "tool_dispatch":
                return i
        return len(events) // 2

    def _truncate(events: list[dict[str, Any]], budget: int) -> str:
        target_idx = _find_target(events)
        # Collect events in expanding window around target
        collected: list[dict[str, Any]] = []
        left, right = target_idx, target_idx + 1
        while left >= 0 or right < len(events):
            if left >= 0:
                collected.insert(0, events[left])
                left -= 1
            text = json.dumps(collected, ensure_ascii=False, default=str)
            if _estimate_tokens(text) >= budget:
                collected.pop(0)
                break
            if right < len(events):
                collected.append(events[right])
                right += 1
            text = json.dumps(collected, ensure_ascii=False, default=str)
            if _estimate_tokens(text) >= budget:
                collected.pop()
                break
            if left < 0 and right >= len(events):
                break
        return json.dumps(collected, ensure_ascii=False, default=str)

    base_text = _truncate(baseline_events, apc_token_budget)
    var_text = _truncate(variant_events, apc_token_budget)
    return base_text, var_text


# ---------------------------------------------------------------------------
# Structured JSON (flat, no contracts)
# ---------------------------------------------------------------------------

def build_structured_json(
    events: list[dict[str, Any]],
    target_tool_use_id: str,
    window_radius: int = 4,
) -> dict[str, Any]:
    """Build a flat structured JSON table from events near the target action.

    No OEG, no contracts, no graph structure — just structured fields with
    event indices.  Matches the information content of APC but without
    contract semantics.
    """
    # Find target action index
    target_idx = -1
    for i, ev in enumerate(events):
        eid = ev.get("id") or ev.get("event_id") or ""
        if eid == target_tool_use_id or ev.get("type") == "tool_dispatch":
            target_idx = i
            break
    if target_idx < 0:
        target_idx = len(events) // 2

    start = max(0, target_idx - window_radius)
    end = min(len(events), target_idx + window_radius + 1)

    window = []
    for i in range(start, end):
        ev = events[i]
        structured = _extract_structured_fields(ev)
        structured["_event_index"] = i
        structured["_is_target"] = (i == target_idx)
        window.append(structured)

    return {
        "target_action_index": target_idx,
        "window_radius": window_radius,
        "events": window,
        "total_events_in_trace": len(events),
    }


# ---------------------------------------------------------------------------
# Token estimator for APC package
# ---------------------------------------------------------------------------

def estimate_apc_tokens(version: dict[str, Any]) -> int:
    """Estimate token count of an APC representation."""
    text = json.dumps(version, ensure_ascii=False, default=str)
    return _estimate_tokens(text)


# ---------------------------------------------------------------------------
# Deterministic diff → model-compatible prediction
# ---------------------------------------------------------------------------

# Mapping from changed field names to component types
FIELD_TO_COMPONENT: dict[str, str] = {
    # entity
    "entity_id": "entity", "product_id": "entity", "customer_id": "entity",
    "target_id": "entity", "contact_id": "entity", "ticket_id": "entity",
    "name": "entity", "subject": "entity", "recipient": "entity",
    "entity_name": "entity", "target_name": "entity",
    # provenance
    "parameters": "provenance", "params": "provenance", "arguments": "provenance",
    "observation_id": "provenance", "source": "provenance",
    "input": "provenance", "context_evidence": "provenance",
    # time
    "timestamp": "time", "time": "time", "created_at": "time",
    "updated_at": "time", "latency_ms": "time",
    # effect
    "result": "effect", "response": "effect", "effect": "effect",
    "output": "effect", "response_body": "effect", "response_status": "effect",
    "request_body": "effect",
    # state
    "status": "state", "state": "state", "version": "state",
}

CONFLICT_KEYWORDS = {"product_id", "entity_id", "customer_id", "status", "state", "version"}


def _component_from_changes(changed_fields: list[dict[str, Any]]) -> str:
    """Map changed fields to a single component type."""
    components: set[str] = set()
    for cf in changed_fields:
        fields = cf.get("fields", [])
        if isinstance(fields, list):
            for f in fields:
                f_lower = f.lower().rsplit(".", 1)[-1]
                comp = FIELD_TO_COMPONENT.get(f_lower, "")
                if comp:
                    components.add(comp)
    if not components:
        return "none"
    # Priority: entity > provenance > state > effect > time
    for priority in ("entity", "provenance", "state", "effect", "time"):
        if priority in components:
            return priority
    return sorted(components)[0]


def _status_from_changes(changed_fields: list[dict[str, Any]]) -> str:
    """Infer relation status from the nature of changes."""
    if not changed_fields:
        return "unchanged"
    for cf in changed_fields:
        change_type = cf.get("change", "modified")
        fields = cf.get("fields", [])
        if isinstance(fields, list):
            for f in fields:
                f_lower = f.lower().rsplit(".", 1)[-1]
                if f_lower in CONFLICT_KEYWORDS:
                    return "conflicting"
        if change_type == "removed":
            return "missing"
    return "unknown"


def diff_as_model_prediction(
    clean_events: list[dict[str, Any]],
    variant_events: list[dict[str, Any]],
    case_token: str,
) -> dict[str, Any]:
    """Run deterministic diff and produce model-compatible prediction."""
    diff_summary, changed_indices = deterministic_diff(clean_events, variant_events)
    changed_fields = diff_summary.get("changed_fields", [])
    return {
        "case_token": case_token,
        "representation_id": "R14",
        "condition": "deterministic_diff",
        "provider": "rule",
        "model": "deterministic_structured_diff",
        "relation_status": _status_from_changes(changed_fields),
        "changed_component": _component_from_changes(changed_fields),
        "source_event_ids": sorted(changed_indices),
        "change_present": len(changed_indices) > 0,
        "absence_is_scope_bound": False,
        "confidence": 1.0,
        "rationale": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _score_predictions(
    predictions: list[dict[str, Any]],
    gt_by_token: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Score predictions against ground truth with the same metrics as model eval."""
    fault_cases = []
    benign_cases = []
    status_correct = 0
    component_correct = 0
    total = 0

    for pred in predictions:
        case_token = pred["case_token"]
        gt = gt_by_token.get(case_token)
        if not gt:
            continue
        total += 1

        # Status accuracy
        expected_status = gt.get("expected_relation_status", "")
        pred_status = pred.get("relation_status", "")
        if pred_status == expected_status:
            status_correct += 1

        # Component accuracy
        expected_comp = gt.get("changed_component", "none")
        pred_comp = pred.get("changed_component", "none")
        if pred_comp == expected_comp:
            component_correct += 1

        gt_ids = set(gt.get("source_event_ids", []))
        pred_ids = set(pred.get("source_event_ids", []))
        tp = len(gt_ids & pred_ids)
        fp = len(pred_ids - gt_ids)
        fn = len(gt_ids - pred_ids)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        entry = {
            "case_token": case_token,
            "status_correct": int(pred_status == expected_status),
            "component_correct": int(pred_comp == expected_comp),
            "source_exact": int(tp == len(gt_ids) and fp == 0),
            "source_f1": f1,
            "gt_ids": sorted(gt_ids),
            "pred_ids": sorted(pred_ids),
        }

        if gt.get("change_present", True):
            fault_cases.append(entry)
        else:
            # Benign: flagged if component != "none" or source IDs non-empty
            benign_cases.append(entry)

    n_fault = len(fault_cases) if fault_cases else 1
    n_benign = len(benign_cases) if benign_cases else 1

    return {
        "total": total,
        "total_fault": len(fault_cases),
        "total_benign": len(benign_cases),
        "status_accuracy": round(status_correct / total, 4) if total else 0,
        "component_accuracy": round(component_correct / total, 4) if total else 0,
        "source_exact": sum(c["source_exact"] for c in fault_cases),
        "source_exact_rate": round(sum(c["source_exact"] for c in fault_cases) / n_fault, 4) if fault_cases else 0,
        "source_f1_fault": round(sum(c["source_f1"] for c in fault_cases) / n_fault, 4) if fault_cases else 0,
        "benign_component_correct_rate": round(
            sum(c["component_correct"] for c in benign_cases) / n_benign, 4
        ),
        "benign_source_fp_rate": round(
            sum(1 for c in benign_cases if c.get("pred_ids")) / n_benign, 4
        ),
        # source_f1_benign: benign cases should have source_ids=[]; F1=1 if pred is empty
        "source_f1_benign": round(
            sum(1.0 for c in benign_cases if not c.get("pred_ids")) / n_benign, 4
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("results/current_downstream_fact_recovery_inputs"),
        help="Directory containing existing R*.jsonl and ground_truth.jsonl",
    )
    parser.add_argument(
        "--apc-representation", type=str, default="R4",
        help="Which representation to use as APC token budget reference (default: R4=complete_audit_package)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/baselines_v2"),
    )
    parser.add_argument("--window-radius", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt_path = args.input_dir / "ground_truth.jsonl"
    gt_rows = _read_jsonl(gt_path)
    gt_by_token: dict[str, dict[str, Any]] = {r["case_token"]: r for r in gt_rows}
    print(f"Loaded {len(gt_rows)} ground truth cases")

    # Load APC representation for token budget
    apc_path = args.input_dir / f"{args.apc_representation}.jsonl"
    apc_rows = _read_jsonl(apc_path)
    apc_by_token: dict[str, dict[str, Any]] = {r["case_token"]: r for r in apc_rows}
    print(f"Loaded {len(apc_rows)} APC ({args.apc_representation}) cases for token budget")

    # Load raw trace representation (R3) for the trace data
    raw_path = args.input_dir / "R3.jsonl"
    raw_rows = _read_jsonl(raw_path)
    raw_by_token: dict[str, dict[str, Any]] = {r["case_token"]: r for r in raw_rows}
    print(f"Loaded {len(raw_rows)} raw trace (R3) cases")

    # ---- Deterministic diff ----
    print("\n" + "=" * 60)
    print("Deterministic Structured Diff Baseline")
    print("=" * 60)

    diff_results: list[dict[str, Any]] = []
    skipped_diff = 0

    for case_token, gt in gt_by_token.items():
        raw_row = raw_by_token.get(case_token)
        if not raw_row:
            skipped_diff += 1
            continue

        # Parse events from raw trace representation
        try:
            version_a = raw_row.get("version_a", {})
            version_b = raw_row.get("version_b", {})
            # Raw trace stores events as a field in the representation
            events_a = version_a if isinstance(version_a, list) else version_a.get("events", [])
            events_b = version_b if isinstance(version_b, list) else version_b.get("events", [])

            if not isinstance(events_a, list) or not isinstance(events_b, list):
                skipped_diff += 1
                continue
        except Exception:
            skipped_diff += 1
            continue

        diff_summary, predicted = deterministic_diff(events_a, events_b)
        scores = score_deterministic_diff(predicted, gt)
        scores["diff_summary"] = diff_summary
        diff_results.append(scores)

    # Aggregate diff metrics
    if diff_results:
        avg_f1 = sum(r["f1"] for r in diff_results) / len(diff_results)
        avg_prec = sum(r["precision"] for r in diff_results) / len(diff_results)
        avg_rec = sum(r["recall"] for r in diff_results) / len(diff_results)
        exact_count = sum(r["source_exact"] for r in diff_results)
        benign_fp = sum(1 for r in diff_results if r["fp"] > 0 and r["gt_ids"] == [])

        print(f"Cases evaluated:  {len(diff_results)}")
        print(f"Skipped:          {skipped_diff}")
        print(f"Avg Precision:    {avg_prec:.4f}")
        print(f"Avg Recall:       {avg_rec:.4f}")
        print(f"Avg F1:           {avg_f1:.4f}")
        print(f"Source Exact:     {exact_count}/{len(diff_results)} ({exact_count/len(diff_results):.3f})")
        print(f"Benign FP rate:   {benign_fp}/{sum(1 for r in gt_rows if not r.get('change_present', True))}")

        diff_summary = {
            "method": "deterministic_structured_diff",
            "cases": len(diff_results),
            "avg_precision": round(avg_prec, 4),
            "avg_recall": round(avg_rec, 4),
            "avg_f1": round(avg_f1, 4),
            "source_exact_rate": round(exact_count / len(diff_results), 4),
            "per_case": diff_results,
        }
        (args.output_dir / "deterministic_diff_results.json").write_text(
            json.dumps(diff_summary, ensure_ascii=False, indent=2))

    # ---- R14: Deterministic diff as model-compatible prediction ----
    print("\n" + "=" * 60)
    print("R14: Deterministic Diff as Model-Compatible Prediction")
    print("=" * 60)

    diff_predictions: list[dict[str, Any]] = []
    for case_token, gt in gt_by_token.items():
        raw_row = raw_by_token.get(case_token)
        if not raw_row:
            continue
        version_a = raw_row.get("version_a", {})
        version_b = raw_row.get("version_b", {})
        events_a = version_a if isinstance(version_a, list) else version_a.get("events", [])
        events_b = version_b if isinstance(version_b, list) else version_b.get("events", [])
        if not isinstance(events_a, list) or not isinstance(events_b, list):
            continue
        pred = diff_as_model_prediction(events_a, events_b, case_token)
        diff_predictions.append(pred)

    # Score against ground truth using the same three-dimensional metrics
    r14_scores = _score_predictions(diff_predictions, gt_by_token)
    r14_path = args.output_dir / "R14_predictions.jsonl"
    with open(r14_path, "w") as fh:
        for pred in diff_predictions:
            fh.write(json.dumps(pred, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "R14_scores.json").write_text(
        json.dumps(r14_scores, ensure_ascii=False, indent=2))
    print(f"Predictions: {r14_path} ({len(diff_predictions)} cases)")
    print(f"Status accuracy:   {r14_scores['status_accuracy']:.4f}")
    print(f"Component accuracy: {r14_scores['component_accuracy']:.4f}")
    print(f"Source exact:      {r14_scores['source_exact']}/{r14_scores['total_fault']} = {r14_scores['source_exact_rate']:.4f}")
    print(f"Source F1 (fault):  {r14_scores['source_f1_fault']:.4f}")
    print(f"Benign comp correct: {r14_scores['benign_component_correct_rate']:.4f}")
    print(f"Benign source FP:    {r14_scores['benign_source_fp_rate']:.4f}")

    # ---- Structured JSON representation ----
    print("\n" + "=" * 60)
    print("Structured JSON Baseline (no contracts)")
    print("=" * 60)

    structured_rows: list[dict[str, Any]] = []
    representation_id = "R12"  # structured_json_no_contracts

    for case_token, gt in gt_by_token.items():
        raw_row = raw_by_token.get(case_token)
        apc_row_s = apc_by_token.get(case_token) or {}
        if not raw_row:
            continue

        version_a = raw_row.get("version_a", {})
        version_b = raw_row.get("version_b", {})

        # Extract the actual event lists from the raw trace representation
        events_a = version_a if isinstance(version_a, list) else version_a.get("events", [])
        events_b = version_b if isinstance(version_b, list) else version_b.get("events", [])

        if not isinstance(events_a, list) or not isinstance(events_b, list):
            continue

        # Target action ID from APC representation (R4), which correctly identifies it
        target_id = apc_row_s.get("root_action_id", "")
        if not target_id:
            apc_va = apc_row_s.get("version_a", {})
            if isinstance(apc_va, dict):
                target_id = apc_va.get("root_action_id", "") or apc_va.get("contract", {}).get("root_action_id", "")

        struct_a = build_structured_json(events_a, target_id, args.window_radius)
        struct_b = build_structured_json(events_b, target_id, args.window_radius)

        structured_rows.append({
            "case_token": case_token,
            "representation_id": representation_id,
            "task": "Compare version A with version B for the target action. Report only recorded-fact changes; do not judge whether the action is safe, correct, or compliant.",
            "output_schema": {
                "change_present": "boolean",
                "changed_component": "provenance|entity|time|effect|state|none",
                "relation_status": "supported|conflicting|missing|not_applicable|unchanged|unknown",
                "source_event_ids": "list of integer event indices from the representation",
                "absence_is_scope_bound": "boolean",
                "confidence": "number from 0 to 1",
                "rationale": "empty string; not evaluated",
            },
            "version_a": struct_a,
            "version_b": struct_b,
        })

    structured_path = args.output_dir / f"{representation_id}.jsonl"
    with open(structured_path, "w") as fh:
        for row in structured_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(structured_rows)} cases to {structured_path}")
    print(f"  → Run models with: --representation {representation_id}")

    # ---- Equal-token raw trace ----
    print("\n" + "=" * 60)
    print("Equal-Token Raw Trace Baseline")
    print("=" * 60)

    eq_token_rows: list[dict[str, Any]] = []
    representation_id_eq = "R13"  # equal_token_raw_trace

    for case_token, gt in gt_by_token.items():
        raw_row = raw_by_token.get(case_token)
        apc_row = apc_by_token.get(case_token)
        if not raw_row or not apc_row:
            continue

        version_a = raw_row.get("version_a", {})
        version_b = raw_row.get("version_b", {})

        events_a = version_a if isinstance(version_a, list) else version_a.get("events", [])
        events_b = version_b if isinstance(version_b, list) else version_b.get("events", [])

        if not isinstance(events_a, list) or not isinstance(events_b, list):
            continue

        # Estimate APC token budget from complete audit package
        apc_va = apc_row.get("version_a", {})
        apc_text = json.dumps(apc_va, ensure_ascii=False, default=str) if apc_va else "{}"
        token_budget = _estimate_tokens(apc_text)

        # Target action ID from APC representation (R4)
        target_id_eq = apc_row.get("root_action_id", "")
        if not target_id_eq:
            apc_va_eq = apc_row.get("version_a", {})
            if isinstance(apc_va_eq, dict):
                target_id_eq = apc_va_eq.get("root_action_id", "") or apc_va_eq.get("contract", {}).get("root_action_id", "")

        eq_a, eq_b = build_equal_token_raw(events_a, events_b, target_id_eq, token_budget)

        eq_token_rows.append({
            "case_token": case_token,
            "representation_id": representation_id_eq,
            "task": "Compare version A with version B for the target action. Report only recorded-fact changes; do not judge whether the action is safe, correct, or compliant.",
            "output_schema": {
                "change_present": "boolean",
                "changed_component": "provenance|entity|time|effect|state|none",
                "relation_status": "supported|conflicting|missing|not_applicable|unchanged|unknown",
                "source_event_ids": "list of integer event indices from the representation",
                "absence_is_scope_bound": "boolean",
                "confidence": "number from 0 to 1",
                "rationale": "empty string; not evaluated",
            },
            "version_a": eq_a,
            "version_b": eq_b,
        })

    eq_path = args.output_dir / f"{representation_id_eq}.jsonl"
    with open(eq_path, "w") as fh:
        for row in eq_token_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(eq_token_rows)} cases to {eq_path}")
    print(f"  → Run models with: --representation {representation_id_eq}")

    # ---- Summary ----
    summary = {
        "protocol": "downstream-baselines-v2",
        "deterministic_diff_source_only": {
            "cases_evaluated": len(diff_results),
            "avg_f1": round(sum(r["f1"] for r in diff_results) / len(diff_results), 4) if diff_results else None,
            "note": "source_event_ids only, naive field comparison",
        },
        "deterministic_diff_full": {
            "representation_id": "R14",
            "cases": r14_scores["total"],
            "status_accuracy": r14_scores["status_accuracy"],
            "component_accuracy": r14_scores["component_accuracy"],
            "source_exact_rate": r14_scores["source_exact_rate"],
            "source_f1_fault": r14_scores["source_f1_fault"],
            "benign_component_correct": r14_scores["benign_component_correct_rate"],
            "benign_source_fp": r14_scores["benign_source_fp_rate"],
            "note": "Three-dimensional evaluation (status+component+source), no ML",
            "needs_model": False,
        },
        "structured_json": {
            "representation_id": "R12",
            "cases_generated": len(structured_rows),
            "description": "Flat structured JSON table, source pointers, no contracts",
            "needs_model": True,
        },
        "equal_token_raw": {
            "representation_id": "R13",
            "cases_generated": len(eq_token_rows),
            "description": "Raw trace truncated to APC token budget",
            "needs_model": True,
        },
        "model_run_instructions": """
To run models on R12 and R13:

    python scripts/run_downstream_fact_recovery_models_v1.py \\
      --input-dir results/baselines_v2 \\
      --representation R12 --representation R13 \\
      --provider deepseek --output-dir results/baselines_v2/deepseek

R14 (deterministic diff) is already scored — no model needed.
Merge with existing current_downstream_fact_recovery_inputs results for the
complete comparison table.
""",
    }

    summary_path = args.output_dir / "baselines_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSummary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
