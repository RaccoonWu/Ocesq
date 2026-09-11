#!/usr/bin/env python3
"""Deterministic (no-LLM) consumer for the downstream fact-recovery inputs.

This script reuses the frozen blinded
representations produced by `build_downstream_fact_recovery_v1.py` and scores
them with the existing scorer `evaluate_downstream_fact_recovery_v1.py`. It
introduces NO language model, makes NO network calls, and is fully
deterministic and reproducible.

Motivation
----------
The paper reports that an LLM consumer recovers source-event IDs far better
from the COMPLETE-APC package than from the RAW-TRACE representation
(fault-source F1 0.214 -> 0.859 for DeepSeek, 0.240 -> 0.766 for Qwen). A
skeptical reviewer worries this conflates the package's value with LLM
competence. This script tests whether a *deterministic, rule-based* consumer
also recovers sources much better from the COMPLETE package than from the RAW
trace, which would attribute the gain to the data interface (contract-selected
paths + source pointers) rather than to LLM competence.

Representations (identified from private_condition_map.json, not from gold)
--------------------------------------------------------------------------
- raw_trace                 -> the RAW-TRACE form  (version_a/version_b = full
                              raw event streams, no target marker, no pointers).
- complete_audit_package    -> the COMPLETE-PACKAGE form (contract + selected
                              nodes/edges + retained fact ids; every support /
                              conflict fact carries `source_event_pointers` and
                              witness `node_ids`, and every node carries a
                              `source_event_idx`).
- apc_without_status        -> the COMPLETE package with the per-obligation
                              `status` field stripped (a controlled ablation
                              that the manifest reports for the LLM consumer).

Deterministic rules (documented; identical for every case; no family labels)
===========================================================================

COMPLETE-PACKAGE consumer (complete_audit_package, has `status`)
----------------------------------------------------------------
For every obligation *type* present in version_a and/or version_b contracts:
  - A_sup : set of source_event_pointers over version_a support_facts
  - A_con : set of source_event_pointers over version_a conflict_facts
  - B_sup, B_con : same for version_b
  - status_a, status_b : per-obligation normalized status (may be absent)
  - mc_a / mc_b : presence of a missing_certificate

A change is a *cause* transition (not a downstream consequence):
  (M) into-missing      : status_b == "missing" and status_a != "missing"
                          (or missing_certificate appeared in B while A had
                           non-empty support)
      -> predicted source += A_sup           # evidence that disappeared
                                       # (version_a index space; matches gold
                                       #  for deletion / removal families)
  (C) into-conflicting  : status_b == "conflicting" and status_a != "conflicting"
                          (or B gained conflict_facts not present in A)
      -> predicted source += B_con           # newly-conflicting evidence
                                       # (version_b index space; matches gold
                                       #  for replace / inject-conflict / stale)
  Transitions OUT of conflicting (conflicting -> supported) are treated as
  downstream consequences and do NOT contribute source pointers.

Outputs per case:
  - source_event_ids   : sorted union of cause pointers above (ints)
  - change_present      : True iff any cause transition exists
  - relation_status     : "missing" if any (M) cause; "conflicting" if any (C)
                          cause and no (M); else "unchanged"
  - changed_component   : mapped from the *cause* obligation type, priority
                          effect_evidence->effect,
                          entity_state_consistency/state_freshness->state,
                          target_entity_support->entity,
                          parameter_provenance/input_observation->provenance,
                          else "none"
  - absence_is_scope_bound : True iff any (M) cause (scope-bound absence)

APC-WITHOUT-STATUS consumer (apc_without_status, no `status`)
--------------------------------------------------------------
Identical to the COMPLETE consumer, except the cause transitions are detected
*structurally* (no status field is read):
  (M) into-missing      : B has no support_facts AND no conflict_facts for the
                          obligation while A had non-empty support_facts (lost
                          evidence), or a missing_certificate appeared in B
      -> predicted source += A_sup
  (C) into-conflicting  : B has conflict_facts whose source_event_pointers are
                          not a subset of A's conflict pointers (new conflict)
      -> predicted source += B_con
The relation_status / component are inferred from structure exactly as above.

RAW-TRACE consumer (raw_trace: full event lists, no target marker)
-----------------------------------------------------------------
The raw representation exposes only two event streams and does NOT mark a
target action. The deterministic consumer does a best-effort structural diff
of version_a.events vs version_b.events and localizes the changed evidence:
  - Align `tool_dispatch` events by `tool_use_id` across the two streams;
    align tool_use *declarations* (message parts of type "tool_use") by id.
  - For each tool_use_id present in both whose dispatch's
    (request_body, response_body) differs      -> MODIFIED  (version_b index)
  - tool_use_id present in version_a only         -> DELETED   (version_a index)
  - tool_use_id present in version_b only         -> INSERTED  (version_b index)
  - dispatch present in both with identical content but whose position moved
    across the trace                             -> MOVED     (version_b index)
  Predicted source_event_ids  = indices (in the matching version space) of the
    changed dispatch plus its tool_use declaration. Deleted -> version_a index;
    modified/inserted/moved -> version_b index.
  change_present = True iff any structural diff exists beyond completely
    identical content at every aligned position (i.e. ANY modified / inserted /
    deleted / moved dispatch OR any inserted/deleted non-dispatch event OR any
    pure reorder). This is intentionally honest: a raw diff cannot tell benign
    reorders / wrappers / noise from real faults, so benign cases are flagged.
  changed_component / relation_status : inferred from the kind of change
    (deleted->missing/provenance, modified-response-loss->missing/effect,
     modified-request->conflicting/entity, inserted->conflicting/entity,
     moved->conflicting/time; else unchanged/none).

NOTE: The RAW consumer does not consult the contract, the source pointers, or
the family label; it only diffs the raw events the representation exposes.
The key point: the COMPLETE package hands the consumer the deciding
evidence plus integer source pointers, while the RAW trace forces the consumer
to reconstruct both the target and the evidence from a bare event diff.

Scoring
-------
Predictions are written in the exact schema consumed by
`evaluate_downstream_fact_recovery_v1.py`, which is then invoked as a
subprocess (the scorer is NOT modified). Source F1 / exact / component-F1 /
benign-FP are computed by that frozen scorer against ground_truth.jsonl.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

CONDITIONS_OF_INTEREST = (
    "raw_trace",
    "complete_audit_package",
    "apc_without_status",
)

COMPONENT_BY_OBLIGATION = {
    "effect_evidence": "effect",
    "entity_state_consistency": "state",
    "state_freshness": "state",
    "target_entity_support": "entity",
    "parameter_provenance": "provenance",
    "input_observation": "provenance",
    "verification_evidence": "time",
    "channel_boundary": "entity",
    "output_grounding": "entity",
    "task_constraint": "entity",
    "approval_dependency": "state",
}

COMPONENT_PRIORITY = (
    "effect",
    "state",
    "entity",
    "time",
    "provenance",
    "none",
)

STATUS_LABELS = ("supported", "conflicting", "missing", "not_applicable", "unchanged", "unknown")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def fact_pointers(facts: list[dict[str, Any]]) -> set[int]:
    out: set[int] = set()
    for fact in facts or []:
        for value in fact.get("source_event_pointers", []) or []:
            if isinstance(value, int):
                out.add(value)
    return out


def obligation_maps(version: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ob in version.get("contract", {}).get("obligations", []):
        out[ob["type"]] = {
            "status": ob.get("status"),
            "support_ptrs": fact_pointers(ob.get("support_facts", [])),
            "conflict_ptrs": fact_pointers(ob.get("conflict_facts", [])),
            "has_missing_cert": ob.get("missing_certificate") is not None,
            "support_facts": ob.get("support_facts", []),
            "conflict_facts": ob.get("conflict_facts", []),
        }
    return out


def component_for(obligation_types: set[str]) -> str:
    components = {COMPONENT_BY_OBLIGATION.get(name, "none") for name in obligation_types}
    for comp in COMPONENT_PRIORITY:
        if comp in components:
            return comp
    return "none"


def predict_complete(version_a: dict[str, Any], version_b: dict[str, Any], *, has_status: bool) -> dict[str, Any]:
    """Deterministic contract-deciding-evidence consumer.

    When `has_status` is False the `status` field is ignored and cause
    transitions are detected structurally (used for apc_without_status).
    """
    obs_a = obligation_maps(version_a)
    obs_b = obligation_maps(version_b)
    types = sorted(set(obs_a) | set(obs_b))

    cause_missing: set[str] = set()
    cause_conflict: set[str] = set()
    source: set[int] = set()

    for name in types:
        a = obs_a.get(name)
        b = obs_b.get(name)
        if a is None:
            a = {"status": None, "support_ptrs": set(), "conflict_ptrs": set(),
                 "has_missing_cert": False, "support_facts": [], "conflict_facts": []}
        if b is None:
            b = {"status": None, "support_ptrs": set(), "conflict_ptrs": set(),
                 "has_missing_cert": False, "support_facts": [], "conflict_facts": []}

        into_missing = False
        into_conflict = False

        if has_status:
            sa, sb = a["status"], b["status"]
            if sb == "missing" and sa != "missing":
                into_missing = True
            if sb == "conflicting" and sa != "conflicting":
                into_conflict = True
            if sb != "conflicting" and sa == "conflicting":
                pass  # downstream consequence; ignore
        else:
            # structural cause detection (apc_without_status)
            if b["has_missing_cert"] and not a["has_missing_cert"] and a["support_ptrs"]:
                into_missing = True
            elif (not b["support_ptrs"] and not b["conflict_ptrs"]) and a["support_ptrs"] and (
                b["has_missing_cert"] or not a["has_missing_cert"]
            ):
                into_missing = True
            if b["conflict_ptrs"] and not b["conflict_ptrs"].issubset(a["conflict_ptrs"]):
                into_conflict = True
            elif b["conflict_ptrs"] and not a["conflict_ptrs"] and b["conflict_ptrs"]:
                into_conflict = True

        # Conflict-fact growth for the has_status path (covers stale cases where
        # status did not flip but new conflict evidence appeared).
        if has_status and b["conflict_ptrs"] and not b["conflict_ptrs"].issubset(a["conflict_ptrs"]):
            into_conflict = True
        # Missing-certificate appearance without a status flip in the has_status
        # path (covers removal cases where status already reported missing).
        if has_status and b["has_missing_cert"] and not b["support_ptrs"] and a["support_ptrs"]:
            if sb == "missing":
                into_missing = True

        if into_missing:
            cause_missing.add(name)
            source |= a["support_ptrs"]  # version_a index space
        if into_conflict:
            cause_conflict.add(name)
            source |= b["conflict_ptrs"]  # version_b index space

    change_present = bool(cause_missing) or bool(cause_conflict)
    if cause_missing:
        relation_status = "missing"
    elif cause_conflict:
        relation_status = "conflicting"
    else:
        relation_status = "unchanged"
    component = component_for(cause_missing | cause_conflict)
    return {
        "source_event_ids": sorted(source),
        "change_present": change_present,
        "relation_status": relation_status,
        "changed_component": component if change_present else "none",
        "absence_is_scope_bound": bool(cause_missing),
    }


def _dispatch_index(events: list[dict[str, Any]], tool_use_id: str) -> int | None:
    matches = [
        i for i, e in enumerate(events)
        if e.get("type") == "tool_dispatch" and e.get("tool_use_id") == tool_use_id
    ]
    if len(matches) == 1:
        return matches[0]
    return matches[0] if matches else None


def _tooluse_decl_index(events: list[dict[str, Any]], tool_use_id: str) -> int | None:
    matches = []
    for i, e in enumerate(events):
        if e.get("type") != "message":
            continue
        for part in (e.get("message", {}).get("content", []) or []):
            if isinstance(part, dict) and part.get("type") == "tool_use" and part.get("id") == tool_use_id:
                matches.append(i)
                break
    if len(matches) == 1:
        return matches[0]
    return matches[0] if matches else None


def _dispatch_signature(event: dict[str, Any]) -> tuple:
    return (
        json.dumps(event.get("request_body", {}), sort_keys=True),
        json.dumps(event.get("response_body", {}), sort_keys=True),
    )


def _event_canonical(event: dict[str, Any]) -> str:
    stripped = copy.deepcopy(event)
    stripped.pop("timestamp", None)
    stripped.pop("latency_ms", None)
    return json.dumps(stripped, sort_keys=True, ensure_ascii=False)


def predict_raw(version_a: dict[str, Any], version_b: dict[str, Any]) -> dict[str, Any]:
    """Deterministic raw-trace diff consumer (no target marker, no pointers)."""
    events_a = version_a.get("events", [])
    events_b = version_b.get("events", [])

    dispatch_ids_a = {
        e.get("tool_use_id"): i
        for i, e in enumerate(events_a) if e.get("type") == "tool_dispatch" and e.get("tool_use_id")
    }
    dispatch_ids_b = {
        e.get("tool_use_id"): i
        for i, e in enumerate(events_b) if e.get("type") == "tool_dispatch" and e.get("tool_use_id")
    }

    # Pure-reorder detection (events identical in content but order changed).
    canon_a = [_event_canonical(e) for e in events_a]
    canon_b = [_event_canonical(e) for e in events_b]
    reordered = canon_a != canon_b and sorted(canon_a) == sorted(canon_b)

    source: set[int] = set()
    cause_kind: str = "none"  # one of deleted|modified|inserted|moved|reorder|none
    deleted_ids = []
    inserted_ids = []
    modified_ids = []
    moved_ids = []

    for tu_id, idx_a in dispatch_ids_a.items():
        if tu_id not in dispatch_ids_b:
            deleted_ids.append(tu_id)
        else:
            idx_b = dispatch_ids_b[tu_id]
            ev_a = events_a[idx_a]
            ev_b = events_b[idx_b]
            if _dispatch_signature(ev_a) != _dispatch_signature(ev_b):
                modified_ids.append(tu_id)
            elif idx_a != idx_b:
                moved_ids.append(tu_id)
    for tu_id in dispatch_ids_b:
        if tu_id not in dispatch_ids_a:
            inserted_ids.append(tu_id)

    # Build predicted source pointers in the matching index space.
    if deleted_ids:
        cause_kind = "deleted"
        for tu_id in deleted_ids:
            di = dispatch_ids_a.get(tu_id)
            decl = _tooluse_decl_index(events_a, tu_id)
            if di is not None:
                source.add(int(di))
            if decl is not None:
                source.add(int(decl))
    if modified_ids:
        cause_kind = "modified" if cause_kind == "none" else cause_kind
        for tu_id in modified_ids:
            di = dispatch_ids_b.get(tu_id)
            decl = _tooluse_decl_index(events_b, tu_id)
            if di is not None:
                source.add(int(di))
            if decl is not None:
                source.add(int(decl))
    if inserted_ids:
        cause_kind = "inserted" if cause_kind == "none" else cause_kind
        for tu_id in inserted_ids:
            di = dispatch_ids_b.get(tu_id)
            decl = _tooluse_decl_index(events_b, tu_id)
            if di is not None:
                source.add(int(di))
            if decl is not None:
                source.add(int(decl))
    if moved_ids and cause_kind == "none":
        cause_kind = "moved"
        for tu_id in moved_ids:
            di = dispatch_ids_b.get(tu_id)
            decl = _tooluse_decl_index(events_b, tu_id)
            if di is not None:
                source.add(int(di))
            if decl is not None:
                source.add(int(decl))

    structural_change = (
        bool(deleted_ids) or bool(modified_ids) or bool(inserted_ids)
        or bool(moved_ids) or reordered or len(events_a) != len(events_b)
    )
    # Detect inserted/removed non-dispatch noise events (benign families).
    if not (deleted_ids or modified_ids or inserted_ids or moved_ids):
        if set(canon_a) != set(canon_b):
            structural_change = True

    change_present = structural_change

    if not change_present:
        return {
            "source_event_ids": [],
            "change_present": False,
            "relation_status": "unchanged",
            "changed_component": "none",
            "absence_is_scope_bound": False,
        }

    if cause_kind == "deleted":
        relation_status = "missing"
        component = "provenance"
    elif cause_kind == "modified":
        relation_status = "missing"
        component = "effect"
    elif cause_kind == "inserted":
        relation_status = "conflicting"
        component = "entity"
    elif cause_kind == "moved":
        relation_status = "conflicting"
        component = "time"
    else:
        relation_status = "unchanged"
        component = "none"
    return {
        "source_event_ids": sorted(source),
        "change_present": True,
        "relation_status": relation_status,
        "changed_component": component,
        "absence_is_scope_bound": relation_status == "missing",
    }


PREDICTORS = {
    "raw_trace": lambda a, b: predict_raw(a, b),
    "complete_audit_package": lambda a, b: predict_complete(a, b, has_status=True),
    "apc_without_status": lambda a, b: predict_complete(a, b, has_status=False),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=Path("results/current_downstream_fact_recovery_inputs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/deterministic_consumer_202607"),
    )
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    inputs = args.inputs_dir
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    condition_map = json.loads((inputs / "private_condition_map.json").read_text())
    alias = condition_map["condition_alias"]
    alias_to_condition = {v: k for k, v in alias.items()}

    records: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS_OF_INTEREST:
        rname = alias[condition]
        path = inputs / f"{rname}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing representation file: {path}")
        rows = read_jsonl(path)
        by_token = {}
        for row in rows:
            tok = row["case_token"]
            by_token[tok] = {
                "version_a": row["version_a"],
                "version_b": row["version_b"],
                "representation_id": row.get("representation_id"),
            }
        records[condition] = by_token

    tokens = sorted(set.intersection(*[set(v) for v in records.values()]))
    if not tokens:
        raise RuntimeError("no common case tokens across representations")

    headers = {
        "protocol": "deterministic-consumer-v1",
        "downstream_inputs": str(inputs),
        "condition_alias_used": {c: alias[c] for c in CONDITIONS_OF_INTEREST},
        "n_cases": len(tokens),
        "llm_used": False,
        "network_used": False,
        "rules": "see script header docstring",
        "seed": args.seed,
    }
    (output / "run_headers.json").write_text(json.dumps(headers, indent=2, sort_keys=True) + "\n")

    prediction_files: list[str] = []
    for condition in CONDITIONS_OF_INTEREST:
        predict = PREDICTORS[condition]
        by_token = records[condition]
        out_rows: list[dict[str, Any]] = []
        for tok in tokens:
            data = by_token[tok]
            pred = predict(data["version_a"], data["version_b"])
            out_rows.append({
                "case_token": tok,
                "source_event_ids": pred["source_event_ids"],
                "changed_component": pred["changed_component"],
                "relation_status": pred["relation_status"],
                "change_present": pred["change_present"],
                "absence_is_scope_bound": pred["absence_is_scope_bound"],
            })
        path = output / f"predictions_{condition}.jsonl"
        write_jsonl(path, out_rows)
        prediction_files.append(f"{condition}={path}")

    scorer = Path(__file__).resolve().parent / "evaluate_downstream_fact_recovery_v1.py"
    cmd = [
        sys.executable, str(scorer),
        "--ground-truth", str(inputs / "ground_truth.jsonl"),
        "--output-dir", str(output / "scored"),
        "--baseline", "raw_trace",
    ]
    for spec in prediction_files:
        cmd.extend(["--prediction", spec])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (output / "scorer_stdout.json").write_text(proc.stdout)
    if proc.stderr:
        (output / "scorer_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    # Emit a compact summary of the headline metrics.
    summary = json.loads(proc.stdout)
    reps = summary["representations"]
    report: dict[str, Any] = {"n_cases": len(tokens), "representations": {}}
    for name in (CONDITIONS_OF_INTEREST):
        if name in reps:
            r = reps[name]
            report["representations"][name] = {
                "source_F1_fault": r.get("source_event_f1_on_faults"),
                "source_exact_fault": r.get("source_event_exact_on_faults"),
                "component_macro_f1": r.get("changed_component_macro_f1"),
                "status_accuracy": r.get("status_accuracy"),
                "benign_FP": r.get("benign_false_positive_rate"),
                "source_F1_all": r.get("source_event_f1"),
            }
    if "raw_trace" in reps and "complete_audit_package" in reps:
        a = reps["raw_trace"].get("source_event_f1_on_faults") or 0.0
        b = reps["complete_audit_package"].get("source_event_f1_on_faults") or 0.0
        report["delta_source_F1_complete_minus_raw"] = b - a
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())