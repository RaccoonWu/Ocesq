"""Analyze compiled behavior graph outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def analyze_output_dir(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    rows: list[dict[str, Any]] = []
    card_rows: list[dict[str, Any]] = []
    total_nodes = Counter()
    total_edges = Counter()
    total_candidates = Counter()
    total_signals = Counter()

    for graph_path in sorted(root.glob("*.graph.json")):
        task_id = graph_path.name.removesuffix(".graph.json")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        candidates = _read_json(root / f"{task_id}.candidates.json", default=[])
        cards = _read_cards(root / f"{task_id}.monitor_cards.jsonl")
        grading = graph.get("metadata", {}).get("grading_result", {})

        node_types = Counter(node.get("type") for node in graph.get("nodes", []))
        edge_types = Counter(edge.get("type") for edge in graph.get("edges", []))
        candidate_kinds = Counter(candidate.get("kind") for candidate in candidates)
        signal_types = Counter(
            signal.get("type")
            for card in cards
            for signal in card.get("structural_signals", [])
        )
        for card in cards:
            card_rows.append(_summarize_card(task_id, grading, card))

        total_nodes.update(node_types)
        total_edges.update(edge_types)
        total_candidates.update(candidate_kinds)
        total_signals.update(signal_types)

        not_passed = grading.get("passed") is False
        imperfect_score = grading.get("task_score") is not None and grading.get("task_score") < 1.0
        rows.append(
            {
                "task_id": task_id,
                "task_score": grading.get("task_score"),
                "passed": grading.get("passed"),
                "not_passed": not_passed,
                "imperfect_score": imperfect_score,
                "nodes": len(graph.get("nodes", [])),
                "edges": len(graph.get("edges", [])),
                "tool_actions": node_types.get("ToolAction", 0),
                "tool_observations": node_types.get("ToolObservation", 0),
                "external_effects": node_types.get("ExternalEffect", 0),
                "requirements": node_types.get("Requirement", 0),
                "output_assertions": node_types.get("OutputAssertion", 0),
                "cards": len(cards),
                "not_passed_hit": not_passed and len(cards) > 0,
                "imperfect_hit": imperfect_score and len(cards) > 0,
                "candidate_kinds": dict(candidate_kinds),
                "signal_types": dict(signal_types),
            }
        )

    not_passed_rows = [row for row in rows if row["not_passed"]]
    not_passed_hit_rows = [row for row in not_passed_rows if row["not_passed_hit"]]
    imperfect_rows = [row for row in rows if row["imperfect_score"]]
    imperfect_hit_rows = [row for row in imperfect_rows if row["imperfect_hit"]]
    return {
        "summary": {
            "tasks": len(rows),
            "not_passed_tasks": len(not_passed_rows),
            "not_passed_hits": len(not_passed_hit_rows),
            "not_passed_hit_rate": round(len(not_passed_hit_rows) / len(not_passed_rows), 4) if not_passed_rows else None,
            "imperfect_tasks": len(imperfect_rows),
            "imperfect_hits": len(imperfect_hit_rows),
            "imperfect_hit_rate": round(len(imperfect_hit_rows) / len(imperfect_rows), 4) if imperfect_rows else None,
            "total_cards": sum(row["cards"] for row in rows),
            "node_types": dict(total_nodes),
            "edge_types": dict(total_edges),
            "candidate_kinds": dict(total_candidates),
            "signal_types": dict(total_signals),
        },
        "rows": rows,
        "cards": sorted(card_rows, key=lambda row: (-row["priority_score"], row["task_id"], row["card_id"])),
    }


def write_analysis(analysis: dict[str, Any], output_dir: str | Path) -> None:
    root = Path(output_dir)
    (root / "graph_analysis_summary.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = analysis["rows"]
    if not rows:
        return
    fieldnames = [
        "task_id",
        "task_score",
        "passed",
        "not_passed",
        "not_passed_hit",
        "imperfect_score",
        "imperfect_hit",
        "cards",
        "tool_actions",
        "tool_observations",
        "external_effects",
        "requirements",
        "output_assertions",
        "nodes",
        "edges",
        "candidate_kinds",
        "signal_types",
    ]
    with (root / "graph_analysis_tasks.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    card_rows = analysis.get("cards", [])
    if card_rows:
        (root / "graph_analysis_cards.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in card_rows) + "\n",
            encoding="utf-8",
        )
        card_fields = [
            "task_id",
            "card_id",
            "priority",
            "priority_score",
            "candidate_kind",
            "risk_level",
            "signals",
            "binding_statuses",
            "coverage_gap_count",
            "max_missing_ratio",
            "task_score",
            "passed",
            "summary",
            "label",
            "label_notes",
        ]
        with (root / "graph_analysis_cards.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=card_fields)
            writer.writeheader()
            for row in card_rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in card_fields})


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_cards(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summarize_card(task_id: str, grading: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    candidate = card.get("candidate", {})
    context = card.get("behavior_context", {})
    signals = [signal.get("type") for signal in card.get("structural_signals", [])]
    bindings = context.get("parameter_bindings", [])
    binding_statuses = Counter(binding.get("binding_status") for binding in bindings)
    coverage_gaps = context.get("coverage_gaps", [])
    max_missing_ratio = max((gap.get("missing_ratio", 0.0) for gap in coverage_gaps), default=0.0)
    priority_score = _priority_score(candidate, signals, binding_statuses, coverage_gaps, grading)
    return {
        "task_id": task_id,
        "card_id": card.get("card_id"),
        "priority": _priority_label(priority_score),
        "priority_score": priority_score,
        "candidate_kind": candidate.get("kind"),
        "risk_level": candidate.get("risk_level"),
        "signals": signals,
        "binding_statuses": dict(binding_statuses),
        "coverage_gap_count": len(coverage_gaps),
        "max_missing_ratio": round(max_missing_ratio, 3),
        "task_score": grading.get("task_score"),
        "passed": grading.get("passed"),
        "summary": candidate.get("summary"),
        "payload": candidate.get("payload", {}),
        "behavior_context": context,
        "label": "",
        "label_notes": "",
    }


def _priority_score(
    candidate: dict[str, Any],
    signals: list[str],
    binding_statuses: Counter,
    coverage_gaps: list[dict[str, Any]],
    grading: dict[str, Any],
) -> int:
    score = 0
    kind = candidate.get("kind")
    if grading.get("passed") is False:
        score += 3
    task_score = grading.get("task_score")
    if isinstance(task_score, (int, float)) and task_score < 0.5:
        score += 2
    if kind == "effectful_tool_action":
        score += 2
    if kind == "requirement_coverage_gap":
        score += 1
    if "ambiguous_entity_binding" in signals:
        score += 4
    if "requirement_coverage_gap" in signals:
        score += 1
    if binding_statuses.get("ambiguous", 0):
        score += 3
    if binding_statuses.get("unmatched", 0):
        score += 1
    if any(gap.get("missing_ratio", 0) >= 0.5 for gap in coverage_gaps):
        score += 2
    return score


def _priority_label(score: int) -> str:
    if score >= 8:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze compiled OCESQ behavior graph outputs.")
    parser.add_argument("output_dir", help="Directory containing *.graph.json, *.candidates.json, and *.monitor_cards.jsonl.")
    parser.add_argument("--top-k", type=int, default=0, help="Print the top-k prioritized cards for quick review.")
    args = parser.parse_args(argv)

    analysis = analyze_output_dir(args.output_dir)
    write_analysis(analysis, args.output_dir)
    summary = analysis["summary"]
    print(
        "tasks={tasks} not_passed={not_passed_tasks} not_passed_hits={not_passed_hits} "
        "not_passed_hit_rate={not_passed_hit_rate} imperfect={imperfect_tasks} "
        "imperfect_hits={imperfect_hits} imperfect_hit_rate={imperfect_hit_rate} "
        "cards={total_cards}".format(**summary)
    )
    print("candidate_kinds=", json.dumps(summary["candidate_kinds"], ensure_ascii=False, sort_keys=True))
    print("signal_types=", json.dumps(summary["signal_types"], ensure_ascii=False, sort_keys=True))
    if args.top_k:
        print("top_cards:")
        for row in analysis.get("cards", [])[: args.top_k]:
            print(
                "{priority:6s} score={priority_score:<2d} task={task_id} "
                "kind={candidate_kind} signals={signals} passed={passed} task_score={task_score}".format(**row)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
