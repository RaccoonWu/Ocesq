"""Build a stratified sample from ATBench-Claw high-impact action cards."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/processed/behavior_governance/atbench_claw/high_impact_actions.jsonl")
DEFAULT_OUTPUT = Path("data/processed/behavior_governance/atbench_claw/high_impact_actions_sample100.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=100)
    args = parser.parse_args()

    rows = _read_jsonl(args.input)
    sample = stratified_sample(rows, args.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(sample)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"sampled: {len(sample)}")
    print(f"output: {args.output}")
    print(f"summary: {summary_path}")


def stratified_sample(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        action = row.get("action", {})
        labels = row.get("labels", {})
        key = (
            str(action.get("mechanism")),
            str(action.get("channel")),
            str(labels.get("is_safe")),
        )
        groups[key].append(row)

    for group_rows in groups.values():
        group_rows.sort(key=_stable_row_key)

    sample: list[dict[str, Any]] = []
    seen: set[str] = set()
    ordered_keys = sorted(groups, key=lambda key: (-len(groups[key]), key))
    while len(sample) < size:
        added = False
        for key in ordered_keys:
            if not groups[key]:
                continue
            row = groups[key].pop(0)
            row_id = _stable_row_key(row)
            if row_id in seen:
                continue
            sample.append(row)
            seen.add(row_id)
            added = True
            if len(sample) >= size:
                break
        if not added:
            break
    return sample


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {
        "mechanisms": collections.Counter(),
        "channels": collections.Counter(),
        "is_safe": collections.Counter(),
        "failure_modes": collections.Counter(),
        "tools": collections.Counter(),
    }
    for row in rows:
        action = row.get("action", {})
        labels = row.get("labels", {})
        counters["mechanisms"][str(action.get("mechanism"))] += 1
        counters["channels"][str(action.get("channel"))] += 1
        counters["is_safe"][str(labels.get("is_safe"))] += 1
        counters["failure_modes"][str(labels.get("failure_mode"))] += 1
        counters["tools"][str(action.get("tool_name"))] += 1
    return {
        "rows": len(rows),
        "mechanisms": dict(counters["mechanisms"].most_common()),
        "channels": dict(counters["channels"].most_common()),
        "is_safe": dict(counters["is_safe"].most_common()),
        "failure_modes": dict(counters["failure_modes"].most_common()),
        "tools_top50": dict(counters["tools"].most_common(50)),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stable_row_key(row: dict[str, Any]) -> str:
    action = row.get("action", {})
    return f"{row.get('episode_id')}::{row.get('card_id')}::{action.get('tool_use_id')}"


if __name__ == "__main__":
    main()
