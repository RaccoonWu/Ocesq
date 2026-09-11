"""Export ATBench-Claw high-impact action cards for action-local labeling."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocesq.behavior_graph.trace_compiler import compile_trace_file  # noqa: E402


DEFAULT_RAW = Path("data/raw/atbench_claw/test.json")
DEFAULT_TRACES = Path("data/processed/behavior_governance/atbench_claw/traces_jsonl")
DEFAULT_OUTPUT = Path("data/processed/behavior_governance/atbench_claw/high_impact_actions.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-kind", default="effectful_tool_action")
    args = parser.parse_args()

    raw_rows = _load_json(args.raw)
    if not isinstance(raw_rows, list):
        raise ValueError(f"Expected list rows in {args.raw}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary = Summary()

    trace_paths = sorted(args.traces_dir.glob("atbench_claw_*.jsonl"))
    for trace_path in trace_paths:
        episode_idx = int(trace_path.stem.rsplit("_", 1)[-1])
        raw_row = raw_rows[episode_idx]
        result = compile_trace_file(trace_path)
        for card in result.cards:
            if card.candidate.kind != args.candidate_kind:
                continue
            row = _card_row(trace_path, episode_idx, raw_row, card)
            rows.append(row)
            summary.add(row)

    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.output.with_suffix(".summary.json")
    _write_json(summary_path, summary.to_dict())

    print(f"high-impact action cards: {len(rows)}")
    print(f"output: {args.output}")
    print(f"summary: {summary_path}")


class Summary:
    def __init__(self) -> None:
        self.total_cards = 0
        self.episodes: set[str] = set()
        self.unsafe_cards = 0
        self.safe_cards = 0
        self.tools: collections.Counter[str] = collections.Counter()
        self.mechanisms: collections.Counter[str] = collections.Counter()
        self.channels: collections.Counter[str] = collections.Counter()
        self.failure_modes: collections.Counter[str] = collections.Counter()
        self.risk_sources: collections.Counter[str] = collections.Counter()

    def add(self, row: dict[str, Any]) -> None:
        labels = row.get("labels") or {}
        action = row.get("action") or {}
        self.total_cards += 1
        self.episodes.add(str(row.get("episode_id")))
        self.unsafe_cards += int(labels.get("is_safe") is False)
        self.safe_cards += int(labels.get("is_safe") is True)
        self.tools[str(action.get("tool_name"))] += 1
        self.mechanisms[str(action.get("mechanism"))] += 1
        self.channels[str(action.get("channel"))] += 1
        self.failure_modes[str(labels.get("failure_mode"))] += 1
        self.risk_sources[str(labels.get("risk_source"))] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cards": self.total_cards,
            "episodes_with_cards": len(self.episodes),
            "unsafe_cards": self.unsafe_cards,
            "safe_cards": self.safe_cards,
            "tools_top50": dict(self.tools.most_common(50)),
            "mechanisms": dict(self.mechanisms.most_common()),
            "channels": dict(self.channels.most_common()),
            "failure_modes": dict(self.failure_modes.most_common()),
            "risk_sources": dict(self.risk_sources.most_common()),
        }


def _card_row(trace_path: Path, episode_idx: int, raw_row: dict[str, Any], card: Any) -> dict[str, Any]:
    action = card.current_action.get("attrs", {})
    trace_context = _trace_context(trace_path, action.get("tool_use_id"), card.current_action.get("source_event_idx"))
    return {
        "episode_id": f"atbench_claw_{episode_idx:04d}",
        "trace_file": str(trace_path),
        "card_id": card.card_id,
        "candidate_id": card.candidate.id,
        "candidate_kind": card.candidate.kind,
        "candidate_risk_level": card.candidate.risk_level,
        "action": {
            "node_id": card.current_action.get("id"),
            "tool_use_id": action.get("tool_use_id"),
            "tool_name": action.get("tool_name"),
            "mechanism": action.get("mechanism"),
            "risk_level": action.get("risk_level"),
            "channel": action.get("channel"),
            "approval_sensitive": action.get("approval_sensitive"),
            "external_visibility": action.get("external_visibility"),
            "input": action.get("input", {}),
        },
        "trace_context": trace_context,
        "behavior_context": card.behavior_context,
        "structural_signals": [asdict(signal) for signal in card.structural_signals],
        "labels": raw_row.get("labels", {}),
        "reason": raw_row.get("reason", ""),
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _trace_context(trace_path: Path, tool_use_id: str | None, action_source_event_idx: int | None) -> dict[str, Any]:
    events = [_load_json_line(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    user_text = ""
    action_idx = None
    for idx, event in enumerate(events):
        message = event.get("message", {})
        if not user_text and event.get("type") == "message" and message.get("role") == "user":
            user_text = _message_text(message)
        if event.get("type") == "message":
            for part in message.get("content", []) or []:
                if part.get("type") == "tool_use" and part.get("id") == tool_use_id:
                    action_idx = idx
    if action_idx is None and action_source_event_idx is not None:
        action_idx = next((idx for idx, event in enumerate(events) if event.get("source_event_idx") == action_source_event_idx), None)
    if action_idx is None:
        action_idx = 0

    # The runtime monitor decides before the current action is executed. Keep
    # enough prior context plus the assistant message containing the proposed
    # tool call, but exclude the current action's tool result and later events.
    start = max(0, action_idx - 12)
    end = min(len(events), action_idx + 1)
    return {
        "user_message_excerpt": _trim_text(user_text, 5000),
        "local_events": [_compact_event(event) for event in events[start:end]],
        "tool_result": None,
        "context_window": {
            "policy": "pre_action_only",
            "action_event_idx": action_idx,
            "start_event_idx": start,
            "end_event_idx_exclusive": end,
        },
    }


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("type") == "message":
        message = event.get("message", {})
        parts = []
        for part in message.get("content", []) or []:
            if part.get("type") == "tool_use":
                parts.append(
                    {
                        "type": "tool_use",
                        "id": part.get("id"),
                        "name": part.get("name"),
                        "input": _trim_value(part.get("input", {}), 1200),
                    }
                )
            elif part.get("type") in {"text", "thinking"}:
                parts.append({"type": part.get("type"), "text": _trim_text(part.get("text") or part.get("thinking") or "", 1200)})
        return {
            "type": "message",
            "role": message.get("role"),
            "source_event_idx": event.get("source_event_idx"),
            "content": parts,
        }
    if event.get("type") == "tool_dispatch":
        return {
            "type": "tool_dispatch",
            "source_event_idx": event.get("source_event_idx"),
            "tool_use_id": event.get("tool_use_id"),
            "tool_name": event.get("tool_name"),
            "request_body": _trim_value(event.get("request_body", {}), 1200),
            "response_status": event.get("response_status"),
            "response_body": _trim_value(event.get("response_body"), 1200),
        }
    return {key: event.get(key) for key in ("type", "source_event_idx", "trace_id") if key in event}


def _message_text(message: dict[str, Any]) -> str:
    chunks = []
    for part in message.get("content", []) or []:
        if part.get("type") == "text":
            chunks.append(str(part.get("text", "")))
    return "\n".join(chunks)


def _trim_value(value: Any, limit: int) -> Any:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return value
    return {"truncated_json": text[:limit] + "..."}


def _trim_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _load_json_line(line: str) -> dict[str, Any]:
    return json.loads(line)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
