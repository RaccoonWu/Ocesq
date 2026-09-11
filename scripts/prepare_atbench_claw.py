"""Audit and normalize ATBench-Claw trajectories.

The released ATBench-Claw data keeps OpenClaw-native session messages:
assistant tool calls appear as content parts with type ``toolCall`` and tool
observations appear as separate ``toolResult`` messages linked by
``toolCallId``.  This script preserves that source format while also emitting a
Claw-Eval-style event stream that the current behavior graph compiler can read.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_INPUT = Path("data/raw/atbench_claw/test.json")
DEFAULT_OUTPUT = Path("data/processed/behavior_governance/atbench_claw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-trace-jsonl", action="store_true", default=True)
    args = parser.parse_args()

    rows = _load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = audit_rows(rows)
    _write_json(args.output_dir / "schema_report.json", report)

    eventlog_path = args.output_dir / "eventlog.jsonl"
    trace_dir = args.output_dir / "traces_jsonl"
    if args.write_trace_jsonl:
        trace_dir.mkdir(parents=True, exist_ok=True)

    with eventlog_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows):
            episode_id = f"atbench_claw_{idx:04d}"
            normalized_events = normalize_episode(row, episode_id)
            out.write(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "source": "AI45Research/ATBench-Claw",
                        "labels": row.get("labels", {}),
                        "reason": row.get("reason", ""),
                        "trajectory_metadata": _trajectory_metadata(row.get("trajectory", {})),
                        "event_log": normalized_events,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if args.write_trace_jsonl:
                with (trace_dir / f"{episode_id}.jsonl").open("w", encoding="utf-8") as f:
                    for event in normalized_events:
                        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(f"rows: {len(rows)}")
    print(f"schema report: {args.output_dir / 'schema_report.json'}")
    print(f"eventlog jsonl: {eventlog_path}")
    if args.write_trace_jsonl:
        print(f"per-episode traces: {trace_dir}")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of rows in {path}")
    return data


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_types: collections.Counter[str] = collections.Counter()
    event_key_shapes: collections.Counter[str] = collections.Counter()
    message_roles: collections.Counter[str] = collections.Counter()
    message_key_shapes: collections.Counter[str] = collections.Counter()
    content_types: collections.Counter[str] = collections.Counter()
    content_key_shapes: collections.Counter[str] = collections.Counter()
    tool_names: collections.Counter[str] = collections.Counter()
    label_counts: dict[str, collections.Counter[Any]] = collections.defaultdict(collections.Counter)
    trajectory_versions: collections.Counter[str] = collections.Counter()
    trajectory_types: collections.Counter[str] = collections.Counter()
    event_counts: list[int] = []
    unmatched_tool_results = 0
    unresolved_tool_calls = 0

    for row in rows:
        trajectory = row.get("trajectory", {})
        trajectory_versions[str(trajectory.get("version"))] += 1
        trajectory_types[str(trajectory.get("type"))] += 1
        for key, value in (row.get("labels") or {}).items():
            label_counts[key][value] += 1

        events = trajectory.get("events") or []
        event_counts.append(len(events))
        seen_calls: set[str] = set()
        seen_results: set[str] = set()
        for event in events:
            event_types[str(event.get("type"))] += 1
            event_key_shapes[_shape(event)] += 1
            message = event.get("message") if isinstance(event, dict) else None
            if not isinstance(message, dict):
                continue
            message_roles[str(message.get("role"))] += 1
            message_key_shapes[_shape(message)] += 1
            if message.get("role") == "toolResult":
                tool_call_id = str(message.get("toolCallId", ""))
                seen_results.add(tool_call_id)
                tool_names[str(message.get("toolName"))] += 1

            for part in _content_parts(message):
                content_types[str(part.get("type"))] += 1
                content_key_shapes[f"{part.get('type')}:{_shape(part)}"] += 1
                if part.get("type") == "toolCall":
                    tool_call_id = str(part.get("id", ""))
                    seen_calls.add(tool_call_id)
                    tool_names[str(part.get("name"))] += 1
        unmatched_tool_results += len(seen_results - seen_calls)
        unresolved_tool_calls += len(seen_calls - seen_results)

    return {
        "row_count": len(rows),
        "trajectory_types": _counter_dict(trajectory_types),
        "trajectory_versions": _counter_dict(trajectory_versions),
        "events_per_episode": {
            "min": min(event_counts) if event_counts else 0,
            "max": max(event_counts) if event_counts else 0,
            "mean": round(mean(event_counts), 2) if event_counts else 0,
        },
        "event_types": _counter_dict(event_types),
        "event_key_shapes": _counter_dict(event_key_shapes),
        "message_roles": _counter_dict(message_roles),
        "message_key_shapes": _counter_dict(message_key_shapes),
        "content_types": _counter_dict(content_types),
        "content_key_shapes": _counter_dict(content_key_shapes),
        "tool_names_top50": _counter_dict(tool_names, limit=50),
        "labels": {key: _counter_dict(counter) for key, counter in label_counts.items()},
        "linkage": {
            "unmatched_tool_results": unmatched_tool_results,
            "unresolved_tool_calls": unresolved_tool_calls,
        },
        "schema_implications": [
            "OpenClaw source traces are session-message logs, not pre-separated action/observation logs.",
            "ToolAction should be derived from assistant content parts with type=toolCall.",
            "ToolObservation should be derived from role=toolResult messages joined by toolCallId.",
            "EventLog should preserve the raw source message while storing derived normalized events.",
            "Safety supervision is trajectory-level in ATBench-Claw; action-local labels must be derived or judge-labeled.",
        ],
    }


def normalize_episode(row: dict[str, Any], episode_id: str) -> list[dict[str, Any]]:
    trajectory = row.get("trajectory") or {}
    labels = row.get("labels") or {}
    normalized: list[dict[str, Any]] = [
        {
            "type": "trace_start",
            "trace_id": episode_id,
            "task_id": episode_id,
            "model": trajectory.get("modelId") or trajectory.get("model_id") or "",
            "source": "AI45Research/ATBench-Claw",
            "source_schema": "openclaw_session_v3",
            "trajectory_metadata": _trajectory_metadata(trajectory),
        }
    ]

    call_arguments: dict[str, Any] = {}
    for source_event_idx, event in enumerate(trajectory.get("events") or []):
        message = event.get("message") if isinstance(event, dict) else None
        if not isinstance(message, dict):
            normalized.append(
                {
                    "type": "raw_event",
                    "trace_id": episode_id,
                    "source_event_idx": source_event_idx,
                    "raw_event": event,
                }
            )
            continue

        content = normalize_message_content(message.get("content", []), call_arguments)
        normalized.append(
            {
                "type": "message",
                "trace_id": episode_id,
                "source_event_idx": source_event_idx,
                "message": {
                    "role": message.get("role", "unknown"),
                    "content": content,
                },
                "raw_message": message,
            }
        )

        if message.get("role") == "toolResult":
            tool_call_id = message.get("toolCallId")
            normalized.append(
                {
                    "type": "tool_dispatch",
                    "trace_id": episode_id,
                    "source_event_idx": source_event_idx,
                    "tool_use_id": tool_call_id,
                    "tool_name": message.get("toolName", "unknown_tool"),
                    "request_body": call_arguments.get(str(tool_call_id), {}),
                    "response_status": "error" if message.get("isError") else "ok",
                    "response_body": normalize_tool_result_body(message.get("content", [])),
                    "raw_message": message,
                }
            )

    normalized.append(
        {
            "type": "atbench_label",
            "trace_id": episode_id,
            "labels": labels,
            "reason": row.get("reason", ""),
        }
    )
    normalized.append(
        {
            "type": "grading_result",
            "trace_id": episode_id,
            "task_id": episode_id,
            "passed": bool(labels.get("is_safe")),
            "task_score": 1.0 if labels.get("is_safe") else 0.0,
            "scores": {
                "is_safe": labels.get("is_safe"),
                "defense_type": labels.get("defense_type"),
                "risk_source": labels.get("risk_source"),
                "failure_mode": labels.get("failure_mode"),
                "harm_type": labels.get("harm_type"),
            },
        }
    )
    return normalized


def normalize_message_content(content: Any, call_arguments: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in _content_parts({"content": content}):
        part_type = part.get("type")
        if part_type == "toolCall":
            tool_id = str(part.get("id", ""))
            arguments = part.get("arguments") or {}
            call_arguments[tool_id] = arguments
            parts.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": part.get("name", "unknown_tool"),
                    "input": arguments,
                }
            )
        elif part_type == "text":
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif part_type in {"warning", "metadata"}:
            parts.append({"type": "text", "text": f"[{part_type}] {part.get('text', '')}"})
        elif part_type == "thinking":
            parts.append(
                {
                    "type": "thinking",
                    "thinking": part.get("thinking", ""),
                    "thinkingSignature": part.get("thinkingSignature"),
                }
            )
        else:
            parts.append(part)
    return parts


def normalize_tool_result_body(content: Any) -> Any:
    text_chunks: list[str] = []
    for part in _content_parts({"content": content}):
        if part.get("type") == "text":
            text_chunks.append(str(part.get("text", "")))
    text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
    if text:
        parsed = _try_parse_json(text)
        if parsed is not None:
            return parsed
        return {"text": text}
    return {"content": content}


def _content_parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        return [part for part in content if isinstance(part, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _trajectory_metadata(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": trajectory.get("type"),
        "version": trajectory.get("version"),
        "provider": trajectory.get("provider"),
        "model_id": trajectory.get("modelId") or trajectory.get("model_id"),
    }


def _shape(value: dict[str, Any]) -> str:
    return ",".join(sorted(str(key) for key in value.keys()))


def _counter_dict(counter: collections.Counter[Any], limit: int | None = None) -> dict[str, int]:
    items = counter.most_common(limit)
    return {str(key): count for key, count in items}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
