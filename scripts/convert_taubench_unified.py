"""Convert unified tau-bench trajectories into one OCESQ event JSONL per trace.

The unified export stores assistant tool calls as ``{name, args}`` and tool
responses as consecutive ``role=tool`` messages without call ids.  We pair
responses FIFO with pending assistant calls, preserving the actual arguments
as ``request_body`` so correspondence and entity contracts remain meaningful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _message_text(message: dict[str, Any]) -> str:
    return str(message.get("text") or message.get("content") or "")


def convert_row(row: dict[str, Any], ordinal: int) -> list[dict[str, Any]]:
    task = str(row.get("task") or f"unknown_{ordinal}")
    model = str(row.get("model") or "unknown_model")
    trace_id = f"taubench_{model}_{task}_{ordinal}"
    messages = row.get("messages") or []
    events: list[dict[str, Any]] = [
        {"type": "trace_start", "trace_id": trace_id, "task_id": task, "model": model}
    ]
    pending: list[tuple[str, str, dict[str, Any]]] = []
    counter = 0
    for message in messages:
        role = str(message.get("role") or "")
        if role == "system":
            continue
        if role == "assistant":
            parts: list[dict[str, Any]] = []
            text = _message_text(message)
            if text:
                parts.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or []:
                counter += 1
                tool_id = f"tool_use_{counter}"
                name = str(call.get("name") or "unknown_tool")
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                parts.append({"type": "tool_use", "id": tool_id, "name": name, "input": args})
                pending.append((tool_id, name, args))
            if parts:
                events.append({"type": "message", "message": {"role": "assistant", "content": parts}})
        elif role == "user" and pending:
            # Some model exports label the environment/tool response as a
            # user message.  A pending call makes the intended role
            # unambiguous because tau-bench emits one response immediately
            # after each call.
            tool_id, tool_name, request = pending.pop(0)
            raw = _message_text(message)
            try:
                response = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                response = {"raw": raw}
            events.append({
                "type": "tool_dispatch",
                "tool_use_id": tool_id,
                "tool_name": tool_name,
                "request_body": request,
                "response_body": response,
                "response_status": "error" if "error" in raw.lower() or message.get("is_error") else "success",
            })
        elif role == "user":
            text = _message_text(message)
            if text:
                events.append({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": text}]}})
        elif role == "tool":
            if pending:
                tool_id, tool_name, request = pending.pop(0)
            else:
                counter += 1
                tool_id, tool_name, request = f"orphan_tool_{counter}", str(message.get("name") or "unknown_tool"), {}
            raw = _message_text(message)
            try:
                response: Any = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                response = {"raw": raw}
            events.append({
                "type": "tool_dispatch",
                "tool_use_id": tool_id,
                "tool_name": tool_name,
                "request_body": request,
                "response_body": response,
                "response_status": "error" if message.get("is_error") else "success",
            })
    events.append({
        "type": "trace_end",
        "task_score": row.get("score"),
        "passed": row.get("success"),
        "scores": {"official_score": row.get("score"), "official_success": row.get("success")},
    })
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-traces", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8") as source:
        for ordinal, line in enumerate(source):
            if args.max_traces is not None and count >= args.max_traces:
                break
            row = json.loads(line)
            task = str(row.get("task") or f"unknown_{ordinal}")
            model = str(row.get("model") or "unknown_model").replace("/", "_")
            out = args.output_dir / f"{ordinal:04d}_{model}_{task}.json"
            out.write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in convert_row(row, ordinal)) + "\n", encoding="utf-8")
            count += 1
    print(json.dumps({"traces_written": count, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
