#!/usr/bin/env python3
"""Convert AgentDoG ATBench conversations into Claw-Eval-style trace JSONL.

AgentDoG format:
  {id, tool_used: [...], contents: [[{role, thought?, action?, content?}, ...]], label, ...}

Target Claw-Eval format (per-episode JSONL, one event per line):
  trace_start → message (user/assistant with tool_use parts) → tool_dispatch (observations) → trace_end

Key: tool calls must be embedded in assistant messages as content parts with type="tool_use"
so that TraceCompiler._compile_message creates ToolAction nodes BEFORE _compile_tool_dispatch
links observations to them.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any


def parse_agent_action(action: str | dict) -> dict[str, Any]:
    """Parse an agent action into {type: tool_call|complete, ...}."""
    if isinstance(action, dict):
        return _parse_action_dict(action)
    if isinstance(action, str):
        action = action.strip()
        if action.startswith("Complete"):
            inner = action[len("Complete"):].strip()
            try:
                body = json.loads(inner)
            except json.JSONDecodeError:
                body = {"raw": inner}
            return {"type": "complete", "body": body}
        try:
            return _parse_action_dict(json.loads(action))
        except json.JSONDecodeError:
            return {"type": "unknown", "raw": action}
    return {"type": "unknown", "raw": str(action)}


def _parse_action_dict(d: dict) -> dict[str, Any]:
    if "name" in d:
        return {
            "type": "tool_call",
            "name": d["name"],
            "arguments": d.get("arguments", {}),
            "raw": d,
        }
    return {"type": "unknown", "raw": d}


def convert_episode(episode: dict, episode_id: str, episode_idx: int) -> list[dict[str, Any]]:
    """Convert one AgentDoG episode to Claw-Eval event stream."""
    events: list[dict[str, Any]] = []

    # trace_start
    events.append({
        "type": "trace_start",
        "trace_id": episode_id,
        "task_id": episode_id,
        "model": "agentdog_atbench",
        "source": "AgentDoG/ATBench",
        "source_schema": "agentdog_v1",
        "metadata": {
            "agentdog_id": episode.get("id"),
            "label": episode.get("label"),
            "risk_source": episode.get("risk_source", ""),
            "failure_mode": episode.get("failure_mode", ""),
        },
    })

    tool_call_counter = 0
    # Map from tool_call_id → request_body for later tool_dispatch events
    pending_calls: dict[str, dict[str, Any]] = {}
    event_idx = 1

    for turn in episode.get("contents", []):
        for msg in turn:
            role = msg.get("role", "")

            if role == "user":
                events.append({
                    "type": "message",
                    "trace_id": episode_id,
                    "source_event_idx": event_idx,
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": msg.get("content", "")}],
                    },
                })
                event_idx += 1

            elif role == "agent":
                thought = msg.get("thought", "")
                action_raw = msg.get("action", "")
                action = parse_agent_action(action_raw)

                # Build content parts for this assistant message
                content_parts: list[dict[str, Any]] = []

                if thought:
                    content_parts.append({"type": "text", "text": thought})

                if action["type"] == "tool_call":
                    tool_name = action["name"]
                    tool_call_id = f"call_{episode_idx:04d}_{tool_call_counter}"
                    tool_call_counter += 1

                    content_parts.append({
                        "type": "tool_use",
                        "id": tool_call_id,
                        "name": tool_name,
                        "input": action.get("arguments", {}),
                    })

                    # Remember for when the environment responds
                    pending_calls[tool_call_id] = {
                        "tool_name": tool_name,
                        "request_body": action.get("arguments", {}),
                    }

                elif action["type"] == "complete":
                    content_parts.append({
                        "type": "text",
                        "text": json.dumps(action.get("body", {}), ensure_ascii=False),
                    })

                # Emit assistant message with tool_use and/or text parts
                if content_parts:
                    events.append({
                        "type": "message",
                        "trace_id": episode_id,
                        "source_event_idx": event_idx,
                        "message": {
                            "role": "assistant",
                            "content": content_parts,
                        },
                    })
                    event_idx += 1

            elif role == "environment":
                content_str = msg.get("content", "")
                try:
                    env_result = json.loads(content_str) if isinstance(content_str, str) else content_str
                except json.JSONDecodeError:
                    env_result = {"raw": content_str}

                # Match to the most recent pending call
                if pending_calls:
                    # AgentDoG env responses are sequential, match to oldest pending
                    call_id, call_info = next(iter(pending_calls.items()))
                    del pending_calls[call_id]

                    events.append({
                        "type": "tool_dispatch",
                        "trace_id": episode_id,
                        "source_event_idx": event_idx,
                        "tool_use_id": call_id,
                        "tool_name": call_info["tool_name"],
                        "request_body": call_info["request_body"],
                        "response_status": "ok" if env_result.get("status") == "success" else "error",
                        "response_body": env_result.get("result", env_result),
                        "raw_response": env_result,
                    })
                    event_idx += 1

    # trace_end
    events.append({
        "type": "trace_end",
        "trace_id": episode_id,
        "finish_reason": "stop",
    })

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="AgentDoG test.jsonl file")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for trace JSONL files")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max episodes to convert (0=all)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output_dir / "traces_jsonl"
    trace_dir.mkdir(parents=True, exist_ok=True)

    episodes: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    if args.limit > 0:
        episodes = episodes[:args.limit]

    eventlog_path = args.output_dir / "eventlog.jsonl"
    stats = {
        "total_episodes": len(episodes),
        "total_events": 0,
        "tool_dispatch_count": 0,
        "message_count": 0,
        "tool_use_count": 0,
        "label_distribution": {"0": 0, "1": 0},
    }

    with open(eventlog_path, "w", encoding="utf-8") as elog:
        for idx, ep in enumerate(episodes):
            episode_id = f"agentdog_atbench_{idx:04d}"
            events = convert_episode(ep, episode_id, idx)

            trace_path = trace_dir / f"{episode_id}.jsonl"
            with open(trace_path, "w", encoding="utf-8") as tf:
                for event in events:
                    tf.write(json.dumps(event, ensure_ascii=False) + "\n")

            elog.write(json.dumps({
                "episode_id": episode_id,
                "source": "AgentDoG/ATBench",
                "agentdog_id": ep.get("id"),
                "label": ep.get("label"),
                "risk_source": ep.get("risk_source", ""),
                "failure_mode": ep.get("failure_mode", ""),
                "tool_count": len(ep.get("tool_used", [])),
                "event_count": len(events),
            }, ensure_ascii=False) + "\n")

            stats["total_events"] += len(events)
            for e in events:
                t = e.get("type", "")
                if t == "tool_dispatch":
                    stats["tool_dispatch_count"] += 1
                elif t == "message":
                    stats["message_count"] += 1
                    for part in e.get("message", {}).get("content", []) or []:
                        if part.get("type") == "tool_use":
                            stats["tool_use_count"] += 1
            label_str = str(ep.get("label", "0"))
            if label_str in stats["label_distribution"]:
                stats["label_distribution"][label_str] += 1

    stats_path = args.output_dir / "conversion_stats.json"
    with open(stats_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    print(f"Episodes converted: {stats['total_episodes']}")
    print(f"Total events: {stats['total_events']}")
    print(f"Tool dispatches (observations): {stats['tool_dispatch_count']}")
    print(f"Tool use parts (in messages): {stats['tool_use_count']}")
    print(f"Messages: {stats['message_count']}")
    print(f"Label distribution: {stats['label_distribution']}")
    print(f"Trace JSONL dir: {trace_dir}")
    print(f"Eventlog: {eventlog_path}")


if __name__ == "__main__":
    main()
