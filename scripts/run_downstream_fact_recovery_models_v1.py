#!/usr/bin/env python3
"""Run blinded downstream fact-recovery cases with OpenAI-compatible providers.

The runner is resumable and records errors without converting them into labels.
Formal result directories should use one frozen model and decoding configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ENV = Path(__file__).resolve().parents[1] / ".env"
SYSTEM_PROMPT = """You are a blinded fact-recovery auditor. Compare version A and version B only from the supplied representation. Identify changes in recorded facts, not whether the task succeeded or the action is safe, correct, appropriate, or compliant. Source event IDs must be integers explicitly supported by the representation. A missing conclusion is scope-bound, never a claim about the outside world. Return exactly one JSON object matching the requested output schema. Set rationale to an empty string. Do not output markdown or hidden reasoning."""
ALLOWED_STATUS = {"supported", "conflicting", "missing", "not_applicable", "unchanged", "unknown"}
ALLOWED_COMPONENT = {"provenance", "entity", "time", "effect", "state", "none"}


def load_env() -> dict[str, str]:
    values = {}
    for line in PROJECT_ENV.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


def env_value(env: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        for candidate in (key, key.replace("-", "_"), key.upper(), key.lower()):
            if env.get(candidate) or os.environ.get(candidate):
                return env.get(candidate) or os.environ.get(candidate)
    return None


def provider_config(provider: str, env: dict[str, str]) -> dict[str, str]:
    if provider == "deepseek":
        config = {
            "model": env_value(env, "deepseek_model", "DEEPSEEK_MODEL") or "deepseek-v4-flash",
            "base_url": env_value(env, "base_url", "DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            "api_key": env_value(env, "DEEPSEEK-API_KEY", "DEEPSEEK_API_KEY"),
        }
    elif provider == "qwen":
        config = {
            "model": env_value(env, "qwen_model", "DASHSCOPE_MODEL") or "qwen-plus",
            "base_url": env_value(env, "qwen_base_url", "DASHSCOPE_BASE_URL"),
            "api_key": env_value(env, "qwen_api_key", "DASHSCOPE_API_KEY"),
        }
    else:
        raise ValueError(f"unsupported provider: {provider}")
    if not all(config.values()):
        raise RuntimeError(f"missing .env configuration for {provider}")
    return {key: str(value) for key, value in config.items()}


def request_options(model: str) -> dict[str, Any] | None:
    if model.startswith("deepseek"):
        return {"thinking": {"type": "disabled"}}
    if model.startswith("qwen"):
        return {"enable_thinking": False}
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact_jsonl(path: Path, key: str = "case_token") -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    unique = {row[key]: row for row in rows if key in row}
    compacted = list(unique.values())
    if len(compacted) != len(rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in compacted:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return compacted


def usage_dict(usage: Any) -> dict[str, int]:
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def normalize_prediction(case_token: str, value: dict[str, Any]) -> dict[str, Any]:
    status = str(value.get("relation_status", "unknown")).lower().replace("-", "_")
    component = str(value.get("changed_component", "none")).lower().replace("-", "_")
    pointers = []
    for item in value.get("source_event_ids", []) if isinstance(value.get("source_event_ids", []), list) else []:
        if isinstance(item, int):
            pointers.append(item)
        elif isinstance(item, str) and item.strip().isdigit():
            pointers.append(int(item.strip()))
    return {
        "case_token": case_token,
        "change_present": bool(value.get("change_present", component != "none")),
        "relation_status": status if status in ALLOWED_STATUS else "unknown",
        "changed_component": component if component in ALLOWED_COMPONENT else "none",
        "source_event_ids": sorted(set(pointers)),
        "absence_is_scope_bound": bool(value.get("absence_is_scope_bound", False)),
        "confidence": min(1.0, max(0.0, float(value.get("confidence", 0.0) or 0.0))),
        "rationale": str(value.get("rationale", ""))[:500],
    }


def call_one(client: OpenAI, config: dict[str, str], row: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(row, ensure_ascii=False, separators=(",", ":"))},
        ],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body=request_options(config["model"]),
    )
    latency_ms = (time.perf_counter() - started) * 1000
    content = response.choices[0].message.content if response.choices else ""
    try:
        parsed = extract_json(content or "")
    except Exception as exc:
        # Preserve the bounded model response for diagnosing provider-side JSON
        # violations without repeating a long-input request blindly.
        snippet = (content or "")[:2000].replace("\n", "\\n")
        raise ValueError(f"{exc}; raw_response={snippet}") from exc
    prediction = normalize_prediction(row["case_token"], parsed)
    usage = usage_dict(response.usage)
    prediction.update({
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
        "latency_ms": latency_ms,
        "cost_usd": None,
    })
    return prediction


def process_row(
    client: OpenAI,
    config: dict[str, str],
    row: dict[str, Any],
    condition: str,
    alias: str,
    max_input_chars: int,
    max_output_tokens: int,
    retries: int,
) -> tuple[bool, dict[str, Any]]:
    input_chars = len(json.dumps(row, ensure_ascii=False))
    if input_chars > max_input_chars:
        return False, {
            "case_token": row["case_token"],
            "error": "input_above_frozen_limit",
            "input_chars": input_chars,
        }
    last_error = None
    for attempt in range(retries + 1):
        try:
            prediction = call_one(client, config, row, max_output_tokens)
            prediction.update({
                "provider": config["provider"],
                "model": config["model"],
                "condition": condition,
                "representation_id": alias,
                "input_chars": input_chars,
                "attempt": attempt + 1,
            })
            return True, prediction
        except Exception as exc:
            last_error = str(exc)[:1000]
            if attempt < retries:
                time.sleep(min(2 ** attempt, 4))
    return False, {
        "case_token": row["case_token"],
        "error": last_error,
        "input_chars": input_chars,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("deepseek", "qwen"), required=True)
    parser.add_argument("--condition", action="append", help="Private condition name; repeat as needed")
    parser.add_argument("--max-cases", type=int, default=183)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-input-chars", type=int, default=500000)
    parser.add_argument("--max-output-tokens", type=int, default=400)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    condition_map = json.loads((args.input_dir / "private_condition_map.json").read_text())["condition_alias"]
    selected = args.condition or list(condition_map)
    unknown = sorted(set(selected) - set(condition_map))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    env = load_env()
    config = provider_config(args.provider, env)
    config["provider"] = args.provider
    preflight = {
        "provider": args.provider,
        "model": config["model"],
        "conditions": selected,
        "max_cases": args.max_cases,
        "offset": args.offset,
        "max_input_chars": args.max_input_chars,
        "max_output_tokens": args.max_output_tokens,
        "temperature": 0,
        "workers": args.workers,
        "system_prompt": SYSTEM_PROMPT,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.provider}.preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=args.timeout_s,
        max_retries=0,
    )
    for condition in selected:
        alias = condition_map[condition]
        rows = read_jsonl(args.input_dir / f"{alias}.jsonl")[args.offset:args.offset + args.max_cases]
        output = args.output_dir / f"{args.provider}.{condition}.jsonl"
        errors = args.output_dir / f"{args.provider}.{condition}.errors.jsonl"
        completed = {row["case_token"] for row in compact_jsonl(output)}
        pending = [row for row in rows if row["case_token"] not in completed]
        with output.open("a", encoding="utf-8") as out, errors.open("a", encoding="utf-8") as err:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        process_row,
                        client,
                        config,
                        row,
                        condition,
                        alias,
                        args.max_input_chars,
                        args.max_output_tokens,
                        args.retries,
                    ): row["case_token"]
                    for row in pending
                }
                for future in as_completed(futures):
                    try:
                        ok, payload = future.result()
                    except BaseException as exc:
                        ok = False
                        payload = {
                            "case_token": futures[future],
                            "error": f"worker_exception:{type(exc).__name__}:{str(exc)[:800]}",
                        }
                    handle = out if ok else err
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
