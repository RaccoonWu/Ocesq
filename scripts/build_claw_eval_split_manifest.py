"""Create a leakage-safe Claw-Eval task split manifest.

This script only partitions task metadata. It does not run agents, collect
trajectories, or generate training examples.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TASKS_DIR = Path("examples/claw-eval/tasks")
DEFAULT_OUTPUT = Path("data/processed/behavior_governance/claw_eval_splits")
DEFAULT_RATIOS = {
    "train_seed": 0.4,
    "dev": 0.2,
    "test": 0.2,
    "closed_loop_official": 0.2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ratios", default="train_seed=0.4,dev=0.2,test=0.2,closed_loop_official=0.2")
    args = parser.parse_args()

    ratios = parse_ratios(args.ratios)
    tasks = load_tasks(args.tasks_dir)
    groups = group_tasks(tasks)
    split_groups = assign_groups(groups, ratios)
    manifest = build_manifest(args, ratios, tasks, groups, split_groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "manifest.json", manifest)
    for split_name, split_tasks in manifest["splits"].items():
        (args.output_dir / f"{split_name}_tasks.txt").write_text(
            "\n".join(task["task_id"] for task in split_tasks) + "\n",
            encoding="utf-8",
        )
        _write_json(args.output_dir / f"{split_name}_tasks.json", split_tasks)
    _write_readme(args.output_dir / "README.md", manifest)

    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


def parse_ratios(text: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        name, value = part.split("=", 1)
        ratios[name.strip()] = float(value)
    missing = set(DEFAULT_RATIOS) - set(ratios)
    if missing:
        raise ValueError(f"Missing split ratios: {sorted(missing)}")
    total = sum(ratios.values())
    if not 0.999 <= total <= 1.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    return ratios


def load_tasks(tasks_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for task_yaml in sorted(tasks_dir.glob("*/task.yaml")):
        raw = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
        task_id = str(raw.get("task_id") or task_yaml.parent.name)
        prompt = raw.get("prompt") or {}
        if not isinstance(prompt, dict):
            prompt = {}
        tools = raw.get("tools") or []
        services = raw.get("services") or []
        safety_checks = raw.get("safety_checks") or []
        expected_actions = raw.get("expected_actions") or []
        primary_dimensions = raw.get("primary_dimensions") or []
        tasks.append(
            {
                "task_id": task_id,
                "task_dir": str(task_yaml.parent),
                "task_name": raw.get("task_name"),
                "version": raw.get("version"),
                "category": raw.get("category"),
                "difficulty": raw.get("difficulty"),
                "tags": raw.get("tags") or [],
                "language": prompt.get("language"),
                "family_key": family_key(task_id),
                "task_type": task_type(task_id),
                "has_user_agent": bool((raw.get("user_agent") or {}).get("enabled")),
                "tools": [tool.get("name") for tool in tools if isinstance(tool, dict)],
                "services": [service.get("name") for service in services if isinstance(service, dict)],
                "safety_check_count": len(safety_checks),
                "expected_action_count": len(expected_actions),
                "primary_dimensions": primary_dimensions,
            }
        )
    return tasks


def family_key(task_id: str) -> str:
    head = task_type(task_id)
    suffix = task_id.split("_", 1)[1] if "_" in task_id else task_id
    suffix = re.sub(r"^(zh|en)_", "", suffix)
    return f"{head}:{suffix}"


def task_type(task_id: str) -> str:
    match = re.match(r"([A-Za-z]+)", task_id)
    return match.group(1)[0].upper() if match else "X"


def group_tasks(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for task in tasks:
        groups[task["family_key"]].append(task)
    return dict(groups)


def assign_groups(groups: dict[str, list[dict[str, Any]]], ratios: dict[str, float]) -> dict[str, list[str]]:
    total_tasks = sum(len(tasks) for tasks in groups.values())
    targets = {split: total_tasks * ratio for split, ratio in ratios.items()}
    split_groups: dict[str, list[str]] = {split: [] for split in ratios}
    split_counts = {split: 0 for split in ratios}

    ordered_groups = sorted(groups.items(), key=lambda item: stable_hash(item[0]))
    for group_key, group_tasks_ in ordered_groups:
        group_size = len(group_tasks_)
        split = min(
            ratios,
            key=lambda name: (
                split_counts[name] / targets[name] if targets[name] else float("inf"),
                split_counts[name],
                name,
            ),
        )
        split_groups[split].append(group_key)
        split_counts[split] += group_size
    return split_groups


def build_manifest(
    args: argparse.Namespace,
    ratios: dict[str, float],
    tasks: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    split_groups: dict[str, list[str]],
) -> dict[str, Any]:
    splits: dict[str, list[dict[str, Any]]] = {}
    task_to_split: dict[str, str] = {}
    family_to_split: dict[str, str] = {}

    for split_name, group_keys in split_groups.items():
        split_tasks: list[dict[str, Any]] = []
        for group_key in sorted(group_keys):
            family_to_split[group_key] = split_name
            for task in groups[group_key]:
                task_copy = dict(task)
                task_copy["split"] = split_name
                split_tasks.append(task_copy)
                task_to_split[task["task_id"]] = split_name
        split_tasks.sort(key=lambda task: task["task_id"])
        splits[split_name] = split_tasks

    family_split_violations = []
    for group_key, group_tasks_ in groups.items():
        observed = {task_to_split[task["task_id"]] for task in group_tasks_}
        if len(observed) > 1:
            family_split_violations.append({"family_key": group_key, "splits": sorted(observed)})

    summary = {
        "tasks_dir": str(args.tasks_dir),
        "output_dir": str(args.output_dir),
        "total_tasks": len(tasks),
        "total_families": len(groups),
        "ratios": ratios,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_family_counts": {name: len(split_groups[name]) for name in split_groups},
        "family_split_violations": family_split_violations,
        "by_split_category": {
            split: dict(collections.Counter(task.get("category") for task in rows).most_common())
            for split, rows in splits.items()
        },
        "by_split_language": {
            split: dict(collections.Counter(task.get("language") for task in rows).most_common())
            for split, rows in splits.items()
        },
        "by_split_task_type": {
            split: dict(collections.Counter(task.get("task_type") for task in rows).most_common())
            for split, rows in splits.items()
        },
        "by_split_user_agent": {
            split: dict(collections.Counter(str(task.get("has_user_agent")) for task in rows).most_common())
            for split, rows in splits.items()
        },
    }

    return {
        "schema_version": "claw_eval_split_manifest_v1",
        "summary": summary,
        "leakage_policy": {
            "train_seed": "only this split may be used for strong-actor seed trajectories and perturbation construction",
            "dev": "threshold tuning and error analysis only; no training or template mining after test is fixed",
            "test": "held-out report set; never used for training, perturbation seeds, or prompt development",
            "closed_loop_official": "reserved for closed-loop benchmark reporting; never used for dataset construction",
            "family_grouping": "tasks sharing a family_key are assigned to the same split",
        },
        "family_key_rule": "task_type + suffix after first underscore, e.g. T:customer_followup",
        "family_to_split": family_to_split,
        "task_to_split": task_to_split,
        "splits": splits,
    }


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_readme(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        "# Claw-Eval Split Manifest\n\n"
        "Leakage-safe task split manifest. This file partitions task metadata only; it does not contain trajectories or labels.\n\n"
        "```json\n" + json.dumps(manifest["summary"], ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
