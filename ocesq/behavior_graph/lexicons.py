"""Lexicon and tool-registry resources for the behavior-graph pipeline.

Word lists, keyword sets, alias maps, and the tool-semantics override registry
live as data under ``resources/`` rather than as literals scattered across
modules.  Keeping the vocabulary in reviewable JSON files lets the algorithms
stay readable and lets the term sets be extended or audited without editing
code.

Every name here is a frozen, read-only view of the JSON.  Membership tests and
iteration behave exactly as before; callers must not mutate the returned
containers.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_RESOURCES = Path(__file__).resolve().parent / "resources"


@lru_cache(maxsize=None)
def _load(name: str) -> dict[str, Any]:
    with (_RESOURCES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


_LEX = _load("lexicons.json")
_TOOL = _load("tool_semantics.json")

# --- trace compilation lexicons -------------------------------------------
IDENTITY_KEYS: tuple[str, ...] = tuple(_LEX["identity_keys"])
ANCHOR_KEYWORDS: tuple[str, ...] = tuple(_LEX["anchor_keywords"])
FREE_TEXT_KEYS: frozenset[str] = frozenset(_LEX["free_text_keys"])
FREE_TEXT_PARAMETERS: frozenset[str] = FREE_TEXT_KEYS
FREE_TEXT_PARAMETER_MARKERS: tuple[str, ...] = tuple(_LEX["free_text_parameter_markers"])
ALWAYS_REVIEW_EFFECTS: frozenset[str] = frozenset(_LEX["always_review_effects"])
TEXT_SCAN_PARAMETERS: frozenset[str] = frozenset(_LEX["text_scan_parameters"])
TEXT_SCAN_PARAMETER_MARKERS: tuple[str, ...] = tuple(_LEX["text_scan_parameter_markers"])
STOPWORDS: frozenset[str] = frozenset(_LEX["stopwords"])
CONCEPT_ALIASES: dict[str, str] = dict(_LEX["concept_aliases"])
CRITICAL_REQUIREMENT_CONCEPTS: frozenset[str] = frozenset(
    _LEX["critical_requirement_concepts"]
)
ENTITY_TYPE_MAP: dict[str, str] = dict(_LEX["entity_type_map"])
ANCHOR_ALIASES: dict[str, tuple[str, ...]] = {
    key: tuple(values) for key, values in _LEX["anchor_aliases"].items()
}
ENTITY_REFERENCE_PATTERNS: list[tuple[re.Pattern[str], str, int]] = [
    (re.compile(item["pattern"]), item["label"], item["priority"])
    for item in _LEX["entity_reference_patterns"]
]

# --- contract evaluation lexicons -----------------------------------------
HIGH_IMPACT_MECHANISMS: frozenset[str] = frozenset(_LEX["high_impact_mechanisms"])
VERSION_SENSITIVE_MECHANISMS: frozenset[str] = frozenset(
    _LEX["version_sensitive_mechanisms"]
)
CONTENT_DEPENDENT_MECHANISMS: frozenset[str] = frozenset(
    _LEX["content_dependent_mechanisms"]
)
VERIFICATION_SENSITIVE_MECHANISMS: frozenset[str] = frozenset(
    _LEX["verification_sensitive_mechanisms"]
)
SAVE_DRAFT_WEAK_ANCHOR_KEYS: frozenset[str] = frozenset(
    _LEX["save_draft_weak_anchor_keys"]
)
SAVE_DRAFT_STRONG_ANCHOR_KEYS: frozenset[str] = frozenset(
    _LEX["save_draft_strong_anchor_keys"]
)
TASK_AUTHORIZATION_TERMS: dict[str, tuple[str, ...]] = {
    mechanism: tuple(terms)
    for mechanism, terms in _LEX["task_authorization_terms"].items()
}

# --- tool semantics / optional judge utility ------------------------------
URL_PARAM_KEYS: frozenset[str] = frozenset(_LEX["url_param_keys"])
JUDGE_LABELS: frozenset[str] = frozenset(_LEX["judge_labels"])

# --- tool-semantics tables and override registry --------------------------
# (keyword, mechanism, edge_type, risk_level, channel)
EFFECT_KEYWORDS: tuple[tuple[str, str, str, str, str | None], ...] = tuple(
    (keyword, mechanism, edge_type, risk, channel)
    for keyword, mechanism, edge_type, risk, channel in _TOOL["effect_keywords"]
)
READ_KEYWORDS: tuple[str, ...] = tuple(_TOOL["read_keywords"])
READ_ONLY_PREFIXES: tuple[str, ...] = tuple(_TOOL["read_only_prefixes"])
# (keyword, channel)
CHANNEL_KEYWORDS: tuple[tuple[str, str], ...] = tuple(
    (keyword, channel) for keyword, channel in _TOOL["channel_keywords"]
)
EXTERNAL_DATA_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in _TOOL["external_data_patterns"]
)
EXTERNAL_DATA_KEYWORDS: tuple[str, ...] = tuple(_TOOL["external_data_keywords"])

# Raw constructor arguments; ``tool_semantics`` builds ToolSemantics objects
# from these after defining the dataclass (avoids an import cycle).
EXACT_OVERRIDE_SPECS: dict[str, dict[str, Any]] = dict(_TOOL["exact_overrides"])
PREFIX_OVERRIDE_SPECS: tuple[tuple[str, dict[str, Any]], ...] = tuple(
    (prefix, dict(spec)) for prefix, spec in _TOOL["prefix_overrides"]
)
