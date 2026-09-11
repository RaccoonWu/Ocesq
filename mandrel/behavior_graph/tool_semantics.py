"""Mechanism-level tool semantics for workspace/service agents.

Classification is **soft**: the registry provides overrides for known tools,
but unknown tools are classified automatically from name keywords, parameter
shape, and response patterns.  No per-dataset hardcoding required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .lexicons import (
    CHANNEL_KEYWORDS as _CHANNEL_KEYWORDS,
    EFFECT_KEYWORDS as _EFFECT_KEYWORDS,
    EXACT_OVERRIDE_SPECS,
    EXTERNAL_DATA_KEYWORDS as _EXTERNAL_DATA_KEYWORDS,
    EXTERNAL_DATA_PATTERNS as _EXTERNAL_DATA_PATTERNS,
    PREFIX_OVERRIDE_SPECS,
    READ_KEYWORDS as _READ_KEYWORDS,
    READ_ONLY_PREFIXES as _READ_ONLY_PREFIXES,
    URL_PARAM_KEYS as _URL_PARAM_KEYS,
)


@dataclass(frozen=True)
class ToolSemantics:
    mechanism: str
    edge_type: str
    risk_level: str
    effectful: bool = False
    channel: str | None = None
    approval_sensitive: bool = False
    external_visibility: bool = False


# ---------------------------------------------------------------------------
# Keyword-based classification (works for any tool, any dataset)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Channel inference from tool name
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_tool(
    tool_name: str,
    request_body: dict[str, Any] | None = None,
    response_body: Any = None,
) -> ToolSemantics:
    """Classify a tool by name + optional request/response hints.

    Order: exact match → prefix match → keyword-based auto-classify.
    Request/response hints can upgrade a read-only tool to effectful when
    the body contains identity-bearing fields (to, recipient, etc.).
    """
    normalized = _normalize_name(tool_name)

    # 1. Exact match override (for known tools with special semantics)
    if normalized in _EXACT_OVERRIDES:
        return _EXACT_OVERRIDES[normalized]

    # 2. Prefix match override
    for prefix, semantics in _PREFIX_OVERRIDES:
        if normalized.startswith(prefix):
            return semantics

    # 3. Auto-classify from name keywords
    semantics = _classify_by_name(normalized)

    # 3b. Upgrade to external_visibility for web/external-data tools
    if not semantics.external_visibility and _is_external_data_tool(normalized, request_body):
        semantics = ToolSemantics(
            mechanism=semantics.mechanism,
            edge_type=semantics.edge_type,
            risk_level="medium" if semantics.risk_level == "low" else semantics.risk_level,
            effectful=semantics.effectful,
            channel=semantics.channel or "web",
            approval_sensitive=True,
            external_visibility=True,
        )

    # 4. Upgrade to effectful if parameters suggest external effect
    if not semantics.effectful and _has_effect_parameters(request_body):
        semantics = ToolSemantics(
            mechanism="update",
            edge_type="updates",
            risk_level="medium",
            effectful=True,
            channel=semantics.channel,
        )

    return semantics


def service_name(tool_name: str) -> str:
    """Guess the service name from a tool name."""
    name = _normalize_name(tool_name)
    # Check channel keywords for service mapping
    for keyword, channel in _CHANNEL_KEYWORDS:
        if keyword in name:
            return channel
    # Fallback: first underscore-separated segment, or first word
    if "_" in tool_name:
        return tool_name.split("_")[0]
    parts = tool_name.strip().split()
    return parts[0] if parts else tool_name


# ---------------------------------------------------------------------------
# Internal: name-based classification
# ---------------------------------------------------------------------------

def _classify_by_name(normalized: str) -> ToolSemantics:
    """Infer semantics purely from the tool name string."""
    # Read-only check first (explicit read prefixes)
    for prefix in _READ_ONLY_PREFIXES:
        if normalized.startswith(prefix):
            return ToolSemantics("read", "reads", "low", effectful=False)

    # Effect keywords (order matters: first match wins)
    for keyword, mechanism, edge_type, risk, channel in _EFFECT_KEYWORDS:
        if _word_match(keyword, normalized):
            # Infer channel from tool name if not provided by keyword
            ch = channel or _infer_channel(normalized)
            approval = mechanism in {"send", "export", "delete", "execute", "api_call"}
            external = mechanism in {"send", "export", "api_call"}
            return ToolSemantics(
                mechanism=mechanism,
                edge_type=edge_type,
                risk_level=risk,
                effectful=True,
                channel=ch,
                approval_sensitive=approval,
                external_visibility=external,
            )

    # Check read keywords as substring fallback
    for keyword in _READ_KEYWORDS:
        if _word_match(keyword, normalized):
            return ToolSemantics("read", "reads", "low", effectful=False)

    # Default: generic tool call, assume read
    return ToolSemantics("tool_call", "depends_on", "low", effectful=False)


def _word_match(keyword: str, name: str) -> bool:
    """Match *keyword* as a word-boundary token inside *name*."""
    return bool(re.search(rf"(?<![a-z]){re.escape(keyword)}(?![a-z])", name))


# ---------------------------------------------------------------------------
# Internal: helpers
# ---------------------------------------------------------------------------

def _infer_channel(normalized: str) -> str | None:
    for keyword, channel in _CHANNEL_KEYWORDS:
        if keyword in normalized:
            return channel
    return None



# Tools that look like web API endpoints (start with /)
_EXTERNAL_ENDPOINT_PATTERN = re.compile(r"^/\w+")

def _is_external_data_tool(
    normalized: str,
    request_body: dict[str, Any] | None = None,
) -> bool:
    """Check if a tool fetches data from external sources (URLs, web, transcripts)."""
    # Check compiled patterns
    for pat in _EXTERNAL_DATA_PATTERNS:
        if pat.search(normalized):
            return True
    # Check keyword substrings
    for kw in _EXTERNAL_DATA_KEYWORDS:
        if kw in normalized:
            return True
    # Check if tool looks like a web endpoint
    if _EXTERNAL_ENDPOINT_PATTERN.match(normalized):
        return True
    # Check if request body contains URL parameters (strong signal)
    if isinstance(request_body, dict):
        for key in request_body:
            tail = key.rsplit(".", 1)[-1].lower() if "." in key else key.lower()
            tail = re.sub(r"\[\d+\]$", "", tail)
            if tail in _URL_PARAM_KEYS:
                return True
    return False


def _has_effect_parameters(body: dict[str, Any] | None) -> bool:
    """Check whether request parameters suggest an external effect.

    Identity-bearing fields like ``to``, ``recipient``, ``attendees``
    indicate the action targets an external entity — a strong signal
    that the tool is effectful even if its name doesn't suggest it.
    """
    if not isinstance(body, dict):
        return False
    effect_param_keys = {
        "to", "recipient", "recipients", "attendee", "attendees",
        "customer_id", "customer_ids", "ticket_id", "ticket_ids",
        "assignee", "assigned_to", "target", "destination",
    }
    for key in body:
        tail = key.rsplit(".", 1)[-1].lower() if "." in key else key.lower()
        tail = re.sub(r"\[\d+\]$", "", tail)
        if tail in effect_param_keys:
            return True
    return False


def _normalize_name(tool_name: str) -> str:
    return " ".join(tool_name.strip().lower().split())


# ---------------------------------------------------------------------------
# Override registry — optional, for tools with non-obvious semantics.
# Unknown tools auto-classify via the keyword rules above.
# ---------------------------------------------------------------------------

_EXACT_OVERRIDES: dict[str, ToolSemantics] = {
    name: ToolSemantics(**spec) for name, spec in EXACT_OVERRIDE_SPECS.items()
}

_PREFIX_OVERRIDES: tuple[tuple[str, ToolSemantics], ...] = tuple(
    (prefix, ToolSemantics(**spec)) for prefix, spec in PREFIX_OVERRIDE_SPECS
)
