"""Compile tool-calling JSONL traces into behavior graphs and MonitorCards."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import (
    BehaviorGraph,
    CandidateBehavior,
    CompilationResult,
    MonitorCard,
    StructuralSignal,
    EDGE_RESOLVES,
    EDGE_MENTIONS,
    EDGE_SAME_AS,
    EDGE_CONTRADICTS,
    EDGE_CONSTRAINS,
    EDGE_SATISFIES,
    EDGE_AUTHORIZES,
    EDGE_TRIGGERS,
    EDGE_INVALIDATES,
    ENTITY_TYPE_MAP,
)
from .ocesq import (
    analysis_report,
    build_ocei,
    graph_to_eventlog_rows,
    index_stats,
    mine_closure_break_motifs,
    query_results_summary,
    run_batch_ocesq,
)
from .tool_semantics import classify_tool, service_name

from .lexicons import (
    ALWAYS_REVIEW_EFFECTS,
    ANCHOR_KEYWORDS,
    CONCEPT_ALIASES,
    CRITICAL_REQUIREMENT_CONCEPTS,
    ENTITY_REFERENCE_PATTERNS,
    FREE_TEXT_PARAMETER_MARKERS,
    FREE_TEXT_PARAMETERS,
    IDENTITY_KEYS,
    STOPWORDS,
    TEXT_SCAN_PARAMETER_MARKERS,
    TEXT_SCAN_PARAMETERS,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def compile_trace_file(path: str | Path) -> CompilationResult:
    return TraceCompiler(load_jsonl(path)).compile()


class TraceCompiler:
    def __init__(self, events: list[dict[str, Any]]):
        self.events = events
        start = next((e for e in events if e.get("type") == "trace_start"), {})
        self.graph = BehaviorGraph(
            trace_id=start.get("trace_id", "unknown_trace"),
            task_id=start.get("task_id", "unknown_task"),
            metadata={"model": start.get("model"), "source": "tool_call_jsonl"},
        )
        self.tool_action_nodes: dict[str, str] = {}
        self.tool_observation_nodes: dict[str, str] = {}
        self.action_records: list[dict[str, Any]] = []
        self.output_nodes: list[str] = []
        self.signals: list[StructuralSignal] = []
        self.candidates: list[CandidateBehavior] = []

    def compile(self) -> CompilationResult:
        task_node = self._add_task_instruction()
        previous_message: str | None = task_node

        for idx, event in enumerate(self.events):
            typ = event.get("type")
            if typ == "message":
                previous_message = self._compile_message(event, idx, previous_message)
            elif typ == "tool_dispatch":
                self._compile_tool_dispatch(event, idx)
            elif typ == "audit_snapshot":
                self._compile_audit_snapshot(event, idx)
            elif typ in {"trace_end", "grading_result"}:
                self.graph.metadata[typ] = self._compact_result(event)

        self._link_late_observation_conflicts()
        self._deduplicate_entities()
        text_refs_added = self._extract_textual_entity_references()
        if text_refs_added:
            self.graph.metadata.setdefault("text_extraction_stats", {})["total_edges"] = text_refs_added

        # Route B Layer 3: temporal consistency detection
        contradicts_added = self._detect_temporal_inconsistencies()
        if contradicts_added:
            self.graph.metadata.setdefault("contradiction_stats", {})["total_contradicts"] = contradicts_added

        self._mine_signals_and_candidates()
        cards = [self._make_card(candidate) for candidate in self.candidates]
        return CompilationResult(self.graph, self.signals, self.candidates, cards)

    def _add_task_instruction(self) -> str:
        user_text = ""
        for event in self.events:
            if event.get("type") == "message" and event.get("message", {}).get("role") == "user":
                user_text = _message_text(event["message"])
                if user_text:
                    break
        task_node = self.graph.add_node(
            "TaskInstruction",
            self.graph.task_id,
            {"text": user_text},
            node_id="task_instruction",
            source_event_idx=0,
        )
        for req_idx, sentence in enumerate(_split_requirements(user_text), start=1):
            req_node = self.graph.add_node(
                "Requirement",
                f"requirement_{req_idx}",
                {"text": sentence},
                node_id=f"req_{req_idx}",
            )
            self.graph.add_edge(task_node, req_node, "requires")
        return task_node

    def _compile_message(self, event: dict[str, Any], idx: int, previous_message: str | None) -> str:
        message = event.get("message", {})
        role = message.get("role", "unknown")
        node_id = self.graph.add_node(
            "AssistantMessage" if role == "assistant" else "Message",
            role,
            {"role": role, "text": _message_text(message), "content": message.get("content", [])},
            source_event_idx=idx,
        )
        if previous_message:
            self.graph.add_edge(previous_message, node_id, "depends_on")

        for part in message.get("content", []) or []:
            if part.get("type") == "tool_use":
                self._compile_tool_use(part, idx, node_id)
            elif role == "assistant" and part.get("type") == "text":
                self._compile_submission(part.get("text", ""), idx, node_id)
        return node_id

    def _compile_tool_use(self, part: dict[str, Any], idx: int, message_node: str) -> None:
        tool_use_id = part.get("id") or f"tool_use_{len(self.tool_action_nodes) + 1}"
        tool_name = part.get("name", "unknown_tool")
        semantics = classify_tool(tool_name)
        node_id = self.graph.add_node(
            "ToolAction",
            tool_name,
            {
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "service": service_name(tool_name),
                "input": part.get("input", {}),
                "mechanism": semantics.mechanism,
                "risk_level": semantics.risk_level,
                "effectful": semantics.effectful,
                "channel": semantics.channel,
                "approval_sensitive": semantics.approval_sensitive,
                "external_visibility": semantics.external_visibility,
            },
            source_event_idx=idx,
        )
        self.tool_action_nodes[tool_use_id] = node_id
        self.graph.add_edge(message_node, node_id, "depends_on")
        self.graph.add_edge(node_id, "task_instruction", "depends_on")
        if semantics.effectful:
            self._link_prior_observations(node_id, part.get("input", {}), idx)
        self.action_records.append({"node_id": node_id, "tool_use_id": tool_use_id, "tool_name": tool_name})

    def _link_prior_observations(self, action_node: str, action_input: Any, source_event_idx: int) -> None:
        """Link an effectful action to prior observations by exact entity keys."""

        action_values = _identity_values(action_input)
        if not action_values:
            return
        aligned_observations: list[tuple[GraphNode, list[str]]] = []
        conflicting_observations: list[tuple[GraphNode, list[str]]] = []
        for observation in self.graph.nodes:
            if observation.type != "ToolObservation":
                continue
            if observation.source_event_idx is None or observation.source_event_idx >= source_event_idx:
                continue
            observed_values = _identity_values(observation.attrs.get("response_body"))
            aligned = sorted(
                key for key, values in action_values.items()
                if values & observed_values.get(key, set())
            )
            if aligned:
                aligned_observations.append((observation, aligned))
                continue
            conflicting = sorted(
                key for key, values in action_values.items()
                if key in observed_values and not (values & observed_values[key])
            )
            if conflicting:
                conflicting_observations.append((observation, conflicting))
        contextual_conflicts = conflicting_observations
        if aligned_observations:
            contextual_conflicts = [
                (observation, keys)
                for observation, keys in conflicting_observations
                if any(
                    _same_observation_context(observation, aligned)
                    for aligned, _ in aligned_observations
                )
            ]
        selected = [
            (observation, keys, "exact_entity_alignment")
            for observation, keys in aligned_observations
        ] + [
            (observation, keys, "entity_key_conflict")
            for observation, keys in contextual_conflicts
        ]
        for observation, keys, derivation in selected:
            self.graph.add_edge(
                action_node,
                observation.id,
                "depends_on",
                {"derivation": derivation, "alignment_keys": keys},
            )

    def _link_late_observation_conflicts(self) -> None:
        """Retain observations that occur after a matching effectful action."""

        observations = [node for node in self.graph.nodes if node.type == "ToolObservation"]
        for action in self.graph.nodes:
            if action.type != "ToolAction" or not action.attrs.get("effectful"):
                continue
            if action.source_event_idx is None:
                continue
            action_values = _identity_values(action.attrs.get("input", {}))
            for observation in observations:
                if observation.source_event_idx is None or observation.source_event_idx <= action.source_event_idx:
                    continue
                observed_values = _identity_values(observation.attrs.get("response_body"))
                aligned = sorted(
                    key for key, values in action_values.items()
                    if values & observed_values.get(key, set())
                )
                if aligned:
                    self.graph.add_edge(
                        action.id,
                        observation.id,
                        "depends_on",
                        {"derivation": "late_entity_alignment", "alignment_keys": aligned, "temporal_conflict": True},
                    )

    def _compile_tool_dispatch(self, event: dict[str, Any], idx: int) -> None:
        tool_use_id = event.get("tool_use_id")
        action_node = self.tool_action_nodes.get(tool_use_id)
        tool_name = event.get("tool_name", "unknown_tool")
        semantics = classify_tool(tool_name)
        response_body = event.get("response_body")
        records = _walk_records(response_body)
        result_count = len(records)
        result_truncated = _response_is_truncated(response_body, result_count)
        obs_node = self.graph.add_node(
            "ToolObservation",
            f"{tool_name}_result",
            {
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "status": event.get("response_status"),
                "request_body": event.get("request_body", {}),
                "response_body": response_body,
                "result_count": result_count,
                "result_truncated": result_truncated,
            },
            source_event_idx=idx,
        )
        self.tool_observation_nodes[tool_use_id] = obs_node
        if action_node:
            self.graph.add_edge(action_node, obs_node, "observes")
            self.graph.add_edge(action_node, obs_node, semantics.edge_type)
            self._add_parameter_entities(action_node, event.get("request_body", {}), source_event_idx=idx)
            self._add_response_entities(obs_node, event.get("response_body"), source_event_idx=idx, tool_name=tool_name)
            if semantics.effectful:
                effect_node = self.graph.add_node(
                    "ExternalEffect",
                    tool_name,
                    {
                        "tool_name": tool_name,
                        "mechanism": semantics.mechanism,
                        "request_body": event.get("request_body", {}),
                        "response_body": event.get("response_body"),
                        "channel": semantics.channel,
                        "approval_sensitive": semantics.approval_sensitive,
                        "external_visibility": semantics.external_visibility,
                    },
                    source_event_idx=idx,
                )
                self.graph.add_edge(action_node, effect_node, semantics.edge_type)

    def _compile_audit_snapshot(self, event: dict[str, Any], idx: int) -> None:
        self.graph.add_node(
            "EntityVersion",
            f"{event.get('service_name', 'service')}_audit",
            {"service_name": event.get("service_name"), "audit_data": event.get("audit_data", {})},
            source_event_idx=idx,
        )

    def _compile_submission(self, text: str, idx: int, message_node: str) -> None:
        submission = self.graph.add_node("Submission", "assistant_final", {"text": text}, source_event_idx=idx)
        self.graph.add_edge(message_node, submission, "submits")
        for assertion_idx, assertion in enumerate(_split_requirements(text), start=1):
            assertion_node = self.graph.add_node(
                "OutputAssertion",
                f"assertion_{len(self.output_nodes) + assertion_idx}",
                {"text": assertion},
                source_event_idx=idx,
            )
            self.graph.add_edge(submission, assertion_node, "grounds")
            self.output_nodes.append(assertion_node)

    def _add_parameter_entities(self, action_node: str, body: Any, *, source_event_idx: int) -> None:
        for key, value in _walk_scalar_items(body):
            if _is_identity_key(key):
                entity_type = _classify_entity_type(key)
                entity = self.graph.add_node(
                    "ServiceEntity",
                    f"{key}:{value}",
                    {
                        "key": key,
                        "value": value,
                        "source": "request_parameter",
                        "entity_type": entity_type,
                        "resolution_method": "reference",
                        "candidate_count": None,
                    },
                    source_event_idx=source_event_idx,
                )
                self.graph.add_edge(action_node, entity, "depends_on", {"parameter": key})
                self.graph.add_edge(action_node, entity, EDGE_MENTIONS, {"parameter": key})

    def _add_response_entities(self, obs_node: str, body: Any, *, source_event_idx: int, tool_name: str = "") -> None:
        records = _walk_records(body)
        total = len(records)
        # A ``resolves`` edge is only valid when the observation *discovers* the
        # entity (read/query).  Write/send/export observations reflect the effect
        # of the action, not the source of the entity.
        obs_is_read = bool(tool_name) and not _is_effectful_tool(tool_name)
        for item in records:
            identity = _record_identity(item)
            if identity:
                key, value = identity.split(":", 1) if ":" in identity else ("id", identity)
                entity_type = _classify_entity_type(key)
                entity = self.graph.add_node(
                    "ServiceEntity",
                    identity,
                    {
                        "key": key,
                        "value": value,
                        "record": item,
                        "source": "tool_response",
                        "entity_type": entity_type,
                        "source_observation_id": obs_node,
                        "candidate_count": total,
                        "resolution_method": "exact_match" if (total == 1 and obs_is_read) else ("ambiguous" if total > 1 else "reference"),
                    },
                    source_event_idx=source_event_idx,
                )
                self.graph.add_edge(obs_node, entity, "observes")
                if total == 1 and obs_is_read:
                    self.graph.add_edge(obs_node, entity, EDGE_RESOLVES)

    def _deduplicate_entities(self) -> None:
        """Link ServiceEntity nodes that refer to the same real-world object.

        Scans all ServiceEntity nodes, groups by (entity_type, normalized_value),
        and adds ``same_as`` edges within each group.  This cross-links entities
        created from different observations / parameters so that OCESQ can traverse
        entity identity chains.
        """
        from collections import defaultdict

        by_identity: dict[str, list[str]] = defaultdict(list)
        for node in self.graph.nodes:
            if node.type != "ServiceEntity":
                continue
            etype = node.attrs.get("entity_type", "entity")
            value = node.attrs.get("value", "") or node.label.split(":", 1)[-1] if ":" in node.label else node.label
            norm = _normalize_value(value)
            if not norm or len(norm) < 3:
                continue
            identity = f"{etype}:{norm}"
            by_identity[identity].append(node.id)

        same_as_count = 0
        for identity, node_ids in by_identity.items():
            if len(node_ids) < 2:
                continue
            first = node_ids[0]
            for other in node_ids[1:]:
                self.graph.add_edge(first, other, EDGE_SAME_AS)
                same_as_count += 1
        if same_as_count:
            self.graph.metadata.setdefault("dedup_stats", {})["same_as_edges"] = same_as_count

    # ------------------------------------------------------------------
    # Route B Layer 2: free-text entity reference extraction
    # ------------------------------------------------------------------

    def _extract_textual_entity_references(self) -> int:
        """Scan text-rich parameters of every ToolAction for entity references.

        Builds a known-entity index from all ServiceEntity nodes, then scans
        the text-valued parameters of each action for ID patterns, emails,
        phone numbers, and known entity-name substrings.  Matches are linked
        to the action via ``mentions`` edges.

        Returns the total number of new ``mentions`` edges created.
        """
        entity_index = self._build_entity_name_index()
        if not entity_index:
            return 0

        total_added = 0
        stats: dict[str, int] = {"id_match": 0, "email_match": 0, "name_match": 0, "phone_match": 0, "url_match": 0}

        for node in self.graph.nodes:
            if node.type != "ToolAction":
                continue
            text_params = self._get_text_scan_parameters(node)
            if not text_params:
                continue

            added_for_action = 0
            for param_key, text_value in text_params:
                added = self._scan_text_for_entities(
                    text_value, node.id, param_key, entity_index,
                )
                added_for_action += added

            if added_for_action:
                total_added += added_for_action

        # Record stats for experiment analysis
        stats["total_edges"] = total_added
        self.graph.metadata.setdefault("text_extraction_stats", {}).update(stats)

        # Deduplicate: keep only the best (source, target) edge, preferring
        # id/email patterns over name_substring matches.
        removed = self._deduplicate_text_mentions()
        stats["dedup_removed"] = removed

        return total_added

    def _deduplicate_text_mentions(self) -> int:
        """Remove duplicate text-extraction ``mentions`` edges, keeping the
        highest-priority pattern match for each (source, target) pair.

        Returns the number of edges removed.
        """
        _pattern_priority = {"id_pattern": 0, "email": 1, "phone": 2, "url": 3, "name_substring": 4}

        # Collect text-extracted mentions edges, grouped by (source, target)
        groups: dict[tuple[str, str], list[int]] = {}  # (src, tgt) -> [edge_indices]
        for i, edge in enumerate(self.graph.edges):
            if edge.attrs.get("source") != "text_extraction":
                continue
            key = (edge.source, edge.target)
            groups.setdefault(key, []).append(i)

        removed = 0
        for (src, tgt), indices in groups.items():
            if len(indices) <= 1:
                continue
            # Sort by pattern priority; keep the best (lowest priority number)
            def _priority(idx: int) -> int:
                pat = self.graph.edges[idx].attrs.get("pattern", "name_substring")
                return _pattern_priority.get(pat, 99)

            indices.sort(key=_priority)
            # Mark all but the best for removal
            for idx in indices[1:]:
                self.graph.edges[idx] = None  # type: ignore[assignment]
                removed += 1

        if removed:
            self.graph.edges = [e for e in self.graph.edges if e is not None]
        return removed

    def _build_entity_name_index(self) -> dict[str, list[tuple[str, str]]]:
        """Build a lookup from normalised entity names/values to entity nodes.

        Returns ``{normalized_key: [(entity_node_id, match_type), ...]}``
        where *match_type* is one of ``id``, ``email``, ``name``.
        """
        index: dict[str, list[tuple[str, str]]] = {}

        for node in self.graph.nodes:
            if node.type != "ServiceEntity":
                continue

            value = node.attrs.get("value", "")
            key = node.attrs.get("key", "")
            record = node.attrs.get("record", {}) or {}
            entity_type = node.attrs.get("entity_type", "entity")

            # Index the canonical identity value (e.g. "CUS-008")
            if value:
                norm = _normalize_value(value)
                if norm and len(norm) >= 3:
                    index.setdefault(norm, []).append((node.id, "id"))

            # Index email values separately
            if entity_type == "email" and value and "@" in str(value):
                norm = _normalize_value(value)
                if norm:
                    index.setdefault(norm, []).append((node.id, "email"))

            # Index human-readable names from the source record
            for name_key in ("name", "subject", "title"):
                name_val = record.get(name_key)
                if name_val and isinstance(name_val, str) and len(name_val) >= 4:
                    norm = _normalize_value(name_val)
                    if norm:
                        index.setdefault(norm, []).append((node.id, "name"))

            # Also index by the record's full identity string (e.g. "customer_id:CUS-008")
            identity = _record_identity(record)
            if identity:
                norm = _normalize_value(identity)
                if norm:
                    index.setdefault(norm, []).append((node.id, "id"))

        return index

    @staticmethod
    def _get_text_scan_parameters(action_node: Any) -> list[tuple[str, str]]:
        """Return the action's text-rich parameter values worth scanning.

        Looks at the action's ``input`` dict and selects entries whose key
        tail suggests free-form text (body, content, message, etc.) and
        whose value is a non-empty string long enough for pattern matching.
        """
        params: list[tuple[str, str]] = []
        raw_input = action_node.attrs.get("input", {})
        if not isinstance(raw_input, dict):
            return params

        for key, value in _walk_scalar_items(raw_input):
            tail = key.rsplit(".", 1)[-1].lower() if "." in key else key.lower()
            # Skip already-structured identity keys — those are handled by
            # _add_parameter_entities via _is_identity_key.
            if _is_identity_key(tail):
                continue
            # Only scan parameters whose key name looks text-like.
            if tail not in TEXT_SCAN_PARAMETERS and not any(
                marker in tail for marker in TEXT_SCAN_PARAMETER_MARKERS
            ):
                continue
            text = str(value).strip()
            if len(text) >= 20:  # require a minimum length to be worth scanning
                params.append((key, text))
        return params

    def _scan_text_for_entities(
        self,
        text: str,
        action_node_id: str,
        param_key: str,
        entity_index: dict[str, list[tuple[str, str]]],
    ) -> int:
        """Scan *text* for entity references and create ``mentions`` edges.

        Strategy (in priority order):
        1. Regex patterns (ID patterns, email, phone, URL) — exact span match
           against the entity index.
        2. Known-entity-name substring matching — search for each entity name
           in the text.

        Returns the number of new edges created.
        """
        added = 0
        matched_entities: set[str] = set()
        matched_values: set[str] = set()  # dedup by entity identity, not node id

        # --- Pass 1: regex patterns ---
        # Track matched spans to avoid double-counting overlapping matches.
        matched_spans: set[tuple[int, int]] = set()

        for pattern, pattern_type, _priority in ENTITY_REFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                span = match.span()
                # Skip if this span overlaps an already-processed match.
                if any(s[0] < span[1] and span[0] < s[1] for s in matched_spans):
                    continue
                matched_spans.add(span)

                match_text = match.group(0)
                norm = _normalize_value(match_text)

                # Look up in entity index
                candidates = entity_index.get(norm, [])
                if not candidates:
                    # Try partial matching: check if any entity value is a
                    # substring of the match or vice versa.
                    for ent_norm, ent_list in entity_index.items():
                        if norm in ent_norm or ent_norm in norm:
                            candidates.extend(ent_list)

                for entity_id, _match_type in candidates:
                    entity = self.graph.node(entity_id)
                    entity_identity = entity.attrs.get("value", entity_id) if entity else entity_id
                    if entity_id not in matched_entities and entity_identity not in matched_values:
                        self.graph.add_edge(
                            action_node_id,
                            entity_id,
                            EDGE_MENTIONS,
                            {
                                "parameter": param_key,
                                "source": "text_extraction",
                                "pattern": pattern_type,
                                "matched_text": match_text[:120],
                            },
                        )
                        matched_entities.add(entity_id)
                        matched_values.add(entity_identity)
                        added += 1

        # --- Pass 2: known-entity name substring matching ---
        # For entities with a human-readable name, check whether that name
        # appears in the text as a substring.  Only do this for entities
        # whose value was NOT already matched in Pass 1.
        text_lower = text.lower()
        for ent_norm, ent_list in entity_index.items():
            if len(ent_norm) < 5:  # skip very short names (high false-positive risk)
                continue
            if ent_norm not in text_lower:
                continue
            for entity_id, match_type in ent_list:
                if match_type != "name":
                    continue
                if entity_id in matched_entities:
                    continue
                entity = self.graph.node(entity_id)
                entity_identity = entity.attrs.get("value", entity_id) if entity else entity_id
                if entity_identity in matched_values:
                    continue
                # Skip name_substring matching for identity-key entities
                # (emails, message_ids, etc.) — they should only match via
                # id_pattern or email patterns, not free-text name overlap.
                ent_key = (entity.attrs.get("key") or "").lower() if entity else ""
                if ent_key in IDENTITY_KEYS:
                    continue
                self.graph.add_edge(
                    action_node_id,
                    entity_id,
                    EDGE_MENTIONS,
                    {
                        "parameter": param_key,
                        "source": "text_extraction",
                        "pattern": "name_substring",
                        "matched_text": ent_norm[:120],
                    },
                )
                matched_entities.add(entity_id)
                matched_values.add(entity_identity)
                added += 1

        return added

    # ------------------------------------------------------------------
    # Route B Layer 3: temporal consistency detection
    # ------------------------------------------------------------------

    # State fields whose values are compared across same_as-linked entity
    # observations.  A difference in any of these fields between two
    # observations produces a ``contradicts`` edge.
    INCONSISTENCY_STATE_FIELDS: tuple[str, ...] = (
        "status", "priority", "resolution", "stage", "state",
        "current_stock", "is_read",
    )

    def _detect_temporal_inconsistencies(self) -> int:
        """Compare entities linked by ``same_as`` edges for state differences.

        For each cluster of same_as-linked ServiceEntity nodes:
        1. Only consider entities sourced from tool responses (real observations).
        2. Compare state fields (status, priority, etc.) across observations.
        3. If any field has different non-empty values, add a ``contradicts``
           edge between the two entity nodes.

        Returns the number of new ``contradicts`` edges created.
        """
        from collections import defaultdict

        # --- Build same_as clusters ---
        parent: dict[str, str] = {}
        entity_nodes = [n for n in self.graph.nodes if n.type == "ServiceEntity"]
        for n in entity_nodes:
            parent[n.id] = n.id
        for e in self.graph.edges:
            if e.type == EDGE_SAME_AS and e.source in parent and e.target in parent:
                ra = parent[e.source]
                rb = parent[e.target]
                for k in parent:
                    if parent[k] == ra:
                        parent[k] = rb

        clusters: dict[str, list[Any]] = defaultdict(list)
        for n in entity_nodes:
            clusters[parent[n.id]].append(n)

        added = 0
        stats: dict[str, int] = defaultdict(int)

        for _root, members in clusters.items():
            if len(members) < 2:
                continue

            # Keep only tool_response entities (they carry actual state data).
            response_members = [
                m for m in members if m.attrs.get("source") == "tool_response"
            ]
            if len(response_members) < 2:
                continue

            # Compare each state field across members.
            for sf in self.INCONSISTENCY_STATE_FIELDS:
                val_to_members: dict[str, list[Any]] = defaultdict(list)
                for m in response_members:
                    rec = m.attrs.get("record", {}) or {}
                    val = rec.get(sf)
                    if val is not None and str(val).strip() != "":
                        val_to_members[str(val).strip()].append(m)

                if len(val_to_members) < 2:
                    continue

                # One contradicts edge per pair of differing observations.
                members_by_val = list(val_to_members.items())
                for i in range(len(members_by_val)):
                    for j in range(i + 1, len(members_by_val)):
                        val_a, ents_a = members_by_val[i]
                        val_b, ents_b = members_by_val[j]
                        # Connect the first entity from each value group.
                        ea = ents_a[0]
                        eb = ents_b[0]
                        self.graph.add_edge(
                            ea.id,
                            eb.id,
                            EDGE_CONTRADICTS,
                            {
                                "field": sf,
                                "value_in_source": val_a,
                                "value_in_target": val_b,
                                "entity_type": ea.attrs.get("entity_type", "entity"),
                                "entity_value": ea.attrs.get("value", ""),
                                "source_observation_a": ea.attrs.get("source_observation_id", ""),
                                "source_observation_b": eb.attrs.get("source_observation_id", ""),
                            },
                        )
                        added += 1
                        stats[f"{sf}_contradiction"] += 1

        if added:
            self.graph.metadata.setdefault("contradiction_stats", {}).update(stats)
            self.graph.metadata["contradiction_stats"]["total_contradicts"] = added

        return added

    def _mine_signals_and_candidates(self) -> None:
        for node in self.graph.nodes:
            if node.type == "ToolAction" and node.attrs.get("effectful"):
                signal_ids = self._signals_for_effect(node.id)
                context = self._behavior_context_for_action(node)
                if not self._should_emit_effect_candidate(node, signal_ids, context):
                    continue
                risk_level = "high" if signal_ids or node.attrs.get("risk_level") == "high" else "medium"
                self.candidates.append(
                    CandidateBehavior(
                        id=f"cand_{len(self.candidates) + 1}",
                        kind="effectful_tool_action",
                        risk_level=risk_level,
                        action_node_id=node.id,
                        summary=f"Review {node.attrs.get('mechanism')} action via {node.attrs.get('tool_name')}",
                        signal_ids=signal_ids,
                        node_ids=[node.id],
                        payload=node.attrs,
                    )
                )

        self._mine_requirement_coverage_gaps()

    @staticmethod
    def _should_emit_effect_candidate(action: Any, signal_ids: list[str], context: dict[str, Any]) -> bool:
        mechanism = action.attrs.get("mechanism")
        if signal_ids:
            return True
        if action.attrs.get("approval_sensitive") or action.attrs.get("external_visibility"):
            return True
        if action.attrs.get("risk_level") == "high" and action.attrs.get("effectful"):
            return True
        if mechanism not in ALWAYS_REVIEW_EFFECTS:
            return False
        binding_statuses = {binding.get("binding_status") for binding in context.get("parameter_bindings", [])}
        if binding_statuses and binding_statuses <= {"matched_unique", "disambiguated"}:
            return False
        return bool(context.get("missing_facts"))

    def _signals_for_effect(self, action_node_id: str) -> list[str]:
        action = self.graph.node(action_node_id)
        if not action:
            return []
        signal_ids: list[str] = []
        context = self._behavior_context_for_action(action)
        blockers = context.get("structural_blockers", [])
        ambiguous_bindings = [
            binding
            for binding in context.get("parameter_bindings", [])
            if binding.get("binding_status") == "ambiguous"
        ]
        if ambiguous_bindings:
            signal = self._add_signal(
                "ambiguous_entity_binding",
                "high",
                "Effectful action binds an action parameter to one item from a multi-entity observation without a clear disambiguation step.",
                [action_node_id],
                {"parameter_bindings": ambiguous_bindings},
            )
            signal_ids.append(signal.id)
        for blocker in blockers:
            signal_type = blocker.get("type")
            if not signal_type:
                continue
            severity = "high" if blocker.get("status") in {"conflict", "missing"} else blocker.get("severity", "medium")
            signal = self._add_signal(
                signal_type,
                severity,
                blocker.get("summary") or f"Structural blocker detected: {signal_type}",
                [action_node_id, *blocker.get("evidence_node_ids", [])],
                blocker,
            )
            signal_ids.append(signal.id)
        if action.attrs.get("mechanism") in {"delete", "export", "send", "share"}:
            if not self._has_read_before(action_node_id):
                signal = self._add_signal(
                    "weak_observation_support",
                    "medium",
                    "High-risk effect has little preceding read/query support in the trace graph.",
                    [action_node_id],
                )
                signal_ids.append(signal.id)
        return signal_ids

    def _mine_requirement_coverage_gaps(self) -> None:
        covered_context = self._coverage_text()
        grading = self.graph.metadata.get("grading_result", {})
        passed = grading.get("passed")
        task_score = grading.get("task_score")
        for req in self.graph.nodes:
            if req.type != "Requirement":
                continue
            text = req.attrs.get("text", "")
            req_concepts = _content_concepts(text)
            missing_concepts = [concept for concept in req_concepts if concept not in covered_context]
            critical_missing = [concept for concept in missing_concepts if concept in CRITICAL_REQUIREMENT_CONCEPTS]
            missing_ratio = len(missing_concepts) / max(len(req_concepts), 1)
            if not self._should_emit_coverage_gap(
                critical_missing=critical_missing,
                missing_ratio=missing_ratio,
                passed=passed,
                task_score=task_score,
            ):
                continue
            signal = self._add_signal(
                "requirement_coverage_gap",
                "medium",
                "A task requirement has weak support for behavior-critical concepts in observed actions, observations, and final assertions.",
                [req.id],
                {
                    "requirement": text,
                    "missing_concepts": critical_missing[:12],
                    "missing_ratio": round(missing_ratio, 3),
                },
            )
            self.candidates.append(
                CandidateBehavior(
                    id=f"cand_{len(self.candidates) + 1}",
                    kind="requirement_coverage_gap",
                    risk_level="medium",
                    action_node_id=req.id,
                    summary="Review whether the trajectory covers a task requirement.",
                    signal_ids=[signal.id],
                    node_ids=[req.id],
                    payload={
                        "requirement": text,
                        "missing_concepts": critical_missing[:12],
                        "missing_ratio": round(missing_ratio, 3),
                    },
                )
            )

    def _coverage_text(self) -> set[str]:
        chunks: list[str] = []
        for node in self.graph.nodes:
            if node.type in {"ToolAction", "ToolObservation", "OutputAssertion", "Submission"}:
                chunks.append(node.label)
                chunks.append(json.dumps(node.attrs, ensure_ascii=False))
        return set(_content_concepts("\n".join(chunks)))

    @staticmethod
    def _should_emit_coverage_gap(
        *,
        critical_missing: list[str],
        missing_ratio: float,
        passed: Any,
        task_score: Any,
    ) -> bool:
        if len(critical_missing) < 2:
            return False
        if passed is True and isinstance(task_score, (int, float)) and task_score >= 0.8:
            return missing_ratio >= 0.5 and len(critical_missing) >= 3
        return missing_ratio >= 0.3

    def _behavior_context_for_action(self, action: Any) -> dict[str, Any]:
        bindings = []
        support_facts = []
        missing_facts = []
        anchors = _action_anchors(action.attrs.get("input", {}))
        if not anchors and self._requires_target_binding(action):
            missing_facts.append(
                {
                    "type": "missing_target_anchor",
                    "summary": "No structured target parameter was available for binding in a high-impact action.",
                }
            )
        for key, value in anchors:
            binding = self._bind_parameter(action, key, value)
            bindings.append(binding)
            if binding["binding_status"] == "unmatched":
                missing_facts.append({"type": "unmatched_parameter", "parameter": key, "value": value})
            else:
                support_facts.append(
                    {
                        "type": "parameter_binding",
                        "parameter": key,
                        "value": value,
                        "source_observation": binding.get("source_observation"),
                        "binding_status": binding.get("binding_status"),
                    }
                )
        structural_blockers = self._structural_blockers_for_action(action, bindings)
        missing_facts.extend(
            {
                "type": blocker.get("type"),
                "status": blocker.get("status"),
                "summary": blocker.get("summary"),
                "failure_site": blocker.get("failure_site"),
                "repair_target": blocker.get("repair_target"),
                "evidence_node_ids": blocker.get("evidence_node_ids", []),
            }
            for blocker in structural_blockers
        )
        allow_certificate = self._allow_certificate_for_action(action, bindings, structural_blockers)
        return {
            "parameter_bindings": bindings,
            "support_facts": support_facts,
            "missing_facts": missing_facts,
            "coverage_gaps": [],
            "structural_blockers": structural_blockers,
            "allow_certificate": allow_certificate,
            "certificate_complete": all(
                item.get("status") in {"satisfied", "not_required"} for item in allow_certificate
            ),
        }

    @staticmethod
    def _requires_target_binding(action: Any) -> bool:
        return bool(
            action.attrs.get("approval_sensitive")
            or action.attrs.get("external_visibility")
            or action.attrs.get("risk_level") == "high"
        )

    def _structural_blockers_for_action(self, action: Any, bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        blockers: list[dict[str, Any]] = []
        if self._requires_target_binding(action):
            unmatched = [binding for binding in bindings if binding.get("binding_status") in {"unmatched", "ambiguous"}]
            if unmatched:
                blockers.append(
                    {
                        "type": "target_binding_not_certified",
                        "status": "missing",
                        "failure_site": "target_binding",
                        "repair_target": "run_verification",
                        "summary": "A high-impact action has unmatched or ambiguous target parameters.",
                        "evidence_node_ids": [action.id],
                        "parameter_bindings": unmatched,
                    }
                )
        return blockers

    def _allow_certificate_for_action(
        self,
        action: Any,
        bindings: list[dict[str, Any]],
        blockers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        requirements: list[dict[str, Any]] = []

        def add(name: str, failure_site: str, repair_target: str, *, required: bool, evidence: list[str] | None = None) -> None:
            related = [blocker for blocker in blockers if blocker.get("failure_site") == failure_site]
            if related:
                status = "conflict" if any(blocker.get("status") == "conflict" for blocker in related) else "missing"
                summary = related[0].get("summary", "")
                evidence_node_ids = sorted({node_id for blocker in related for node_id in blocker.get("evidence_node_ids", [])})
            elif required:
                status = "satisfied" if evidence else "missing"
                summary = "Requirement satisfied by graph evidence." if evidence else "Required evidence is missing from the graph."
                evidence_node_ids = evidence or []
            else:
                status = "not_required"
                summary = "Requirement is not applicable to this action."
                evidence_node_ids = []
            requirements.append(
                {
                    "name": name,
                    "required": required,
                    "status": status,
                    "failure_site": failure_site,
                    "repair_target": repair_target,
                    "summary": summary,
                    "evidence_node_ids": evidence_node_ids,
                }
            )

        good_bindings = [
            binding
            for binding in bindings
            if binding.get("binding_status") in {"matched_unique", "disambiguated"}
            and binding.get("source_observation")
        ]
        target_required = self._requires_target_binding(action)
        read_support = [node.id for node in self.graph.nodes if node.type == "ToolObservation" and (node.source_event_idx or 0) < (action.source_event_idx or 0)]

        add(
            "target_binding",
            "target_binding",
            "revise_target",
            required=target_required,
            evidence=[binding["source_observation"] for binding in good_bindings if binding.get("source_observation")],
        )
        add("input_observation", "input_observation", "run_verification", required=target_required, evidence=read_support[-4:])
        add(
            "authorization",
            "authorization",
            "request_authorization",
            required=bool(action.attrs.get("approval_sensitive") or action.attrs.get("external_visibility")),
            evidence=[],
        )
        add(
            "effect_safety",
            "destructive_effect",
            "abort_action",
            required=bool(action.attrs.get("effectful") and action.attrs.get("risk_level") == "high"),
            evidence=[action.id] if not blockers else [],
        )
        add(
            "policy_consistency",
            "policy_conflict",
            "abort_action",
            required=False,
            evidence=[],
        )
        return requirements

    def _bind_parameter(self, action: Any, key: str, value: Any) -> dict[str, Any]:
        action_idx = action.source_event_idx or 0
        value_text = _normalize_value(value)
        base = {
            "parameter": key,
            "value": value,
            "binding_status": "unmatched",
            "source_observation": None,
            "matched_record": None,
            "alternative_entities": [],
            "disambiguation_evidence": [],
        }
        if not value_text:
            return base

        best: dict[str, Any] | None = None
        for obs in self.graph.nodes:
            if obs.type != "ToolObservation" or (obs.source_event_idx or 0) >= action_idx:
                continue
            records = _walk_records(obs.attrs.get("response_body"))
            matches = [record for record in records if _record_contains_value(record, value_text)]
            if not matches:
                continue
            alternatives = [_record_summary(record) for record in records if _record_summary(record)]
            is_stable_id = _is_stable_identifier_key(key)
            candidate = {
                **base,
                "binding_status": "ambiguous" if len(alternatives) > 1 and not is_stable_id else "matched_unique",
                "source_observation": obs.id,
                "source_tool": obs.attrs.get("tool_name"),
                "matched_record": _record_summary(matches[0]),
                "alternative_entities": alternatives[:12],
                "disambiguation_evidence": self._disambiguation_between(obs, action, value_text),
            }
            candidate["disambiguation_evidence"].extend(
                _record_context_disambiguation(matches[0], action.attrs.get("input", {}), obs.id)
            )
            if candidate["binding_status"] == "ambiguous" and candidate["disambiguation_evidence"]:
                candidate["binding_status"] = "disambiguated"
            if best is None or candidate["binding_status"] != "ambiguous":
                best = candidate
            if best and best["binding_status"] in {"matched_unique", "disambiguated"}:
                break
        return best or base

    def _disambiguation_between(self, obs: Any, action: Any, value_text: str) -> list[dict[str, Any]]:
        evidence = []
        start = obs.source_event_idx or 0
        end = action.source_event_idx or 0
        for node in self.graph.nodes:
            idx = node.source_event_idx or 0
            if not (start < idx < end):
                continue
            if node.type == "ToolObservation":
                body = node.attrs.get("response_body")
                total = _reported_total(body)
                text = json.dumps(body, ensure_ascii=False).lower()
                if value_text in text and (total == 1 or total is None):
                    evidence.append({"node_id": node.id, "tool_name": node.attrs.get("tool_name"), "kind": "narrowing_observation"})
            elif node.type == "Message":
                text = node.attrs.get("text", "").lower()
                if value_text in text:
                    evidence.append({"node_id": node.id, "kind": "user_confirmation"})
        return evidence[:6]

    def _has_read_before(self, action_node_id: str) -> bool:
        action = self.graph.node(action_node_id)
        action_idx = action.source_event_idx or 0 if action else 0
        for node in self.graph.nodes:
            if node.type == "ToolObservation" and (node.source_event_idx or 0) < action_idx:
                return True
        return False

    def _add_signal(
        self,
        signal_type: str,
        severity: str,
        summary: str,
        node_ids: list[str],
        attrs: dict[str, Any] | None = None,
    ) -> StructuralSignal:
        signal = StructuralSignal(
            id=f"sig_{len(self.signals) + 1}",
            type=signal_type,
            severity=severity,
            summary=summary,
            node_ids=node_ids,
            attrs=attrs or {},
        )
        self.signals.append(signal)
        signal_node = self.graph.add_node(
            "StructuralSignals",
            signal_type,
            {"severity": severity, "summary": summary, **(attrs or {})},
        )
        for node_id in node_ids:
            self.graph.add_edge(signal_node, node_id, "depends_on")
        return signal

    def _make_card(self, candidate: CandidateBehavior) -> MonitorCard:
        action_node = self.graph.node(candidate.action_node_id) if candidate.action_node_id else None
        signal_ids = set(candidate.signal_ids)
        signals = [signal for signal in self.signals if signal.id in signal_ids]
        neighborhood = self._neighborhood(candidate.node_ids)
        behavior_context = self._behavior_context_for_action(action_node) if action_node and action_node.type == "ToolAction" else {}
        if candidate.kind == "requirement_coverage_gap":
            behavior_context = {
                "parameter_bindings": [],
                "support_facts": [],
                "missing_facts": [],
                "coverage_gaps": [candidate.payload],
            }
        return MonitorCard(
            card_id=f"card_{candidate.id}",
            trace_id=self.graph.trace_id,
            task_id=self.graph.task_id,
            candidate=candidate,
            current_action=asdict(action_node) if action_node else {},
            relevant_history=self._recent_tool_history(action_node.source_event_idx if action_node else None),
            graph_context=neighborhood,
            structural_signals=signals,
            behavior_context=behavior_context,
            official_result={
                "task_score": self.graph.metadata.get("grading_result", {}).get("task_score"),
                "passed": self.graph.metadata.get("grading_result", {}).get("passed"),
                "scores": self.graph.metadata.get("grading_result", {}).get("scores"),
            },
        )

    def _recent_tool_history(self, before_idx: int | None, limit: int = 6) -> list[dict[str, Any]]:
        rows = [
            node
            for node in self.graph.nodes
            if node.type in {"ToolAction", "ToolObservation"} and (before_idx is None or (node.source_event_idx or 0) <= before_idx)
        ]
        return [asdict(node) for node in rows[-limit:]]

    def _neighborhood(self, node_ids: list[str]) -> dict[str, Any]:
        ids = set(node_ids)
        for edge in self.graph.edges:
            if edge.source in ids or edge.target in ids:
                ids.add(edge.source)
                ids.add(edge.target)
        return {
            "nodes": [asdict(node) for node in self.graph.nodes if node.id in ids],
            "edges": [asdict(edge) for edge in self.graph.edges if edge.source in ids or edge.target in ids],
        }

    @staticmethod
    def _compact_result(event: dict[str, Any]) -> dict[str, Any]:
        keys = ("task_id", "task_score", "passed", "scores", "failure_modes", "total_turns", "total_tokens")
        return {key: event.get(key) for key in keys if key in event}


def _message_text(message: dict[str, Any]) -> str:
    chunks: list[str] = []
    for part in message.get("content", []) or []:
        if part.get("type") == "text":
            chunks.append(str(part.get("text", "")))
    return "\n".join(chunks).strip()


def _split_requirements(text: str) -> list[str]:
    lines = [line.strip(" -*\t") for line in text.splitlines()]
    parts = [line for line in lines if line]
    if len(parts) <= 1:
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return parts[:12]


def _walk_scalar_items(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_walk_scalar_items(item, next_key))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            rows.extend(_walk_scalar_items(item, f"{prefix}[{idx}]"))
    elif value is not None and not isinstance(value, (dict, list)):
        rows.append((prefix, value))
    return rows


def _walk_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if _record_identity(value):
            records.append(value)
        for item in value.values():
            records.extend(_walk_records(item))
    elif isinstance(value, list):
        for item in value:
            records.extend(_walk_records(item))
    return records


def _action_anchors(body: Any) -> list[tuple[str, Any]]:
    anchors: list[tuple[str, Any]] = []
    for key, value in _walk_scalar_items(body):
        tail = key.rsplit(".", 1)[-1].lower()
        if tail in FREE_TEXT_PARAMETERS:
            continue
        text = str(value).strip()
        if not text or len(text) < 3:
            continue
        if any(marker in tail for marker in ANCHOR_KEYWORDS) or "@" in text:
            anchors.append((key, value))
    return anchors


def _record_identity(record: dict[str, Any]) -> str | None:
    for key, value in record.items():
        if _is_stable_identifier_key(key) and value:
            return f"{key}:{value}"
    for key in IDENTITY_KEYS:
        if key in record and record[key]:
            return f"{key}:{record[key]}"
    return None


def _record_summary(record: dict[str, Any]) -> dict[str, Any]:
    keys = [key for key in record if _is_stable_identifier_key(key)]
    keys.extend(key for key in IDENTITY_KEYS if key in record and key not in keys)
    if not keys:
        keys = list(record.keys())[:5]
    return {key: record.get(key) for key in keys if record.get(key) is not None}


def _record_contains_value(record: dict[str, Any], value_text: str) -> bool:
    for _, item in _walk_scalar_items(record):
        if _normalize_value(item) == value_text:
            return True
    return False


def _record_context_disambiguation(record: dict[str, Any], action_input: Any, node_id: str) -> list[dict[str, Any]]:
    action_text = json.dumps(action_input, ensure_ascii=False).lower()
    evidence: list[dict[str, Any]] = []
    for key, value in _walk_scalar_items(record):
        tail = key.rsplit(".", 1)[-1].lower()
        if tail in {"email", "phone", "id"} or tail.endswith("_id"):
            continue
        text = str(value).strip().lower()
        if len(text) >= 2 and text in action_text:
            evidence.append({"node_id": node_id, "kind": "action_context_matches_record", "field": key})
    return evidence[:3]


def _normalize_value(value: Any) -> str:
    return str(value).strip().lower()


def _strip_array_suffix(key: str) -> str:
    """Normalise ``key[0]`` → ``key`` so array-flattened parameters
    match identity-key checks and entity-type classification."""
    return re.sub(r"\[\d+\]$", "", key)


def _identity_key_tail(key: str) -> str:
    """Normalize nested response paths to their identity-key tail."""
    return _strip_array_suffix(key.rsplit(".", 1)[-1]).lower()


def _identity_values(value: Any) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for key, item in _walk_scalar_items(value):
        normalized = _normalize_value(item)
        if not _is_identity_key(key) or not normalized:
            continue
        values.setdefault(_identity_key_tail(key), set()).add(normalized)
    return values


def _same_observation_context(left: GraphNode, right: GraphNode) -> bool:
    """Return whether two observations answer the same tool request."""
    return (
        left.attrs.get("tool_name") == right.attrs.get("tool_name")
        and left.attrs.get("request_body") == right.attrs.get("request_body")
    )


def _classify_entity_type(key: str) -> str:
    """Map a parameter/identity key tail to a semantic entity type."""
    tail = key.rsplit(".", 1)[-1].lower()
    tail = _strip_array_suffix(tail)
    if tail in ENTITY_TYPE_MAP:
        return ENTITY_TYPE_MAP[tail]
    if tail.endswith("_ids"):
        return tail[:-4] if len(tail) > 4 else "entity"
    if tail.endswith("_id"):
        return tail[:-3] if len(tail) > 3 else "entity"
    return "entity"


def _is_effectful_tool(tool_name: str) -> bool:
    """Check if a tool is effectful (write/send/export/delete), not read-only.

    ``resolves`` edges should only come from read observations.
    """
    semantics = classify_tool(tool_name)
    return bool(semantics.effectful or semantics.mechanism in ALWAYS_REVIEW_EFFECTS)


def _response_is_truncated(body: Any, record_count: int) -> bool:
    """Check if a response was truncated (total > returned records)."""
    if not isinstance(body, dict):
        return False
    total = body.get("total")
    if isinstance(total, int) and total > record_count:
        return True
    if isinstance(total, str) and total.isdigit() and int(total) > record_count:
        return True
    # Also check nested totals
    for value in body.values():
        if isinstance(value, dict):
            t = value.get("total")
            if isinstance(t, int) and t > record_count:
                return True
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    t = item.get("total")
                    if isinstance(t, int) and t > record_count:
                        return True
    return False


def _is_identity_key(key: str) -> bool:
    tail = key.rsplit(".", 1)[-1].lower()
    tail = _strip_array_suffix(tail)
    return tail in IDENTITY_KEYS or tail.endswith("_id") or tail.endswith("_ids")


def _is_stable_identifier_key(key: str) -> bool:
    tail = key.rsplit(".", 1)[-1].lower()
    tail = _strip_array_suffix(tail)
    return tail == "id" or tail.endswith("_id") or tail.endswith("_ids")


def _reported_total(body: Any) -> int | None:
    if isinstance(body, dict):
        total = body.get("total")
        if isinstance(total, int):
            return total
        for value in body.values():
            nested = _reported_total(value)
            if nested is not None:
                return nested
    return None


def _content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def _content_concepts(text: str) -> list[str]:
    concepts: list[str] = []
    for token in _content_tokens(text):
        concept = CONCEPT_ALIASES.get(token, token)
        if concept in STOPWORDS:
            continue
        if concept not in concepts:
            concepts.append(concept)
    return concepts


def write_compilation(result: CompilationResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = result.graph.task_id or result.graph.trace_id
    res_rows = run_batch_ocesq(result.graph)
    ocei = build_ocei(result.graph, res_rows)

    (output / f"{stem}.graph.json").write_text(
        json.dumps(result.graph.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / f"{stem}.candidates.json").write_text(
        json.dumps([asdict(item) for item in result.candidates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output / f"{stem}.monitor_cards.jsonl").open("w", encoding="utf-8") as f:
        for card in result.cards:
            f.write(json.dumps(asdict(card), ensure_ascii=False) + "\n")
    _write_experiment_outputs(output, stem, result, res_rows, ocei)


def _write_experiment_outputs(output: Path, stem: str, result: CompilationResult, res_rows: list[Any], ocei: dict[str, Any]) -> None:
    """Write OEG/OCESQ/RES artifacts required by the experiment plan."""

    eventlog_rows = graph_to_eventlog_rows(result.graph)
    audit_candidates = [
        {
            "action_id": res.action_id,
            "trace_id": res.trace_id,
            "task_id": res.task_id,
            "action_summary": res.action_summary,
            "obligation_statuses": res.obligation_statuses,
        }
        for res in res_rows
    ]
    candidate_paths = [
        path.to_dict()
        for res in res_rows
        for path in res.candidate_paths
    ]
    query_results = query_results_summary(result.graph, res_rows)
    motifs = mine_closure_break_motifs(res_rows)
    report = analysis_report(result.graph, res_rows)
    idx_stats = index_stats(ocei)

    (output / f"{stem}.eventlog.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in eventlog_rows) + ("\n" if eventlog_rows else ""),
        encoding="utf-8",
    )
    (output / f"{stem}.oeg.json").write_text(
        json.dumps(result.graph.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / f"{stem}.audit_candidates.json").write_text(
        json.dumps(audit_candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / f"{stem}.candidate_paths.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in candidate_paths) + ("\n" if candidate_paths else ""),
        encoding="utf-8",
    )
    (output / f"{stem}.res.jsonl").write_text(
        "\n".join(json.dumps(res.to_dict(), ensure_ascii=False, sort_keys=True) for res in res_rows) + ("\n" if res_rows else ""),
        encoding="utf-8",
    )
    (output / f"{stem}.closure_break_motifs.json").write_text(
        json.dumps(motifs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / f"{stem}.query_results.json").write_text(
        json.dumps(query_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / f"{stem}.analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / f"{stem}.ocei.json").write_text(
        json.dumps(ocei, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / f"{stem}.index_stats.json").write_text(
        json.dumps(idx_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile tool-calling JSONL traces into behavior graphs.")
    parser.add_argument("trace", nargs="+", help="Trace JSONL file(s) or directories containing JSONL files.")
    parser.add_argument("--output-dir", default="behavior_graph_out", help="Directory for graph/candidate/card outputs.")
    args = parser.parse_args(argv)

    traces: list[Path] = []
    for raw in args.trace:
        path = Path(raw)
        if path.is_dir():
            traces.extend(sorted(path.glob("*.jsonl")))
        else:
            traces.append(path)

    total_cards = 0
    for trace in traces:
        result = compile_trace_file(trace)
        write_compilation(result, args.output_dir)
        total_cards += len(result.cards)
        print(f"{trace.name}: nodes={len(result.graph.nodes)} edges={len(result.graph.edges)} cards={len(result.cards)}")
    print(f"compiled={len(traces)} cards={total_cards} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
