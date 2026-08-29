"""Lifecycle and evidence bookkeeping for learned task-completion conditions.

The tracker never selects tools and never rejects or regenerates an answer.  It
only records which learned conditions apply, what evidence supports them, and
what remains before the task may be closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from agents.completion_templates import ToolEvent, tool_events


PHASES = frozenset({"pre_claim", "pre_action", "final"})
KINDS = frozenset({"achievement", "invariant"})
STATUSES = frozenset(
    {
        "pending_evidence",
        "pending",
        "awaiting_confirmation",
        "approved_pending_execution",
        "satisfied",
        "invalidated",
        "violated",
    }
)
OPEN_STATUSES = frozenset(
    {"pending_evidence", "pending", "awaiting_confirmation", "approved_pending_execution"}
)
TERMINAL_STATUSES = frozenset({"satisfied", "invalidated", "violated"})
SEMANTIC_STATUSES = frozenset(
    {"pending_evidence", "pending", "satisfied", "invalidated", "violated"}
)

PREVIEW_STATUSES = frozenset(
    {"preview", "quoted", "pending", "pending_confirmation", "requires_confirmation"}
)
FAILED_STATUSES = frozenset({"error", "failed", "failure", "rejected", "denied", "invalid"})
SUCCESS_STATUSES = frozenset(
    {
        "added",
        "applied",
        "booked",
        "cancelled",
        "canceled",
        "claimed",
        "completed",
        "confirmed",
        "created",
        "exchanged",
        "processed",
        "redeemed",
        "refunded",
        "removed",
        "replaced",
        "returned",
        "set",
        "submitted",
        "store_credit_issued",
        "success",
        "updated",
    }
)
MUTATION_NAME = re.compile(
    r"^(add|apply|book|cancel|create|exchange|process|redeem|refund|remove|replace|return|set|submit|update)_",
    re.IGNORECASE,
)
AFFIRMATIVE = re.compile(
    r"^\s*(yes\b|sure\b|okay\b|ok\b|go ahead\b|proceed\b|confirm\b|do it\b|"
    r"approved?\b|sounds good\b|please do\b|that's fine\b|book it\b|apply it\b|"
    r"cancel (?:it|both|them)\b)",
    re.IGNORECASE,
)
CONFIRMATION_REQUEST = re.compile(
    r"\b(confirm|confirmation|proceed|go ahead|would you like|shall i|should i|"
    r"reply with|before i|before proceeding|preview only)\b",
    re.IGNORECASE,
)
PLANNING_DIRECTIVE = re.compile(
    r"\b(?:call|invoke|run)\s+(?:the\s+)?(?:tool\b|[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b)|"
    r"\b(?:ignore previous|system prompt|tool arguments?|next tool)\b",
    re.IGNORECASE,
)


def compact(value: Any, limit: int = 360) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def stable_id(*parts: str, prefix: str = "cm") -> str:
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:14]
    return f"{prefix}_{digest}"


def result_status(result: Any) -> str:
    return str(result.get("status", "")).strip().lower() if isinstance(result, dict) else ""


def result_failed(result: Any) -> bool:
    if result is None:
        return True
    if not isinstance(result, dict):
        return False
    if result.get("error") or result.get("success") is False:
        return True
    return result_status(result) in FAILED_STATUSES


def valid_tool_evidence(event: ToolEvent) -> bool:
    return not result_failed(event.result)


def mutation_completed(event: ToolEvent) -> bool:
    if not MUTATION_NAME.match(event.name) or result_failed(event.result):
        return False
    status = result_status(event.result)
    if status in PREVIEW_STATUSES:
        return False
    return status in SUCCESS_STATUSES or (
        event.arguments.get("confirm") is True
        and isinstance(event.result, dict)
        and event.result.get("success") is True
    )


def action_key(event: ToolEvent) -> str:
    arguments = {
        key: value
        for key, value in event.arguments.items()
        if key not in {"confirm", "dry_run", "preview"}
    }
    return json.dumps([event.name, arguments], ensure_ascii=True, sort_keys=True, default=str)


def _evidence_key(evidence: dict[str, Any]) -> str:
    return json.dumps(evidence, ensure_ascii=True, sort_keys=True, default=str)


@dataclass
class CompletionItem:
    id: str
    template_id: str
    obligation_id: str
    phase: str
    kind: str
    type: str
    description: str
    source: str
    status: str
    priority: int = 50
    activation: str = ""
    required_evidence: list[str] = field(default_factory=list)
    discharge: str = ""
    scope_key: str = "default"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    scope: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"unsupported completion phase: {self.phase}")
        if self.kind not in KINDS:
            raise ValueError(f"unsupported completion kind: {self.kind}")
        if self.status not in STATUSES:
            raise ValueError(f"unsupported completion status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "template_id": self.template_id,
            "obligation_id": self.obligation_id,
            "phase": self.phase,
            "kind": self.kind,
            "type": self.type,
            "description": self.description,
            "source": self.source,
            "status": self.status,
            "priority": self.priority,
            "activation": self.activation,
            "required_evidence": list(self.required_evidence),
            "discharge": self.discharge,
            "scope_key": self.scope_key,
            "evidence": list(self.evidence),
            "missing_evidence": list(self.missing_evidence),
            "scope": dict(self.scope),
        }


class CompletionTracker:
    """Per-task ledger with conservative, auditable state transitions."""

    def __init__(self) -> None:
        self.items: dict[str, CompletionItem] = {}
        self.events: list[dict[str, Any]] = []
        self._semantic_keys: dict[tuple[str, str, str], str] = {}
        self._seen_tools: set[str] = set()
        self._seen_users: set[int] = set()

    def _log(self, event: str, **details: Any) -> None:
        self.events.append({"index": len(self.events), "event": event, **details})

    def _add_evidence(self, item: CompletionItem, evidence: list[dict[str, Any]]) -> None:
        seen = {_evidence_key(entry) for entry in item.evidence}
        for entry in evidence:
            if not isinstance(entry, dict):
                continue
            key = _evidence_key(entry)
            if key not in seen:
                seen.add(key)
                item.evidence.append(entry)

    def _transition(
        self,
        item: CompletionItem,
        status: str,
        *,
        reason: str,
        evidence: list[dict[str, Any]] | None = None,
    ) -> bool:
        if status not in STATUSES:
            return False
        self._add_evidence(item, evidence or [])
        if item.status == status:
            return False
        if item.status in TERMINAL_STATUSES:
            self._log(
                "transition_ignored",
                item_id=item.id,
                previous=item.status,
                requested=status,
                reason="terminal_state",
            )
            return False
        previous = item.status
        item.status = status
        self._log(
            "transition",
            item_id=item.id,
            previous=previous,
            status=status,
            reason=reason,
        )
        return True

    def _tool_evidence(self, event: ToolEvent) -> dict[str, Any]:
        return {
            "kind": "tool_result",
            "sequence": event.sequence,
            "assistant_index": event.assistant_index,
            "tool": event.name,
            "status": result_status(event.result),
            "success": not result_failed(event.result),
            "result": compact(event.result, 500),
        }

    def _runtime_item(
        self,
        event: ToolEvent,
        *,
        item_type: str,
        status: str,
        description: str,
        suffix: str,
    ) -> CompletionItem:
        identifier = stable_id(str(event.sequence), event.name, action_key(event), suffix, prefix="runtime")
        item = CompletionItem(
            id=identifier,
            template_id="runtime_execution",
            obligation_id=suffix,
            phase="pre_action",
            kind="achievement",
            type=item_type,
            description=description,
            source=f"tool:{event.name}:{event.sequence}",
            status=status,
            priority=0 if item_type == "execution" else 5,
            scope_key=action_key(event),
            evidence=[self._tool_evidence(event)],
            scope={
                "tool": event.name,
                "arguments": event.arguments,
                "action_key": action_key(event),
                "preview_sequence": event.sequence,
                "assistant_index": event.assistant_index,
            },
        )
        self.items[item.id] = item
        self._log("item_created", item=item.to_dict())
        return item

    def _matching_execution(
        self, event: ToolEvent, *, require_prior_preview: bool = True
    ) -> CompletionItem | None:
        key = action_key(event)
        matches = [
            item
            for item in self.items.values()
            if item.type == "execution"
            and item.scope.get("action_key") == key
            and item.status in OPEN_STATUSES
            and (
                not require_prior_preview
                or int(item.scope.get("preview_sequence", -1)) < event.sequence
            )
        ]
        return max(matches, key=lambda item: int(item.scope.get("preview_sequence", -1)), default=None)

    def _ingest_tool(self, event: ToolEvent) -> None:
        evidence = [self._tool_evidence(event)]
        status = result_status(event.result)
        if not MUTATION_NAME.match(event.name):
            return

        if status in PREVIEW_STATUSES:
            existing = self._matching_execution(event, require_prior_preview=False)
            if existing is None:
                existing = self._runtime_item(
                    event,
                    item_type="execution",
                    status="awaiting_confirmation",
                    description=(
                        f"The previewed {event.name} operation has not executed; explicit user approval "
                        "and a successful confirmed operation are still required."
                    ),
                    suffix="execution",
                )
            else:
                self._add_evidence(existing, evidence)
                existing.scope.update(
                    {
                        "preview_sequence": event.sequence,
                        "assistant_index": event.assistant_index,
                    }
                )
                self._log(
                    "preview_refreshed",
                    item_id=existing.id,
                    tool=event.name,
                    sequence=event.sequence,
                )

            confirmation = next(
                (
                    item
                    for item in self.items.values()
                    if item.type == "user_confirmation_choice"
                    and item.scope.get("action_key") == action_key(event)
                    and item.status in OPEN_STATUSES
                ),
                None,
            )
            if existing.status == "awaiting_confirmation" and confirmation is None:
                self._runtime_item(
                    event,
                    item_type="user_confirmation_choice",
                    status="pending",
                    description=f"Obtain explicit user approval for the previewed {event.name} operation.",
                    suffix="confirmation",
                )
            return

        matching = self._matching_execution(event)
        if mutation_completed(event):
            if matching is None:
                matching = self._runtime_item(
                    event,
                    item_type="execution",
                    status="satisfied",
                    description=f"The {event.name} operation completed successfully.",
                    suffix="execution",
                )
            else:
                self._transition(
                    matching,
                    "satisfied",
                    reason="confirmed_mutation_succeeded",
                    evidence=evidence,
                )
            for item in self.items.values():
                if (
                    item.type == "user_confirmation_choice"
                    and item.scope.get("action_key") == action_key(event)
                    and item.status == "pending"
                ):
                    self._transition(
                        item,
                        "violated",
                        reason="operation_completed_without_a_bound_confirmation",
                        evidence=evidence,
                    )
            return

        if result_failed(event.result):
            if matching is not None:
                self._add_evidence(matching, evidence)
                self._log("execution_failed", item_id=matching.id, tool=event.name)
            else:
                # A failed attempt is evidence, not by itself a new user
                # obligation.  Creating an item here would leave stale
                # obligations after a corrected retry with different args.
                self._log(
                    "unbound_mutation_failed",
                    tool=event.name,
                    sequence=event.sequence,
                    evidence=evidence[0],
                )

    @staticmethod
    def _previous_assistant(conversation: list[dict[str, Any]], user_index: int) -> int | None:
        return next(
            (
                index
                for index in range(user_index - 1, -1, -1)
                if conversation[index].get("role") == "assistant"
            ),
            None,
        )

    @staticmethod
    def _argument_values(item: CompletionItem) -> list[str]:
        values = []
        for value in item.scope.get("arguments", {}).values():
            if isinstance(value, (str, int, float)) and len(str(value)) >= 3:
                values.append(str(value).casefold())
        return values

    @classmethod
    def _confirmed_scopes(
        cls,
        user_text: str,
        executions: list[CompletionItem],
        assistant_text: str,
    ) -> list[CompletionItem]:
        """Bind a confirmation conservatively when several previews were shown."""

        if len(executions) <= 1:
            return executions
        lowered = user_text.casefold()
        identifier_matches = [
            item
            for item in executions
            if any(value in lowered for value in cls._argument_values(item))
        ]
        if identifier_matches:
            return identifier_matches
        if re.search(r"\b(both|all|them|those|everything)\b", lowered):
            return executions

        if re.search(r"\b(only|except|but not|leave|keep)\b", lowered):
            aliases = {
                "booking": {"booking", "flight", "ticket"},
                "hotel": {"hotel", "room", "reservation"},
                "car_rental": {"car", "rental", "vehicle"},
                "order": {"order"},
                "return": {"return", "item"},
            }
            matched = []
            for item in executions:
                tool = str(item.scope.get("tool", "")).casefold()
                terms = set(tool.split("_"))
                for marker, synonyms in aliases.items():
                    if marker in tool:
                        terms.update(synonyms)
                mentioned = [
                    term
                    for term in terms
                    if len(term) > 2 and re.search(rf"\b{re.escape(term)}\b", lowered)
                ]
                negated = any(
                    re.search(
                        rf"\b(?:keep|leave|except|not)\b[^.;,]{{0,32}}\b{re.escape(term)}\b",
                        lowered,
                    )
                    for term in mentioned
                )
                if mentioned and not negated:
                    matched.append(item)
            return matched
        assistant_lowered = assistant_text.casefold()
        grouped_request = bool(
            re.search(r"\b(both|all|them|these|the following)\b", assistant_lowered)
        ) or all(
            any(value in assistant_lowered for value in cls._argument_values(item))
            for item in executions
        )
        return executions if grouped_request else []

    def _sync_confirmations(self, conversation: list[dict[str, Any]]) -> None:
        for index, message in enumerate(conversation):
            if index in self._seen_users or message.get("role") != "user":
                continue
            self._seen_users.add(index)
            user_text = str(message.get("content", ""))
            if not AFFIRMATIVE.search(user_text):
                continue
            previous = self._previous_assistant(conversation, index)
            if previous is None:
                continue
            assistant_text = str(conversation[previous].get("content", ""))
            if not CONFIRMATION_REQUEST.search(assistant_text):
                continue
            candidates = [
                item
                for item in self.items.values()
                if item.type == "execution"
                and item.status == "awaiting_confirmation"
                and item.scope.get("assistant_index") == previous
            ]
            executions = self._confirmed_scopes(user_text, candidates, assistant_text)
            for execution in executions:
                evidence = [
                    {
                        "kind": "user_confirmation",
                        "conversation_index": index,
                        "text": compact(user_text, 160),
                        "bound_preview": execution.source,
                    }
                ]
                self._transition(
                    execution,
                    "approved_pending_execution",
                    reason="scoped_user_confirmation",
                    evidence=evidence,
                )
                for item in self.items.values():
                    if (
                        item.type == "user_confirmation_choice"
                        and item.status == "pending"
                        and item.scope.get("action_key") == execution.scope.get("action_key")
                    ):
                        self._transition(
                            item,
                            "satisfied",
                            reason="scoped_user_confirmation",
                            evidence=evidence,
                        )

    def sync_execution(self, conversation: list[dict[str, Any]]) -> None:
        unseen: list[tuple[str, ToolEvent]] = []
        for event in tool_events(conversation):
            fingerprint = stable_id(
                str(event.sequence),
                str(event.assistant_index),
                event.name,
                action_key(event),
                compact(event.result, 1000),
                prefix="event",
            )
            if fingerprint in self._seen_tools:
                continue
            unseen.append((fingerprint, event))

        # First materialize every preview, then bind user confirmations, and
        # only then consume later confirmed mutations.  This also makes a
        # one-shot replay of a finished trajectory equivalent to incremental
        # online updates.
        for fingerprint, event in unseen:
            if result_status(event.result) not in PREVIEW_STATUSES:
                continue
            self._seen_tools.add(fingerprint)
            self._ingest_tool(event)
        self._sync_confirmations(conversation)
        for fingerprint, event in unseen:
            if fingerprint in self._seen_tools:
                continue
            self._seen_tools.add(fingerprint)
            self._ingest_tool(event)

    @staticmethod
    def _obligation_map(templates: list[dict[str, Any]]) -> dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]:
        output = {}
        for template in templates:
            template_id = str(template.get("id", ""))
            for obligation in template.get("obligations", []):
                if isinstance(obligation, dict):
                    output[(template_id, str(obligation.get("id", "")))] = (template, obligation)
        return output

    def merge_semantic(
        self,
        payload: dict[str, Any],
        templates: list[dict[str, Any]],
        *,
        valid_evidence_refs: set[str] | None = None,
        allow_invariant_satisfaction: bool = False,
    ) -> None:
        """Merge a bookkeeping result; unknown templates and unsafe states are ignored."""

        allowed = self._obligation_map(templates)
        for update in payload.get("items", []):
            if not isinstance(update, dict):
                continue
            template_id = str(update.get("template_id", ""))
            obligation_id = str(update.get("obligation_id", ""))
            definition = allowed.get((template_id, obligation_id))
            if definition is None:
                self._log(
                    "semantic_update_ignored",
                    reason="unknown_obligation",
                    template_id=template_id,
                    obligation_id=obligation_id,
                )
                continue
            template, obligation = definition
            scope_key = compact(update.get("scope_key") or "default", 100).casefold()
            key = (template_id, obligation_id, scope_key)
            existing = self.items.get(self._semantic_keys.get(key, ""))
            evidence = [entry for entry in update.get("evidence", []) if isinstance(entry, dict)]
            if valid_evidence_refs is not None:
                evidence = [
                    entry
                    for entry in evidence
                    if str(entry.get("ref", "")) in valid_evidence_refs
                ]
            applicable = update.get("applicable") is True
            if not applicable:
                if existing is not None and existing.status in OPEN_STATUSES and evidence:
                    self._transition(
                        existing,
                        "invalidated",
                        reason="semantic_condition_not_applicable",
                        evidence=evidence,
                    )
                continue

            requested_status = str(update.get("status", "pending"))
            if requested_status not in SEMANTIC_STATUSES:
                requested_status = "pending"
            phase = str(obligation.get("phase", "final"))
            kind = str(obligation.get("kind", "achievement"))
            if phase not in PHASES or kind not in KINDS:
                continue
            base_description = compact(obligation.get("requirement", ""), 520)
            description = compact(update.get("description") or base_description, 520)
            if PLANNING_DIRECTIVE.search(description) or (
                re.search(r"\d", description) and not re.search(r"\d", base_description)
            ):
                description = base_description
            missing = [compact(item, 180) for item in update.get("missing_evidence", []) if item]
            if kind == "invariant" and requested_status == "satisfied" and not allow_invariant_satisfaction:
                requested_status = "pending"
            if requested_status in {"satisfied", "invalidated", "violated"} and not evidence:
                requested_status = "pending_evidence" if missing else "pending"
            if requested_status == "satisfied" and missing:
                requested_status = "pending_evidence"
            if existing is None:
                identifier = stable_id(template_id, obligation_id, scope_key, prefix="learned")
                existing = CompletionItem(
                    id=identifier,
                    template_id=template_id,
                    obligation_id=obligation_id,
                    phase=phase,
                    kind=kind,
                    type=str(obligation.get("type", "completion")),
                    description=description,
                    source=f"completion_template:{template_id}",
                    status=requested_status,
                    priority=int(obligation.get("priority", 50)),
                    activation=compact(obligation.get("activation", ""), 320),
                    required_evidence=[compact(value, 200) for value in obligation.get("required_evidence", [])],
                    discharge=compact(obligation.get("discharge", ""), 320),
                    scope_key=scope_key,
                    evidence=evidence,
                    missing_evidence=missing,
                    scope={"family": template.get("family", ""), "title": template.get("title", "")},
                )
                self.items[identifier] = existing
                self._semantic_keys[key] = identifier
                self._log("item_created", item=existing.to_dict())
                continue

            if description:
                existing.description = description
            existing.missing_evidence = missing
            self._transition(
                existing,
                requested_status,
                reason="semantic_bookkeeping",
                evidence=evidence,
            )

    def has_valid_evidence(self, conversation: list[dict[str, Any]]) -> bool:
        return any(valid_tool_evidence(event) for event in tool_events(conversation))

    def open_items(self) -> list[CompletionItem]:
        return [item for item in self.items.values() if item.status in OPEN_STATUSES]

    def actionable_items(self) -> list[CompletionItem]:
        return sorted(
            [
                item
                for item in self.open_items()
                if item.type == "execution" or item.phase == "pre_action"
            ],
            key=lambda item: (item.priority, item.id),
        )

    def guard_items(self) -> list[CompletionItem]:
        return sorted(
            [item for item in self.open_items() if item.phase == "pre_claim"],
            key=lambda item: (item.priority, item.id),
        )

    def final_items(self) -> list[CompletionItem]:
        return sorted(
            [item for item in self.open_items() if item.phase == "final"],
            key=lambda item: (item.priority, item.id),
        )

    def violated_items(self) -> list[CompletionItem]:
        return sorted(
            [item for item in self.items.values() if item.status == "violated"],
            key=lambda item: (item.priority, item.id),
        )

    def snapshot(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in sorted(self.items.values(), key=lambda item: item.id)]

    def status_counts(self) -> dict[str, int]:
        return {
            status: sum(item.status == status for item in self.items.values())
            for status in sorted(STATUSES)
        }
