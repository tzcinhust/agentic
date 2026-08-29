"""Domain-agnostic task-closure memory for the archived PWM agent.

Procedure memory remains responsible for tool planning.  This module only
tracks what may still need to be covered before a task is declared complete.
It does not call an LLM, select tools, or inspect benchmark oracle fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


COMPLETION_TYPES = frozenset(
    {
        "comparison",
        "explanation_rationale",
        "cost_amount_reporting",
        "proactive_disclosure",
        "user_confirmation_choice",
        "boundary_must_not",
        "final_state_reporting",
        "execution",
    }
)
COMPLETION_STATUSES = frozenset({"pending", "satisfied", "invalidated"})
COMMUNICATION_TYPES = frozenset(
    {
        "comparison",
        "explanation_rationale",
        "cost_amount_reporting",
        "proactive_disclosure",
        "final_state_reporting",
    }
)

TYPE_LABELS = {
    "comparison": "comparison",
    "explanation_rationale": "explanation or rationale",
    "cost_amount_reporting": "cost or amount reporting",
    "proactive_disclosure": "material disclosure",
    "user_confirmation_choice": "confirmation or choice",
    "boundary_must_not": "must-not boundary",
    "final_state_reporting": "final-state reporting",
    "execution": "execution outcome",
}

TYPE_PATTERNS = {
    "comparison": re.compile(
        r"\b(compare|comparison|difference|versus|vs\.?|which (?:one|option|path)|cheaper|better)\b",
        re.IGNORECASE,
    ),
    "explanation_rationale": re.compile(
        r"\b(why|explain|reason|rationale|basis|break\s*down|breakdown|how .*calculat)\b",
        re.IGNORECASE,
    ),
    "cost_amount_reporting": re.compile(
        r"(?:\$|\b(cost|fee|refund|price|amount|total|discount|credit|points?|fare difference)\b)",
        re.IGNORECASE,
    ),
    "proactive_disclosure": re.compile(
        r"\b(warn|warning|disclose|disclosure|mention|inform|tell|report|show|state|summari[sz]e|notify)\b",
        re.IGNORECASE,
    ),
    "user_confirmation_choice": re.compile(
        r"\b(confirm|confirmation|approval|consent|choose|choice|ask (?:me|the user)|"
        r"before (?:you|doing|changing|cancelling|canceling)|wait for|until (?:I|the user))\b",
        re.IGNORECASE,
    ),
    "boundary_must_not": re.compile(
        r"\b(do not|don't|must not|never|without|leave .* unchanged|keep .* unchanged|only|"
        r"should not|cannot|can't|not allowed)\b",
        re.IGNORECASE,
    ),
    "final_state_reporting": re.compile(
        r"\b(final|final state|final status|after (?:the )?(?:change|update|return|refund|"
        r"exchange|cancellation|booking)|completed|done|remains? active|result)\b",
        re.IGNORECASE,
    ),
}

SECTION = re.compile(r"^(Verify first|Procedure|Branches|Avoid):\s*$", re.IGNORECASE)
MUTATION_NAME = re.compile(
    r"^(add|apply|book|cancel|create|process|redeem|remove|set|update)_",
    re.IGNORECASE,
)
AMOUNT_FIELD = re.compile(
    r"(amount|balance|cost|credit|discount|fare|fee|points|price|refund|subtotal|total)",
    re.IGNORECASE,
)
DISCLOSURE_FIELD = re.compile(
    r"(deadline|eligib|expires?|penalty|policy|reason|restriction|warning)",
    re.IGNORECASE,
)
FAILED_STATUSES = frozenset({"error", "failed", "rejected"})
PREVIEW_STATUSES = frozenset({"pending_confirmation", "preview", "quoted"})
SUCCESS_STATUSES = frozenset(
    {
        "applied",
        "booked",
        "cancelled",
        "canceled",
        "completed",
        "created",
        "exchanged",
        "processed",
        "redeemed",
        "refunded",
        "removed",
        "returned",
        "success",
        "updated",
    }
)

AFFIRMATIVE = re.compile(
    r"^\s*(yes|yes[, ]|sure|okay|ok\b|go ahead|proceed|confirm|do it|"
    r"option\s+\d+|the first|the second|cancel both|book it|apply it)",
    re.IGNORECASE,
)
CONFIRMATION_REQUEST = re.compile(
    r"\b(confirm|confirmation|proceed|go ahead|which|choose|choice|if you want|"
    r"would you like|shall I|reply with)\b",
    re.IGNORECASE,
)


def _compact(text: Any, limit: int = 320) -> str:
    normalized = " ".join(str(text or "").split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


def _stable_id(item_type: str, description: str, source: str) -> str:
    payload = f"{item_type}\n{description.lower()}\n{source}".encode("utf-8")
    return f"cm_{hashlib.sha1(payload).hexdigest()[:12]}"


@dataclass
class CompletionItem:
    id: str
    type: str
    description: str
    source: str
    status: str = "pending"
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.type not in COMPLETION_TYPES:
            raise ValueError(f"unsupported completion type: {self.type}")
        if self.status not in COMPLETION_STATUSES:
            raise ValueError(f"unsupported completion status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "source": self.source,
            "status": self.status,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ToolEvent:
    sequence: int
    name: str
    arguments: dict[str, Any]
    result: Any


def result_succeeded(result: Any) -> bool:
    if not isinstance(result, dict):
        return result is not None
    if result.get("error") or result.get("success") is False:
        return False
    return str(result.get("status", "")).lower() not in FAILED_STATUSES


def tool_events(conversation: Iterable[dict[str, Any]]) -> list[ToolEvent]:
    events: list[ToolEvent] = []
    for message in conversation:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            events.append(
                ToolEvent(
                    sequence=len(events),
                    name=str(call.get("name", "")),
                    arguments=call.get("arguments") or {},
                    result=call.get("result"),
                )
            )
    return events


def has_valid_tool_evidence(conversation: Iterable[dict[str, Any]]) -> bool:
    return any(result_succeeded(event.result) for event in tool_events(conversation))


def classify_completion_types(text: str) -> list[str]:
    return [item_type for item_type, pattern in TYPE_PATTERNS.items() if pattern.search(text)]


def latest_user_completion_types(conversation: Iterable[dict[str, Any]]) -> set[str]:
    messages = [
        str(item.get("content", ""))
        for item in conversation
        if item.get("role") == "user" and "[TASK_DONE]" not in str(item.get("content", ""))
    ]
    return set(classify_completion_types(messages[-1])) if messages else set()


def _workflow_bullets(text: str) -> list[tuple[str, str]]:
    section = ""
    bullets: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = SECTION.match(line)
        if match:
            section = match.group(1).lower()
            continue
        if line.endswith(":") and not line.startswith("-"):
            section = ""
        if section in {"branches", "avoid"} and line.startswith("-"):
            bullets.append((section, _compact(line[1:])))
    return bullets


def static_completion_requirements(
    workflows: list[str],
    conversation: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[str]:
    """Return untracked text requirements for the static ablation.

    This intentionally does not instantiate CompletionItem or maintain status
    and evidence.  It is a prompt-only baseline.
    """

    requirements: list[str] = []
    seen: set[str] = set()
    for item in conversation:
        if item.get("role") != "user" or "[TASK_DONE]" in str(item.get("content", "")):
            continue
        content = _compact(item.get("content", ""))
        for item_type in classify_completion_types(content):
            requirement = f"Address the user's {TYPE_LABELS[item_type]}: {content}"
            key = requirement.lower()
            if key not in seen:
                seen.add(key)
                requirements.append(requirement)
    for workflow in workflows:
        for _section, bullet in _workflow_bullets(workflow):
            if not classify_completion_types(bullet):
                continue
            key = bullet.lower()
            if key not in seen:
                seen.add(key)
                requirements.append(bullet)
            if len(requirements) >= limit:
                return requirements
    return requirements[:limit]


def _field_names(value: Any) -> set[str]:
    names: set[str] = set()

    def visit(current: Any) -> None:
        if isinstance(current, dict):
            for key, child in current.items():
                names.add(str(key))
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    return names


def _status(result: Any) -> str:
    return str(result.get("status", "")).lower() if isinstance(result, dict) else ""


def _mutation_completed(event: ToolEvent) -> bool:
    if not result_succeeded(event.result) or _status(event.result) in PREVIEW_STATUSES:
        return False
    return _status(event.result) in SUCCESS_STATUSES or event.arguments.get("confirm") is True


def _preview_sources_for_assistant(
    conversation: list[dict[str, Any]], assistant_index: int
) -> set[str]:
    sources: set[str] = set()
    sequence = 0
    for message_index, message in enumerate(conversation):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if message_index == assistant_index and _status(call.get("result")) in PREVIEW_STATUSES:
                sources.add(f"tool:{call.get('name', '')}:{sequence}")
            sequence += 1
    return sources


class CompletionMemory:
    """Persistent per-task ledger of structured completion requirements."""

    def __init__(self, *, max_workflow_items: int = 12):
        self.items: list[CompletionItem] = []
        self.max_workflow_items = max_workflow_items
        self._keys: set[tuple[str, ...]] = set()
        self._seen_tool_events: set[str] = set()
        self._seen_user_messages: set[str] = set()

    def add(
        self,
        item_type: str,
        description: str,
        source: str,
        *,
        status: str = "pending",
        evidence: list[dict[str, Any]] | None = None,
    ) -> CompletionItem | None:
        description = _compact(description)
        # Repeated tool operations can carry independent confirmation and
        # execution obligations even when their human-readable text matches.
        source_scoped = source.startswith("tool:") and item_type in {
            "execution",
            "user_confirmation_choice",
            "final_state_reporting",
        }
        key = (
            (item_type, description.lower(), source)
            if source_scoped
            else (item_type, description.lower())
        )
        if not description or key in self._keys:
            return None
        self._keys.add(key)
        item = CompletionItem(
            id=_stable_id(item_type, description, source),
            type=item_type,
            description=description,
            source=source,
            status=status,
            evidence=list(evidence or []),
        )
        self.items.append(item)
        return item

    def ingest_user_messages(self, conversation: list[dict[str, Any]]) -> None:
        user_index = 0
        for item in conversation:
            if item.get("role") != "user" or "[TASK_DONE]" in str(item.get("content", "")):
                continue
            content = _compact(item.get("content", ""))
            fingerprint = f"{user_index}:{content}"
            if fingerprint not in self._seen_user_messages:
                self._seen_user_messages.add(fingerprint)
                for item_type in classify_completion_types(content):
                    self.add(
                        item_type,
                        f"Address the user's {TYPE_LABELS[item_type]}: {content}",
                        f"user:turn{user_index}",
                        evidence=[{"kind": "user_request", "turn": user_index}],
                    )
            user_index += 1
        self._satisfy_explicit_confirmations(conversation)

    def ingest_workflows(self, workflows: list[str]) -> None:
        existing = sum(item.source.startswith("workflow:") for item in self.items)
        remaining = self.max_workflow_items - existing
        if remaining <= 0:
            return
        added = 0
        for rank, workflow in enumerate(workflows, start=1):
            for section, bullet in _workflow_bullets(workflow):
                for item_type in classify_completion_types(bullet):
                    if self.add(
                        item_type,
                        bullet,
                        f"workflow:rank{rank}:{section}",
                        evidence=[{"kind": "learned_workflow", "rank": rank, "section": section}],
                    ):
                        added += 1
                    if added >= remaining:
                        return

    def sync_evidence(self, conversation: list[dict[str, Any]]) -> None:
        self.ingest_user_messages(conversation)
        for event in tool_events(conversation):
            fingerprint = hashlib.sha1(
                json.dumps(
                    [event.sequence, event.name, event.arguments, event.result],
                    sort_keys=True,
                    ensure_ascii=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            if fingerprint in self._seen_tool_events:
                continue
            self._seen_tool_events.add(fingerprint)
            self._ingest_tool_event(event)
        self._satisfy_explicit_confirmations(conversation)
        completed_mutations = [
            event
            for event in tool_events(conversation)
            if MUTATION_NAME.match(event.name) and _mutation_completed(event)
        ]
        for item in self.items:
            match = re.fullmatch(r"tool:(.+):(\d+)", item.source)
            if item.type != "user_confirmation_choice" or item.status != "pending" or not match:
                continue
            preview_tool, preview_sequence = match.group(1), int(match.group(2))
            completion = next(
                (
                    event
                    for event in completed_mutations
                    if event.name == preview_tool and event.sequence > preview_sequence
                ),
                None,
            )
            if completion is not None:
                item.status = "invalidated"
                item.evidence.append(
                    {
                        "kind": "condition_passed",
                        "reason": "preview was followed by a completed operation",
                        "tool": completion.name,
                        "sequence": completion.sequence,
                    }
                )

    def _ingest_tool_event(self, event: ToolEvent) -> None:
        success = result_succeeded(event.result)
        evidence = {
            "kind": "tool_result",
            "sequence": event.sequence,
            "tool": event.name,
            "success": success,
            "status": _status(event.result),
        }
        is_mutation = bool(MUTATION_NAME.match(event.name))
        completed_mutation = is_mutation and _mutation_completed(event)
        if completed_mutation:
            self.add(
                "execution",
                f"The operation represented by {event.name} is complete.",
                f"tool:{event.name}:{event.sequence}",
                status="satisfied",
                evidence=[evidence],
            )
        elif is_mutation and not success:
            self.add(
                "execution",
                f"The operation represented by {event.name} is not complete; report the failure if ending.",
                f"tool:{event.name}:{event.sequence}",
                status="pending",
                evidence=[evidence],
            )
        if not success:
            return

        fields = _field_names(event.result)
        if any(AMOUNT_FIELD.search(name) for name in fields):
            self.add(
                "cost_amount_reporting",
                f"Report material amounts returned by {event.name} when relevant to the user's request.",
                f"tool:{event.name}:{event.sequence}",
                evidence=[evidence | {"fields": sorted(name for name in fields if AMOUNT_FIELD.search(name))[:12]}],
            )
        if any(DISCLOSURE_FIELD.search(name) for name in fields):
            self.add(
                "proactive_disclosure",
                f"Disclose material constraints or warnings returned by {event.name} when relevant.",
                f"tool:{event.name}:{event.sequence}",
                evidence=[evidence | {"fields": sorted(name for name in fields if DISCLOSURE_FIELD.search(name))[:12]}],
            )
        if _status(event.result) in PREVIEW_STATUSES:
            self.add(
                "user_confirmation_choice",
                f"Do not describe the preview from {event.name} as completed without the user's choice or confirmation.",
                f"tool:{event.name}:{event.sequence}",
                evidence=[evidence],
            )
        if completed_mutation:
            self.add(
                "final_state_reporting",
                f"Report the tool-confirmed final outcome of {event.name} without inventing additional changes.",
                f"tool:{event.name}:{event.sequence}",
                evidence=[evidence],
            )

    def _satisfy_explicit_confirmations(self, conversation: list[dict[str, Any]]) -> None:
        for index, item in enumerate(conversation):
            if item.get("role") != "user" or not AFFIRMATIVE.search(str(item.get("content", ""))):
                continue
            previous_assistant_index = next(
                (
                    prior
                    for prior in range(index - 1, -1, -1)
                    if conversation[prior].get("role") == "assistant"
                ),
                None,
            )
            if previous_assistant_index is None:
                continue
            previous_assistant = str(
                conversation[previous_assistant_index].get("content", "")
            )
            if not CONFIRMATION_REQUEST.search(previous_assistant):
                continue
            preview_sources = _preview_sources_for_assistant(
                conversation, previous_assistant_index
            )
            if not preview_sources:
                continue
            for completion in self.items:
                if (
                    completion.type == "user_confirmation_choice"
                    and completion.status == "pending"
                    and completion.source in preview_sources
                ):
                    completion.status = "satisfied"
                    completion.evidence.append(
                        {
                            "kind": "user_confirmation",
                            "conversation_index": index,
                            "bound_preview_source": completion.source,
                            "text": _compact(item.get("content", ""), 120),
                        }
                    )

    def pending(self) -> list[CompletionItem]:
        return [item for item in self.items if item.status == "pending"]

    def prompt_items(self, *, limit: int = 6) -> list[CompletionItem]:
        priority = {
            "execution": 0,
            "boundary_must_not": 1,
            "comparison": 2,
            "explanation_rationale": 3,
            "cost_amount_reporting": 4,
            "proactive_disclosure": 5,
            "user_confirmation_choice": 6,
            "final_state_reporting": 7,
        }
        ordered = sorted(
            self.pending(),
            key=lambda item: (
                0 if item.source.startswith("user:") else 1 if item.source.startswith("tool:") else 2,
                priority[item.type],
                item.id,
            ),
        )
        selected: list[CompletionItem] = []
        seen_descriptions: set[str] = set()
        for item in ordered:
            normalized = re.sub(r"\s+", " ", item.description).strip().rstrip(".").casefold()
            if normalized in seen_descriptions:
                continue
            seen_descriptions.add(normalized)
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def snapshot(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.items]
