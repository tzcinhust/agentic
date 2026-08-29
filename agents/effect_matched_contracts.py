"""Effect-matched closure contracts and deterministic runtime evaluation.

The module is deliberately independent from Process Workflow Memory (PWM).
PWM decides how to proceed.  A closure contract describes user-level
conditions that must hold at a response boundary.  Contract states are derived
from the current observable transcript; they are not appended turn after turn.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


PREVIEW_STATUSES = frozenset(
    {"preview", "quoted", "pending", "pending_confirmation", "requires_confirmation"}
)
FAILED_STATUSES = frozenset(
    {"error", "failed", "failure", "rejected", "denied", "invalid", "unavailable"}
)
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
        "issued",
        "processed",
        "redeemed",
        "refunded",
        "removed",
        "replaced",
        "returned",
        "set",
        "submitted",
        "success",
        "updated",
    }
)
SUCCESS_STATUS_SUFFIXES = (
    "_issued",
    "_completed",
    "_confirmed",
    "_created",
    "_updated",
)
MUTATION_NAME = re.compile(
    r"^(?:add|apply|book|cancel|create|exchange|issue|process|redeem|refund|remove|"
    r"replace|return|set|submit|update)_",
    re.IGNORECASE,
)
AFFIRMATIVE = re.compile(
    r"^\s*(?:yes\b|sure\b|okay\b|ok\b|go ahead\b|proceed\b|confirm\b|do it\b|"
    r"approved?\b|sounds good\b|please do\b|that's fine\b|book it\b|apply it\b|"
    r"that works\b|cancel (?:it|both|them)\b)",
    re.IGNORECASE,
)
NEGATIVE = re.compile(
    r"^\s*(?:no\b|nope\b|do not\b|don't\b|stop\b|never mind\b|nevermind\b|"
    r"decline\b|skip (?:it|that|this|them|those)\b)",
    re.IGNORECASE,
)
CONFIRMATION_REQUEST = re.compile(
    r"\b(?:confirm|confirmation|proceed|go ahead|would you like|shall i|should i|"
    r"reply with|before i|before proceeding|preview only|want me to|ready to|"
    r"if (?:you want|you would like|you'd like)[^.!?]{0,80}i can)\b",
    re.IGNORECASE,
)
ENTITY_ID = re.compile(r"\b[A-Z]{1,8}[-_][A-Z0-9]{2,}\b")
NUMBER_CLAIM = re.compile(
    r"(?:[$£€]\s*\d|\b\d+(?:[.,]\d+)?\s*(?:%|USD|dollars?|euros?|pounds?|"
    r"hours?|days?|nights?|points?)\b)",
    re.IGNORECASE,
)
DATE_LITERAL = re.compile(
    r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|"
    r"sep|oct|nov|dec)[a-z]*\s+\d{1,2})\b",
    re.IGNORECASE,
)
BARE_NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])")
STATUS_CLAIM = re.compile(
    r"\b(?:cancelled|canceled|booked|confirmed|refunded|returned|exchanged|applied|"
    r"completed|processed|eligible|ineligible|approved|denied)\b",
    re.IGNORECASE,
)
CAUSAL_PATTERN = re.compile(
    r"\b(?:because|due to|based on|which means|therefore|so|as a result|the reason)\b",
    re.IGNORECASE,
)
COMPARISON_MARKERS = (
    "compared",
    "comparison",
    "difference",
    "versus",
    " vs ",
    "whereas",
    "while",
    "both",
    "each",
)
CONTROL_ARGUMENTS = frozenset({"confirm", "dry_run", "preview"})
IDENTITY_ARGUMENTS = frozenset(
    {
        "id",
        "code",
        "sku",
        "promo_code",
        "coupon_code",
        "order_number",
        "confirmation_number",
    }
)
ACTION_SCOPE_STOPWORDS = frozenset(
    {
        "add",
        "apply",
        "book",
        "cancel",
        "create",
        "exchange",
        "id",
        "item",
        "process",
        "request",
        "return",
        "set",
        "submit",
        "update",
    }
)
RETRIEVAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "amount_or_threshold",
        "are",
        "as",
        "at",
        "be",
        "before",
        "can",
        "current",
        "date",
        "do",
        "for",
        "from",
        "help",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "need",
        "number",
        "of",
        "on",
        "please",
        "report",
        "task",
        "that",
        "the",
        "this",
        "to",
        "user",
        "entity",
        "entity_id",
        "want",
        "with",
        "you",
    }
)


def compact(value: Any, limit: int = 500) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    )


def stable_hash(value: Any, *, prefix: str = "") -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.casefold())


def char_ngrams(text: str, n: int = 4) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return {
        normalized[index : index + n]
        for index in range(max(0, len(normalized) - n + 1))
    }


def _retrieval_content_token(token: str) -> bool:
    return token not in RETRIEVAL_STOPWORDS and (
        len(token) >= 3 or bool(re.search(r"[\u4e00-\u9fff]", token))
    )


def normalize_retrieval_query(value: Any, limit: int = 2500) -> str:
    text = compact(value, limit * 2)
    text = ENTITY_ID.sub(" entity_id ", text)
    text = NUMBER_CLAIM.sub(" amount_or_threshold ", text)
    text = DATE_LITERAL.sub(" date ", text)
    text = BARE_NUMBER.sub(" number ", text)
    return compact(text, limit)


@dataclass(frozen=True)
class ToolEvent:
    sequence: int
    name: str
    arguments: dict[str, Any]
    result: Any
    assistant_index: int

    @property
    def status(self) -> str:
        if not isinstance(self.result, dict):
            return ""
        return str(self.result.get("status", "")).strip().casefold()


def _tool_records(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    if isinstance(content, dict):
        return [content]
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            return []
        return _tool_records(payload)
    return []


def tool_events(conversation: list[dict[str, Any]]) -> list[ToolEvent]:
    """Normalize folded and explicit tool-result transcript layouts."""

    events: list[ToolEvent] = []
    sequence = 0
    pending: tuple[int, list[dict[str, Any]]] | None = None
    for index, message in enumerate(conversation):
        role = message.get("role")
        if role == "assistant":
            calls = [
                item
                for item in (message.get("tool_calls") or [])
                if isinstance(item, dict)
            ]
            unresolved: list[dict[str, Any]] = []
            for call in calls:
                if "result" not in call:
                    unresolved.append(call)
                    continue
                events.append(
                    ToolEvent(
                        sequence=sequence,
                        name=str(call.get("name", "")),
                        arguments=(
                            call.get("arguments")
                            if isinstance(call.get("arguments"), dict)
                            else {}
                        ),
                        result=call.get("result"),
                        assistant_index=index,
                    )
                )
                sequence += 1
            pending = (index, unresolved) if unresolved else None
            continue

        if role != "tool":
            continue
        # STATE-Bench's working conversation repeats already-executed folded
        # records in a following role=tool message. Only consume a tool message
        # when the preceding assistant calls were genuinely unresolved.
        if pending is None:
            continue
        assistant_index, calls = pending
        for position, record in enumerate(_tool_records(message.get("content"))):
            call = calls[position] if position < len(calls) else {}
            events.append(
                ToolEvent(
                    sequence=sequence,
                    name=str(record.get("name") or call.get("name") or ""),
                    arguments=(
                        record.get("arguments")
                        if isinstance(record.get("arguments"), dict)
                        else call.get("arguments")
                        if isinstance(call.get("arguments"), dict)
                        else {}
                    ),
                    result=record.get("result", record),
                    assistant_index=assistant_index,
                )
            )
            sequence += 1
        pending = None
    return events


def _assistant_text_is_user_visible(
    conversation: list[dict[str, Any]], index: int
) -> bool:
    """STATE-Bench hides assistant text from intermediate tool generations."""

    return not (
        index + 1 < len(conversation) and conversation[index + 1].get("role") == "tool"
    )


def result_failed(result: Any) -> bool:
    if result is None:
        return True
    if not isinstance(result, dict):
        return False
    if result.get("error") or result.get("success") is False:
        return True
    return str(result.get("status", "")).strip().casefold() in FAILED_STATUSES


def mutation_completed(event: ToolEvent) -> bool:
    if not MUTATION_NAME.match(event.name) or result_failed(event.result):
        return False
    if event_is_preview(event):
        return False
    if event.status in SUCCESS_STATUSES or event.status.endswith(
        SUCCESS_STATUS_SUFFIXES
    ):
        return True
    # Some STATE-Bench tools omit a status after a confirmed mutation.  A
    # confirmed, non-error result is the conservative fallback.
    return event.arguments.get("confirm") is True and isinstance(event.result, dict)


def _control_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _tool_parameters(tool: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    definition = (
        tool.get("function") if isinstance(tool.get("function"), dict) else tool
    )
    name = str(definition.get("name", ""))
    parameters = next(
        (
            definition.get(key)
            for key in ("parameters", "input_schema", "schema")
            if isinstance(definition.get(key), dict)
        ),
        {},
    )
    return name, parameters


def proposed_effect_kind(
    call: dict[str, Any], tools: list[dict[str, Any]] | None = None
) -> str:
    """Classify a proposed call without mistaking a preview for a write.

    Explicit call arguments take precedence.  When a control argument is
    omitted, a JSON-schema default may establish that the tool is preview-only.
    Descriptive prose is deliberately ignored because it is not executable
    evidence of the call's effect.
    """

    name = str(call.get("name", ""))
    if not MUTATION_NAME.match(name):
        return "read_only"
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    if "confirm" in arguments and _control_value(arguments.get("confirm")) is False:
        return "preview"
    if "dry_run" in arguments and _control_value(arguments.get("dry_run")) is True:
        return "preview"
    if "preview" in arguments and _control_value(arguments.get("preview")) is True:
        return "preview"

    schema = next(
        (
            parameters
            for tool in tools or []
            if isinstance(tool, dict)
            for tool_name, parameters in [_tool_parameters(tool)]
            if tool_name == name
        ),
        {},
    )
    properties = (
        schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    )
    defaults = {
        key: definition.get("default")
        for key, definition in properties.items()
        if isinstance(definition, dict) and "default" in definition
    }
    if "confirm" not in arguments and _control_value(defaults.get("confirm")) is False:
        return "preview"
    if "dry_run" not in arguments and _control_value(defaults.get("dry_run")) is True:
        return "preview"
    if "preview" not in arguments and _control_value(defaults.get("preview")) is True:
        return "preview"
    return "potential_mutation"


def event_is_preview(event: ToolEvent) -> bool:
    return (
        event.status in PREVIEW_STATUSES
        or proposed_effect_kind({"name": event.name, "arguments": event.arguments})
        == "preview"
    )


def _non_control_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in arguments.items() if key not in CONTROL_ARGUMENTS
    }


def _identity_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in arguments.items()
        if key not in CONTROL_ARGUMENTS
        and (key.casefold().endswith("_id") or key.casefold() in IDENTITY_ARGUMENTS)
    }


def _material_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    identity = set(_identity_arguments(arguments))
    return {
        key: value
        for key, value in _non_control_arguments(arguments).items()
        if key not in identity
    }


def action_scope_key(event: ToolEvent) -> str:
    return canonical_json([event.name, _identity_arguments(event.arguments)])


def action_key(event: ToolEvent) -> str:
    return canonical_json(
        [
            event.name,
            _identity_arguments(event.arguments),
            _material_arguments(event.arguments),
        ]
    )


def _derived_argument_values(result: Any) -> dict[str, list[Any]]:
    output: dict[str, list[Any]] = {}
    for path, value in _flatten(result):
        if not path or isinstance(value, (dict, list)):
            continue
        leaf = path.rsplit(".", 1)[-1].casefold()
        output.setdefault(leaf, []).append(value)
    return output


def effect_fingerprint(event: ToolEvent) -> str:
    """Fingerprint a realized mutation without treating previews as effects."""

    return stable_hash(
        {
            "tool": event.name,
            "arguments": {
                key: value
                for key, value in event.arguments.items()
                if key not in CONTROL_ARGUMENTS
            },
            "status": event.status,
            "result": event.result,
        },
        prefix="effect_",
    )


def effect_signatures(conversation: list[dict[str, Any]]) -> dict[int, str]:
    """Return conservative cumulative effect signatures at assistant checkpoints.

    Equality means no additional successful mutation occurred and the exact
    realized-mutation history is unchanged.  It intentionally permits false
    negatives rather than pairing trajectories with different external effects.
    """

    by_index: dict[int, list[ToolEvent]] = {}
    for event in tool_events(conversation):
        by_index.setdefault(event.assistant_index, []).append(event)
    realized: list[str] = []
    output: dict[int, str] = {}
    for index, message in enumerate(conversation):
        if message.get("role") != "assistant":
            continue
        for event in by_index.get(index, []):
            if mutation_completed(event):
                realized.append(effect_fingerprint(event))
        output[index] = stable_hash(realized, prefix="state_")
    return output


class TruthValue(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class EvidenceFact:
    ref: str
    source: str
    value: Any
    path: str
    tool: str = ""
    outcome: str = "text"
    sequence: int = -1
    conversation_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "source": self.source,
            "tool": self.tool,
            "outcome": self.outcome,
            "path": self.path,
            "value": self.value,
            "sequence": self.sequence,
            "conversation_index": self.conversation_index,
        }


def _flatten(value: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    if depth > 6:
        return [(prefix, compact(value, 200))]
    output = [(prefix, value)] if prefix else []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.extend(_flatten(child, path, depth + 1))
    elif isinstance(value, list):
        for index, child in enumerate(value[:50]):
            path = f"{prefix}.{index}" if prefix else str(index)
            output.extend(_flatten(child, path, depth + 1))
    return output


def _path_pattern(value: Any) -> str:
    pattern = str(value or "*").strip()
    if pattern.startswith("$."):
        pattern = pattern[2:]
    pattern = pattern.replace("[*]", ".*")
    pattern = re.sub(r"\[(\d+)\]", r".\1", pattern)
    return pattern or "*"


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return compact(left, 1000).casefold() == compact(right, 1000).casefold()


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", str(value))
    return float(match.group(0)) if match else None


def _predicate_result(value: Any, selector: dict[str, Any]) -> bool:
    operator = str(selector.get("operator", "exists"))
    expected = selector.get("value")
    values = selector.get("values") if isinstance(selector.get("values"), list) else []
    if operator == "exists":
        return True
    if operator == "nonempty":
        return value not in (None, "", [], {})
    if operator == "truthy":
        return bool(value)
    if operator == "falsy":
        return not bool(value)
    if operator == "equals":
        return _same_value(value, expected)
    if operator == "not_equals":
        return not _same_value(value, expected)
    if operator == "in":
        return any(_same_value(value, item) for item in values)
    if operator in {"contains", "contains_any", "contains_all"}:
        lowered = compact(value, 4000).casefold()
        needles = (
            [str(expected)]
            if operator == "contains"
            else [str(item) for item in values]
        )
        present = [needle.casefold() in lowered for needle in needles if needle]
        return bool(present) and (
            all(present) if operator == "contains_all" else any(present)
        )
    if operator in {"gt", "gte", "lt", "lte"}:
        current, target = _numeric(value), _numeric(expected)
        if current is None or target is None:
            return False
        return {
            "gt": current > target,
            "gte": current >= target,
            "lt": current < target,
            "lte": current <= target,
        }[operator]
    raise ValueError(f"unsupported evidence operator: {operator}")


class EvidenceLedger:
    """Immutable evidence view reconstructed from observable messages and tools."""

    def __init__(self, conversation: list[dict[str, Any]]):
        self.conversation = conversation
        self.events = tool_events(conversation)
        self.facts: list[EvidenceFact] = []
        self._build()

    def _build(self) -> None:
        for index, message in enumerate(self.conversation):
            role = str(message.get("role", ""))
            if role not in {"user", "assistant"}:
                continue
            if role == "assistant" and not _assistant_text_is_user_visible(
                self.conversation, index
            ):
                continue
            self.facts.append(
                EvidenceFact(
                    ref=f"M{index}",
                    source=f"{role}_text",
                    value=str(message.get("content", "")),
                    path="content",
                    conversation_index=index,
                )
            )
        for event in self.events:
            outcome = (
                "failure"
                if result_failed(event.result)
                else "preview"
                if event_is_preview(event)
                else "success"
            )
            for path, value in _flatten(event.arguments):
                self.facts.append(
                    EvidenceFact(
                        ref=f"T{event.sequence}:arg:{path}",
                        source="tool_argument",
                        tool=event.name,
                        outcome=outcome,
                        path=path,
                        value=value,
                        sequence=event.sequence,
                        conversation_index=event.assistant_index,
                    )
                )
            for path, value in _flatten(event.result):
                self.facts.append(
                    EvidenceFact(
                        ref=f"T{event.sequence}:result:{path}",
                        source="tool_result",
                        tool=event.name,
                        outcome=outcome,
                        path=path,
                        value=value,
                        sequence=event.sequence,
                        conversation_index=event.assistant_index,
                    )
                )

    def matching(
        self, selector: dict[str, Any], *, trustworthy_only: bool = False
    ) -> list[EvidenceFact]:
        source = str(selector.get("source", "tool_result"))
        tool_pattern = str(selector.get("tool", "*") or "*")
        path_pattern = _path_pattern(selector.get("path"))
        requested_outcome = str(selector.get("outcome", "any"))
        return [
            fact
            for fact in self.facts
            if fact.source == source
            and (requested_outcome == "any" or fact.outcome == requested_outcome)
            and not (
                trustworthy_only
                and fact.outcome == "failure"
                and requested_outcome != "failure"
            )
            and fnmatch.fnmatchcase(fact.tool.casefold(), tool_pattern.casefold())
            and fnmatch.fnmatchcase(fact.path.casefold(), path_pattern.casefold())
        ]

    def evaluate(
        self, selector: dict[str, Any]
    ) -> tuple[TruthValue, list[EvidenceFact]]:
        facts = self.matching(selector)
        if not facts:
            return TruthValue.UNKNOWN, []
        results = [_predicate_result(fact.value, selector) for fact in facts]
        quantifier = str(selector.get("quantifier", "any"))
        if quantifier == "all":
            return (TruthValue.TRUE if all(results) else TruthValue.FALSE), facts
        if quantifier == "consistent":
            if all(results):
                return TruthValue.TRUE, facts
            if not any(results):
                return TruthValue.FALSE, facts
            return TruthValue.CONFLICT, facts
        return (TruthValue.TRUE if any(results) else TruthValue.FALSE), facts

    def satisfying(
        self, selector: dict[str, Any], *, trustworthy_only: bool = False
    ) -> list[EvidenceFact]:
        return [
            fact
            for fact in self.matching(selector, trustworthy_only=trustworthy_only)
            if _predicate_result(fact.value, selector)
        ]

    def compact_facts(
        self, facts: list[EvidenceFact], limit: int = 6
    ) -> list[dict[str, Any]]:
        return [
            {
                "ref": fact.ref,
                "source": fact.source,
                "tool": fact.tool,
                "outcome": fact.outcome,
                "path": fact.path,
                "value": compact(fact.value, 180),
                "conversation_index": fact.conversation_index,
            }
            for fact in facts[:limit]
        ]


@dataclass
class ActionRecord:
    key: str
    scope_key: str
    tool: str
    arguments: dict[str, Any]
    state: str
    preview_sequence: int
    preview_assistant_index: int
    preview_fingerprint: str = ""
    derived_arguments: dict[str, list[Any]] = field(default_factory=dict)
    executed_arguments: dict[str, Any] = field(default_factory=dict)
    confirmation_obtained: bool = False
    violation: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "scope_key": self.scope_key,
            "tool": self.tool,
            "arguments": self.arguments,
            "derived_arguments": self.derived_arguments,
            "executed_arguments": self.executed_arguments,
            "state": self.state,
            "preview_sequence": self.preview_sequence,
            "preview_assistant_index": self.preview_assistant_index,
            "preview_fingerprint": self.preview_fingerprint,
            "confirmation_obtained": self.confirmation_obtained,
            "violation": self.violation,
            "evidence": list(self.evidence),
        }


class ActionLedger:
    """Conservative preview -> approval -> execution state reconstructed from a trace."""

    def __init__(self, conversation: list[dict[str, Any]]):
        self.records: list[ActionRecord] = []
        self._conversation = conversation
        self._events_by_assistant: dict[int, list[ToolEvent]] = {}
        for event in tool_events(conversation):
            self._events_by_assistant.setdefault(event.assistant_index, []).append(
                event
            )
        self._build()

    @staticmethod
    def _argument_values(record: ActionRecord) -> list[str]:
        return [
            str(value).casefold()
            for value in _identity_arguments(record.arguments).values()
            if isinstance(value, (str, int, float)) and len(str(value)) >= 3
        ]

    @staticmethod
    def _scope_terms(record: ActionRecord) -> set[str]:
        terms = {
            token
            for token in tokens(
                " ".join(
                    [record.tool, *record.arguments.keys()]
                    + [
                        str(value)
                        for value in record.arguments.values()
                        if isinstance(value, str)
                        and len(value) <= 40
                        and not ENTITY_ID.fullmatch(value)
                    ]
                ).replace("_", " ")
            )
            if len(token) >= 3 and token not in ACTION_SCOPE_STOPWORDS
        }
        return terms

    @classmethod
    def _scoped_mentions(
        cls, user_text: str, candidates: list[ActionRecord]
    ) -> tuple[list[ActionRecord], list[ActionRecord]]:
        positive: list[ActionRecord] = []
        negative: list[ActionRecord] = []
        terms_by_record = [cls._scope_terms(record) for record in candidates]
        term_counts = Counter(
            term for record_terms in terms_by_record for term in record_terms
        )
        clauses = re.split(r"[;,]|\bbut\b", user_text.casefold())
        for clause in clauses:
            is_negative = bool(
                re.search(r"\b(?:except|keep|leave|not|do not|don't|skip)\b", clause)
            )
            for record, record_terms in zip(candidates, terms_by_record):
                identifiers = cls._argument_values(record)
                terms = {term for term in record_terms if term_counts[term] == 1}
                mentioned = any(value in clause for value in identifiers) or any(
                    re.search(rf"\b{re.escape(term)}\b", clause) for term in terms
                )
                if not mentioned:
                    continue
                target = negative if is_negative else positive
                if record not in target:
                    target.append(record)
        return positive, negative

    def _open_match(self, event: ToolEvent) -> ActionRecord | None:
        matches = [
            record
            for record in self.records
            if record.scope_key == action_scope_key(event)
            and self._materially_compatible(record, event.arguments)
            and record.state in {"awaiting_confirmation", "approved_pending_execution"}
            and record.preview_sequence < event.sequence
        ]
        return max(matches, key=lambda item: item.preview_sequence, default=None)

    @staticmethod
    def _materially_compatible(
        record: ActionRecord, proposed_arguments: dict[str, Any]
    ) -> bool:
        approved = _material_arguments(record.arguments)
        proposed = _material_arguments(proposed_arguments)
        for key in approved.keys() | proposed.keys():
            if key not in proposed:
                return False
            if key in approved and _same_value(approved[key], proposed[key]):
                continue
            lowered_key = key.casefold()
            derived = [
                value
                for result_key, values in record.derived_arguments.items()
                if result_key == lowered_key
                or result_key.endswith(f"_{lowered_key}")
                or lowered_key.endswith(f"_{result_key}")
                for value in values
            ]
            if not any(_same_value(proposed[key], value) for value in derived):
                return False
        return True

    def _ingest_event(self, event: ToolEvent) -> None:
        if not MUTATION_NAME.match(event.name):
            return
        if event_is_preview(event):
            preview_fingerprint = stable_hash(
                {
                    "tool": event.name,
                    "arguments": _non_control_arguments(event.arguments),
                    "result": event.result,
                },
                prefix="preview_",
            )
            existing = next(
                (
                    record
                    for record in reversed(self.records)
                    if record.scope_key == action_scope_key(event)
                    and record.state
                    in {"awaiting_confirmation", "approved_pending_execution"}
                ),
                None,
            )
            if existing is None:
                self.records.append(
                    ActionRecord(
                        key=action_key(event),
                        scope_key=action_scope_key(event),
                        tool=event.name,
                        arguments=_non_control_arguments(event.arguments),
                        state="awaiting_confirmation",
                        preview_sequence=event.sequence,
                        preview_assistant_index=event.assistant_index,
                        preview_fingerprint=preview_fingerprint,
                        derived_arguments=_derived_argument_values(event.result),
                        confirmation_obtained=False,
                        evidence=[f"T{event.sequence}:preview"],
                    )
                )
            else:
                if (
                    existing.state == "approved_pending_execution"
                    and existing.preview_fingerprint != preview_fingerprint
                ):
                    existing.state = "awaiting_confirmation"
                    existing.confirmation_obtained = False
                existing.key = action_key(event)
                existing.arguments = _non_control_arguments(event.arguments)
                existing.preview_sequence = event.sequence
                existing.preview_assistant_index = event.assistant_index
                existing.preview_fingerprint = preview_fingerprint
                existing.derived_arguments = _derived_argument_values(event.result)
                existing.evidence.append(f"T{event.sequence}:preview")
            return
        matching = self._open_match(event)
        if mutation_completed(event):
            if matching is None:
                scope_conflict = any(
                    record.scope_key == action_scope_key(event)
                    and record.state
                    in {"awaiting_confirmation", "approved_pending_execution"}
                    for record in self.records
                )
                self.records.append(
                    ActionRecord(
                        key=action_key(event),
                        scope_key=action_scope_key(event),
                        tool=event.name,
                        arguments=_non_control_arguments(event.arguments),
                        state="executed",
                        preview_sequence=event.sequence,
                        preview_assistant_index=event.assistant_index,
                        preview_fingerprint="",
                        executed_arguments=_non_control_arguments(event.arguments),
                        confirmation_obtained=False,
                        violation=(
                            "material_arguments_changed_after_preview"
                            if scope_conflict
                            else ""
                        ),
                        evidence=[f"T{event.sequence}:executed"],
                    )
                )
            else:
                if not matching.confirmation_obtained:
                    matching.violation = "executed_without_bound_confirmation"
                matching.state = "executed"
                matching.executed_arguments = _non_control_arguments(event.arguments)
                matching.evidence.append(f"T{event.sequence}:executed")
        elif result_failed(event.result) and matching is not None:
            matching.evidence.append(f"T{event.sequence}:failed")

    def _scope_confirmation(
        self, user_text: str, assistant_text: str, candidates: list[ActionRecord]
    ) -> list[ActionRecord]:
        if len(candidates) <= 1:
            return candidates
        lowered = user_text.casefold()
        scoped_positive, scoped_negative = self._scoped_mentions(user_text, candidates)
        if scoped_positive:
            return [item for item in scoped_positive if item not in scoped_negative]
        if scoped_negative and re.search(
            r"\b(?:yes|sure|okay|ok|go ahead|proceed|all|both)\b", lowered
        ):
            return [item for item in candidates if item not in scoped_negative]
        if re.search(r"\b(?:except|keep|leave|not|do not|don't|skip)\b", lowered):
            return []
        identifiers = [
            record
            for record in candidates
            if any(value in lowered for value in self._argument_values(record))
        ]
        if identifiers:
            return identifiers
        if re.search(r"\b(?:both|all|them|those|everything)\b", lowered):
            return candidates
        assistant_lowered = assistant_text.casefold()
        assistant_scoped = [
            record
            for record in candidates
            if any(
                value in assistant_lowered for value in self._argument_values(record)
            )
        ]
        if len(assistant_scoped) == 1:
            return assistant_scoped
        # With several previews, ungrounded noun matching is unsafe: shared
        # verbs and negated phrases that mention another action
        # can bind the wrong action.  If no stable identifier or explicit
        # all/both scope is present, preserve every item as pending.
        grouped = bool(
            re.search(r"\b(?:both|all|them|the following)\b", assistant_lowered)
        )
        return candidates if grouped else []

    def _scope_rejection(
        self, user_text: str, candidates: list[ActionRecord]
    ) -> list[ActionRecord]:
        if len(candidates) <= 1:
            return candidates
        lowered = user_text.casefold()
        identifiers = [
            record
            for record in candidates
            if any(value in lowered for value in self._argument_values(record))
        ]
        if identifiers:
            return identifiers
        if re.search(r"\b(?:both|all|them|those|everything)\b", lowered):
            return candidates
        return []

    def _build(self) -> None:
        for index, message in enumerate(self._conversation):
            if message.get("role") == "assistant":
                for event in self._events_by_assistant.get(index, []):
                    self._ingest_event(event)
                continue
            if message.get("role") != "user":
                continue
            text = str(message.get("content", ""))
            affirmative = bool(AFFIRMATIVE.search(text))
            negative = bool(NEGATIVE.search(text))
            if not affirmative and not negative:
                continue
            previous = next(
                (
                    prior
                    for prior in range(index - 1, -1, -1)
                    if self._conversation[prior].get("role") == "assistant"
                ),
                None,
            )
            if previous is None:
                continue
            assistant_text = str(self._conversation[previous].get("content", ""))
            if not CONFIRMATION_REQUEST.search(assistant_text):
                continue
            candidates = [
                record
                for record in self.records
                if record.state == "awaiting_confirmation"
                and record.preview_assistant_index == previous
            ]
            if negative:
                for record in self._scope_rejection(text, candidates):
                    record.state = "invalidated"
                    record.evidence.append(f"M{index}:rejection")
                continue
            selected = self._scope_confirmation(text, assistant_text, candidates)
            _, excluded = self._scoped_mentions(text, candidates)
            for record in selected:
                record.state = "approved_pending_execution"
                record.confirmation_obtained = True
                record.evidence.append(f"M{index}:confirmation")
            for record in excluded:
                record.state = "invalidated"
                record.evidence.append(f"M{index}:excluded_choice")
            if selected and re.search(r"\bonly\b", text, re.IGNORECASE):
                for record in candidates:
                    if record not in selected and record.state != "invalidated":
                        record.state = "invalidated"
                        record.evidence.append(f"M{index}:excluded_choice")

    def evaluate(
        self, requirement: dict[str, Any]
    ) -> tuple[TruthValue, list[ActionRecord]]:
        tool_pattern = str(requirement.get("tool", "*") or "*")
        expected = requirement.get("state", "executed")
        expected_states = (
            {str(item) for item in expected}
            if isinstance(expected, list)
            else {str(expected)}
        )
        matches = [
            record
            for record in self.records
            if fnmatch.fnmatchcase(record.tool.casefold(), tool_pattern.casefold())
        ]
        if not matches:
            return TruthValue.UNKNOWN, []
        states = [
            record.state in expected_states
            or (
                expected_states == {"approved_pending_execution"}
                and record.state == "executed"
            )
            for record in matches
        ]
        if requirement.get("confirmation_required") is True:
            states = [
                state and record.confirmation_obtained
                for state, record in zip(states, matches)
            ]
        quantifier = str(requirement.get("quantifier", "any"))
        satisfied = all(states) if quantifier == "all" else any(states)
        return (TruthValue.TRUE if satisfied else TruthValue.FALSE), matches


def _selector_nodes(value: Any):
    if isinstance(value, dict):
        if {"source", "path", "operator"}.issubset(value):
            yield value
        for child in value.values():
            yield from _selector_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _selector_nodes(child)


def _runtime_contract_is_safe(contract: dict[str, Any]) -> bool:
    applicability = contract.get("applicability") or {}
    if any(
        isinstance(selector, dict) and selector.get("source") == "assistant_text"
        for selector in applicability.get("predicates", [])
    ):
        return False
    for obligation in contract.get("obligations", []):
        if not isinstance(obligation, dict):
            continue
        for group in obligation.get("evidence_requirements", []):
            if any(
                isinstance(selector, dict)
                and selector.get("source") == "assistant_text"
                for selector in (group.get("any_of", []) if isinstance(group, dict) else [])
            ):
                return False
        for clause in obligation.get("response_requirements", []):
            if not isinstance(clause, dict):
                continue
            if clause.get("kind") == "mention_evidence" and any(
                isinstance(selector, dict)
                and selector.get("source") == "assistant_text"
                for selector in [
                    *clause.get("selectors", []),
                    *[
                        member
                        for group in clause.get("selector_groups", [])
                        if isinstance(group, list)
                        for member in group
                    ],
                ]
            ):
                return False
    for selector in _selector_nodes(contract):
        value = selector.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                support = int(selector.get("value_support", 0))
            except (TypeError, ValueError):
                return False
            value_evidence = selector.get("value_evidence")
            if (
                selector.get("value_kind") != "structural_constant"
                or not isinstance(value_evidence, dict)
                or value_evidence.get("source") != "tool_result"
                or not value_evidence.get("tool")
                or not value_evidence.get("path")
                or support < 2
                or not str(selector.get("value_provenance_sha256", ""))
            ):
                return False
        values = selector.get("values")
        if isinstance(values, list) and any(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in values
        ):
            return False
    return True


class EffectMatchedContractIndex:
    """Independent one-shot retriever for learned closure contracts."""

    def __init__(
        self,
        artifact: dict[str, Any],
        *,
        domain: str | None,
        contract_set: str = "runtime",
        top_k: int = 3,
        relative_threshold: float = 0.32,
        minimum_score: float = 0.25,
    ):
        if int(artifact.get("version", 0)) < 4:
            raise ValueError("effect-matched contract artifact must have version >= 4")
        if artifact.get("kind") != "effect_matched_closure_contracts":
            raise ValueError("unexpected closure contract artifact kind")
        if contract_set not in {"runtime", "monitor"}:
            raise ValueError("contract_set must be runtime or monitor")
        self.top_k = max(1, int(top_k))
        self.relative_threshold = max(0.0, min(1.0, float(relative_threshold)))
        self.minimum_score = max(0.0, float(minimum_score))
        source_contracts = (
            artifact.get("monitor_contracts", [])
            if contract_set == "monitor" and "monitor_contracts" in artifact
            else artifact.get("contracts", [])
        )
        available = [
            item
            for item in source_contracts
            if isinstance(item, dict)
            and item.get("obligations")
            and (domain is None or item.get("domain") == domain)
        ]
        unsafe = [
            str(item.get("id", "unknown"))
            for item in available
            if not _runtime_contract_is_safe(item)
        ]
        if unsafe:
            raise ValueError(
                "runtime contract contains an unsafe or unvalidated selector: "
                + ", ".join(unsafe)
            )
        self.contracts = available
        self.document_frequency = Counter(
            token for item in self.contracts for token in set(item.get("tokens", []))
        )
        self.average_length = sum(
            len(item.get("tokens", [])) for item in self.contracts
        ) / max(len(self.contracts), 1)
        self.ngrams = [
            char_ngrams(str(item.get("search_text", ""))) for item in self.contracts
        ]

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        domain: str | None,
        contract_set: str = "runtime",
        top_k: int = 3,
        relative_threshold: float = 0.32,
        minimum_score: float = 0.25,
    ) -> "EffectMatchedContractIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            payload,
            domain=domain,
            contract_set=contract_set,
            top_k=top_k,
            relative_threshold=relative_threshold,
            minimum_score=minimum_score,
        )

    def _score(self, query: str, index: int, contract: dict[str, Any]) -> float:
        query_counts = Counter(tokens(query))
        document_counts = Counter(contract.get("tokens", []))
        query_ngrams = char_ngrams(query)
        query_content = {
            token for token in query_counts if _retrieval_content_token(token)
        }
        document_content = {
            token for token in document_counts if _retrieval_content_token(token)
        }
        lexical = 0.0
        for token, query_frequency in query_counts.items():
            if not _retrieval_content_token(token):
                continue
            frequency = document_counts.get(token, 0)
            if not frequency:
                continue
            frequency_in_documents = self.document_frequency.get(token, 0)
            inverse = math.log(
                1
                + (len(self.contracts) - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            denominator = frequency + 1.3 * (
                0.25
                + 0.75 * sum(document_counts.values()) / max(self.average_length, 1)
            )
            lexical += inverse * frequency * 2.3 / denominator * min(query_frequency, 2)
        character = len(query_ngrams & self.ngrams[index]) / max(len(query_ngrams), 1)
        if not (query_content & document_content) and character < 0.035:
            return 0.0
        confidence = float(contract.get("confidence", 0.5))
        support = math.log1p(max(1, int(contract.get("support", 1))))
        heldout = float(contract.get("validation", {}).get("precision", 0.5))
        specificity = float(contract.get("validation", {}).get("specificity", 0.5))
        return (
            lexical
            + 7.0 * character
            + 0.18 * support
            + 0.25 * confidence
            + 0.12 * heldout
            + 0.1 * specificity
        )

    def retrieve_with_scores(self, query: str) -> list[tuple[float, dict[str, Any]]]:
        if not query.strip() or not self.contracts:
            return []
        ranked = sorted(
            (
                (self._score(query, index, contract), contract)
                for index, contract in enumerate(self.contracts)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] <= 0:
            return []
        floor = max(self.minimum_score, ranked[0][0] * self.relative_threshold)
        selected: list[tuple[float, dict[str, Any]]] = []
        family_counts: Counter[str] = Counter()
        for score, contract in ranked:
            family = str(contract.get("family", ""))
            if score < floor or score <= 0 or family_counts[family] >= 2:
                continue
            selected.append((score, contract))
            family_counts[family] += 1
            if len(selected) >= self.top_k:
                break
        return selected

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        return [contract for _, contract in self.retrieve_with_scores(query)]


@dataclass
class ObligationState:
    contract_id: str
    obligation_id: str
    deadline: str
    type: str
    requirement: str
    priority: int
    status: str
    applicability: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    missing_evidence_tools: list[str] = field(default_factory=list)
    failed_response_requirements: list[str] = field(default_factory=list)
    claim_types: list[str] = field(default_factory=list)
    claim_terms: list[str] = field(default_factory=list)

    @property
    def open(self) -> bool:
        return self.status in {"pending_evidence", "pending_communication", "violated"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "obligation_id": self.obligation_id,
            "deadline": self.deadline,
            "type": self.type,
            "requirement": self.requirement,
            "priority": self.priority,
            "status": self.status,
            "applicability": self.applicability,
            "evidence": self.evidence,
            "missing_evidence": self.missing_evidence,
            "missing_evidence_tools": self.missing_evidence_tools,
            "failed_response_requirements": self.failed_response_requirements,
            "claim_types": self.claim_types,
            "claim_terms": self.claim_terms,
        }


@dataclass
class GateDecision:
    should_recover: bool
    boundary: str
    obligations: list[ObligationState]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_recover": self.should_recover,
            "boundary": self.boundary,
            "reason": self.reason,
            "obligations": [item.to_dict() for item in self.obligations],
        }


def _combine_truth(values: list[TruthValue], mode: str) -> TruthValue:
    if not values:
        return TruthValue.TRUE
    if mode == "any":
        if TruthValue.TRUE in values:
            return TruthValue.TRUE
        if all(value == TruthValue.FALSE for value in values):
            return TruthValue.FALSE
        return (
            TruthValue.CONFLICT if TruthValue.CONFLICT in values else TruthValue.UNKNOWN
        )
    if TruthValue.FALSE in values:
        return TruthValue.FALSE
    if all(value == TruthValue.TRUE for value in values):
        return TruthValue.TRUE
    return TruthValue.CONFLICT if TruthValue.CONFLICT in values else TruthValue.UNKNOWN


def _assistant_text(
    conversation: list[dict[str, Any]],
    proposed_text: str = "",
    *,
    after_index: int = -1,
) -> str:
    parts = [
        str(message.get("content", ""))
        for index, message in enumerate(conversation)
        if index >= after_index
        and message.get("role") == "assistant"
        and _assistant_text_is_user_visible(conversation, index)
        and message.get("content")
    ]
    if proposed_text:
        parts.append(proposed_text)
    return "\n".join(parts)


def _number_variants(value: float) -> set[str]:
    output = {f"{value:g}", f"{value:.1f}", f"{value:.2f}"}
    if value.is_integer():
        output.add(str(int(value)))
    return output


def _value_is_mentioned(value: Any, text: str, mode: str = "any") -> bool:
    lowered = text.casefold()
    if value is None or isinstance(value, bool) or isinstance(value, (dict, list)):
        return False
    if isinstance(value, (int, float)):
        normalized = lowered.replace(",", "")
        return any(
            re.search(rf"(?<!\d){re.escape(variant)}(?!\d)", normalized)
            for variant in _number_variants(float(value))
        )
    rendered = compact(value, 500).casefold()
    if mode == "numeric":
        values = re.findall(r"[-+]?\d+(?:[.,]\d+)?", rendered)
        return any(
            re.search(
                rf"(?<!\d){re.escape(item.replace(',', ''))}(?!\d)",
                lowered.replace(",", ""),
            )
            for item in values
        )
    if mode == "identifier":
        return rendered in lowered
    if len(rendered) <= 100 and rendered in lowered:
        return True
    salient = {
        token
        for token in tokens(rendered)
        if len(token) >= 4
        and token
        not in {
            "status",
            "reason",
            "result",
            "true",
            "false",
            "none",
            "with",
            "from",
            "that",
            "this",
            "have",
        }
    }
    if not salient:
        return False
    overlap = len(salient & set(tokens(lowered))) / len(salient)
    return overlap >= (0.55 if mode == "text" else 0.7)


def _contains_claim(text: str, claim_types: list[str], terms: list[str]) -> bool:
    lowered = text.casefold()
    if terms and any(term.casefold() in lowered for term in terms):
        return True
    checks = {
        "amount": bool(NUMBER_CLAIM.search(text)),
        "percentage": bool(re.search(r"\b\d+(?:\.\d+)?\s*%", text)),
        "duration": bool(
            re.search(r"\b\d+(?:\.\d+)?\s*(?:hours?|days?|nights?)\b", text, re.I)
        ),
        "status": bool(STATUS_CLAIM.search(text)),
        "identifier": bool(ENTITY_ID.search(text)),
    }
    return any(checks.get(item, False) for item in claim_types)


class ContractEvaluator:
    """Derive contract states and gate only a proposed claim/action/final response."""

    def __init__(
        self, contracts: list[dict[str, Any]], conversation: list[dict[str, Any]]
    ):
        self.contracts = contracts
        self.conversation = conversation
        self.evidence = EvidenceLedger(conversation)
        self.actions = ActionLedger(conversation)

    def _applicability(
        self, contract: dict[str, Any], evidence_view: EvidenceLedger
    ) -> tuple[TruthValue, list[EvidenceFact]]:
        definition = contract.get("applicability") or {}
        predicates = [
            item for item in definition.get("predicates", []) if isinstance(item, dict)
        ]
        evaluations = [evidence_view.evaluate(item) for item in predicates]
        truth = _combine_truth(
            [value for value, _ in evaluations], str(definition.get("mode", "all"))
        )
        facts = [fact for _, found in evaluations for fact in found]
        return truth, facts

    def _required_evidence(
        self, obligation: dict[str, Any], evidence_view: EvidenceLedger
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        evidence: list[dict[str, Any]] = []
        missing: list[str] = []
        missing_tools: list[str] = []
        for position, group in enumerate(obligation.get("evidence_requirements", [])):
            if not isinstance(group, dict):
                continue
            selectors = (
                [item for item in group.get("any_of", []) if isinstance(item, dict)]
                if isinstance(group.get("any_of"), list)
                else [group.get("selector")]
                if isinstance(group.get("selector"), dict)
                else [group]
            )
            found: list[EvidenceFact] = []
            for selector in selectors:
                found.extend(evidence_view.satisfying(selector, trustworthy_only=True))
            if found:
                evidence.extend(evidence_view.compact_facts(found))
            elif group.get("required", True):
                missing.append(
                    compact(
                        group.get("description") or f"evidence_group_{position}", 180
                    )
                )
                missing_tools.extend(
                    str(selector.get("tool", ""))
                    for selector in selectors
                    if selector.get("source") in {"tool_argument", "tool_result"}
                    and str(selector.get("tool", "")) not in {"", "*"}
                )
        return evidence, missing, list(dict.fromkeys(missing_tools))

    def _response_clause(
        self,
        clause: dict[str, Any],
        text: str,
        proposed_text: str,
        evidence_view: EvidenceLedger,
    ) -> bool:
        kind = str(clause.get("kind", ""))
        if kind in {"mention_any", "mention_all"}:
            terms = [str(item).casefold() for item in clause.get("terms", []) if item]
            matches = [term in text.casefold() for term in terms]
            return bool(matches) and (
                all(matches) if kind == "mention_all" else any(matches)
            )
        if kind == "causal_explanation":
            markers = [str(item).casefold() for item in clause.get("terms", []) if item]
            if markers:
                return any(marker in text.casefold() for marker in markers)
            return bool(CAUSAL_PATTERN.search(text))
        if kind == "comparison":
            markers = [str(item).casefold() for item in clause.get("terms", []) if item]
            markers = markers or list(COMPARISON_MARKERS)
            return any(marker in text.casefold() for marker in markers)
        if kind == "mention_evidence":
            selector_groups = [
                [item for item in group if isinstance(item, dict)]
                for group in clause.get("selector_groups", [])
                if isinstance(group, list)
            ]
            selector_groups = [group for group in selector_groups if group]
            if selector_groups:
                mode = str(clause.get("value_mode", "any"))
                mentioned_groups = 0
                for group in selector_groups:
                    facts = [
                        fact
                        for selector in group
                        for fact in evidence_view.satisfying(
                            selector, trustworthy_only=True
                        )
                    ]
                    if any(_value_is_mentioned(fact.value, text, mode) for fact in facts):
                        mentioned_groups += 1
                required_groups = max(
                    1,
                    int(
                        clause.get(
                            "min_groups", clause.get("min_mentions", 1)
                        )
                    ),
                )
                return mentioned_groups >= required_groups
            selectors = [
                item for item in clause.get("selectors", []) if isinstance(item, dict)
            ]
            if isinstance(clause.get("selector"), dict):
                selectors.append(clause["selector"])
            facts = [
                fact
                for selector in selectors
                for fact in evidence_view.satisfying(selector, trustworthy_only=True)
            ]
            mode = str(clause.get("value_mode", "any"))
            mentions = sum(
                _value_is_mentioned(fact.value, text, mode) for fact in facts
            )
            return mentions >= max(1, int(clause.get("min_mentions", 1)))
        if kind == "action_state":
            truth, _ = self.actions.evaluate(clause.get("requirement") or clause)
            return truth == TruthValue.TRUE
        if kind == "claim_requires_evidence":
            claim_types = [str(item) for item in clause.get("claim_types", [])]
            terms = [str(item) for item in clause.get("terms", [])]
            if not _contains_claim(proposed_text, claim_types, terms):
                return True
            selectors = [
                item
                for item in clause.get("evidence_any_of", [])
                if isinstance(item, dict)
            ]
            return any(
                evidence_view.satisfying(selector, trustworthy_only=True)
                for selector in selectors
            )
        return False

    def states(
        self,
        proposed_text: str = "",
        proposed_calls: list[dict[str, Any]] | None = None,
    ) -> list[ObligationState]:
        # Applicability is fixed before the candidate response.  Proposed text
        # is evaluated only by response clauses and can never self-activate a
        # contract. Proposed tool arguments remain visible for deterministic
        # pre-action contracts, but are not committed evidence.
        applicability_view = self.evidence
        if proposed_calls:
            applicability_view = EvidenceLedger(self.conversation)
        for call_position, call in enumerate(proposed_calls or []):
            for path, value in _flatten(call.get("arguments") or {}):
                applicability_view.facts.append(
                    EvidenceFact(
                        ref=f"P{call_position}:arg:{path}",
                        source="tool_argument",
                        tool=str(call.get("name", "")),
                        outcome="proposed",
                        value=value,
                        path=path,
                        conversation_index=len(self.conversation),
                    )
                )
        output: list[ObligationState] = []
        for contract in self.contracts:
            applicability, applicability_facts = self._applicability(
                contract, applicability_view
            )
            unknown_policy = str(
                (contract.get("applicability") or {}).get("unknown_policy", "inactive")
            )
            for obligation in contract.get("obligations", []):
                if not isinstance(obligation, dict):
                    continue
                evidence, missing, missing_tools = self._required_evidence(
                    obligation, self.evidence
                )
                evidence = [
                    *applicability_view.compact_facts(applicability_facts),
                    *evidence,
                ]
                evidence_indices = [
                    int(item.get("conversation_index", -1))
                    for item in evidence
                    if int(item.get("conversation_index", -1)) >= 0
                    and item.get("ref") != "DRAFT"
                    and not str(item.get("ref", "")).startswith("P")
                ]
                all_text = _assistant_text(
                    self.conversation,
                    proposed_text,
                    # Communication can discharge an obligation only after all
                    # currently selected applicability/grounding evidence is
                    # available. Earlier assertions must not be retroactively
                    # legitimized by a later tool result.
                    after_index=max(evidence_indices, default=-1),
                )
                failed: list[str] = []
                response_clauses = [
                    item
                    for item in obligation.get("response_requirements", [])
                    if isinstance(item, dict)
                ]
                claim_types = list(
                    dict.fromkeys(
                        str(value)
                        for clause in response_clauses
                        if clause.get("kind") == "claim_requires_evidence"
                        for value in clause.get("claim_types", [])
                    )
                )
                claim_terms = list(
                    dict.fromkeys(
                        str(value)
                        for clause in response_clauses
                        if clause.get("kind") == "claim_requires_evidence"
                        for value in clause.get("terms", [])
                    )
                )
                if applicability == TruthValue.FALSE:
                    status = "inactive"
                elif applicability in {TruthValue.UNKNOWN, TruthValue.CONFLICT}:
                    status = (
                        "pending_evidence"
                        if unknown_policy == "require_resolution"
                        else "inactive"
                    )
                    if status == "pending_evidence" and not missing:
                        missing = [
                            compact(
                                (contract.get("applicability") or {}).get(
                                    "unknown_description",
                                    "evidence needed to decide whether this contract applies",
                                ),
                                180,
                            )
                        ]
                    if status == "pending_evidence":
                        missing_tools = list(
                            dict.fromkeys(
                                [
                                    *missing_tools,
                                    *[
                                        str(selector.get("tool", ""))
                                        for selector in (
                                            contract.get("applicability") or {}
                                        ).get("predicates", [])
                                        if isinstance(selector, dict)
                                        and selector.get("source")
                                        in {"tool_argument", "tool_result"}
                                        and str(selector.get("tool", ""))
                                        not in {"", "*"}
                                    ],
                                ]
                            )
                        )
                elif missing:
                    status = "pending_evidence"
                else:
                    clauses = response_clauses
                    for position, clause in enumerate(clauses):
                        if not self._response_clause(
                            clause, all_text, proposed_text, self.evidence
                        ):
                            failed.append(
                                compact(
                                    clause.get("description")
                                    or f"response_clause_{position}",
                                    180,
                                )
                            )
                    status = (
                        "satisfied"
                        if clauses and not failed
                        else "pending_communication"
                    )
                output.append(
                    ObligationState(
                        contract_id=str(contract.get("id", "")),
                        obligation_id=str(obligation.get("id", "")),
                        deadline=str(obligation.get("deadline", "before_final")),
                        type=str(obligation.get("type", "completion")),
                        requirement=compact(obligation.get("requirement", ""), 520),
                        priority=int(obligation.get("priority", 50)),
                        status=status,
                        applicability=applicability.value,
                        evidence=evidence[:10],
                        missing_evidence=missing,
                        missing_evidence_tools=missing_tools,
                        failed_response_requirements=failed,
                        claim_types=claim_types,
                        claim_terms=claim_terms,
                    )
                )
        return output

    @staticmethod
    def _calls(response: Any) -> list[dict[str, Any]]:
        output = []
        for call in getattr(response, "tool_calls", []) or []:
            if isinstance(call, dict):
                output.append(
                    {
                        "name": str(call.get("name", "")),
                        "arguments": call.get("arguments") or {},
                    }
                )
            else:
                output.append(
                    {
                        "name": str(getattr(call, "name", "")),
                        "arguments": getattr(call, "arguments", {}) or {},
                    }
                )
        return output

    def gate(
        self, response: Any, tools: list[dict[str, Any]] | None = None
    ) -> GateDecision:
        proposed_text = str(getattr(response, "text", "") or "")
        calls = self._calls(response)
        effect_kinds = [proposed_effect_kind(call, tools) for call in calls]
        mutating = "potential_mutation" in effect_kinds
        if calls and not mutating:
            reason = (
                "preview_tool_proposal"
                if "preview" in effect_kinds
                else "read_only_tool_proposal"
            )
            return GateDecision(False, "tool", [], reason)
        # STATE-Bench does not expose text attached to an intermediate tool-call
        # generation to the user. Such text cannot discharge a communication
        # condition before the proposed mutation is executed.
        states = self.states(
            proposed_text="" if calls else proposed_text, proposed_calls=calls,
        )
        if mutating:
            blocked = [
                state
                for state in states
                if state.deadline == "before_action" and state.open
            ]
            return GateDecision(
                bool(blocked),
                "before_action",
                sorted(blocked, key=lambda item: (item.priority, item.contract_id)),
                "unresolved_pre_action_contract"
                if blocked
                else "action_contracts_satisfied",
            )
        claim_blocked = [
            state
            for state in states
            if state.deadline == "before_claim"
            and state.open
            and _contains_claim(proposed_text, state.claim_types, state.claim_terms,)
        ]
        final_blocked = [
            state for state in states if state.deadline == "before_final" and state.open
        ]
        blocked = sorted(
            [*claim_blocked, *final_blocked],
            key=lambda item: (item.priority, item.contract_id, item.obligation_id),
        )
        return GateDecision(
            bool(blocked),
            "before_claim" if claim_blocked else "before_final",
            blocked,
            "unresolved_response_contract"
            if blocked
            else "closure_contracts_satisfied",
        )


def opening_query(conversation: list[dict[str, Any]]) -> str:
    """Build a one-shot retrieval query without tool results or hidden runtime context."""

    return next(
        (
            normalize_retrieval_query(message.get("content", ""), 2500)
            for message in conversation
            if message.get("role") == "user"
            and "[TASK_DONE]" not in str(message.get("content", ""))
        ),
        "",
    )
