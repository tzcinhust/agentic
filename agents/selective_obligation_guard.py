"""Low-interference obligation guidance and high-precision runtime guards.

The module deliberately leaves workflow retrieval and normal generation to the
archived PWM agent.  It adds no semantic judge.  Its two responsibilities are:

1. turn applicable communication clauses from retrieved workflow text into a
   compact, soft obligation ledger; and
2. flag only machine-checkable execution mistakes before a candidate is used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


WRITE_TOOLS = frozenset(
    {
        "add_to_cart",
        "apply_promo",
        "book_car_rental",
        "book_hotel",
        "cancel_booking",
        "cancel_car_rental",
        "cancel_hotel_reservation",
        "cancel_loyalty_redemption",
        "cancel_order",
        "create_booking",
        "process_exchange",
        "process_refund",
        "process_return",
        "process_warranty_claim",
        "redeem_loyalty_points",
        "remove_from_cart",
        "remove_promo",
        "set_shipping_option",
        "update_booking",
        "update_cart_item",
    }
)

TWO_STEP_TOOLS = frozenset(
    {
        "cancel_booking",
        "cancel_car_rental",
        "cancel_hotel_reservation",
        "cancel_order",
        "process_exchange",
        "process_refund",
        "process_return",
        "process_warranty_claim",
        "update_booking",
    }
)

# These reads are required only when a previous mutation made an earlier read
# stale for the same entity.  They are not required before an ordinary first
# write, which keeps the guard from turning common workflows into long loops.
REFRESH_TOOL_BY_WRITE = {
    "cancel_booking": "get_booking",
    "cancel_car_rental": "get_car_rental",
    "cancel_hotel_reservation": "get_hotel_reservation",
    "cancel_order": "get_order",
    "process_exchange": "get_order",
    "process_refund": "get_order",
    "process_return": "get_order",
    "process_warranty_claim": "get_order",
    "remove_from_cart": "get_cart",
    "remove_promo": "get_cart",
    "update_cart_item": "get_cart",
}

IDENTIFIER_KEYS = frozenset(
    {
        "booking_id",
        "car_id",
        "customer_id",
        "flight_id",
        "hotel_id",
        "item_id",
        "order_id",
        "product_id",
        "rental_id",
        "reservation_id",
        "user_id",
        "variant_id",
        "warranty_id",
    }
)

COMMUNICATION_TERMS = re.compile(
    r"\b(ask|break\s*down|calculate|clarify|compare|confirm|disclose|explain|"
    r"inform|mention|offer|quote|recommend|report|show|state|summari[sz]e|tell)\b",
    re.IGNORECASE,
)
TOKEN = re.compile(r"[a-z0-9_]+")
SECTION = re.compile(r"^(Verify first|Procedure|Branches|Avoid):\s*$", re.IGNORECASE)
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "for",
        "from",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "then",
        "to",
        "use",
        "user",
        "when",
        "with",
    }
)


@dataclass(frozen=True)
class ToolEvent:
    sequence: int
    name: str
    arguments: dict[str, Any]
    result: Any


def _result_succeeded(result: Any) -> bool:
    if not isinstance(result, dict):
        return True
    if result.get("error") or result.get("success") is False:
        return False
    return str(result.get("status", "")).lower() not in {
        "error",
        "failed",
        "rejected",
    }


def _is_preview(name: str, arguments: dict[str, Any], result: Any = None) -> bool:
    if isinstance(result, dict):
        status = str(result.get("status", "")).lower()
        if status in {"pending_confirmation", "preview", "quoted"}:
            return True
        if status in {
            "cancelled",
            "completed",
            "exchanged",
            "processed",
            "refunded",
            "returned",
            "success",
            "updated",
        }:
            return False
    if name == "update_booking" and not arguments.get("flight_id"):
        return False
    return name in TWO_STEP_TOOLS and not bool(arguments.get("confirm"))


def _collect_values(value: Any) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}

    def visit(current: Any, key: str = "") -> None:
        if isinstance(current, dict):
            for child_key, child in current.items():
                visit(child, str(child_key))
        elif isinstance(current, list):
            for child in current:
                visit(child, key)
        elif current is not None and key:
            normalized = key[:-1] if key.endswith("_ids") else key
            if normalized in IDENTIFIER_KEYS:
                values.setdefault(normalized, set()).add(str(current))

    visit(value)
    return values


def _scope(arguments: dict[str, Any], result: Any = None) -> dict[str, set[str]]:
    return _collect_values({"arguments": arguments, "result": result})


def _same_entity(
    left_arguments: dict[str, Any],
    left_result: Any,
    right_arguments: dict[str, Any],
) -> bool:
    left = _scope(left_arguments, left_result)
    right = _scope(right_arguments)
    shared = set(left) & set(right)
    return bool(shared) and all(left[key] & right[key] for key in shared)


def tool_events(conversation: Iterable[dict[str, Any]]) -> list[ToolEvent]:
    events: list[ToolEvent] = []
    sequence = 0
    for message in conversation:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            events.append(
                ToolEvent(
                    sequence=sequence,
                    name=str(call.get("name", "")),
                    arguments=call.get("arguments") or {},
                    result=call.get("result"),
                )
            )
            sequence += 1
    return events


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
        if section and line.startswith("-"):
            bullets.append((section, line[1:].strip()))
    return bullets


def _relevance(text: str, user_tokens: set[str]) -> tuple[int, int]:
    tokens = {token for token in TOKEN.findall(text.lower()) if token not in STOPWORDS}
    return len(tokens & user_tokens), int(bool(COMMUNICATION_TERMS.search(text)))


def build_obligation_prompt(
    workflows: list[str],
    conversation: list[dict[str, Any]],
    *,
    max_communication: int = 4,
) -> str:
    """Build a compact soft ledger without inferring hidden task requirements."""

    user_messages = [
        " ".join(str(item.get("content", "")).split())
        for item in conversation
        if item.get("role") == "user" and "[TASK_DONE]" not in str(item.get("content", ""))
    ]
    if not user_messages:
        return ""
    user_text = " ".join(user_messages).lower()
    user_tokens = {token for token in TOKEN.findall(user_text) if token not in STOPWORDS}

    ranked: list[tuple[int, int, int, str]] = []
    seen: set[str] = set()
    # The highest-ranked archived card supplies the focused checklist.  Lower
    # ranked cards remain in the original PWM prompt, but are not allowed to
    # turn weak retrieval matches into extra obligations.
    for workflow_index, workflow in enumerate(workflows[:1]):
        for section, bullet in _workflow_bullets(workflow):
            if not COMMUNICATION_TERMS.search(bullet):
                continue
            normalized = re.sub(r"\s+", " ", bullet).strip()
            key = normalized.lower()
            if key in seen:
                continue
            overlap, communication = _relevance(normalized, user_tokens)
            if overlap == 0:
                continue
            seen.add(key)
            ranked.append((overlap, communication, -workflow_index, normalized))
    ranked.sort(reverse=True)
    communication = [item[-1] for item in ranked[:max_communication]]

    events = tool_events(conversation)
    successful = list(
        dict.fromkeys(event.name for event in events if _result_succeeded(event.result))
    )
    failed = list(
        dict.fromkeys(event.name for event in events if not _result_succeeded(event.result))
    )
    original_message = user_messages[0]
    latest_message = user_messages[-1]
    original = original_message[:420]
    latest = latest_message[:320]
    lines = [
        "Selective obligation ledger (soft guidance; live tools remain authoritative):",
        f"- Original objective: {original}",
    ]
    if latest_message != original_message:
        lines.append(f"- Latest user instruction: {latest}")
    lines.append(f"- Successful evidence tools: {', '.join(successful[-10:]) or 'none yet'}")
    if failed:
        lines.append(
            f"- Failed tools: {', '.join(failed[-6:])}; never describe these calls as successful."
        )
    if communication:
        lines.append("- Potential communication obligations, only when directly applicable:")
        lines.extend(f"  * {item[:360]}" for item in communication)
    lines.extend(
        [
            "- Final coverage: directly answer the latest user request with the exact tool-grounded "
            "result, necessary rationale, and any requested comparison or time/amount warning.",
            "- Bounded closure: report only the requested outcome, necessary policy basis, and "
            "tool-confirmed changes. Do not add unrelated account-wide totals or balances.",
            "- Do not introduce alternative workflows, refund paths, conversions, future services, "
            "or speculative options unless the user explicitly requested that exact information "
            "and authoritative tools confirm it is available.",
            "- When a requested action is unsupported, clearly state the limitation and supported "
            "outcome; do not volunteer a workaround unless the user asks for alternatives.",
            "Do not perform an action merely because it appears in this ledger.",
        ]
    )
    return "\n".join(lines)


def _candidate_calls(response: Any) -> list[tuple[str, dict[str, Any]]]:
    calls = []
    for call in getattr(response, "tool_calls", []) or []:
        if isinstance(call, dict):
            name = str(call.get("name", ""))
            arguments = call.get("arguments") or {}
        else:
            name = str(getattr(call, "name", ""))
            arguments = getattr(call, "arguments", {}) or {}
        calls.append((name, arguments if isinstance(arguments, dict) else {}))
    return calls


def _matching_preview(
    events: list[ToolEvent], name: str, arguments: dict[str, Any]
) -> ToolEvent | None:
    for event in reversed(events):
        if (
            event.name == name
            and _result_succeeded(event.result)
            and _is_preview(event.name, event.arguments, event.result)
            and _same_entity(event.arguments, event.result, arguments)
        ):
            return event
    return None


def _numeric_preview_mismatch(
    name: str, arguments: dict[str, Any], preview: ToolEvent
) -> str | None:
    result = preview.result if isinstance(preview.result, dict) else {}
    aliases = {
        "amount": ("amount", "refund_amount", "net_refund", "refund"),
        "cash_amount": ("cash_amount", "remaining_cash_payment"),
        "points_used": ("points_used",),
    }
    for argument_name, result_names in aliases.items():
        supplied = arguments.get(argument_name)
        if not isinstance(supplied, (int, float)) or isinstance(supplied, bool):
            continue
        expected = next(
            (
                result.get(result_name)
                for result_name in result_names
                if isinstance(result.get(result_name), (int, float))
                and not isinstance(result.get(result_name), bool)
            ),
            None,
        )
        if expected is not None and float(supplied) != float(expected):
            return (
                f"Do not confirm {name}: {argument_name}={supplied} conflicts with "
                f"the same-entity preview value {expected}."
            )
    return None


def _stale_read_feedback(
    events: list[ToolEvent], name: str, arguments: dict[str, Any]
) -> str | None:
    refresh_tool = REFRESH_TOOL_BY_WRITE.get(name)
    if not refresh_tool:
        return None
    relevant_mutations = [
        event
        for event in events
        if event.name in WRITE_TOOLS
        and not _is_preview(event.name, event.arguments, event.result)
        and _result_succeeded(event.result)
        and _same_entity(event.arguments, event.result, arguments)
    ]
    if not relevant_mutations:
        return None
    last_mutation = relevant_mutations[-1]
    refreshed = any(
        event.sequence > last_mutation.sequence
        and event.name == refresh_tool
        and _result_succeeded(event.result)
        and _same_entity(event.arguments, event.result, arguments)
        for event in events
    )
    if refreshed:
        return None
    return (
        f"The same entity changed after the latest authoritative read. Call {refresh_tool} "
        f"once before executing {name}; do not reuse pre-mutation state."
    )


def _claims_success_after_failure(text: str, events: list[ToolEvent]) -> str | None:
    if (
        not text
        or not events
        or events[-1].name not in WRITE_TOOLS
        or _result_succeeded(events[-1].result)
    ):
        return None
    lowered = text.lower()
    if re.search(r"\b(not|unable|couldn['’]?t|didn['’]?t|failed|rejected)\b", lowered):
        return None
    claims_success = re.search(
        r"\b(done|completed successfully|successfully (?:cancelled|canceled|updated|processed|"
        r"refunded|returned|exchanged)|(?:has|have|was|were) been (?:successfully )?"
        r"(?:cancelled|canceled|updated|processed|refunded|returned|exchanged))\b",
        lowered,
    )
    if not claims_success:
        return None
    return (
        f"The latest {events[-1].name} call failed. State the observed failure instead of "
        "describing the action as completed."
    )


def guard_feedback(response: Any, conversation: list[dict[str, Any]]) -> str | None:
    """Return feedback only for a narrow, machine-checkable violation."""

    events = tool_events(conversation)
    calls = _candidate_calls(response)
    if not calls:
        return _claims_success_after_failure(str(getattr(response, "text", "")), events)

    candidate_previews = [
        (name, arguments)
        for name, arguments in calls
        if name in TWO_STEP_TOOLS and _is_preview(name, arguments)
    ]
    for name, arguments in calls:
        if name not in WRITE_TOOLS:
            continue
        if _is_preview(name, arguments):
            continue
        if name in TWO_STEP_TOOLS and arguments.get("confirm") is True:
            has_in_batch_preview = any(
                preview_name == name
                and _same_entity(preview_arguments, None, arguments)
                for preview_name, preview_arguments in candidate_previews
            )
            if not has_in_batch_preview:
                preview = _matching_preview(events, name, arguments)
                if preview is not None:
                    mismatch = _numeric_preview_mismatch(name, arguments, preview)
                    if mismatch:
                        return mismatch
        stale = _stale_read_feedback(events, name, arguments)
        if stale:
            return stale
    return None


def compact_audit_record(feedback: str, corrected: bool) -> str:
    return json.dumps(
        {"event": "selective_guard", "feedback": feedback, "corrected": corrected},
        ensure_ascii=True,
        sort_keys=True,
    )
