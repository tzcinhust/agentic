"""Compact, deterministic facts extracted from canonical tool observations.

The ledger deliberately does not inspect ``AgentRuntimeContext``.  It accepts
both folded trajectory calls and Responses API call/output pairs, retains the
source of every fact, and distinguishes an unread value from an explicit null
or numeric zero.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class FactState(str, Enum):
    UNKNOWN = "UNKNOWN"
    NULL = "NULL"
    VALUE = "VALUE"
    ZERO = "ZERO"


@dataclass(frozen=True, slots=True)
class Fact:
    value: Any = None
    state: FactState = FactState.UNKNOWN
    source_tool: str = ""
    source_call_id: str | None = None
    turn: int = -1
    entity_id: str | None = None
    stale: bool = False


@dataclass(frozen=True, slots=True)
class ToolRecord:
    name: str
    arguments: dict[str, Any]
    result: Any
    call_id: str | None
    turn: int
    is_write: bool


WRITE_TOOLS = frozenset(
    {
        "add_to_cart",
        "update_cart_item",
        "remove_from_cart",
        "apply_promo",
        "remove_promo",
        "redeem_loyalty_points",
        "cancel_loyalty_redemption",
        "set_shipping_option",
        "create_booking",
        "update_booking",
        "cancel_booking",
        "book_hotel",
        "cancel_hotel_reservation",
        "book_car_rental",
        "cancel_car_rental",
        "process_return",
        "process_refund",
        "cancel_order",
        "process_exchange",
        "process_warranty_claim",
    }
)

ENTITY_KEYS = (
    "product_id",
    "cart_item_id",
    "item_id",
    "order_id",
    "booking_id",
    "flight_id",
    "reservation_id",
    "rental_id",
    "customer_id",
    "user_id",
)

ALIASES = {
    "cart_subtotal": "cart_subtotal",
    "subtotal": "cart_subtotal",
    "cart_total": "cart_total",
    "total": "cart_total",
    "loyalty_tier": "tier",
    "membership_tier": "tier",
    "first_time": "is_first_time",
    "first_time_customer": "is_first_time",
    "promo_codes": "applied_promo_codes",
    "gift_wrap_fee": "gift_wrap_fee",
    "refund": "refund_amount",
    "refund_total": "refund_amount",
    "return_eligible": "return_eligibility",
    "warranty_eligible": "warranty_eligibility",
    "remaining_warranty_claims": "warranty_remaining_claims",
    "remedies": "available_remedies",
    "options": "available_remedies",
    "flight_cancelled": "cancelled",
    "flight_delayed": "delayed",
    "disruption_weather": "weather_disruption",
    "fee": "change_fee",
}

COMPACT_FIELDS = (
    "product_id",
    "brand",
    "category",
    "price",
    "quantity",
    "gift_wrap",
    "line_total",
    "cart_subtotal",
    "cart_total",
    "discount_amount",
    "applied_promo_codes",
    "is_first_time",
    "tier",
    "loyalty_points_redeemed",
    "shipping_cost",
    "booking_id",
    "flight_id",
    "flight_status",
    "cancelled",
    "delayed",
    "schedule_changed",
    "weather_disruption",
    "change_fee",
    "change_reason",
    "order_id",
    "order_status",
    "shipment_status",
    "delivery_status",
    "lost",
    "damaged",
    "investigation_status",
    "refund_amount",
    "return_eligibility",
    "warranty_eligibility",
    "warranty_remaining_claims",
    "available_remedies",
)


def _state(value: Any) -> FactState:
    if value is None:
        return FactState.NULL
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        return FactState.ZERO
    return FactState.VALUE


def _decode(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _walk(value: Any, inherited_entity: str | None = None) -> Iterable[tuple[dict[str, Any], str | None]]:
    if isinstance(value, dict):
        entity = next(
            (str(value[key]) for key in ENTITY_KEYS if value.get(key) not in (None, "")),
            inherited_entity,
        )
        yield value, entity
        for child in value.values():
            yield from _walk(child, entity)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child, inherited_entity)


class RuntimeFactLedger:
    """Latest facts and entity joins derived solely from conversation evidence."""

    def __init__(self, conversation: list[Any] | None = None):
        self.facts: dict[str, Fact] = {}
        self.entities: dict[str, dict[str, Fact]] = {}
        self.records: list[ToolRecord] = []
        if conversation is not None:
            self.ingest(conversation)

    @classmethod
    def from_conversation(cls, conversation: list[Any]) -> "RuntimeFactLedger":
        return cls(conversation)

    @staticmethod
    def _events(conversation: list[Any]) -> list[ToolRecord]:
        calls: dict[str, tuple[str, dict[str, Any], int]] = {}
        pending: list[tuple[str, dict[str, Any], str | None, int]] = []
        events: list[ToolRecord] = []
        for turn, item in enumerate(conversation):
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "function_call":
                call_id = str(item.get("call_id") or item.get("id") or "")
                args = _decode(item.get("arguments") or {})
                calls[call_id] = (str(item.get("name", "")), args if isinstance(args, dict) else {}, turn)
            elif kind == "function_call_output":
                call_id = str(item.get("call_id") or "")
                name, args, call_turn = calls.get(call_id, ("", {}, turn))
                events.append(
                    ToolRecord(name, args, _decode(item.get("output")), call_id or None, call_turn, name in WRITE_TOOLS)
                )

            for call in item.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or (call.get("function") or {}).get("name") or "")
                args = call.get("arguments")
                if args is None:
                    args = (call.get("function") or {}).get("arguments") or {}
                args = _decode(args)
                call_id = str(call.get("call_id") or call.get("id") or "") or None
                if "result" in call:
                    events.append(ToolRecord(name, args if isinstance(args, dict) else {}, call.get("result"), call_id, turn, name in WRITE_TOOLS))
                else:
                    pending.append((name, args if isinstance(args, dict) else {}, call_id, turn))

            if item.get("role") == "tool":
                content = item.get("content")
                rows = content if isinstance(content, list) else [content]
                for row in rows:
                    if isinstance(row, dict):
                        name = str(row.get("name", ""))
                        args = row.get("arguments") or {}
                        result = row.get("result", row.get("output", row.get("content")))
                        call_id = str(row.get("call_id") or row.get("tool_call_id") or "") or None
                    else:
                        name, args, call_id = "", {}, None
                        result = row
                    if not name and pending:
                        name, args, call_id, call_turn = pending.pop(0)
                    else:
                        call_turn = turn
                        if name:
                            for index, queued in enumerate(pending):
                                if (call_id and queued[2] == call_id) or queued[0] == name:
                                    _, queued_args, queued_id, call_turn = pending.pop(index)
                                    args = args or queued_args
                                    call_id = call_id or queued_id
                                    break
                    events.append(ToolRecord(name, args if isinstance(args, dict) else {}, _decode(result), call_id, call_turn, name in WRITE_TOOLS))
        return events

    def ingest(self, conversation: list[Any]) -> None:
        for record in self._events(conversation):
            self.records.append(record)
            if record.is_write:
                self._mark_stale(record)
            self._extract_record(record)

    def _mark_stale(self, record: ToolRecord) -> None:
        ids = {str(record.arguments[key]) for key in ENTITY_KEYS if record.arguments.get(key) not in (None, "")}
        # Cart writes change quantities/totals, not immutable catalog metadata.
        # Keeping product brand/category fresh is essential for the post-write
        # join and avoids a redundant details read after every cart mutation.
        non_account_cart_writes = {
            "add_to_cart",
            "update_cart_item",
            "remove_from_cart",
            "apply_promo",
            "remove_promo",
            "set_shipping_option",
        }
        if record.name in non_account_cart_writes:
            ids.clear()
        elif record.name in {"redeem_loyalty_points", "cancel_loyalty_redemption"}:
            ids.discard(str(record.arguments.get("product_id", "")))
        for key, fact in list(self.facts.items()):
            if fact.entity_id in ids or key in {"cart_subtotal", "cart_total", "discount_amount", "shipping_cost", "applied_promo_codes"}:
                self.facts[key] = Fact(fact.value, fact.state, fact.source_tool, fact.source_call_id, fact.turn, fact.entity_id, True)
        for entity_id in ids:
            for key, fact in list(self.entities.get(entity_id, {}).items()):
                self.entities[entity_id][key] = Fact(fact.value, fact.state, fact.source_tool, fact.source_call_id, fact.turn, fact.entity_id, True)

    def _put(self, key: str, value: Any, record: ToolRecord, entity_id: str | None, *, stale: bool | None = None) -> None:
        normalized = ALIASES.get(key.lower(), key.lower())
        if normalized == "status":
            if record.name == "get_flight_status":
                normalized = "flight_status"
            elif "order" in record.name or record.name in {"process_return", "process_refund", "process_exchange"}:
                normalized = "order_status"
        fact = Fact(value, _state(value), record.name, record.call_id, record.turn, entity_id, record.is_write if stale is None else stale)
        self.facts[normalized] = fact
        if entity_id:
            self.entities.setdefault(entity_id, {})[normalized] = fact

    def _extract_record(self, record: ToolRecord) -> None:
        result = _decode(record.result)
        fallback_entity = next(
            (str(record.arguments[key]) for key in ENTITY_KEYS if record.arguments.get(key) not in (None, "")),
            None,
        )
        for row, entity_id in _walk(result, fallback_entity):
            for key, value in row.items():
                if isinstance(value, (dict, list)) and key.lower() not in {"applied_promo_codes", "available_remedies", "remedies", "options"}:
                    continue
                if key.lower() in COMPACT_FIELDS or key.lower() in ALIASES or key.lower() in {
                    "status", "cancelled", "delayed", "schedule_changed", "weather_disruption",
                    "change_reason", "investigation_status", "lost", "damaged", "shipping_cost",
                    "warranty_claims_remaining", "eligible", "refund_amount", "total_price",
                    "grand_total", "selected_total", "gift_wrap_fee", "discount_percent", "discount_rate",
                } or key.lower().endswith("_id"):
                    normalized_key = key
                    if key == "eligible" and "warranty" in record.name:
                        normalized_key = "warranty_eligibility"
                    self._put(normalized_key, value, record, entity_id)

        # Tool names add meaning when a response uses a generic status field.
        status = self.latest("flight_status")
        if record.name == "get_flight_status" and status.state in {FactState.VALUE, FactState.ZERO}:
            lowered = str(status.value).lower()
            self._put("cancelled", lowered == "cancelled", record, fallback_entity, stale=False)
            self._put("delayed", lowered == "delayed", record, fallback_entity, stale=False)

    def latest(self, key: str, entity_id: str | None = None) -> Fact:
        normalized = ALIASES.get(key.lower(), key.lower())
        if entity_id is not None:
            return self.entities.get(str(entity_id), {}).get(normalized, Fact())
        return self.facts.get(normalized, Fact())

    def values(self, key: str, *, fresh_only: bool = True) -> list[Any]:
        normalized = ALIASES.get(key.lower(), key.lower())
        result = []
        for fields in self.entities.values():
            fact = fields.get(normalized)
            if fact and fact.state in {FactState.VALUE, FactState.ZERO} and (not fresh_only or not fact.stale):
                result.append(fact.value)
        return result

    def tool_seen(self, name: str) -> bool:
        return any(record.name == name for record in self.records)

    def latest_record(self, name: str) -> ToolRecord | None:
        return next((record for record in reversed(self.records) if record.name == name), None)

    def policy_text(self, topic: str | None = None) -> str:
        parts: list[str] = []
        for record in self.records:
            if record.name != "get_policies":
                continue
            requested = str(record.arguments.get("topic", "")).lower()
            if topic and topic.lower().replace(" ", "_") not in {requested.replace(" ", "_"), ""}:
                continue
            parts.append(json.dumps(record.result, ensure_ascii=False, default=str))
        return "\n".join(parts)

    def cart_items(self) -> list[dict[str, Any]]:
        record = self.latest_record("get_cart")
        result = record.result if record and isinstance(record.result, dict) else {}
        items = result.get("items") or result.get("cart_items") or []
        return [item for item in items if isinstance(item, dict)]

    def joined_cart_items(self) -> list[dict[str, Any]]:
        joined = []
        for item in self.cart_items():
            product_id = item.get("product_id")
            details = self.entities.get(str(product_id), {}) if product_id is not None else {}
            row = dict(item)
            for key in ("brand", "category", "price", "gift_wrap_available"):
                fact = details.get(key)
                if fact and not fact.stale and fact.state in {FactState.VALUE, FactState.ZERO}:
                    row[key] = fact.value
            joined.append(row)
        return joined

    def compact_summary(self, max_chars: int = 1200) -> str:
        entries: list[str] = []
        for key in COMPACT_FIELDS:
            fact = self.latest(key)
            if fact.state not in {FactState.VALUE, FactState.ZERO} or fact.stale:
                continue
            value = json.dumps(fact.value, ensure_ascii=False, separators=(",", ":"))
            if len(value) > 100:
                continue
            entries.append(f"{key}={value} [{fact.source_tool}]")

        brand_counts: Counter[str] = Counter()
        for item in self.joined_cart_items():
            brand = item.get("brand")
            quantity = item.get("quantity", 1)
            if brand not in (None, ""):
                brand_counts[str(brand)] += int(quantity) if isinstance(quantity, int) and quantity > 0 else 1
        entries.extend(f"brand_count.{brand}={count} [cart+product]" for brand, count in brand_counts.items())

        output = " ".join(entries)
        if len(output) <= max_chars:
            return output
        clipped = output[:max_chars].rsplit(" ", 1)[0]
        return clipped.rstrip()
