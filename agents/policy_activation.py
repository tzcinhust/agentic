"""Three-valued activation of policy branches from canonical runtime facts."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

from state_bench.agents.base import AgentToolCallRequest

from agents.runtime_fact_ledger import FactState, RuntimeFactLedger


class ActivationValue(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class ActivationResult:
    value: ActivationValue
    missing_read_tools: list[AgentToolCallRequest]
    evidence: list[str]


def _user_text(conversation: list[Any]) -> str:
    return " ".join(
        str(item.get("content", ""))
        for item in conversation
        if isinstance(item, dict)
        and item.get("role") == "user"
        and "[task_done]" not in str(item.get("content", "")).lower()
    )


def _threshold_and_rate(text: str, default_threshold: int) -> tuple[int, float | None]:
    threshold = default_threshold
    rate: float | None = None
    count = re.search(r"(\d+)\s*\+?\s*(?:items?|products?|units?)", text, re.I)
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if count:
        threshold = int(count.group(1))
    if percent:
        rate = float(percent.group(1)) / 100
    return threshold, rate


class PolicyActivation:
    """Evaluate only well-grounded branches; absence of a read is UNKNOWN."""

    def __init__(self, policy_topics: list[dict[str, Any]] | None = None):
        self._topics = {str(item.get("topic", "")): item for item in policy_topics or []}

    def _policy_text(self, topic: str, ledger: RuntimeFactLedger) -> str:
        live = ledger.policy_text(topic)
        if live:
            return live
        item = self._topics.get(topic)
        return json.dumps(item, ensure_ascii=False, default=str) if item else ""

    @staticmethod
    def _customer_id(ledger: RuntimeFactLedger) -> str:
        for record in reversed(ledger.records):
            value = record.arguments.get("customer_id")
            if value:
                return str(value)
        return ""

    @staticmethod
    def _product_reads(ledger: RuntimeFactLedger, field: str) -> list[AgentToolCallRequest]:
        reads = []
        for item in ledger.cart_items():
            product_id = item.get("product_id")
            if product_id is None:
                continue
            fact = ledger.latest(field, str(product_id))
            if fact.state == FactState.UNKNOWN or fact.stale:
                reads.append(AgentToolCallRequest("get_product_details", {"product_id": str(product_id)}))
        return reads

    def evaluate(
        self,
        domain: str,
        topic: str,
        ledger: RuntimeFactLedger,
        conversation: list[Any],
    ) -> ActivationResult:
        normalized = topic.lower().replace(" ", "_").replace("gift_wrapping", "gift_wrap")
        if domain == "shopping_assistant":
            if normalized in {"brand_bundle", "category_bundle"}:
                field = "brand" if normalized == "brand_bundle" else "category"
                default = 2 if field == "brand" else 3
                items = ledger.joined_cart_items()
                if not items:
                    if not ledger.tool_seen("get_cart"):
                        cid = self._customer_id(ledger)
                        args = {"customer_id": cid} if cid else {}
                        return ActivationResult(ActivationValue.UNKNOWN, [AgentToolCallRequest("get_cart", args)], ["cart unread"])
                    return ActivationResult(ActivationValue.FALSE, [], ["cart is empty"])
                missing = self._product_reads(ledger, field)
                if missing:
                    return ActivationResult(ActivationValue.UNKNOWN, missing, [f"{field} missing for cart products"])
                policy = self._policy_text(normalized, ledger)
                if not policy:
                    return ActivationResult(
                        ActivationValue.UNKNOWN,
                        [AgentToolCallRequest("get_policies", {"topic": normalized})],
                        ["policy condition unread"],
                    )
                threshold, _ = _threshold_and_rate(policy, default)
                counts = Counter(str(item[field]) for item in items if item.get(field) not in (None, ""))
                winners = {key: count for key, count in counts.items() if count >= threshold}
                if winners:
                    return ActivationResult(ActivationValue.TRUE, [], [f"{field}_counts={dict(counts)}", f"threshold={threshold}"])
                return ActivationResult(ActivationValue.FALSE, [], [f"{field}_counts={dict(counts)}", f"threshold={threshold}"])

            if normalized == "welcome_discount":
                fact = ledger.latest("is_first_time")
                if fact.state == FactState.UNKNOWN or fact.stale:
                    cid = self._customer_id(ledger)
                    args = {"customer_id": cid} if cid else {}
                    return ActivationResult(ActivationValue.UNKNOWN, [AgentToolCallRequest("get_customer_account", args)], ["is_first_time unread"])
                return ActivationResult(
                    ActivationValue.TRUE if fact.value is True else ActivationValue.FALSE,
                    [],
                    [f"is_first_time={fact.value} from {fact.source_tool}"],
                )

            if normalized == "promo_stacking":
                possible = 0
                evidence = []
                for candidate in ("brand_bundle", "category_bundle", "welcome_discount"):
                    result = self.evaluate(domain, candidate, ledger, conversation)
                    if result.value == ActivationValue.TRUE:
                        possible += 1
                        evidence.append(candidate)
                promo_possible = ledger.tool_seen("validate_promo") or ledger.tool_seen("get_promotions") or bool(re.search(r"\bpromo\b|\bcoupon\b", _user_text(conversation), re.I))
                if promo_possible:
                    possible += 1
                    evidence.append("promo")
                return ActivationResult(ActivationValue.TRUE if possible >= 2 else ActivationValue.FALSE, [], evidence)

            if normalized in {"loyalty_points", "gift_wrap", "shipping"}:
                deciding = {
                    "loyalty_points": ("tier", "get_customer_account"),
                    "gift_wrap": ("gift_wrap", "get_cart"),
                    "shipping": ("shipping_cost", "get_shipping_options"),
                }
                field, tool = deciding[normalized]
                fact = ledger.latest(field)
                if fact.state == FactState.UNKNOWN or fact.stale:
                    cid = self._customer_id(ledger)
                    args = {"customer_id": cid} if cid and tool != "get_policies" else {}
                    return ActivationResult(ActivationValue.UNKNOWN, [AgentToolCallRequest(tool, args)], [f"{field} unread"])
                return ActivationResult(ActivationValue.TRUE, [], [f"{field}={fact.value}"])

        if domain == "travel":
            status = str(ledger.latest("flight_status").value or "").lower()
            cancelled = ledger.latest("cancelled")
            schedule = ledger.latest("schedule_changed")
            weather = ledger.latest("weather_disruption")
            if normalized in {"weather_fee_waiver", "weather_waiver"}:
                if weather.state == FactState.UNKNOWN:
                    return ActivationResult(ActivationValue.UNKNOWN, [], ["weather disruption unread"])
                return ActivationResult(ActivationValue.TRUE if weather.value is True else ActivationValue.FALSE, [], [f"weather_disruption={weather.value}"])
            if normalized in {"schedule_change_fee_waiver", "schedule_change_waiver"}:
                known = cancelled.state != FactState.UNKNOWN or schedule.state != FactState.UNKNOWN or bool(status)
                if not known:
                    return ActivationResult(ActivationValue.UNKNOWN, [], ["flight status unread"])
                holds = cancelled.value is True or schedule.value is True or status == "cancelled"
                return ActivationResult(ActivationValue.TRUE if holds else ActivationValue.FALSE, [], [f"flight_status={status}", f"schedule_changed={schedule.value}"])
            if normalized == "voluntary_change":
                evidence = not (weather.value is True or cancelled.value is True or schedule.value is True)
                return ActivationResult(ActivationValue.TRUE if evidence else ActivationValue.FALSE, [], ["no canonical involuntary disruption"])

        if domain == "customer_support":
            aliases = {"shipping_claim": "shipping", "lost_package": "shipping", "replacement": "exchange"}
            branch = aliases.get(normalized, normalized)
            text = _user_text(conversation).lower()
            status = " ".join(
                str(ledger.latest(key).value or "").lower()
                for key in ("order_status", "shipment_status", "delivery_status", "investigation_status")
            )
            cues = {
                "cancellation": ("cancel",),
                "refund": ("refund", "money back"),
                "shipping": ("lost", "missing", "not received", "damaged", "delayed"),
                "return": ("return", "send back"),
                "warranty": ("warranty", "repair"),
                "exchange": ("exchange", "replace", "replacement"),
            }
            holds = any(cue in text or cue in status for cue in cues.get(branch, (branch,)))
            return ActivationResult(ActivationValue.TRUE if holds else ActivationValue.FALSE, [], [f"request/status cues for {branch}: {holds}"])

        return ActivationResult(ActivationValue.UNKNOWN, [], ["topic not implemented"])


def evaluate(
    domain: str,
    topic: str,
    ledger: RuntimeFactLedger,
    conversation: list[Any],
) -> ActivationResult:
    return PolicyActivation().evaluate(domain, topic, ledger, conversation)
