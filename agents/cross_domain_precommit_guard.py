"""Deterministic cross-domain checks applied immediately before mutations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from state_bench.agents.base import AgentToolCallRequest

from agents.policy_activation import ActivationValue, PolicyActivation
from agents.runtime_fact_ledger import FactState, RuntimeFactLedger, WRITE_TOOLS


class GuardAction(str, Enum):
    APPROVE = "APPROVE"
    BLOCK = "BLOCK"
    NEED_READ = "NEED_READ"
    NEED_USER_CHOICE = "NEED_USER_CHOICE"


@dataclass(slots=True)
class GuardResult:
    action: GuardAction
    correction: str = ""
    read_only_tool_calls: list[AgentToolCallRequest] = field(default_factory=list)
    question: str = ""
    evidence: list[str] = field(default_factory=list)


def _call_name(call: Any) -> str:
    return str(call.get("name", "")) if isinstance(call, dict) else str(call.name)


def _call_args(call: Any) -> dict[str, Any]:
    return dict(call.get("arguments") or {}) if isinstance(call, dict) else dict(call.arguments)


def _tool_calls(response: Any) -> list[Any]:
    return list(getattr(response, "tool_calls", None) or (response.get("tool_calls", []) if isinstance(response, dict) else []))


def _text(response: Any) -> str:
    return str(getattr(response, "text", None) or (response.get("text", "") if isinstance(response, dict) else ""))


def _user_text(conversation: list[Any]) -> str:
    return " ".join(
        str(item.get("content", ""))
        for item in conversation
        if isinstance(item, dict)
        and item.get("role") == "user"
        and "[task_done]" not in str(item.get("content", "")).lower()
    ).lower()


def _dedupe(calls: list[AgentToolCallRequest]) -> list[AgentToolCallRequest]:
    seen = set()
    result = []
    for call in calls:
        key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            result.append(call)
    return result


class CrossDomainPrecommitGuard:
    def __init__(self, policy_topics: list[dict[str, Any]] | None = None):
        self.activation = PolicyActivation(policy_topics)

    def check(
        self,
        domain: str,
        conversation: list[Any],
        proposed_response: Any,
        ledger: RuntimeFactLedger,
    ) -> GuardResult:
        calls = _tool_calls(proposed_response)
        if domain == "travel":
            result = self._travel(calls, ledger)
        elif domain == "customer_support":
            result = self._customer_support(calls, _text(proposed_response), conversation, ledger)
        elif domain == "shopping_assistant":
            result = self._shopping(calls, conversation, ledger)
        else:
            result = GuardResult(GuardAction.APPROVE)
        return result

    @staticmethod
    def _existing_flight_id(ledger: RuntimeFactLedger) -> str | None:
        record = ledger.latest_record("get_booking")
        if record and isinstance(record.result, dict):
            value = record.result.get("flight_id")
            if value:
                return str(value)
        return None

    def _travel(self, calls: list[Any], ledger: RuntimeFactLedger) -> GuardResult:
        for call in calls:
            if _call_name(call) != "update_booking":
                continue
            args = _call_args(call)
            if not args.get("flight_id"):
                continue
            reason = str(args.get("change_reason", "personal"))
            if reason not in {"weather", "schedule_change"}:
                continue
            topic = "weather_fee_waiver" if reason == "weather" else "schedule_change_fee_waiver"
            verdict = self.activation.evaluate("travel", topic, ledger, [])
            if verdict.value == ActivationValue.TRUE:
                continue
            if verdict.value == ActivationValue.UNKNOWN:
                flight_id = self._existing_flight_id(ledger)
                if flight_id and not ledger.tool_seen("get_flight_status"):
                    return GuardResult(
                        GuardAction.NEED_READ,
                        read_only_tool_calls=[AgentToolCallRequest("get_flight_status", {"flight_id": flight_id})],
                        evidence=verdict.evidence,
                    )
            return GuardResult(
                GuardAction.BLOCK,
                correction=(
                    f"The proposed {reason} reason is contradicted or unsupported by canonical flight evidence. "
                    "Regenerate the change as a voluntary/personal change, preview it with confirm=false, and use "
                    "the fee returned by the policy/tool rather than claiming a waiver."
                ),
                evidence=verdict.evidence,
            )
        return GuardResult(GuardAction.APPROVE)

    @staticmethod
    def _selected_remedy(user_text: str) -> str | None:
        patterns = {
            "refund": (r"\b(?:i (?:want|choose|prefer)|please|go with) (?:a |the )?refund\b", r"\brefund (?:it|me|please)\b"),
            "reship": (r"\b(?:i (?:want|choose|prefer)|please|go with) (?:a |the )?(?:reship|replacement)\b", r"\b(?:reship|replace) (?:it|this|please)\b"),
            "return": (r"\b(?:i (?:want|choose|prefer)|please) (?:to )?return\b",),
            "exchange": (r"\b(?:i (?:want|choose|prefer)|please) (?:an )?exchange\b",),
        }
        for remedy, regexes in patterns.items():
            if any(re.search(regex, user_text, re.I) for regex in regexes):
                return remedy
        return None

    def _customer_support(
        self,
        calls: list[Any],
        proposed_text: str,
        conversation: list[Any],
        ledger: RuntimeFactLedger,
    ) -> GuardResult:
        writes = [call for call in calls if _call_name(call) in WRITE_TOOLS]
        if not writes and "paid repair" not in proposed_text.lower():
            return GuardResult(GuardAction.APPROVE)

        state_text = " ".join(
            str(ledger.latest(key).value or "")
            for key in ("order_status", "shipment_status", "delivery_status", "investigation_status")
        ).lower()
        policy_text = ledger.policy_text().lower()
        investigation_only = (
            "investigation only" in state_text
            or "investigation_only" in state_text
            or "investigation only" in policy_text
            or ledger.latest("investigation_status").value in {"pending", "open", "required"}
        )
        if investigation_only and writes:
            return GuardResult(
                GuardAction.BLOCK,
                correction="The canonical state is investigation-only. Do not execute refund, reship, return, cancellation, exchange, warranty, or store-credit mutations until the investigation resolves.",
                evidence=[state_text],
            )

        for call in writes:
            name = _call_name(call)
            args = _call_args(call)
            if name == "cancel_order" and any(term in state_text for term in ("lost", "missing", "not received", "delivered", "damaged")):
                return GuardResult(
                    GuardAction.BLOCK,
                    correction="This is a shipment remedy case, not an order-cancellation case. Do not use cancel_order; follow the shipping policy and offer only the grounded refund/reship path.",
                    evidence=[state_text],
                )
            if args.get("amount") == 0 and ledger.latest("refund_amount").state == FactState.NULL:
                return GuardResult(
                    GuardAction.BLOCK,
                    correction="refund_amount is explicitly null, not zero. Do not issue a meaningless amount=0 write; obtain a real preview/calculation or take no mutation.",
                    evidence=["refund_amount=NULL"],
                )

        remedies = ledger.latest("available_remedies")
        remedies_text = json.dumps(remedies.value, ensure_ascii=False).lower() if remedies.state in {FactState.VALUE, FactState.ZERO} else policy_text
        both = "refund" in remedies_text and any(term in remedies_text for term in ("reship", "replacement"))
        selection = self._selected_remedy(_user_text(conversation))
        if both and selection not in {"refund", "reship"} and any(_call_name(call) in {"process_refund", "process_exchange"} for call in writes):
            return GuardResult(
                GuardAction.NEED_USER_CHOICE,
                question="Both a refund and a reshipment are available. Which would you like me to process? I have not changed the order yet.",
                evidence=["refund and reship are both available", "customer choice absent"],
            )

        warranty = ledger.latest("warranty_eligibility")
        return_eligibility = ledger.latest("return_eligibility")
        if (
            ("paid repair" in proposed_text.lower() or any(_call_name(call) == "process_warranty_claim" for call in writes))
            and warranty.state != FactState.UNKNOWN
            and warranty.value is False
            and return_eligibility.state == FactState.UNKNOWN
        ):
            return GuardResult(
                GuardAction.NEED_READ,
                read_only_tool_calls=[AgentToolCallRequest("get_policies", {"topic": "return"})],
                evidence=["warranty unavailable", "return eligibility unread"],
            )
        return GuardResult(GuardAction.APPROVE)

    def _shopping(
        self,
        calls: list[Any],
        conversation: list[Any],
        ledger: RuntimeFactLedger,
    ) -> GuardResult:
        names = {_call_name(call) for call in calls}
        writes = names & WRITE_TOOLS
        if not writes:
            mutation_intent = bool(
                re.search(
                    r"\b(?:add|remove|update|change|apply|cart|buy|purchase|wrap)\b",
                    _user_text(conversation),
                    re.I,
                )
            )
            if "get_cart" in names and mutation_intent and not ledger.tool_seen("get_customer_account"):
                cart_call = next(call for call in calls if _call_name(call) == "get_cart")
                customer_id = _call_args(cart_call).get("customer_id")
                return GuardResult(
                    GuardAction.NEED_READ,
                    read_only_tool_calls=[AgentToolCallRequest("get_customer_account", {"customer_id": customer_id})],
                    evidence=["cart mutation intent with account identity unread"],
                )
            return GuardResult(GuardAction.APPROVE)
        reads: list[AgentToolCallRequest] = []

        if names & {"add_to_cart", "update_cart_item", "apply_promo"}:
            current_ids = {
                str(item.get("product_id"))
                for item in ledger.cart_items()
                if item.get("product_id") not in (None, "")
            }
            proposed_ids = {
                str(_call_args(call).get("product_id"))
                for call in calls
                if _call_name(call) == "add_to_cart"
                and _call_args(call).get("product_id") not in (None, "")
            }
            prospective_count = len(current_ids | proposed_ids)
            topics = []
            if prospective_count >= 2:
                topics.append("brand_bundle")
            if prospective_count >= 3:
                topics.append("category_bundle")
            for topic in topics:
                result = self.activation.evaluate("shopping_assistant", topic, ledger, conversation)
                reads.extend(result.missing_read_tools if result.value == ActivationValue.UNKNOWN else [])
            for call in calls if topics else []:
                if _call_name(call) not in {"add_to_cart", "update_cart_item"}:
                    continue
                product_id = _call_args(call).get("product_id")
                fields = ["brand"] + (["category"] if prospective_count >= 3 else [])
                if product_id and any(ledger.latest(field, str(product_id)).state == FactState.UNKNOWN for field in fields):
                    reads.append(AgentToolCallRequest("get_product_details", {"product_id": str(product_id)}))

        if names & {"add_to_cart", "update_cart_item"} and not ledger.tool_seen("get_customer_account"):
            call = next(call for call in calls if _call_name(call) in {"add_to_cart", "update_cart_item"})
            customer_id = _call_args(call).get("customer_id")
            reads.append(AgentToolCallRequest("get_customer_account", {"customer_id": customer_id}))

        if any(
            _call_name(call) in {"add_to_cart", "update_cart_item"}
            and _call_args(call).get("gift_wrap") is True
            for call in calls
        ) and not ledger.policy_text("gift_wrap"):
            reads.append(AgentToolCallRequest("get_policies", {"topic": "gift_wrap"}))

        if "apply_promo" in names:
            welcome = self.activation.evaluate("shopping_assistant", "welcome_discount", ledger, conversation)
            if welcome.value == ActivationValue.UNKNOWN:
                reads.extend(welcome.missing_read_tools)
            for call in calls:
                if _call_name(call) == "apply_promo":
                    code = _call_args(call).get("promo_code")
                    if code and not ledger.tool_seen("validate_promo"):
                        customer_id = _call_args(call).get("customer_id")
                        reads.append(AgentToolCallRequest("validate_promo", {"customer_id": customer_id, "promo_code": code}))

        if names & {"redeem_loyalty_points", "cancel_loyalty_redemption"}:
            for tool, observed in (
                ("get_customer_account", ledger.tool_seen("get_customer_account")),
                ("get_cart", ledger.tool_seen("get_cart")),
            ):
                if not observed:
                    call = next(call for call in calls if _call_name(call) in {"redeem_loyalty_points", "cancel_loyalty_redemption"})
                    cid = _call_args(call).get("customer_id")
                    reads.append(AgentToolCallRequest(tool, {"customer_id": cid}))

        if "set_shipping_option" in names and not ledger.tool_seen("get_shipping_options"):
            call = next(call for call in calls if _call_name(call) == "set_shipping_option")
            reads.append(AgentToolCallRequest("get_shipping_options", {"customer_id": _call_args(call).get("customer_id")}))

        reads = _dedupe(reads)
        if reads:
            return GuardResult(GuardAction.NEED_READ, read_only_tool_calls=reads, evidence=["policy decision facts are unread"])
        return GuardResult(GuardAction.APPROVE)
