"""Deterministic, append-only completion of policy-sensitive final answers."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from agents.policy_activation import ActivationValue, PolicyActivation
from agents.runtime_fact_ledger import FactState, RuntimeFactLedger


def _user_text(conversation: list[Any]) -> str:
    return " ".join(
        str(item.get("content", ""))
        for item in conversation
        if isinstance(item, dict)
        and item.get("role") == "user"
        and "[task_done]" not in str(item.get("content", "")).lower()
    )


def _money(value: float) -> str:
    return f"${value:,.2f}" if value != int(value) else f"${int(value):,}"


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _percent(text: str, *, near: str = "") -> float | None:
    candidate = text
    if near:
        chunks = [chunk for chunk in re.split(r"[\n.]", text) if near.lower() in chunk.lower()]
        if chunks:
            candidate = " ".join(chunks)
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", candidate)
    return float(match.group(1)) / 100 if match else None


class FinalResponseContract:
    def __init__(self, policy_topics: list[dict[str, Any]] | None = None):
        self._topics = {str(item.get("topic", "")): item for item in policy_topics or []}
        self.activation = PolicyActivation(policy_topics)

    def _policy(self, topic: str, ledger: RuntimeFactLedger) -> str:
        live = ledger.policy_text(topic)
        if live:
            return live
        item = self._topics.get(topic)
        return json.dumps(item, ensure_ascii=False, default=str) if item else ""

    @staticmethod
    def _append(draft: str, sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence or sentence.lower() in draft.lower():
            return draft
        return f"{draft.rstrip()}\n\n{sentence}" if draft.strip() else sentence

    @staticmethod
    def _budget(text: str) -> float | None:
        patterns = (
            r"(?:budget|under|within|no more than|up to)\s*(?:is|of|:)?\s*[$£€]?\s*(\d[\d,]*(?:\.\d+)?)",
            r"[$£€]\s*(\d[\d,]*(?:\.\d+)?)\s*(?:budget|maximum|max)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return float(match.group(1).replace(",", ""))
        return None

    @staticmethod
    def _travel_total(ledger: RuntimeFactLedger) -> float | None:
        for key in ("selected_total", "total_price", "trip_total", "new_total", "total_cost", "price"):
            fact = ledger.latest(key)
            value = _number(fact.value)
            if value is not None and not fact.stale:
                return value
        for name in ("update_booking", "create_booking", "search_flights", "search_hotels", "search_car_rentals"):
            record = ledger.latest_record(name)
            if not record:
                continue
            for row in reversed(list(_walk_dicts(record.result))):
                for key in ("selected_total", "new_total", "total_price", "total_cost", "price"):
                    value = _number(row.get(key))
                    if value is not None:
                        return value
        return None

    def _travel(self, conversation: list[Any], draft: str, ledger: RuntimeFactLedger) -> str:
        budget = self._budget(_user_text(conversation))
        total = self._travel_total(ledger)
        if budget is None or total is None:
            return draft
        delta = budget - total
        if delta >= 0:
            sentence = f"The selected total is {_money(total)}, which is within your {_money(budget)} budget, leaving {_money(delta)} of headroom."
        else:
            sentence = f"The selected total is {_money(total)}, which is {_money(-delta)} over your {_money(budget)} budget."
        if ("within" in draft.lower() or "over" in draft.lower()) and _money(abs(delta)).lower() in draft.lower():
            return draft
        return self._append(draft, sentence)

    @staticmethod
    def _line_total(item: dict[str, Any]) -> float | None:
        direct = _number(item.get("line_total"))
        if direct is not None:
            return direct
        price = _number(item.get("unit_price"))
        if price is None:
            price = _number(item.get("price"))
        quantity = _number(item.get("quantity")) or 1
        return price * quantity if price is not None else None

    def _shopping(self, conversation: list[Any], draft: str, ledger: RuntimeFactLedger) -> str:
        result = draft
        items = ledger.joined_cart_items()

        brand = self.activation.evaluate("shopping_assistant", "brand_bundle", ledger, conversation)
        if brand.value == ActivationValue.TRUE:
            policy = self._policy("brand_bundle", ledger)
            rate = _percent(policy, near="brand")
            counts = Counter(str(item.get("brand")) for item in items if item.get("brand"))
            qualifying = {name for name, count in counts.items() if count >= 2}
            base = sum(self._line_total(item) or 0 for item in items if str(item.get("brand")) in qualifying)
            if rate is not None and base > 0:
                saving = int(base * rate)
                stacking = "It stacks with other eligible discounts." if re.search(r"stack(?:s|ing).{0,40}(?:everything|promo|category)", policy, re.I) and "does not stack" not in policy.lower() else "Its stacking treatment follows the fetched policy."
                sentence = f"The qualifying same-brand lines total {_money(base)}; the {rate * 100:g}% brand-bundle saving is {_money(saving)}. {stacking}"
                if _money(saving) not in result:
                    result = self._append(result, sentence)

        welcome = self.activation.evaluate("shopping_assistant", "welcome_discount", ledger, conversation)
        if welcome.value == ActivationValue.TRUE:
            policy = self._policy("welcome_discount", ledger)
            rate = _percent(policy, near="welcome")
            subtotal = _number(ledger.latest("cart_subtotal").value)
            if rate is not None and subtotal is not None:
                saving = int(subtotal * rate)
                if _money(saving) not in result or "welcome" not in result.lower():
                    result = self._append(result, f"Your first-time welcome discount is {rate * 100:g}% of {_money(subtotal)}, or {_money(saving)}; it cannot be combined with a promo code, so only the better option applies.")

        category = self.activation.evaluate("shopping_assistant", "category_bundle", ledger, conversation)
        promo_record = ledger.latest_record("validate_promo")
        if category.value == ActivationValue.TRUE and promo_record and isinstance(promo_record.result, dict):
            policy = self._policy("category_bundle", ledger)
            rate = _percent(policy, near="category")
            counts = Counter(str(item.get("category")) for item in items if item.get("category"))
            qualifying = {name for name, count in counts.items() if count >= 3}
            base = sum(self._line_total(item) or 0 for item in items if str(item.get("category")) in qualifying)
            applied_discount = _number(ledger.latest("discount_amount").value)
            promo = applied_discount if applied_discount is not None and applied_discount > 0 else _number(promo_record.result.get("estimated_discount"))
            if rate is not None and base > 0 and promo is not None:
                category_saving = int(base * rate)
                better = "the category bundle" if category_saving >= promo else str(promo_record.arguments.get("promo_code", "the promo"))
                sentence = f"The category bundle saves {_money(category_saving)}, while the promo saves {_money(promo)}. They do not stack on the same items, so {better} is the better option."
                if not (_money(category_saving) in result and _money(promo) in result):
                    result = self._append(result, sentence)

        loyalty = self.activation.evaluate("shopping_assistant", "loyalty_points", ledger, conversation)
        tier = str(ledger.latest("tier").value or "")
        total = _number(ledger.latest("cart_total").value)
        if loyalty.value == ActivationValue.TRUE and tier and total is not None:
            policy = self._policy("loyalty_points", ledger)
            tier_line = next((line for line in re.split(r"[\n.]", policy) if tier.lower() in line.lower() and "point" in line.lower()), "")
            rate_match = re.search(r"(\d+(?:\.\d+)?)\s+points?\s+per", tier_line, re.I)
            if rate_match:
                rate = float(rate_match.group(1))
                points = int(round(total * rate))
                if str(points) not in result or "point" not in result.lower():
                    result = self._append(result, f"At the {tier.title()} rate of {rate:g} point{'s' if rate != 1 else ''} per dollar on the final {_money(total)} eligible total, you earn {points:,} points.")

        gift_policy = self._policy("gift_wrap", ledger)
        wrapped = sum(int(item.get("quantity", 1) or 1) for item in items if item.get("gift_wrap") is True)
        fee_match = re.search(r"[$£€]\s*(\d+(?:\.\d+)?)\s*(?:per|/)[- ]?(?:wrapped )?item", gift_policy, re.I)
        if wrapped and fee_match:
            fee = float(fee_match.group(1))
            total_fee = wrapped * fee
            if _money(total_fee) not in result or "wrap" not in result.lower():
                result = self._append(result, f"Gift wrap is {_money(fee)} per wrapped item; {wrapped} wrapped items cost {_money(total_fee)} in total.")
        return result

    def _customer_support(self, draft: str, ledger: RuntimeFactLedger) -> str:
        result = draft
        investigation = str(ledger.latest("investigation_status").value or "").lower()
        if investigation in {"pending", "open", "required", "investigation_only"}:
            sentence = "While the investigation is pending, I cannot issue a refund, replacement, cancellation, exchange, or store credit; no remedy mutation has been made."
            return self._append(result, sentence)

        refund = _number(ledger.latest("refund_amount").value)
        shipping = _number(ledger.latest("shipping_cost").value)
        if refund is not None and "refund" in result.lower():
            shipping_text = (
                f"Shipping of {_money(shipping)} is included in that refund."
                if shipping and _shipping_refundable(ledger)
                else (f"The {_money(shipping)} shipping charge is not refundable." if shipping else "No shipping refund is included.")
            )
            sentence = f"The item refund amount is {_money(refund)}. {shipping_text}"
            if _money(refund) not in result or "shipping" not in result.lower():
                result = self._append(result, sentence)
        return result

    def apply(
        self,
        domain: str,
        conversation: list[Any],
        draft: str,
        ledger: RuntimeFactLedger,
    ) -> str:
        if domain == "travel":
            return self._travel(conversation, draft, ledger)
        if domain == "shopping_assistant":
            return self._shopping(conversation, draft, ledger)
        if domain == "customer_support":
            return self._customer_support(draft, ledger)
        return draft


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _shipping_refundable(ledger: RuntimeFactLedger) -> bool:
    fact = ledger.latest("shipping_refundable")
    if fact.state != FactState.UNKNOWN:
        return fact.value is True
    policy = ledger.policy_text("refund").lower() + ledger.policy_text("return").lower()
    if re.search(r"shipping.{0,35}(?:not|isn't|is not|non[- ]?)\s*refundable", policy):
        return False
    return "shipping" in policy and bool(re.search(r"shipping.{0,30}(?:refundable|included)", policy))


def ground_policy_sensitive_claims(
    domain: str,
    conversation: list[Any],
    text: str,
    ledger: RuntimeFactLedger,
    policy_topics: list[dict[str, Any]] | None = None,
) -> str:
    """Remove only claims that canonical evidence definitely disproves.

    Ordinary catalog prices are intentionally outside this check.  Unknown
    claims are left to the existing minimal reviewer; definite false branches
    are removed without rewriting the rest of the answer.
    """
    activation = PolicyActivation(policy_topics)
    sentences = [part for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part]
    kept = []
    for sentence in sentences:
        lowered = sentence.lower()
        if domain == "shopping_assistant" and "welcome" in lowered and "discount" in lowered:
            if activation.evaluate(domain, "welcome_discount", ledger, conversation).value == ActivationValue.FALSE:
                continue
        if domain == "travel" and ("fee waived" in lowered or "free change" in lowered or "no change fee" in lowered):
            weather = activation.evaluate(domain, "weather_fee_waiver", ledger, conversation).value
            schedule = activation.evaluate(domain, "schedule_change_fee_waiver", ledger, conversation).value
            if weather == ActivationValue.FALSE and schedule == ActivationValue.FALSE:
                continue
        if domain == "customer_support" and "shipping" in lowered and "refund" in lowered:
            if not _shipping_refundable(ledger):
                continue
        kept.append(sentence)
    return " ".join(kept).strip()
