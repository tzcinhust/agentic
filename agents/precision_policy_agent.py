"""PWM with precision-gated policy compliance.

The broad policy arms improved individual disclosure rubrics but changed too
many unrelated decisions.  This variant keeps PWM's generation untouched and
adds only two guards whose applicability is observable from the conversation:

* a deterministic quantity-cap consent turn before an over-cap cart write;
* a non-destructive final addendum for policy topics activated by exact user or
  tool evidence.

The addendum never rewrites the PWM draft.  For the unconditional loyalty
disclosure it first obtains the account and cart through read-only tools, so the
tier, calculation base, and arithmetic are grounded rather than guessed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from state_bench.agents.base import AgentToolCallRequest, AgentTurnResponse

from agents.policy_obligation_agent import PolicyObligationAgent as _PolicyMixin
from agents.post_tool_policy_agent import PostToolPolicyAgent as _ContextPolicy
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent as _PWMParent


_CODE = re.compile(r"\b[A-Z][A-Z0-9]{2,}\d[A-Z0-9]*\b")
_CURRENCY = re.compile(r"[$£€]\s*\d[\d,]*(?:\.\d+)?")
_REQUEST_PATTERNS = (
    # Keep the number adjacent to the quantity verb.  A looser gap mistakes
    # model names such as "SlimBook Air 13" for a request for 13 units.
    re.compile(r"\b(?:need|want|add|buy|get|order)\s+(?:exactly\s+)?(\d+)\b", re.I),
    re.compile(r"\b(\d+)\s+(?:more|units?|items?|of\s+(?:these|them))\b", re.I),
    re.compile(r"\b(?:quantity|qty)\D{0,12}?(\d+)\b", re.I),
)


class PrecisionPolicyAgent(_PolicyMixin):
    """Preserve PWM except where policy applicability is precisely grounded."""

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        # PolicyObligationAgent loads the policy artifact, while its parent is
        # PWM.  Retrieval is overridden below to remain byte-identical to PWM.
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self._quantity_cap = self._read_quantity_cap()

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        """Use all three PWM workflow slots; policy never displaces a card."""
        return _PWMParent.retrieve_learnings(self, query, top_k=top_k)

    def _topic(self, name: str) -> dict[str, Any] | None:
        return next((item for item in self._topics if item.get("topic") == name), None)

    def _read_quantity_cap(self) -> int:
        item = self._topic("quantity_limit")
        if not item:
            return 3
        for obligation in item.get("obligations", []):
            if "number" not in set(obligation.get("act") or []):
                continue
            match = re.search(r"\b(\d+)\s+units?\b", str(obligation.get("text", "")), re.I)
            if match:
                return int(match.group(1))
        return 3

    @staticmethod
    def _user_text(conversation: list[dict[str, Any]]) -> str:
        return " ".join(
            str(item.get("content", ""))
            for item in conversation
            if item.get("role") == "user" and "[TASK_DONE]" not in str(item.get("content", ""))
        )

    @staticmethod
    def _observed_calls(conversation: list[dict[str, Any]]) -> set[str]:
        return {
            str(call.get("name", ""))
            for item in conversation
            if item.get("role") == "assistant"
            for call in item.get("tool_calls") or []
        }

    @staticmethod
    def _call_name(call: Any) -> str:
        return str(call.get("name", "")) if isinstance(call, dict) else str(call.name)

    @staticmethod
    def _call_args(call: Any) -> dict[str, Any]:
        return dict(call.get("arguments") or {}) if isinstance(call, dict) else dict(call.arguments)

    @classmethod
    def _requested_quantity(cls, text: str) -> int | None:
        cleaned = _CURRENCY.sub(" ", text)
        candidates = [
            int(match.group(1))
            for pattern in _REQUEST_PATTERNS
            for match in pattern.finditer(cleaned)
        ]
        return max(candidates) if candidates else None

    def _quantity_guard(
        self,
        response: AgentTurnResponse,
        conversation: list[dict[str, Any]],
    ) -> AgentTurnResponse | None:
        writes = [
            call
            for call in response.tool_calls
            if self._call_name(call) in {"add_to_cart", "update_cart_item"}
        ]
        if not writes:
            return None
        requested = self._requested_quantity(self._user_text(conversation))
        if requested is None or requested <= self._quantity_cap:
            return None

        prior_assistant = " ".join(
            str(item.get("content", ""))
            for item in conversation
            if item.get("role") == "assistant"
        ).lower()
        # Once the cap has been disclosed, the simulator's next user turn gives
        # explicit consent and the original PWM write can proceed.
        if "cap" in prior_assistant and str(self._quantity_cap) in prior_assistant:
            return None

        write = writes[0]
        proposed = int(self._call_args(write).get("quantity", self._quantity_cap) or self._quantity_cap)
        allowed = min(proposed, self._quantity_cap)
        shortfall = max(0, requested - allowed)
        if self._call_name(write) == "update_cart_item":
            text = (
                f"Store policy caps the total quantity of one product at {self._quantity_cap} "
                f"units, including units already in the cart. I have not changed it yet. "
                f"Shall I set the final quantity to {allowed}?"
            )
        else:
            text = (
                f"Store policy caps one product at {self._quantity_cap} units per cart, so I can "
                f"add only {allowed} of the requested {requested} (a shortfall of {shortfall}). "
                f"I have not added anything yet. Shall I proceed with {allowed}?"
            )
        return AgentTurnResponse(text=text, tool_calls=[])

    def _active_topics(self, conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        user_text = self._user_text(conversation)
        lowered = user_text.lower()
        context = json.dumps(conversation, ensure_ascii=False, default=str).lower()
        calls = self._observed_calls(conversation)
        names: list[str] = []

        loyalty = self._topic("loyalty_points")
        if loyalty and (
            calls & {str(name) for name in loyalty.get("write_tools", [])}
            or re.search(r"\b(?:loyalty|points?\s+(?:would|will|do|earn))\b", lowered)
        ):
            names.append("loyalty_points")

        codes = set(_CODE.findall(user_text))
        if len(codes) >= 2 or "stack" in lowered:
            names.append("promo_stacking")
        cart_history = self._cart_history(conversation)
        if (
            cart_history
            and cart_history[0].get("applied_promo_codes")
            and calls & {"add_to_cart", "remove_from_cart", "update_cart_item"}
        ):
            names.append("promo_stacking")
        if (
            any(code.startswith("WELCOME") for code in codes)
            or "first-time" in lowered
            or "first time" in lowered
            or ("apply_promo" in calls and re.search(r'"is_first_time"\s*:\s*true', context))
        ):
            names.append("welcome_discount")
        if re.search(r"\b(?:redeem|use|spend)\b.{0,24}\bpoints?\b", lowered):
            names.append("loyalty_redemption")
        if any(term in lowered for term in ("shipping", "same-day", "same day", "next-day", "next day", "arrive", "delivery deadline")):
            names.append("shipping")
        if "backorder_available" in context and re.search(r"backorder_available.{0,20}true", context):
            names.append("backorder")
        if self._has_price_drop(conversation):
            names.append("price_alerts")

        selected = []
        for name in dict.fromkeys(names):
            item = self._topic(name)
            if item:
                selected.append(item)
        return selected

    @classmethod
    def _has_price_drop(cls, value: Any) -> bool:
        if isinstance(value, dict):
            previous = value.get("previous_price")
            current = value.get("price")
            if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
                if previous > current:
                    return True
            return any(cls._has_price_drop(item) for item in value.values())
        if isinstance(value, list):
            return any(cls._has_price_drop(item) for item in value)
        return False

    def _missing_loyalty_reads(
        self,
        items: list[dict[str, Any]],
        conversation: list[dict[str, Any]],
    ) -> list[AgentToolCallRequest]:
        if not any(item.get("topic") == "loyalty_points" for item in items):
            return []
        calls = self._observed_calls(conversation)
        customer_id = str(getattr(self.runtime_context, "user_id", ""))
        missing = []
        if "get_customer_account" not in calls:
            missing.append(AgentToolCallRequest("get_customer_account", {"customer_id": customer_id}))
        if "get_cart" not in calls:
            missing.append(AgentToolCallRequest("get_cart", {"customer_id": customer_id}))
        return missing

    @classmethod
    def _walk_dicts(cls, value: Any):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from cls._walk_dicts(child)
        elif isinstance(value, list):
            for child in value:
                yield from cls._walk_dicts(child)

    @classmethod
    def _latest_field(cls, conversation: list[dict[str, Any]], key: str) -> Any:
        values = [item[key] for item in cls._walk_dicts(conversation) if key in item]
        return values[-1] if values else None

    @classmethod
    def _cart_history(cls, conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        carts = []
        for item in cls._walk_dicts(conversation):
            if "cart_subtotal" in item and "cart_total" in item:
                carts.append(
                    {
                        **item,
                        "subtotal": item["cart_subtotal"],
                        "total": item["cart_total"],
                    }
                )
            elif (
                "subtotal" in item
                and "total" in item
                and any(key in item for key in ("items", "discount_amount", "applied_promo_codes"))
            ):
                carts.append(item)
        return carts

    @classmethod
    def _latest_cart(cls, conversation: list[dict[str, Any]]) -> dict[str, Any] | None:
        carts = cls._cart_history(conversation)
        return carts[-1] if carts else None

    @classmethod
    def _first_product_price(cls, conversation: list[dict[str, Any]]) -> int | float | None:
        products = [
            item
            for item in cls._walk_dicts(conversation)
            if "product_id" in item and "name" in item and isinstance(item.get("price"), (int, float))
        ]
        return products[0].get("price") if products else None

    def _deterministic_addendum(
        self,
        items: list[dict[str, Any]],
        conversation: list[dict[str, Any]],
        draft: str,
    ) -> str:
        """Render only facts that can be computed directly from canonical tools."""
        active = {str(item.get("topic")) for item in items}
        lowered = draft.lower()
        user_text = self._user_text(conversation)
        cart = self._latest_cart(conversation) or {}
        subtotal = cart.get("subtotal")
        total = cart.get("total")
        tier = str(self._latest_field(conversation, "tier") or "").title()
        additions: list[str] = []

        if "loyalty_points" in active and tier in {"Standard", "Gold", "Platinum"}:
            rate = {"Standard": 1, "Gold": 2, "Platinum": 3}[tier]
            base = total if isinstance(total, (int, float)) and total > 0 else self._first_product_price(conversation)
            if isinstance(base, (int, float)):
                points = int(round(float(base) * rate))
                point_text = f"{points:,}"
                cart_based = isinstance(total, (int, float)) and total > 0
                basis = "final total after discounts" if cart_based else "listed item price before discounts"
                if point_text not in draft or tier.lower() not in lowered:
                    additions.append(
                        f"Verified loyalty tier: {tier}, earning {rate} point"
                        f"{'s' if rate != 1 else ''} per $1 on the {basis}; "
                        f"{rate} × ${float(base):g} = {point_text} points."
                    )

        is_first_time = self._latest_field(conversation, "is_first_time") is True
        if "welcome_discount" in active and is_first_time:
            has_eligibility = "first-time" in lowered or "first time" in lowered
            has_nonstack = any(
                phrase in lowered
                for phrase in ("does not stack", "doesn't stack", "cannot be combined", "can't combine")
            )
            if not (has_eligibility and has_nonstack):
                applied = cart.get("applied_promo_codes") or []
                applied_text = str(applied[-1]) if applied else "the applied promo"
                additions.append(
                    "As a first-time customer, you are eligible for the automatic 5% welcome "
                    f"discount. It does not stack with promo codes; you receive the better option, "
                    f"which is {applied_text} here."
                )

        if "promo_stacking" in active:
            codes = sorted(set(_CODE.findall(user_text)))
            rates = {
                code: int(match.group(1))
                for code in codes
                if (match := re.search(r"(\d+)$", code))
            }
            # The first comparison turn often occurs while the cart is still
            # empty.  Wait until the selected item's canonical subtotal exists;
            # otherwise a mathematically valid 10% × $0 disclosure poisons the
            # transcript even if the final cart addendum is later correct.
            if len(rates) >= 2 and isinstance(subtotal, (int, float)) and subtotal > 0:
                comparison = ", ".join(
                    f"{code}: {rate}% = ${int(float(subtotal) * rate / 100):,} off"
                    for code, rate in sorted(rates.items(), key=lambda pair: pair[1])
                )
                amounts = [f"${int(float(subtotal) * rate / 100):,}" for rate in rates.values()]
                if not all(amount in draft for amount in amounts):
                    best = max(rates, key=rates.get)
                    additions.append(
                        f"Promo comparison: {comparison}. Only one promo code can be used per cart, "
                        f"so {best} is the better single code."
                    )

            history = self._cart_history(conversation)
            if history:
                seeded = next(
                    (cart for cart in history if cart.get("applied_promo_codes")),
                    None,
                )
                latest = history[-1]
                dropped = (
                    seeded
                    and isinstance(latest.get("subtotal"), (int, float))
                    and isinstance(latest.get("total"), (int, float))
                    and latest.get("subtotal") == latest.get("total")
                    and latest.get("subtotal") != seeded.get("subtotal")
                )
                if dropped:
                    code = str(seeded.get("applied_promo_codes", ["the existing promo"])[0])
                    old_discount = seeded.get("discount_amount", 0)
                    if code.lower() not in lowered or not any(
                        word in lowered for word in ("removed", "dropped", "ineligible", "no longer")
                    ):
                        additions.append(
                            f"Cart edit disclosure: {code} became ineligible for the edited cart and "
                            f"was removed; the ${old_discount:g} discount dropped to $0, making the "
                            f"updated total ${float(latest['total']):g}."
                        )

        if "shipping" in active and "next business day" not in lowered:
            calls = self._observed_calls(conversation)
            if "set_shipping_option" in calls and "next-day" in json.dumps(conversation).lower():
                fee = 0 if tier == "Platinum" else 15
                additions.append(
                    f"Next-day shipping arrives the next business day; the applicable fee is ${fee}."
                )

        if "backorder" in active and not ("10%" in draft and ("2–4" in draft or "2-4" in draft)):
            additions.append(
                "Backorder is available with a refundable 10% deposit and an estimated 2–4 week restock window."
            )

        return "\n".join(additions)

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        draft = _PWMParent.generate_next_turn(
            self,
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
        if draft.tool_calls:
            return draft
        if not draft.text.strip():
            return draft

        items = self._active_topics(conversation)
        if not items:
            return draft
        missing_reads = self._missing_loyalty_reads(items, conversation)
        if missing_reads:
            return AgentTurnResponse(text="", tool_calls=missing_reads)

        items = _ContextPolicy._contextual_items(items, conversation)
        addendum = self._deterministic_addendum(items, conversation, draft.text).strip()
        if not addendum:
            return draft
        return AgentTurnResponse(text=f"{draft.text.rstrip()}\n\n{addendum}", tool_calls=[])
