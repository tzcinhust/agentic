"""PWM plus a thin StateAct/CRITIC-style state verification layer.

This is deliberately an adapter, not a replacement agent.  Workflow retrieval
and the first action decision remain ProcessWorkflowMemoryAgent.  After a
shopping write, the adapter requires one canonical ``get_cart`` readback before
allowing the final answer, then supplies a compact chain-of-states to the next
model call.  The verifier therefore uses external environment feedback rather
than an unconstrained second opinion.

Motivation from prior work:
* StateAct (REALM 2025): self-prompting plus chain-of-states improves WebShop.
* TOOLVERIFIER (EMNLP Findings 2024): verify close tool/argument choices.
* CRITIC (ICLR 2024): correction works when grounded in external tool feedback.
"""

from __future__ import annotations

import json
from typing import Any

from state_bench.agents.base import AgentToolCallRequest, AgentTurnResponse

from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent as _Parent


SHOPPING_WRITES = frozenset(
    {
        "add_to_cart",
        "remove_from_cart",
        "update_cart_item",
        "apply_promo",
        "remove_promo",
        "redeem_loyalty_points",
        "cancel_loyalty_redemption",
        "set_shipping_option",
    }
)


class StateVerifiedPWMAgent(_Parent):
    """Add canonical post-write verification and a compact state chain to PWM."""

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self._runtime_domain = getattr(runtime_context, "domain", None)
        self._runtime_user_id = str(getattr(runtime_context, "user_id", ""))

    @staticmethod
    def _records(conversation: list[dict[str, Any]]) -> list[tuple[int, str, Any]]:
        records: list[tuple[int, str, Any]] = []
        for index, item in enumerate(conversation):
            for call in item.get("tool_calls") or []:
                if isinstance(call, dict):
                    records.append((index, str(call.get("name", "")), call.get("result")))
        return records

    @classmethod
    def _needs_cart_audit(cls, conversation: list[dict[str, Any]]) -> bool:
        records = cls._records(conversation)
        last_write = max(
            (index for index, name, _ in records if name in SHOPPING_WRITES),
            default=-1,
        )
        last_view = max(
            (index for index, name, _ in records if name == "get_cart"),
            default=-1,
        )
        return last_write > last_view

    @classmethod
    def _state_card(cls, conversation: list[dict[str, Any]]) -> str:
        records = cls._records(conversation)
        cart = next(
            (
                result
                for _, name, result in reversed(records)
                if name == "get_cart" and isinstance(result, dict)
            ),
            None,
        )
        if not cart:
            return ""
        items = cart.get("items") or []
        compact_items = [
            {
                "id": item.get("product_id"),
                "name": item.get("product_name"),
                "qty": item.get("quantity"),
                "wrap": item.get("gift_wrap"),
                "line_total": item.get("line_total"),
            }
            for item in items
            if isinstance(item, dict)
        ]
        state = {
            "items": compact_items,
            "subtotal": cart.get("subtotal"),
            "discount": cart.get("discount_amount"),
            "promos": cart.get("applied_promo_codes"),
            "loyalty_redeemed": cart.get("loyalty_points_redeemed"),
            "loyalty_discount": cart.get("loyalty_discount"),
            "shipping_option": cart.get("shipping_option"),
            "shipping_cost": cart.get("shipping_cost"),
            "total": cart.get("total"),
        }
        return (
            "Canonical chain-of-states after the latest write (tool-grounded JSON):\n"
            + json.dumps(state, ensure_ascii=False, separators=(",", ":"))
            + "\nUse this state for the final readback. Explicitly disclose any promo, "
            "shipping, redemption, quantity, or total that changed; do not reconstruct "
            "values from an earlier cart snapshot."
        )

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        if self._runtime_domain == "shopping_assistant":
            card = self._state_card(conversation)
            if card:
                system_prompt = f"{system_prompt}\n\n{card}"

        response = super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
        if (
            self._runtime_domain == "shopping_assistant"
            and not response.tool_calls
            and response.text.strip()
            and self._needs_cart_audit(conversation)
        ):
            return AgentTurnResponse(
                text="",
                tool_calls=[
                    AgentToolCallRequest(
                        name="get_cart",
                        arguments={"customer_id": self._runtime_user_id},
                    )
                ],
            )
        return response
