"""PWM plus a compact, risk-gated TOOLVERIFIER-style pre-commit check.

This keeps PWM's retrieval and acting path intact.  A second model call is made
only when the proposed shopping write touches one of the failure families seen
in the train split: quantity semantics, firm-budget bundles, or loyalty state.
The verifier receives a compact evidence packet rather than the full transcript.
"""

from __future__ import annotations

import json
import re
from typing import Any

from state_bench.agents.base import AgentTurnResponse

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

QUANTITY_WRITES = frozenset({"add_to_cart", "update_cart_item"})
LOYALTY_WRITES = frozenset({"redeem_loyalty_points", "cancel_loyalty_redemption"})
LOYALTY_CONTEXT_WRITES = frozenset(
    {"add_to_cart", "remove_from_cart", "update_cart_item", *LOYALTY_WRITES}
)

VERIFY_SYSTEM = """You are a conservative pre-commit verifier for shopping
tool calls. Use only the evidence packet. Return exactly one line:
APPROVE
or
REVISE: <one concrete correction grounded in the packet>

Check only these high-risk invariants:
1. Quantity: distinguish requested increment from final quantity. If a cap or
stock limit may bind, reads/policy and disclosure must occur before the first
write attempt; never probe a known/possible cap by attempting the write.
2. Firm budget bundles: exact item total and remaining headroom must be computed
and disclosed; every proposed item must match the accepted bundle.
3. Loyalty: preserve a still-valid existing redemption. Do not cancel/redeem,
top up, or choose a new amount unless the user explicitly names that exact
amount; account balance and the cap are not permission. Contrast keeping the
current redemption with cancelling/replacing it.

Use REVISE only for a definite violation. Do not demand optional optimization,
invent policy, or revise style."""


class RiskGatedVerifiedPWMAgent(_Parent):
    """Run a compact verifier only for quantity, budget, or loyalty writes."""

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self._runtime_domain = getattr(runtime_context, "domain", None)

    @staticmethod
    def _name(call: Any) -> str:
        return str(call.get("name", "")) if isinstance(call, dict) else str(call.name)

    @staticmethod
    def _args(call: Any) -> dict[str, Any]:
        return dict(call.get("arguments") or {}) if isinstance(call, dict) else dict(call.arguments)

    @staticmethod
    def _user_text(conversation: list[dict[str, Any]]) -> str:
        return " ".join(
            str(item.get("content", ""))
            for item in conversation
            if item.get("role") == "user" and "[TASK_DONE]" not in str(item.get("content", ""))
        ).lower()

    def _risk_family(self, conversation: list[dict[str, Any]], writes: list[Any]) -> str | None:
        names = [self._name(call) for call in writes]
        text = self._user_text(conversation)

        if any(name in LOYALTY_CONTEXT_WRITES for name in names) and re.search(
            r"\b(loyalty|points?|redeem|redemption)\b", text
        ):
            return "loyalty"

        if any(name in QUANTITY_WRITES for name in names) and (
            re.search(r"\b(quantity|qty|units?|more|another|stock|limit|cap)\b", text)
            or re.search(r"\b\d+\b", text)
        ):
            return "quantity"

        add_count = sum(name == "add_to_cart" for name in names)
        if add_count >= 2 and re.search(r"\b(budget|under|maximum|max|cap|total|bundle)\b", text):
            return "budget_bundle"
        return None

    @staticmethod
    def _compact_evidence(
        conversation: list[dict[str, Any]], proposed: AgentTurnResponse, risk_family: str
    ) -> dict[str, Any]:
        users = [
            str(item.get("content", ""))[:1200]
            for item in conversation
            if item.get("role") == "user" and "[TASK_DONE]" not in str(item.get("content", ""))
        ][-4:]
        observations: list[dict[str, Any]] = []
        for item in conversation[-8:]:
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                result = json.dumps(call.get("result"), ensure_ascii=False, separators=(",", ":"))
                observations.append(
                    {
                        "name": str(call.get("name", "")),
                        "arguments": call.get("arguments") or {},
                        "result": result[:1800],
                    }
                )
        return {
            "risk_family": risk_family,
            "user_messages": users,
            "recent_tool_observations": observations[-10:],
            "proposed_assistant_text": proposed.text[:1800],
            "proposed_tool_batch": [
                {
                    "position": index,
                    "name": str(call.get("name", "")) if isinstance(call, dict) else str(call.name),
                    "arguments": (call.get("arguments") or {}) if isinstance(call, dict) else dict(call.arguments),
                }
                for index, call in enumerate(proposed.tool_calls, start=1)
            ],
        }

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        proposed = super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
        writes = [call for call in proposed.tool_calls if self._name(call) in SHOPPING_WRITES]
        if self._runtime_domain != "shopping_assistant" or not writes:
            return proposed

        risk_family = self._risk_family(conversation, writes)
        if risk_family is None:
            return proposed

        packet = self._compact_evidence(conversation, proposed, risk_family)
        verdict = self.client.generate(
            system_prompt=VERIFY_SYSTEM,
            conversation=[
                {
                    "role": "user",
                    "content": json.dumps(packet, ensure_ascii=False, separators=(",", ":")),
                }
            ],
            tools=[],
        )
        usage = verdict.usage
        self.add_token_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )
        text = verdict.text.strip()
        if not text.upper().startswith("REVISE:"):
            return proposed
        correction = text.split(":", 1)[1].strip()
        if not correction:
            return proposed

        revised_prompt = (
            f"{system_prompt}\n\nA risk-gated pre-commit verifier found this definite issue: "
            f"{correction}\nRegenerate only the next step. Do not execute the invalid write. "
            "Use current canonical tool facts and preserve all valid user-authorized state."
        )
        return _Parent.generate_next_turn(
            self,
            system_prompt=revised_prompt,
            conversation=conversation,
            tools=tools,
        )
