"""PWM with a narrow external-state verifier for loyalty replacements.

The module is intentionally dormant unless PWM proposes a loyalty write.  It
implements the useful part of TOOLVERIFIER/CRITIC as a pre-commit check while
leaving retrieval, reads, cart mutations, and every non-shopping domain alone.
"""

from __future__ import annotations

import json
from typing import Any

from state_bench.agents.base import AgentTurnResponse

from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent as _Parent


LOYALTY_WRITES = frozenset({"redeem_loyalty_points", "cancel_loyalty_redemption"})

VERIFY_SYSTEM = """Verify only the proposed loyalty write batch using the
canonical observations in the evidence packet. Return exactly one line:
APPROVE
or
REVISE: <one concrete, evidence-grounded correction>

Rules:
- An existing cart redemption is state, not available balance. Preserve it if
  it remains valid after other cart changes.
- Never cancel, replace, top up, or choose a new redemption merely because the
  user said "maximum", "all", or "make sure". A write requires the user to name
  the exact final points amount.
- Contrast the proposed final redemption with the observed existing redemption.
  If replacing it would lower the discount, drain extra balance, or reinterpret
  a named incremental amount as the final amount, require clarification rather
  than approving the write.
- Account points and a computed cap are facts, not authorization.

Use REVISE only for a definite violation; otherwise APPROVE. Do not invent a
policy or request unrelated optimization."""


def preserve_authorized_removal(
    conversation: list[dict[str, Any]], proposed: AgentTurnResponse
) -> AgentTurnResponse | None:
    """Keep an explicit removal while dropping a rejected loyalty mutation."""
    latest_user = next(
        (
            str(item.get("content", "")).lower()
            for item in reversed(conversation)
            if item.get("role") == "user" and "[task_done]" not in str(item.get("content", "")).lower()
        ),
        "",
    )
    explicitly_remove = any(
        phrase in latest_user for phrase in ("remove", "take out", "delete", "leave out")
    )
    names = [
        str(call.get("name", "")) if isinstance(call, dict) else str(call.name)
        for call in proposed.tool_calls
    ]
    if not explicitly_remove or "remove_from_cart" not in names:
        return None
    filtered = [
        call
        for call in proposed.tool_calls
        if (str(call.get("name", "")) if isinstance(call, dict) else str(call.name))
        not in LOYALTY_WRITES
    ]
    if len(filtered) == len(proposed.tool_calls):
        return None
    return AgentTurnResponse(
        text=(
            "I’m completing the explicitly requested cart removal first and leaving the "
            "existing loyalty redemption unchanged. I’ll use the resulting canonical cart "
            "state before considering any separate loyalty change; no points are being "
            "cancelled or re-redeemed in this step."
        ),
        tool_calls=filtered,
    )


class LoyaltyVerifiedPWMAgent(_Parent):
    """Pre-commit verification only when a loyalty mutation is proposed."""

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self._runtime_domain = getattr(runtime_context, "domain", None)

    @staticmethod
    def _name(call: Any) -> str:
        return str(call.get("name", "")) if isinstance(call, dict) else str(call.name)

    @staticmethod
    def _args(call: Any) -> dict[str, Any]:
        return dict(call.get("arguments") or {}) if isinstance(call, dict) else dict(call.arguments)

    @classmethod
    def _evidence(
        cls, conversation: list[dict[str, Any]], proposed: AgentTurnResponse
    ) -> dict[str, Any]:
        user_messages = [
            str(item.get("content", ""))[:1200]
            for item in conversation
            if item.get("role") == "user" and "[TASK_DONE]" not in str(item.get("content", ""))
        ][-4:]
        observations: list[dict[str, Any]] = []
        for item in conversation[-8:]:
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                if call.get("name") not in {
                    "get_cart",
                    "get_customer_account",
                    "get_policies",
                    "remove_from_cart",
                    "add_to_cart",
                    "update_cart_item",
                    "redeem_loyalty_points",
                    "cancel_loyalty_redemption",
                }:
                    continue
                result = json.dumps(call.get("result"), ensure_ascii=False, separators=(",", ":"))
                observations.append(
                    {
                        "name": str(call.get("name", "")),
                        "arguments": call.get("arguments") or {},
                        "result": result[:2000],
                    }
                )
        return {
            "user_messages": user_messages,
            "canonical_observations": observations[-10:],
            "proposed_assistant_text": proposed.text[:1600],
            "proposed_tool_batch": [
                {"name": cls._name(call), "arguments": cls._args(call)}
                for call in proposed.tool_calls
            ],
        }

    @staticmethod
    def _loyalty_state_note(conversation: list[dict[str, Any]]) -> str:
        """Derive a tiny StateAct-style ledger from the latest canonical cart."""
        latest_cart: dict[str, Any] | None = None
        for item in conversation:
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                if call.get("name") == "get_cart" and isinstance(call.get("result"), dict):
                    latest_cart = call["result"]
        if not latest_cart:
            return ""
        redeemed = latest_cart.get("loyalty_points_redeemed")
        discount = latest_cart.get("loyalty_discount")
        total = latest_cart.get("total")
        if not isinstance(discount, (int, float)) or discount <= 0:
            return ""
        if not isinstance(total, (int, float)):
            return ""
        pre_loyalty_total = total + discount
        cap = pre_loyalty_total * 0.5
        validity = "valid under the 50% cap" if discount <= cap else "above the 50% cap"
        return (
            "\n\nCanonical loyalty state ledger (arithmetic from the latest get_cart): "
            f"pre-loyalty amount = total {total:g} + loyalty_discount {discount:g} = "
            f"{pre_loyalty_total:g}; 50% cap = {cap:g}; existing redemption = "
            f"{redeemed} points / {discount:g} discount, which is {validity}. "
            "Do not recompute the cap from the already-discounted total alone, and do not "
            "treat an account balance as permission to replace this state."
        )

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        acting_prompt = system_prompt + self._loyalty_state_note(conversation)
        proposed = super().generate_next_turn(
            system_prompt=acting_prompt,
            conversation=conversation,
            tools=tools,
        )
        if self._runtime_domain != "shopping_assistant" or not any(
            self._name(call) in LOYALTY_WRITES for call in proposed.tool_calls
        ):
            return proposed

        verdict = self.client.generate(
            system_prompt=VERIFY_SYSTEM,
            conversation=[
                {
                    "role": "user",
                    "content": json.dumps(
                        self._evidence(conversation, proposed),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
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

        separated = preserve_authorized_removal(conversation, proposed)
        if separated is not None:
            return separated

        return _Parent.generate_next_turn(
            self,
            system_prompt=(
                f"{acting_prompt}\n\nA loyalty pre-commit verifier found this definite issue: "
                f"{correction}\nRegenerate only the next step. Preserve valid existing "
                "redemption state and do not execute the invalid loyalty write."
            ),
            conversation=conversation,
            tools=tools,
        )
