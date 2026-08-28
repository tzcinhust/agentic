"""PWM with a TOOLVERIFIER-style pre-commit check for shopping writes.

The original PWM proposes the action.  A small tool-free verification pass then
contrasts the proposed target/arguments with alternatives already present in
the canonical tool results.  Only a concrete, evidence-backed REVISE verdict
causes one regeneration; APPROVE returns the original response unchanged.
"""

from __future__ import annotations

import json
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

VERIFY_SYSTEM = """You verify a proposed batch of shopping tool calls. Use only
the user messages and canonical tool results in the transcript. Contrast each
write target and argument with its closest plausible alternative. Check exact
product/cart-item identity, variant, quantity semantics, gift-wrap consent,
promo eligibility and relative savings, loyalty amount authorization, shipping
choice, and whether a required state read is missing. Do not invent policy.

Return exactly one line:
APPROVE
or
REVISE: <one concrete correction grounded in an observed fact>

Use REVISE only for a definite error. Missing optional optimization, style, or
an unsupported suspicion is APPROVE."""


class ContrastiveVerifiedPWMAgent(_Parent):
    """Regenerate a high-risk write once when contrastive verification fails."""

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self._runtime_domain = getattr(runtime_context, "domain", None)

    @staticmethod
    def _name(call: Any) -> str:
        return str(call.get("name", "")) if isinstance(call, dict) else str(call.name)

    @staticmethod
    def _args(call: Any) -> dict[str, Any]:
        return dict(call.get("arguments") or {}) if isinstance(call, dict) else dict(call.arguments)

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

        plan = [
            {"name": self._name(call), "arguments": self._args(call)}
            for call in proposed.tool_calls
        ]
        verdict = self.client.generate(
            system_prompt=VERIFY_SYSTEM,
            conversation=[
                *conversation,
                {
                    "role": "user",
                    "content": "Proposed tool batch:\n" + json.dumps(plan, ensure_ascii=False),
                },
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
            f"{system_prompt}\n\nA contrastive pre-commit verifier found this definite issue "
            f"in the proposed write batch: {correction}\nRegenerate only the next step. "
            "Use canonical tool facts, preserve valid reads, and do not write until the issue is fixed."
        )
        return _Parent.generate_next_turn(
            self,
            system_prompt=revised_prompt,
            conversation=conversation,
            tools=tools,
        )
