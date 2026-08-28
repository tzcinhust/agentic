"""Current best eager-policy agent plus the narrow loyalty verifier."""

from __future__ import annotations

import json
from typing import Any

from state_bench.agents.base import AgentTurnResponse

from agents.loyalty_verified_pwm_agent import (
    LOYALTY_WRITES,
    VERIFY_SYSTEM,
    LoyaltyVerifiedPWMAgent as _LoyaltyLogic,
    preserve_authorized_removal,
)
from agents.policy_obligation_agent import PolicyObligationAgent as _Parent


class LoyaltyVerifiedPolicyAgent(_Parent):
    """PolicyObligationAgent with verification only for loyalty mutations."""

    _name = staticmethod(_LoyaltyLogic._name)
    _args = staticmethod(_LoyaltyLogic._args)
    _evidence = classmethod(_LoyaltyLogic._evidence.__func__)
    _loyalty_state_note = staticmethod(_LoyaltyLogic._loyalty_state_note)

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self._runtime_domain = getattr(runtime_context, "domain", None)

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        acting_prompt = system_prompt + self._loyalty_state_note(conversation)
        proposed = _Parent.generate_next_turn(
            self,
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
