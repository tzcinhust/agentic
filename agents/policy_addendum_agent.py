"""Pure PWM acting followed by a retrieved policy-obligation addendum.

Eager policy injection retrieves the right utterance duties but perturbs tool
execution; fetch-only review preserves execution but misses tasks where PWM
never calls ``get_policies``. This arm separates the layers:

1. act with byte-identical three-card PWM retrieval;
2. after a final draft exists, retrieve policy topics from the completed
   conversation (including observed write-tool names);
3. ask for only the shortest missing customer-facing addendum and concatenate
   it to the untouched draft.

The reviewer cannot delete correct draft content or mutate state. Returning
``NONE`` is a no-op.
"""

from __future__ import annotations

from typing import Any

from state_bench.agents.base import AgentTurnResponse

from agents.late_bound_policy_agent import LateBoundPolicyAgent as _LateParent
from agents.policy_obligation_agent import PolicyObligationAgent as _PolicyParent


ADDENDUM_INSTRUCTIONS = """You generate a policy addendum, not a replacement.
The existing draft will be preserved verbatim. Compare it with the verified
act-typed policy checklist below and the tool-grounded conversation. If every
applicable duty is already satisfied, return exactly NONE. Otherwise return only
the shortest customer-facing sentence(s) needed to add the missing disclosure,
derived figure, or refusal. Do not restate the draft, invent facts or actions,
mention this review, or assert a condition not verified by the tool results."""

ADDENDUM_REQUEST = (
    "Output only NONE or the minimal customer-facing addendum missing from the draft."
)


class PolicyAddendumAgent(_LateParent):
    """Keep PWM's action policy intact and append only missing policy speech acts."""

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        # Bypass the late-bound JIT override. The action-producing call is pure
        # PWM and sees all three workflow cards.
        draft = _PolicyParent.generate_next_turn(
            self,
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
        if draft.tool_calls or not draft.text.strip():
            return draft

        query = self._query_from_conversation(conversation)
        items = self._rank_topics(query)
        card = self._render(items)
        if not card:
            return draft

        review_conversation = [
            *conversation,
            {"role": "assistant", "content": draft.text},
            {"role": "user", "content": ADDENDUM_REQUEST},
        ]
        result = self.client.generate(
            system_prompt=f"{system_prompt}\n\n{ADDENDUM_INSTRUCTIONS}\n\n{card}",
            conversation=review_conversation,
            tools=[],
        )
        usage = result.usage
        self.add_token_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )
        if result.tool_calls:
            return draft
        addendum = result.text.strip()
        if not addendum or addendum.upper().strip("` .\n\t") == "NONE":
            return draft
        return AgentTurnResponse(text=f"{draft.text.rstrip()}\n\n{addendum}", tool_calls=[])
