"""Post-tool policy guidance plus a non-destructive final addendum verifier.

The acting pass is PostToolPolicyAgent: full PWM workflow retrieval, mixed-batch
write deferral, and context-pruned policy guidance after tool results. A final
tool-free verifier then emits only policy facts still missing from the draft;
the wrapper concatenates them, so verified draft content cannot be deleted.
"""

from __future__ import annotations

from typing import Any

from state_bench.agents.base import AgentTurnResponse

from agents.post_tool_policy_agent import PostToolPolicyAgent as _PostToolParent


VERIFIER_SYSTEM = """Use only the tool-grounded conversation and the verified
policy checklist. Never invent facts, choices, or actions. The original draft
will be preserved verbatim, so generate an addendum rather than a rewrite."""

VERIFIER_REQUEST = """Output only the shortest customer-facing addendum needed
to satisfy every applicable checklist duty missing from the draft. For an
applicable numeric or calculation duty, explicitly state the matched branch,
rate/cap/fee, calculation base, arithmetic, and computed result whenever the
inputs are present. Include any required disclosure or refusal. Do not repeat
unrelated draft content. Return NONE only if no applicable duty is missing."""


class VerifiedPostToolPolicyAgent(_PostToolParent):
    """Preserve PWM actions and draft text; append verifier-found obligations."""

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        draft = super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
        if draft.tool_calls or not draft.text.strip():
            return draft

        query = self._query_from_conversation(conversation)
        items = self._contextual_items(self._rank_topics(query), conversation)
        card = self._render(items)
        if not card:
            return draft

        result = self.client.generate(
            system_prompt=f"{system_prompt}\n\n{VERIFIER_SYSTEM}\n\n{card}",
            conversation=[
                *conversation,
                {"role": "assistant", "content": draft.text},
                {"role": "user", "content": VERIFIER_REQUEST},
            ],
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
