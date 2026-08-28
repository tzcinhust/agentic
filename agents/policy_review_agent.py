"""PWM plus a policy-conditioned final-response self-review.

This is the non-invasive counterpart to eager policy injection. The first pass
keeps all three PWM workflow cards and executes tools normally. If the
conversation actually fetched policy text, a second tool-free model call checks
the final draft against the act-typed obligations for those exact topics. The
review may add a missing disclosure, derived figure, ordering explanation, or
refusal, but must preserve tool-grounded facts and the action already taken.

The pattern is a narrow Self-Refine/critic graft at PWM's measured weak layer:
shopping state-only rubric items almost always pass, while utterance obligations
fail. It therefore edits only the utterance and cannot mutate benchmark state.
"""

from __future__ import annotations

from typing import Any

from state_bench.agents.base import AgentTurnResponse

from agents.late_bound_policy_agent import LateBoundPolicyAgent as _LateParent
from agents.policy_obligation_agent import PolicyObligationAgent as _PolicyParent


REVIEW_INSTRUCTIONS = """Policy-conditioned final-response review follows.
Revise the draft only if an obligation below is missing. Preserve every
tool-grounded fact, identifier, price, choice, refusal, and action already taken.
Do not claim an unverified condition, do not invent a new action, do not call a
tool, and do not discuss this review. Return only the final customer-facing
response. Prefer the shortest revision that satisfies the applicable duties."""

REVIEW_REQUEST = (
    "Return the final customer-facing response. Keep the draft unchanged except "
    "for the minimum additions or corrections required by the verified policy checklist."
)


class PolicyReviewAgent(_LateParent):
    """Use PWM for acting, then revise the final utterance against fetched policy."""

    def _fetched_policy_items(self, conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        requested: list[str] = []
        for message in conversation:
            records: list[dict[str, Any]] = []
            if message.get("role") == "assistant":
                records = list(message.get("tool_calls") or [])
            elif message.get("role") == "tool" and isinstance(message.get("content"), list):
                records = list(message["content"])
            for record in records:
                if record.get("name") != "get_policies":
                    continue
                arguments = record.get("arguments")
                if isinstance(arguments, dict):
                    requested.append(str(arguments.get("topic", "")))

        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for topic in requested:
            item = self._topic_item(topic)
            if item is None:
                continue
            name = str(item.get("topic", ""))
            if name in seen:
                continue
            seen.add(name)
            selected.append(item)
        return selected

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        # Bypass LateBoundPolicyAgent.generate_next_turn: the acting pass should
        # be pure PWM, with no JIT prompt inserted before tool decisions.
        draft = _PolicyParent.generate_next_turn(
            self,
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
        if draft.tool_calls or not draft.text.strip():
            return draft

        items = self._fetched_policy_items(conversation)
        card = self._render(items)
        if not card:
            return draft

        review_conversation = [
            *conversation,
            {"role": "assistant", "content": draft.text},
            {"role": "user", "content": REVIEW_REQUEST},
        ]
        result = self.client.generate(
            system_prompt=f"{system_prompt}\n\n{REVIEW_INSTRUCTIONS}\n\n{card}",
            conversation=review_conversation,
            tools=[],
        )
        usage = result.usage
        self.add_token_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )
        # With no tool schemas the reviewer should return text only. If a relay
        # still emits a tool call or an empty response, retain the grounded draft
        # rather than turning review failure into task failure.
        if result.tool_calls or not result.text.strip():
            return draft
        return AgentTurnResponse(text=result.text, tool_calls=[])
