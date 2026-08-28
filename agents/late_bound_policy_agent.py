"""PWM with act-typed policy obligations injected after ``get_policies``.

The eager policy arm spends one of the three retrieval slots on a policy card
before every assistant turn.  Its two train replicates improve failed rubric
items but not conjunctive task pass@1: unrelated workflow/state misses offset
the obligation gains.

This arm keeps PWM retrieval byte-identical (all three slots remain workflow
cards).  When the harness has just executed ``get_policies``, the next tool
round receives a compact act-typed checklist for that exact topic.  The trigger
is an observed domain-tool call, not task text or rubric labels, so it is both
late-bound and domain-general.  The mined artifact remains read-only.
"""

from __future__ import annotations

from typing import Any

from agents.policy_obligation_agent import PolicyObligationAgent as _PolicyParent
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent as _WorkflowParent


JIT_HEADER = (
    "The policy tool just returned the rules below. Before your next write or "
    "final answer, use this act-typed checklist for the fetched topic. Apply only "
    "conditions verified by the current tool result."
)


class LateBoundPolicyAgent(_PolicyParent):
    """Preserve three PWM cards; structure policy text only after it is fetched."""

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        # Both PWM's push path and the public retrieve_learnings tool remain
        # exactly the parent workflow retriever. Policy guidance is late-bound
        # in generate_next_turn and therefore never consumes a top-k slot.
        return _WorkflowParent.retrieve_learnings(self, query, top_k=top_k)

    def _topic_item(self, requested: str) -> dict[str, Any] | None:
        requested = requested.strip().lower()
        if not requested:
            return None
        exact = next(
            (item for item in self._topics if str(item.get("topic", "")).lower() == requested),
            None,
        )
        if exact is not None:
            return exact

        # Tool vocabularies use short action nouns in a few domains
        # (``cancel``), while the mined topic is nominalised
        # (``cancellation``). Prefer a prefix match before a general substring
        # so travel ``cancel`` selects cancellation, not hotel_cancellation.
        prefixed = [
            item
            for item in self._topics
            if str(item.get("topic", "")).lower().startswith(requested)
        ]
        if prefixed:
            return max(prefixed, key=lambda item: int(item.get("trajectories", 0)))
        contained = [
            item
            for item in self._topics
            if requested in str(item.get("topic", "")).lower()
            or str(item.get("topic", "")).lower() in requested
        ]
        return max(contained, key=lambda item: int(item.get("trajectories", 0))) if contained else None

    @staticmethod
    def _just_fetched_topic(conversation: list[dict[str, Any]]) -> str:
        if not conversation or conversation[-1].get("role") != "tool":
            return ""
        content = conversation[-1].get("content")
        records = content if isinstance(content, list) else []
        for record in reversed(records):
            if record.get("name") != "get_policies":
                continue
            arguments = record.get("arguments")
            if isinstance(arguments, dict):
                return str(arguments.get("topic", ""))
        return ""

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ):
        topic = self._just_fetched_topic(conversation)
        item = self._topic_item(topic) if topic else None
        if item is not None:
            card = self._render([item])
            if card:
                # Do not insert a system-role message into ``conversation`` here:
                # the final two items are an assistant tool call and its tool
                # result, and separating that pair makes Chat Completions reject
                # the request. Extending the existing system prompt leaves the
                # canonical tool sequence untouched.
                system_prompt = f"{system_prompt}\n\n{JIT_HEADER}\n\n{card}"
        return super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
