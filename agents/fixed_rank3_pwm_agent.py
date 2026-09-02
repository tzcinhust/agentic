"""Shopping-only fixed rank-3 PWM candidate.

The candidate preserves the original PWM ranking and card text.  It changes
only the Shopping injection set from the original top three cards to the card
at position three (lattice mask 4).  Other domains execute the parent method
directly.
"""

from __future__ import annotations

from typing import Any

from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent as _Parent


class FixedRank3PWMAgent(_Parent):
    """Inject only baseline rank 3 for Shopping; preserve PWM elsewhere."""

    def __init__(
        self,
        client,
        system_prompt,
        tools,
        tool_handlers,
        runtime_context=None,
        **kwargs,
    ):
        super().__init__(
            client,
            system_prompt,
            tools,
            tool_handlers,
            runtime_context,
            **kwargs,
        )
        self._domain = str(getattr(runtime_context, "domain", ""))

    def _baseline_items(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not query.strip() or not self._cards:
            return []
        ranked = sorted(
            (
                (self._score(query, index, item), item)
                for index, item in enumerate(self._cards)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        families: set[str] = set()
        observed_tools: set[str] = set()
        for score, item in ranked:
            family = str(item.get("family", ""))
            if family in families:
                continue
            item_tools = set(item.get("observed_tools", []))
            adjusted = score - 0.08 * len(item_tools & observed_tools)
            if adjusted <= 0 and selected:
                continue
            selected.append(item)
            families.add(family)
            observed_tools.update(item_tools)
            if len(selected) >= min(top_k, self.retrieve_learnings_top_k, 3):
                break
        return selected

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        if self._domain != "shopping_assistant":
            return super().retrieve_learnings(query, top_k)
        selected = self._baseline_items(query, top_k)
        if len(selected) < 3:
            return []
        text_key = {
            "hybrid": "text",
            "awm_only": "awm_text",
            "process_only": "process_text",
        }[self.mode]
        item = selected[2]
        return [str(item.get(text_key, item.get("text", "")))[:2200]]
