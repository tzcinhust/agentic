"""Controlled memory-subset interventions for the LatticeGuard pilot.

This agent deliberately does not implement a learned router.  It exposes the
complete subset lattice of the unchanged PWM top-3 candidates so train-only
paired rollouts can test whether card interactions are real before we fit any
selector.  ``STATE_BENCH_LATTICE_MASK`` is a three-bit integer in [0, 7]; bit
zero controls the original top-1 card, bit one top-2, and bit two top-3.
"""

from __future__ import annotations

import os
from typing import Any

from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent as _Parent


class LatticeSubsetPWMAgent(_Parent):
    """Return an exact subset of the baseline PWM's first three cards."""

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        raw_mask = os.environ.get("STATE_BENCH_LATTICE_MASK", "7").strip()
        try:
            mask = int(raw_mask, 0)
        except ValueError as exc:
            raise ValueError("STATE_BENCH_LATTICE_MASK must be an integer in [0, 7]") from exc
        if mask < 0 or mask > 7:
            raise ValueError("STATE_BENCH_LATTICE_MASK must be an integer in [0, 7]")
        self.lattice_mask = mask

    def _baseline_items(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not query.strip() or not self._cards:
            return []
        ranked = sorted(
            ((self._score(query, index, item), item) for index, item in enumerate(self._cards)),
            key=lambda pair: pair[0],
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        families: set[str] = set()
        tools: set[str] = set()
        for score, item in ranked:
            family = str(item.get("family", ""))
            if family in families:
                continue
            item_tools = set(item.get("observed_tools", []))
            diversity_penalty = 0.08 * len(item_tools & tools)
            adjusted = score - diversity_penalty
            if adjusted <= 0 and selected:
                continue
            selected.append(item)
            families.add(family)
            tools.update(item_tools)
            if len(selected) >= min(top_k, self.retrieve_learnings_top_k, 3):
                break
        return selected

    def candidate_card_ids(self, query: str, top_k: int = 3) -> list[str]:
        """Return stable IDs for audit logs without exposing them to the Actor."""

        return [str(item.get("id", "")) for item in self._baseline_items(query, top_k)]

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        selected = self._baseline_items(query, top_k)
        subset = [item for index, item in enumerate(selected) if self.lattice_mask & (1 << index)]
        text_key = {"hybrid": "text", "awm_only": "awm_text", "process_only": "process_text"}[
            self.mode
        ]
        return [str(item.get(text_key, item.get("text", "")))[:2200] for item in subset]

