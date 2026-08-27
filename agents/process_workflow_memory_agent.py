"""Process-conformant AWM agent for the STATE-Bench learning track."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agents.opencode_agent import OpenCodeAgent as _OpenCodeAgent
from agents.selective_obligation_guard import (
    build_obligation_prompt,
    compact_audit_record,
    guard_feedback,
)


LOGGER = logging.getLogger(__name__)


INTENT_HINTS = {
    "travel": {
        "cancel": ("cancel",),
        "hotel": ("hotel",),
        "car_rental": ("car rental", "rental car"),
        "status": ("status", "delay", "gate", "terminal"),
        "baggage": ("baggage", "bag", "luggage"),
        "seat": ("seat", "window", "aisle"),
        "ancillary": ("meal", "wifi", "legroom", "insurance"),
        "change": ("change", "update", "modify", "move", "switch"),
        "book": ("book a", "reserve a", "new flight"),
    },
    "customer_support": {
        "price_match": ("price match", "price drop", "cheaper"),
        "warranty": ("warranty", "repair"),
        "exchange": ("exchange", "replacement", "replace"),
        "shipping_claim": ("missing", "lost", "damaged", "wrong item", "late", "delivery"),
        "return": ("return", "send back"),
        "cancel": ("cancel",),
        "refund": ("refund", "money back"),
    },
    "shopping_assistant": {
        "promo": ("promo", "coupon", "discount"),
        "loyalty": ("loyalty", "points"),
        "shipping": ("shipping", "delivery option", "expedited"),
        "compatibility": ("compatible", "compatibility", "work with"),
        "remove": ("remove", "delete", "take out"),
        "update_cart": ("quantity", "change cart", "update cart"),
        "add_to_cart": ("add", "buy", "put in cart"),
        "search": ("find", "recommend", "looking for", "search"),
    },
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    compact = re.sub(r"\s+", " ", text.lower()).strip()
    return {compact[index : index + n] for index in range(max(0, len(compact) - n + 1))}


class ProcessWorkflowMemoryAgent(_OpenCodeAgent):
    """Retrieve process-grounded workflow cards without test-task oracle data."""

    memory_path = Path(
        os.environ.get("STATE_BENCH_MEMORY_PATH", "outputs/memory/process_workflows.json")
    )

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self.retrieve_learnings_top_k = int(kwargs.get("retrieve_learnings_top_k", 3))
        self.mode = os.environ.get("STATE_BENCH_MEMORY_MODE", "hybrid")
        if self.mode not in {"hybrid", "awm_only", "process_only"}:
            raise ValueError("STATE_BENCH_MEMORY_MODE must be hybrid, awm_only, or process_only")
        self.obligation_mode = os.environ.get("STATE_BENCH_OBLIGATION_MODE", "on")
        if self.obligation_mode not in {"off", "on"}:
            raise ValueError("STATE_BENCH_OBLIGATION_MODE must be off or on")
        self.guard_mode = os.environ.get("STATE_BENCH_SELECTIVE_GUARD_MODE", "enforce")
        if self.guard_mode not in {"off", "monitor", "enforce"}:
            raise ValueError(
                "STATE_BENCH_SELECTIVE_GUARD_MODE must be off, monitor, or enforce"
            )
        self.guard_audit: list[dict[str, Any]] = []
        artifact = json.loads(self.memory_path.read_text(encoding="utf-8"))
        domain = getattr(runtime_context, "domain", None)
        self._cards = [item for item in artifact.get("cards", []) if item.get("domain") == domain]
        self._document_frequency = Counter(
            token for item in self._cards for token in set(item.get("tokens", []))
        )
        self._avg_len = sum(len(item.get("tokens", [])) for item in self._cards) / max(len(self._cards), 1)
        self._card_ngrams = [_char_ngrams(item.get("search_text", "")) for item in self._cards]

    def _query_from_conversation(self, conversation: list[Any]) -> str:
        user_text = " ".join(
            str(item.get("content", ""))
            for item in conversation
            if item.get("role") == "user" and "[TASK_DONE]" not in str(item.get("content", ""))
        )
        observed_tools = [
            str(call.get("name", ""))
            for item in conversation
            if item.get("role") == "assistant"
            for call in item.get("tool_calls") or []
        ]
        return f"{user_text} {' '.join(observed_tools)}".strip()

    def prepare_conversation(self, conversation: list[Any]) -> list[Any]:
        query = self._query_from_conversation(conversation)
        learnings = self.retrieve_learnings(query, top_k=self.retrieve_learnings_top_k)
        if not learnings:
            return conversation
        memory_prompt = (
            "Process-conformant workflow memory follows. Treat it as procedural guidance, not current-task facts. "
            "Verify identifiers, state, prices, eligibility, and policy with current domain tools. "
            "Before any state-changing call, gather required facts and obtain explicit user approval when the "
            "workflow requires a preview or choice. A valid branch may require no state change.\n\n"
            + "\n\n---\n\n".join(learnings)
        )
        if self.obligation_mode == "on":
            obligation_prompt = build_obligation_prompt(learnings, conversation)
            if obligation_prompt:
                memory_prompt = f"{memory_prompt}\n\n---\n\n{obligation_prompt}"
        return self.inject_system_message(conversation, memory_prompt)

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ):
        """Preserve PWM generation and revise at most once for a precise violation."""

        response = super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
        if self.guard_mode == "off":
            return response
        feedback = guard_feedback(response, conversation)
        if not feedback:
            return response
        record = {"feedback": feedback, "corrected": False}
        self.guard_audit.append(record)
        if self.guard_mode == "monitor":
            LOGGER.warning(compact_audit_record(feedback, corrected=False))
            return response

        correction = {
            "role": "system",
            "content": (
                "The previous candidate was not executed. A high-confidence mechanical check found: "
                f"{feedback} Produce the minimal corrective next step while preserving every other "
                "correct part of the plan and every explicit user constraint."
            ),
        }
        revised = super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=[*conversation, correction],
            tools=tools,
        )
        if guard_feedback(revised, conversation):
            # A failed correction must not replace the archived PWM candidate with
            # a generic dead end or start an unbounded repair loop.
            LOGGER.warning(compact_audit_record(feedback, corrected=False))
            return response
        record["corrected"] = True
        LOGGER.warning(compact_audit_record(feedback, corrected=True))
        return revised

    def memory_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "retrieve_learnings",
                "description": (
                    "Retrieve up to three process-conformant workflows learned from fixed training trajectories. "
                    "Use when the request changes state, has policy branches, or combines multiple operations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]

    def memory_tool_handlers(self) -> dict[str, Any]:
        return {"retrieve_learnings": self.handle_retrieve_learnings}

    def handle_retrieve_learnings(self, args: dict[str, Any]) -> list[str]:
        return self.retrieve_learnings(str(args.get("query", "")), top_k=self.retrieve_learnings_top_k)

    def _score(self, query: str, index: int, item: dict[str, Any]) -> float:
        query_counts = Counter(_tokens(query))
        document_counts = Counter(item.get("tokens", []))
        document_length = sum(document_counts.values())
        total_documents = len(self._cards)
        lexical = 0.0
        for token, query_frequency in query_counts.items():
            frequency = document_counts.get(token, 0)
            if not frequency:
                continue
            document_frequency = self._document_frequency.get(token, 0)
            inverse_frequency = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = frequency + 1.4 * (0.25 + 0.75 * document_length / max(self._avg_len, 1))
            lexical += inverse_frequency * frequency * 2.4 / denominator * min(query_frequency, 2)

        query_ngrams = _char_ngrams(query)
        card_ngrams = self._card_ngrams[index]
        character_similarity = len(query_ngrams & card_ngrams) / max(len(query_ngrams), 1)
        support = math.log1p(max(0, int(item.get("support", 0))))
        conformance = float(item.get("mean_fitness", 0.0))
        quality = float(item.get("quality", 0.0))
        observed_tools = set(re.findall(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b", query.lower()))
        tool_overlap = len(observed_tools & set(item.get("observed_tools", [])))
        domain = item.get("domain", "")
        family_parts = set(str(item.get("family", "")).split("+"))
        lowered_query = query.lower()
        intent_matches = {
            intent
            for intent, phrases in INTENT_HINTS.get(domain, {}).items()
            if any(phrase in lowered_query for phrase in phrases)
        }
        intent_overlap = len(intent_matches & family_parts)
        return (
            lexical
            + 8.0 * character_similarity
            + 0.18 * support
            + 0.35 * conformance
            + 0.15 * quality
            + 0.6 * tool_overlap
            + 1.8 * intent_overlap
        )

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        if not query.strip() or not self._cards:
            return []
        ranked = sorted(
            ((self._score(query, index, item), item) for index, item in enumerate(self._cards)),
            key=lambda pair: pair[0],
            reverse=True,
        )
        selected = []
        families = set()
        tools = set()
        for score, item in ranked:
            family = item.get("family", "")
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
            if len(selected) >= min(top_k, self.retrieve_learnings_top_k):
                break
        text_key = {"hybrid": "text", "awm_only": "awm_text", "process_only": "process_text"}[self.mode]
        return [str(item.get(text_key, item.get("text", "")))[:2200] for item in selected]
