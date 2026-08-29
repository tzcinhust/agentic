"""Process-conformant AWM agent for the STATE-Bench learning track."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agents.completion_memory import (
    COMMUNICATION_TYPES,
    CompletionMemory,
    has_valid_tool_evidence,
    latest_user_completion_types,
    static_completion_requirements,
)
from agents.opencode_agent import OpenCodeAgent as _OpenCodeAgent


COMPLETION_MODES = frozenset({"pwm_only", "generic", "static", "structured"})
SINGLE_CALL_CLOSURE_RULE = (
    "If you choose to call tools in this turn, ignore all closure requirements below. "
    "They must not affect tool selection, tool arguments, or whether another tool call is needed. "
    "Apply them only if you are otherwise ready to answer the user without tool calls."
)
GENERIC_COMPLETENESS_REMINDER = (
    "Before ending, cover the user's explicit request and material outcomes already established by "
    "authoritative tool evidence."
)


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
        artifact = json.loads(self.memory_path.read_text(encoding="utf-8"))
        domain = getattr(runtime_context, "domain", None)
        self._cards = [item for item in artifact.get("cards", []) if item.get("domain") == domain]
        self._document_frequency = Counter(
            token for item in self._cards for token in set(item.get("tokens", []))
        )
        self._avg_len = sum(len(item.get("tokens", [])) for item in self._cards) / max(len(self._cards), 1)
        self._card_ngrams = [_char_ngrams(item.get("search_text", "")) for item in self._cards]
        self.completion_mode = os.environ.get("STATE_BENCH_COMPLETION_MODE", "structured")
        if self.completion_mode not in COMPLETION_MODES:
            raise ValueError(
                "STATE_BENCH_COMPLETION_MODE must be pwm_only, generic, static, or structured"
            )
        self.completion_memory = (
            CompletionMemory() if self.completion_mode == "structured" else None
        )
        self._static_requirements: list[str] = []
        self._initial_completion_items: list[dict[str, Any]] | None = None
        self._generation_log: list[dict[str, Any]] = []

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
        self._update_completion_knowledge(learnings, conversation)
        if not learnings:
            return conversation
        memory_prompt = (
            "Process-conformant workflow memory follows. Treat it as procedural guidance, not current-task facts. "
            "Verify identifiers, state, prices, eligibility, and policy with current domain tools. "
            "Before any state-changing call, gather required facts and obtain explicit user approval when the "
            "workflow requires a preview or choice. A valid branch may require no state change.\n\n"
            + "\n\n---\n\n".join(learnings)
        )
        return self.inject_system_message(conversation, memory_prompt)

    def _update_completion_knowledge(
        self,
        learnings: list[str],
        conversation: list[dict[str, Any]],
    ) -> None:
        if self.completion_mode == "structured" and self.completion_memory is not None:
            self.completion_memory.ingest_workflows(learnings)
            self.completion_memory.ingest_user_messages(conversation)
            if self._initial_completion_items is None:
                self._initial_completion_items = self.completion_memory.snapshot()
        elif self.completion_mode == "static":
            seen = {item.lower() for item in self._static_requirements}
            for requirement in static_completion_requirements(learnings, conversation):
                if requirement.lower() not in seen:
                    seen.add(requirement.lower())
                    self._static_requirements.append(requirement)

    @staticmethod
    def _last_non_system_role(conversation: list[dict[str, Any]]) -> str:
        return next(
            (
                str(item.get("role", ""))
                for item in reversed(conversation)
                if item.get("role") != "system"
            ),
            "",
        )

    def _pending_prompt_data(self) -> tuple[list[str], list[dict[str, Any]]]:
        if self.completion_mode == "generic":
            return [GENERIC_COMPLETENESS_REMINDER], []
        if self.completion_mode == "static":
            return self._static_requirements[:8], []
        if self.completion_mode == "structured" and self.completion_memory is not None:
            items = self.completion_memory.prompt_items()
            return [f"[{item.type}] {item.description}" for item in items], [
                item.to_dict() for item in items
            ]
        return [], []

    def _closure_gate_reason(
        self,
        conversation: list[dict[str, Any]],
        requirements: list[str],
    ) -> str | None:
        if self.completion_mode == "pwm_only" or not requirements:
            return None
        if not has_valid_tool_evidence(conversation):
            return None
        role = self._last_non_system_role(conversation)
        if role == "tool":
            return "post_tool_with_valid_evidence"
        if role == "user" and (
            latest_user_completion_types(conversation) & COMMUNICATION_TYPES
        ):
            return "evidence_backed_communication_followup"
        return None

    def _closure_prompt(self, requirements: list[str]) -> str:
        if self.completion_mode == "structured":
            title = "Remaining task-closure requirements, if supported by current evidence:"
        elif self.completion_mode == "static":
            title = "Static completion text (untracked; no status or evidence is maintained):"
        else:
            title = "Generic completeness reminder:"
        lines = ["Single-call task-closure gate:", SINGLE_CALL_CLOSURE_RULE, title]
        lines.extend(f"- {item}" for item in requirements)
        lines.append(
            "Use only facts already supported by the conversation and tool results; do not invent facts or actions."
        )
        return "\n".join(lines)

    @staticmethod
    def _response_tool_calls(response: Any) -> list[dict[str, Any]]:
        calls = []
        for call in getattr(response, "tool_calls", []) or []:
            if isinstance(call, dict):
                calls.append(
                    {"name": str(call.get("name", "")), "arguments": call.get("arguments") or {}}
                )
            else:
                calls.append(
                    {
                        "name": str(getattr(call, "name", "")),
                        "arguments": getattr(call, "arguments", {}) or {},
                    }
                )
        return calls

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ):
        """Make exactly one model call, with an optional single-call closure gate."""

        if self.completion_memory is not None:
            self.completion_memory.sync_evidence(conversation)
        requirements, structured_items = self._pending_prompt_data()
        gate_reason = self._closure_gate_reason(conversation, requirements)
        closure_prompt = self._closure_prompt(requirements) if gate_reason else ""
        model_conversation = (
            self.inject_system_message(conversation, closure_prompt, before_last_user=False)
            if closure_prompt
            else conversation
        )

        # One and only one model call.  There is no verifier, rejection, or regeneration.
        response = super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=model_conversation,
            tools=tools,
        )
        response_calls = self._response_tool_calls(response)
        pending_items = (
            [item.to_dict() for item in self.completion_memory.pending()]
            if self.completion_memory is not None
            else []
        )
        self._generation_log.append(
            {
                "generation_index": len(self._generation_log),
                "model_calls_for_generation": 1,
                "closure_injected": bool(closure_prompt),
                "closure_gate_reason": gate_reason,
                "closure_prompt_sha256": (
                    hashlib.sha256(closure_prompt.encode("utf-8")).hexdigest()
                    if closure_prompt
                    else None
                ),
                "pending_items": [item["id"] for item in pending_items],
                "injected_items": [item["id"] for item in structured_items]
                if closure_prompt
                else [],
                "injected_requirements": requirements if closure_prompt else [],
                "output_type": "tool_call" if response_calls else "final_text",
                "tool_calls_after_closure": response_calls if closure_prompt and response_calls else [],
            }
        )
        return response

    def ingest_trajectory(self, trajectory: Any) -> None:
        final_items = self.completion_memory.snapshot() if self.completion_memory else []
        trajectory.metadata["completion_memory"] = {
            "version": "task_closure_memory_v1",
            "mode": self.completion_mode,
            "initial_items": self._initial_completion_items or [],
            "static_requirements": list(self._static_requirements),
            "final_items": final_items,
            "final_pending_items": [item["id"] for item in final_items if item["status"] == "pending"],
            "final_satisfied_items": [
                item["id"] for item in final_items if item["status"] == "satisfied"
            ],
            "final_invalidated_items": [
                item["id"] for item in final_items if item["status"] == "invalidated"
            ],
            "generations": list(self._generation_log),
            "summary": {
                "model_generations": len(self._generation_log),
                "model_calls_per_generation": [1 for _ in self._generation_log],
                "regenerations": 0,
                "closure_injections": sum(
                    int(item["closure_injected"]) for item in self._generation_log
                ),
                "closure_injected_tool_calls": sum(
                    int(item["closure_injected"] and item["output_type"] == "tool_call")
                    for item in self._generation_log
                ),
            },
        }

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
