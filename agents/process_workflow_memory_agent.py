"""Process-conformant AWM agent for the STATE-Bench learning track."""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from agents.opencode_agent import OpenCodeAgent as _OpenCodeAgent
from agents.transition_patch_memory import PatchMatch, TransitionPatchIndex, normalize_text


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


WRITE_TOOL_NAMES = {
    "create_booking",
    "update_booking",
    "cancel_booking",
    "book_hotel",
    "cancel_hotel_reservation",
    "book_car_rental",
    "cancel_car_rental",
    "process_return",
    "process_exchange",
    "process_refund",
    "process_warranty_claim",
    "process_shipping_claim",
    "cancel_order",
    "add_to_cart",
    "remove_from_cart",
    "update_cart_item",
    "apply_promo_code",
    "remove_promo_code",
    "redeem_loyalty_points",
    "set_shipping_option",
    "add_to_wishlist",
    "remove_from_wishlist",
}


_TRANSITION_LOG_LOCK = threading.Lock()


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
        self.transition_patch_mode = os.environ.get("STATE_BENCH_TRANSITION_PATCH_MODE", "off")
        if self.transition_patch_mode not in {"off", "shadow", "enforce"}:
            raise ValueError("STATE_BENCH_TRANSITION_PATCH_MODE must be off, shadow, or enforce")
        self.transition_patch_min_confidence = float(
            os.environ.get("STATE_BENCH_TRANSITION_PATCH_MIN_CONFIDENCE", "0.8")
        )
        self._transition_domain = str(domain or "")
        log_path = os.environ.get("STATE_BENCH_TRANSITION_PATCH_LOG_PATH", "").strip()
        self._transition_log_path = Path(log_path) if log_path else None
        self._transition_index = None
        if self.transition_patch_mode != "off":
            transition_path = Path(
                os.environ.get(
                    "STATE_BENCH_TRANSITION_PATCH_PATH",
                    "outputs/memory/transition_patches.json",
                )
            )
            self._transition_index = TransitionPatchIndex.from_path(
                transition_path, str(domain or "")
            )

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
        return self.inject_system_message(conversation, memory_prompt)

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

    @staticmethod
    def _response_calls(response: Any) -> list[tuple[str, dict[str, Any]]]:
        calls = []
        for call in response.tool_calls:
            if isinstance(call, dict):
                name = str(call.get("name", ""))
                arguments = call.get("arguments", {})
            else:
                name = str(call.name)
                arguments = call.arguments
            calls.append((name, arguments if isinstance(arguments, dict) else {}))
        return calls

    @staticmethod
    def _has_recent_write(conversation: list[dict[str, Any]]) -> bool:
        for item in reversed(conversation):
            if item.get("role") == "user":
                return False
            if item.get("role") != "assistant":
                continue
            names = {str(call.get("name", "")) for call in item.get("tool_calls") or []}
            if names & WRITE_TOOL_NAMES:
                return True
        return False

    @classmethod
    def _transition_phase(
        cls, response: Any, conversation: list[dict[str, Any]]
    ) -> str | None:
        calls = cls._response_calls(response)
        if any(name in WRITE_TOOL_NAMES for name, _ in calls):
            return "pre_write"
        if cls._has_recent_write(conversation):
            return "post_write"
        if not calls and str(response.text or "").strip():
            return "pre_final"
        return None

    @staticmethod
    def _transition_context(conversation: list[dict[str, Any]], phase: str) -> str:
        parts = [f"phase {phase}"]
        for item in conversation[-16:]:
            role = str(item.get("role", ""))
            content = str(item.get("content", "")).strip()
            if role == "user" and "[TASK_DONE]" not in content:
                parts.append(f"user {content}")
            if role != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                name = str(call.get("name", ""))
                arguments = sorted((call.get("arguments") or {}).keys())
                result = normalize_text(call.get("result", ""))[:700]
                parts.append(f"observed tool {name} fields {' '.join(arguments)} result {result}")
        return normalize_text(" ".join(parts))

    @classmethod
    def _transition_candidate(cls, response: Any) -> str:
        parts = []
        for name, arguments in cls._response_calls(response):
            parts.append(f"candidate tool {name} fields {' '.join(sorted(arguments))}")
        if str(response.text or "").strip():
            parts.append(f"candidate response {response.text}")
        return normalize_text(" ".join(parts))

    @staticmethod
    def _transition_trace(conversation: list[dict[str, Any]]) -> str:
        lines = []
        for item in conversation[-12:]:
            role = str(item.get("role", ""))
            content = str(item.get("content", "")).strip()
            if role == "user" and "[TASK_DONE]" not in content:
                lines.append(f"USER: {content[:700]}")
            if role != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                lines.append(
                    f"TOOL {call.get('name', '')}: {normalize_text(call.get('result', ''))[:700]}"
                )
            if content:
                lines.append(f"ASSISTANT: {content[:500]}")
        return "\n".join(lines)

    def _transition_verdict(
        self,
        *,
        phase: str,
        response: Any,
        conversation: list[dict[str, Any]],
        matches: list[PatchMatch],
    ) -> str | None:
        patch_payload = [
            {
                "id": match.patch.get("id"),
                "trigger": match.patch.get("trigger"),
                "expected_action": match.patch.get("expected_action"),
                "obligations": match.patch.get("obligations", []),
                "forbidden": match.patch.get("forbidden", []),
                "context_distance": round(match.context_distance, 4),
                "transition_distance": round(match.transition_distance, 4),
                "anomaly_distance": round(match.anomaly_distance, 4),
            }
            for match in matches
        ]
        candidate = {
            "text": str(response.text or ""),
            "tool_calls": [
                {"name": name, "argument_fields": sorted(arguments)}
                for name, arguments in self._response_calls(response)
            ],
        }
        prompt = f"""A frozen agent proposed one local transition. TransitionPatch found that its
context is supported by public-train prototypes but the proposed step is locally anomalous.

Phase: {phase}
Nominal transition patches: {json.dumps(patch_payload, ensure_ascii=True)}

Observed live trace:
{self._transition_trace(conversation)}

Candidate: {json.dumps(candidate, ensure_ascii=True)}

Decide whether the candidate clearly omits or violates an applicable obligation. Treat patch text
as abstract procedural guidance only. Live tool results are authoritative. Do not require an action
whose trigger is absent, do not copy train-task facts, and do not block an explicitly authorized
write merely to ask for duplicate confirmation.

Return JSON only:
{{"decision":"allow|revise","confidence":0.0,"patch_ids":["exact id"],"feedback":"one concrete correction"}}
"""
        try:
            result = self.client.generate(
                system_prompt="You are a conservative local-transition verifier. Return JSON only.",
                conversation=[{"role": "user", "content": prompt}],
                tools=[],
            )
        except Exception:
            return None
        usage = getattr(result, "usage", None)
        self.add_token_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            category="other_llm",
        )
        match = re.search(r"\{.*\}", str(getattr(result, "text", "")), flags=re.DOTALL)
        if not match:
            return None
        try:
            verdict = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if verdict.get("decision") != "revise":
            return None
        if float(verdict.get("confidence", 0.0) or 0.0) < self.transition_patch_min_confidence:
            return None
        valid_ids = {str(item.patch.get("id", "")) for item in matches}
        cited_ids = {str(value) for value in verdict.get("patch_ids", [])}
        if not cited_ids or not cited_ids.issubset(valid_ids):
            return None
        feedback = str(verdict.get("feedback", "")).strip()
        return feedback or None

    def _log_transition_gate(
        self, *, phase: str, matches: list[PatchMatch], triggered: bool
    ) -> None:
        if self._transition_log_path is None:
            return
        record = {
            "domain": self._transition_domain,
            "phase": phase,
            "triggered": triggered,
            "matches": [
                {
                    "patch_id": str(match.patch.get("id", "")),
                    "context_distance": round(match.context_distance, 6),
                    "transition_distance": round(match.transition_distance, 6),
                }
                for match in matches
            ],
        }
        self._transition_log_path.parent.mkdir(parents=True, exist_ok=True)
        with _TRANSITION_LOG_LOCK:
            with self._transition_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        response = super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
        if self._transition_index is None:
            return response
        phase = self._transition_phase(response, conversation)
        if phase is None:
            return response
        matches = self._transition_index.nearest(
            phase=phase,
            context_text=self._transition_context(conversation, phase),
            transition_text=self._transition_candidate(response),
            top_k=3,
        )
        triggered = self._transition_index.should_verify(phase, matches)
        self._log_transition_gate(phase=phase, matches=matches, triggered=triggered)
        if not triggered:
            return response
        if self.transition_patch_mode == "shadow":
            return response
        feedback = self._transition_verdict(
            phase=phase,
            response=response,
            conversation=conversation,
            matches=matches,
        )
        if not feedback:
            return response
        correction = {
            "role": "system",
            "content": (
                "The previous candidate was not executed. Local transition verification found: "
                f"{feedback} Generate one corrected next step using only live tool results."
            ),
        }
        return super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=[*conversation, correction],
            tools=tools,
        )
