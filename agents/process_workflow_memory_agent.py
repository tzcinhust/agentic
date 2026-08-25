"""Process-conformant AWM agent for the STATE-Bench learning track."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agents.opencode_agent import OpenCodeAgent as _OpenCodeAgent
from state_bench.agents.base import AgentTurnResponse

WRITE_TOOL_NAMES = {
    "create_booking",
    "update_booking",
    "cancel_booking",
    "book_hotel",
    "cancel_hotel_reservation",
    "book_car_rental",
    "cancel_car_rental",
    "cancel_order",
    "process_return",
    "process_refund",
    "process_exchange",
    "process_warranty_claim",
    "add_to_cart",
    "remove_from_cart",
    "update_cart_item",
    "apply_promo",
    "remove_promo",
    "redeem_loyalty_points",
    "cancel_loyalty_redemption",
    "set_shipping_option",
}

POLICY_FIELDS = (
    "mandatory_disclosures",
    "confirmation_gates",
    "refresh_after_mutation",
    "forbidden_actions",
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
        "account_history": (
            "again",
            "already",
            "before",
            "bought",
            "first time",
            "looked at",
            "may have",
            "previous",
            "purchase history",
        ),
        "shipping": ("shipping", "delivery option", "expedited"),
        "compatibility": ("compatible", "compatibility", "work with"),
        "remove": ("remove", "delete", "take out"),
        "update_cart": (
            "additional",
            "already have",
            "change cart",
            "in my cart",
            "more of",
            "of those",
            "quantity",
            "update cart",
        ),
        "add_to_cart": ("add", "buy", "put in cart"),
        "search": ("find", "recommend", "looking for", "search"),
    },
}

CARD_INTENT_HINTS = {
    "shopping_assistant": {
        "account_history": (
            "already purchased",
            "bought before",
            "buy again",
            "duplicate",
            "ownership",
            "prior purchase",
            "purchase history",
        ),
        "update_cart": (
            "already in the cart",
            "current cart",
            "current quantity",
            "existing item",
            "existing line",
            "increase an item",
            "item already",
        ),
    }
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
        self.verifier_mode = os.environ.get("STATE_BENCH_VERIFIER_MODE", "off")
        if self.verifier_mode not in {"off", "policy", "full"}:
            raise ValueError("STATE_BENCH_VERIFIER_MODE must be off, policy, or full")
        self.verifier_max_revisions = int(
            os.environ.get("STATE_BENCH_VERIFIER_MAX_REVISIONS", "2")
        )
        self.verifier_min_confidence = float(
            os.environ.get("STATE_BENCH_VERIFIER_MIN_CONFIDENCE", "0.7")
        )
        artifact = json.loads(self.memory_path.read_text(encoding="utf-8"))
        domain = getattr(runtime_context, "domain", None)
        self._cards = [item for item in artifact.get("cards", []) if item.get("domain") == domain]
        self._document_frequency = Counter(
            token for item in self._cards for token in set(item.get("tokens", []))
        )
        self._avg_len = sum(len(item.get("tokens", [])) for item in self._cards) / max(len(self._cards), 1)
        self._card_ngrams = [_char_ngrams(item.get("search_text", "")) for item in self._cards]
        self._active_cards: list[dict[str, Any]] = []

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
        self._active_cards = self._rank_cards(query, top_k=self.retrieve_learnings_top_k)
        learnings = [self._card_text(item) for item in self._active_cards]
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
        query = str(args.get("query", ""))
        self._active_cards = self._rank_cards(query, top_k=self.retrieve_learnings_top_k)
        return [self._card_text(item) for item in self._active_cards]

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
        card_scope = " ".join(
            [
                str(item.get("applies_when", "")),
                " ".join(map(str, item.get("keywords", []))),
            ]
        ).lower()
        specificity_bonus = 0.0
        for intent in intent_matches:
            card_hints = CARD_INTENT_HINTS.get(domain, {}).get(
                intent, INTENT_HINTS.get(domain, {}).get(intent, ())
            )
            if any(phrase in card_scope for phrase in card_hints):
                specificity_bonus += {
                    "account_history": 10.0,
                    "update_cart": 8.0,
                }.get(intent, 3.0)
        return (
            lexical
            + 8.0 * character_similarity
            + 0.18 * support
            + 0.35 * conformance
            + 0.15 * quality
            + 0.6 * tool_overlap
            + 1.8 * intent_overlap
            + specificity_bonus
        )

    def _rank_cards(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
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
        return selected

    def _card_text(self, item: dict[str, Any]) -> str:
        text_key = {"hybrid": "text", "awm_only": "awm_text", "process_only": "process_text"}[self.mode]
        return str(item.get(text_key, item.get("text", "")))[:2200]

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        return [self._card_text(item) for item in self._rank_cards(query, top_k)]

    @staticmethod
    def _tool_calls(response: AgentTurnResponse) -> list[tuple[str, dict[str, Any]]]:
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
    def _conversation_trace(conversation: list[dict[str, Any]]) -> str:
        lines = []
        for item in conversation[-18:]:
            role = str(item.get("role", ""))
            content = item.get("content", "")
            if role == "assistant" and item.get("tool_calls"):
                calls = [
                    {
                        "name": call.get("name", ""),
                        "arguments": call.get("arguments", {}),
                    }
                    for call in item.get("tool_calls", [])
                ]
                lines.append(f"assistant tools: {json.dumps(calls, ensure_ascii=True)[:1600]}")
                if content:
                    lines.append(f"assistant text: {str(content)[:800]}")
            elif role == "tool":
                lines.append(f"tool results: {json.dumps(content, ensure_ascii=True, default=str)[:1800]}")
            elif role in {"user", "assistant"}:
                lines.append(f"{role}: {str(content)[:1200]}")
        return "\n".join(lines)

    def _policy_payload(self) -> list[dict[str, Any]]:
        payload = [
            {
                "workflow": "runtime:tool_grounding",
                "applies_when": "Any candidate makes factual or causal claims about tool-observed state or errors.",
                "support": "runtime invariant",
                "preconditions": [],
                "steps": [],
                "branches": [],
                "avoid": [
                    (
                        "When a tool reports an error or failure reason, repeat only the observed reason. "
                        "Do not add possible, likely, or usual causes that the tool did not report."
                    )
                ],
                "mandatory_disclosures": [],
                "confirmation_gates": [],
                "refresh_after_mutation": [],
                "forbidden_actions": [
                    "Unsupported explanations, causal guesses, and claims that contradict current tool results."
                ],
                "runtime_rules": [],
            }
        ]
        for card in self._active_cards:
            fields = {key: list(map(str, card.get(key, []))) for key in POLICY_FIELDS}
            runtime_rules = [
                rule for rule in card.get("runtime_rules", []) if isinstance(rule, dict)
            ]
            if any(fields.values()) or runtime_rules:
                payload.append(
                    {
                        "workflow": card.get("id", card.get("family", "")),
                        "applies_when": card.get("applies_when", card.get("search_text", ""))[:700],
                        "support": card.get("support", 0),
                        "preconditions": list(map(str, card.get("preconditions", []))),
                        "steps": list(map(str, card.get("steps", []))),
                        "branches": list(map(str, card.get("branches", []))),
                        "avoid": list(map(str, card.get("avoid", []))),
                        **fields,
                        "runtime_rules": runtime_rules,
                    }
                )
        return payload

    @staticmethod
    def _seen_tool_events(conversation: list[dict[str, Any]]) -> list[tuple[int, str, dict[str, Any]]]:
        events = []
        for index, item in enumerate(conversation):
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                events.append((index, str(call.get("name", "")), call.get("arguments", {})))
        return events

    def _deterministic_feedback(
        self,
        *,
        phase: str,
        response: AgentTurnResponse,
        conversation: list[dict[str, Any]],
    ) -> str | None:
        calls = self._tool_calls(response)
        events = self._seen_tool_events(conversation)
        last_user_index = max(
            (index for index, item in enumerate(conversation) if item.get("role") == "user"),
            default=-1,
        )
        recent_events = [event for event in events if event[0] > last_user_index]

        for name, arguments in calls:
            if name not in WRITE_TOOL_NAMES:
                continue
            signature = json.dumps([name, arguments], sort_keys=True, ensure_ascii=True)
            if any(
                json.dumps([old_name, old_arguments], sort_keys=True, ensure_ascii=True)
                == signature
                for _, old_name, old_arguments in recent_events
            ):
                return f"Do not repeat the already executed {name} call with identical arguments."

        if not self._active_cards:
            return None
        card = self._active_cards[0]
        if int(card.get("support", 0)) < 3 or float(card.get("mean_fitness", 0.0)) < 0.8:
            return None
        rules = [rule for rule in card.get("runtime_rules", []) if isinstance(rule, dict)]
        seen_names = [name for _, name, _ in recent_events]
        candidate_names = {name for name, _ in calls}
        for rule in rules:
            rule_phase = rule.get("phase")
            applies_now = rule_phase == phase or (
                phase == "pre_final" and rule_phase == "post_write"
            )
            if rule.get("enforcement") != "deterministic" or not applies_now:
                continue
            required = set(map(str, rule.get("required_tools", [])))
            triggers = set(map(str, rule.get("trigger_tools", [])))
            if phase == "pre_write" and triggers and not (candidate_names & triggers):
                continue
            if rule.get("kind") == "require_tool" and required - set(seen_names):
                missing = ", ".join(sorted(required - set(seen_names)))
                return str(rule.get("feedback") or f"Call the required read tools first: {missing}.")
            if phase == "pre_final" and rule.get("kind") == "refresh" and required:
                write_positions = [
                    index for index, name, _ in recent_events if name in WRITE_TOOL_NAMES
                ]
                if not write_positions:
                    continue
                after_write = {
                    name
                    for index, name, _ in recent_events
                    if index > max(write_positions)
                }
                if required - after_write:
                    missing = ", ".join(sorted(required - after_write))
                    return str(rule.get("feedback") or f"Refresh state before replying: {missing}.")
        return None

    def _semantic_verdict(
        self,
        *,
        phase: str,
        response: AgentTurnResponse,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        policy_payload = self._policy_payload()
        if not policy_payload:
            return True, ""
        candidate = {
            "text": response.text,
            "tool_calls": [
                {"name": name, "arguments": arguments}
                for name, arguments in self._tool_calls(response)
            ],
        }
        allowed_tools = [
            str(tool.get("name") or tool.get("function", {}).get("name", ""))
            for tool in tools
        ]
        prompt = f"""Evaluate one proposed step of a stateful tool-using agent.
The policies below were induced only from fixed training trajectories. Apply a
rule only when it clearly matches the current user request and observed state.
Preserve task utility: approve irrelevant, already-satisfied, or indeterminate
rules. Never infer hidden benchmark requirements.

Phase: {phase}
Allowed tools: {json.dumps(allowed_tools, ensure_ascii=True)}
Retrieved policy cards: {json.dumps(policy_payload, ensure_ascii=True)}

Observed conversation and tool trace:
{self._conversation_trace(conversation)}

Candidate step:
{json.dumps(candidate, ensure_ascii=True)}

For pre_write, check explicit authorization, required reads, stale state,
forbidden actions, and disclosures that must precede the write. For pre_final,
check only applicable disclosures, refresh obligations, and unfinished user
requirements. Do not require a write when the user has not approved one.

Return JSON only:
{{"decision":"allow|revise","confidence":0.0,"workflow_ids":["exact supplied workflow id"],"violations":["..."],"feedback":"one concrete correction"}}
Use revise only for a clear violation supported by the supplied policy and trace.
"""
        try:
            result = self.client.generate(
                system_prompt="You are a conservative runtime process verifier. Return JSON only.",
                conversation=[{"role": "user", "content": prompt}],
                tools=[],
            )
        except Exception:  # noqa: BLE001 - auxiliary verifier failures must not abort the benchmark
            return True, ""
        usage = getattr(result, "usage", None)
        self.add_token_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            category="other_llm",
        )
        match = re.search(r"\{.*\}", str(getattr(result, "text", "")), flags=re.S)
        if not match:
            return True, ""
        try:
            verdict = json.loads(match.group(0))
        except json.JSONDecodeError:
            return True, ""
        confidence = float(verdict.get("confidence", 0.0) or 0.0)
        if verdict.get("decision") != "revise" or confidence < self.verifier_min_confidence:
            return True, ""
        valid_workflows = {
            str(card.get("id", card.get("family", ""))) for card in self._active_cards
        }
        valid_workflows.add("runtime:tool_grounding")
        cited_workflows = set(map(str, verdict.get("workflow_ids", [])))
        if not cited_workflows or not cited_workflows.issubset(valid_workflows):
            return True, ""
        feedback = str(verdict.get("feedback", "")).strip()
        if not feedback:
            feedback = "; ".join(map(str, verdict.get("violations", [])))
        return False, feedback or "Revise the step to satisfy the applicable workflow policy."

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        response = super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
        if self.verifier_mode == "off" or not self._policy_payload():
            return response

        for revision in range(self.verifier_max_revisions + 1):
            calls = self._tool_calls(response)
            has_write = any(name in WRITE_TOOL_NAMES for name, _ in calls)
            if has_write:
                phase = "pre_write"
            elif not calls and self.verifier_mode == "full":
                phase = "pre_final"
            else:
                return response

            feedback = self._deterministic_feedback(
                phase=phase,
                response=response,
                conversation=conversation,
            )
            allowed = feedback is None
            if allowed:
                allowed, feedback = self._semantic_verdict(
                    phase=phase,
                    response=response,
                    conversation=conversation,
                    tools=tools,
                )
            if allowed:
                return response
            if revision >= self.verifier_max_revisions:
                if phase == "pre_write":
                    return AgentTurnResponse(
                        text=response.text
                        or "Before I make that change, I need to verify the required details or confirmation.",
                        tool_calls=[],
                    )
                return response

            correction = {
                "role": "system",
                "content": (
                    "The previous candidate was not executed. Runtime verification found: "
                    f"{feedback} Generate a corrected next step. Use current tool results only, "
                    "do not claim the rejected action happened, and ask the user when confirmation is missing."
                ),
            }
            response = super().generate_next_turn(
                system_prompt=system_prompt,
                conversation=[*conversation, correction],
                tools=tools,
            )
        return response
