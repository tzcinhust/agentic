"""Process-conformant AWM agent for the STATE-Bench learning track."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from state_bench.agents.base import AgentTurnResponse

from agents.opencode_agent import OpenCodeAgent as _OpenCodeAgent
from agents.transition_aware_memory import TransitionAwareMemory

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


@dataclass(frozen=True)
class ObservedToolEvent:
    sequence: int
    message_index: int
    name: str
    arguments: dict[str, Any]
    result: Any


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
        self.tapm_mode = os.environ.get("STATE_BENCH_TAPM_MODE", "enforce")
        if self.tapm_mode not in {"off", "monitor", "enforce"}:
            raise ValueError("STATE_BENCH_TAPM_MODE must be off, monitor, or enforce")
        self.tapm_max_revisions = int(
            os.environ.get("STATE_BENCH_TAPM_MAX_REVISIONS", "1")
        )
        memory_path = Path(
            os.environ.get("STATE_BENCH_MEMORY_PATH", str(self.memory_path))
        )
        artifact = json.loads(memory_path.read_text(encoding="utf-8"))
        domain = getattr(runtime_context, "domain", None)
        self.transition_memory = TransitionAwareMemory(
            domain=domain,
            learned=os.environ.get("STATE_BENCH_TAPM_LEARNED", "on") == "on",
            entity_aware=os.environ.get("STATE_BENCH_TAPM_ENTITY", "on") == "on",
            value_aware=os.environ.get("STATE_BENCH_TAPM_VALUE", "on") == "on",
        )
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
            "The live tool schemas are authoritative for argument names; never copy an argument signature from "
            "workflow prose when it conflicts with the provided schema. An explicit imperative request or an "
            "affirmative answer to a confirmation question already counts as approval; do not ask for redundant "
            "confirmation unless a material choice remains ambiguous. "
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
        def compact(values: Any, limit: int = 5) -> list[str]:
            return [str(value)[:360] for value in list(values or [])[:limit]]

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
                        "Do not add possible, likely, or usual causes that the tool did not report. "
                        "Do not enumerate hypothetical causes even inside an uncertainty disclaimer such as "
                        "'I cannot tell whether X or Y'; say only that no further reason is available."
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
            fields = {key: compact(card.get(key, [])) for key in POLICY_FIELDS}
            runtime_rules = [
                rule for rule in card.get("runtime_rules", []) if isinstance(rule, dict)
            ][:6]
            if any(fields.values()) or runtime_rules:
                payload.append(
                    {
                        "workflow": card.get("id", card.get("family", "")),
                        "applies_when": card.get("applies_when", card.get("search_text", ""))[:700],
                        "support": card.get("support", 0),
                        "preconditions": compact(card.get("preconditions", [])),
                        "steps": compact(card.get("steps", []), limit=6),
                        "branches": compact(card.get("branches", [])),
                        "avoid": compact(card.get("avoid", [])),
                        **fields,
                        "runtime_rules": runtime_rules,
                    }
                )
        return payload

    @staticmethod
    def _seen_tool_events(
        conversation: list[dict[str, Any]],
    ) -> list[ObservedToolEvent]:
        events: list[ObservedToolEvent] = []
        sequence = 0
        for message_index, item in enumerate(conversation):
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                events.append(
                    ObservedToolEvent(
                        sequence=sequence,
                        message_index=message_index,
                        name=str(call.get("name", "")),
                        arguments=call.get("arguments") or {},
                        result=call.get("result"),
                    )
                )
                sequence += 1
        return events

    def _has_required_evidence(
        self,
        *,
        tool_name: str,
        events: list[ObservedToolEvent],
        candidate_arguments: dict[str, Any],
    ) -> bool:
        return any(
            event.name == tool_name
            and self.transition_memory.observation_succeeded(event.result)
            and self.transition_memory.scope_matches(
                event.arguments, event.result, candidate_arguments
            )
            for event in events
        )

    @staticmethod
    def _unsupported_error_speculation(
        response: AgentTurnResponse, conversation: list[dict[str, Any]]
    ) -> bool:
        failed_entities: set[str] = set()
        has_failure = False
        for item in conversation:
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                result_text = json.dumps(
                    call.get("result"), ensure_ascii=True, default=str
                ).lower()
                if not any(
                    marker in result_text
                    for marker in ('"valid": false', '"error"', "failed", "not found")
                ):
                    continue
                has_failure = True
                for value in (call.get("arguments") or {}).values():
                    if isinstance(value, str) and len(value.strip()) >= 3:
                        failed_entities.add(value.strip().lower())
        if not has_failure or not response.text:
            return False

        speculation = re.compile(
            r"\b(could|likely|like|maybe|might|perhaps|possibly|usually|whether|versus|vs)\b"
            r"|\b(such as|for example|e\.g\.)\b",
            flags=re.IGNORECASE,
        )
        failure_terms = ("error", "fail", "not found", "reason", "validator")
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", response.text.lower()):
            if not speculation.search(sentence):
                continue
            if any(entity in sentence for entity in failed_entities) or any(
                term in sentence for term in failure_terms
            ):
                return True
        return False

    @staticmethod
    def _has_explicit_write_authorization(conversation: list[dict[str, Any]]) -> bool:
        latest_user = next(
            (
                str(item.get("content", "")).lower()
                for item in reversed(conversation)
                if item.get("role") == "user" and "[task_done]" not in str(item.get("content", "")).lower()
            ),
            "",
        )
        if not latest_user or re.search(
            r"\b(do not|don't|dont|not yet|wait|hold on|stop|just asking)\b", latest_user
        ):
            return False
        if re.search(r"\b(yes|confirm|confirmed|go ahead|proceed|do it|please try again)\b", latest_user):
            return True
        return bool(
            re.search(
                r"\b(please\s+)?(add|apply|book|cancel|change|create|delete|increase|"
                r"move|purchase|redeem|reduce|remove|replace|return|set|update)\b",
                latest_user,
            )
        )

    @staticmethod
    def _schema_feedback(
        response: AgentTurnResponse, tools: list[dict[str, Any]]
    ) -> str | None:
        schemas: dict[str, dict[str, Any]] = {}
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = str(tool.get("name") or function.get("name", ""))
            parameters = tool.get("parameters") or function.get("parameters") or {}
            if name:
                schemas[name] = parameters

        for name, arguments in ProcessWorkflowMemoryAgent._tool_calls(response):
            schema = schemas.get(name)
            if not schema:
                continue
            properties = set((schema.get("properties") or {}).keys())
            required = set(map(str, schema.get("required") or []))
            supplied = set(arguments)
            missing = sorted(required - supplied)
            unexpected = sorted(supplied - properties) if properties else []
            if missing or (unexpected and schema.get("additionalProperties") is False):
                parts = []
                if missing:
                    parts.append(f"missing required arguments: {', '.join(missing)}")
                if unexpected and schema.get("additionalProperties") is False:
                    parts.append(f"unsupported arguments: {', '.join(unexpected)}")
                return (
                    f"The proposed {name} call does not match the live tool schema ({'; '.join(parts)}). "
                    "Regenerate the call using only the authoritative live schema."
                )
        return None

    @staticmethod
    def _compatibility_evidence_feedback(
        response: AgentTurnResponse,
        conversation: list[dict[str, Any]],
        phase: str,
    ) -> str | None:
        if phase != "pre_final":
            return None
        product_compatibility: dict[str, list[str]] = {}
        unknown_checks: list[str] = []
        for item in conversation:
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                name = str(call.get("name", ""))
                arguments = call.get("arguments") or {}
                product_id = str(arguments.get("product_id", ""))
                if name == "get_product_details" and product_id:
                    result = call.get("result") or {}
                    compatible_with = result.get("compatible_with", []) if isinstance(result, dict) else []
                    product_compatibility[product_id] = [str(value) for value in compatible_with]
                if name != "check_compatibility" or not product_id:
                    continue
                result = call.get("result")
                result_text = json.dumps(result, ensure_ascii=True, default=str).lower()
                if "unknown device" in result_text or "not recognized" in result_text:
                    unknown_checks.append(product_id)
        missing = [product_id for product_id in unknown_checks if product_id not in product_compatibility]
        if not missing:
            candidate = (response.text or "").lower()
            for product_id in unknown_checks:
                supported = product_compatibility.get(product_id, [])
                if not supported:
                    continue
                cites_product_list = (
                    ("compatible_with" in candidate or "compatible with" in candidate)
                    and ("not listed" in candidate or "does not include" in candidate)
                )
                if not cites_product_list:
                    return (
                        f"Do not base the conclusion only on the unknown-device checker error. Use the "
                        f"get_product_details evidence for {product_id}: compatible_with is "
                        f"{json.dumps(supported, ensure_ascii=True)}. Explicitly state that the requested "
                        "device is not listed there and therefore the actual product is incompatible based "
                        "on the product catalog."
                    )
            return None
        return (
            "The compatibility checker returned an unknown/unrecognized device rather than product-level "
            f"compatibility evidence. Call get_product_details for {missing[-1]} and inspect its "
            "compatible_with field before making the final compatibility conclusion."
        )

    @staticmethod
    def _explicit_user_constraint_feedback(
        response: AgentTurnResponse, conversation: list[dict[str, Any]], phase: str
    ) -> str | None:
        if phase != "pre_final" or not response.text:
            return None
        user_text = " ".join(
            str(item.get("content", ""))
            for item in conversation
            if item.get("role") == "user"
        ).lower()
        forbids_alternatives = re.search(
            r"\b(do not|don't|dont|no)\b.{0,40}\b(substitute|alternative|replacement)s?\b",
            user_text,
        )
        suggests_alternatives = re.search(
            r"\b(suggest|recommend|compare|try|find|choose|add|help with)\b.{0,45}"
            r"\b(other|different|alternative|substitute|replacement)\b|"
            r"\b(other|different|alternative|substitute|replacement)\b.{0,45}"
            r"\b(product|item|accessory|dock|option)\b|"
            r"\b(provide|use|try|supply|enter)\b.{0,45}"
            r"\b(canonical|recognized|matching|supported)\b.{0,30}\b(device|name)\b",
            response.text.lower(),
        )
        if forbids_alternatives and suggests_alternatives:
            return (
                "The user explicitly prohibited substitute or alternative suggestions. Remove the "
                "offer, comparison, or recommendation of any different product and complete only the "
                "requested read-only response."
            )
        return None

    @staticmethod
    def _budget_comparison_feedback(
        response: AgentTurnResponse, conversation: list[dict[str, Any]], phase: str
    ) -> str | None:
        if phase != "pre_final" or not response.text:
            return None
        user_text = " ".join(
            str(item.get("content", ""))
            for item in conversation
            if item.get("role") == "user"
        )
        budgets = re.findall(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)", user_text)
        if not budgets or not re.search(
            r"\b(above|over|exceeds?|outside|beyond)\b.{0,35}\bbudget\b|"
            r"\bbudget\b.{0,35}\b(above|over|exceeds?|outside|beyond)\b",
            response.text,
            flags=re.IGNORECASE,
        ):
            return None
        budget = float(budgets[-1].replace(",", ""))
        values = [
            float(value.replace(",", ""))
            for value in re.findall(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)", response.text)
        ]
        prices = [value for value in values if value > budget]
        if not prices:
            return None
        price = max(prices)
        overage = price - budget
        overage_text = f"{overage:g}"
        if re.search(
            rf"\bby\s+\$?{re.escape(overage_text)}\b|"
            rf"\$?{price:g}\s*[-\u2212]\s*\$?{budget:g}\s*=\s*\$?{overage_text}\b",
            response.text.replace(",", ""),
            flags=re.IGNORECASE,
        ):
            return None
        return (
            f"Make the budget violation numerically explicit before recommending the item: "
            f"${price:g} exceeds the user's ${budget:g} budget by ${overage:g}."
        )

    @staticmethod
    def _bundle_completeness_feedback(
        response: AgentTurnResponse, conversation: list[dict[str, Any]], phase: str
    ) -> str | None:
        if phase != "pre_final" or not response.text:
            return None
        item_terms = ("laptop", "backpack", "headphones", "webcam")
        user_text = " ".join(
            str(item.get("content", ""))
            for item in conversation
            if item.get("role") == "user"
        ).lower()
        requested = {term for term in item_terms if term in user_text}
        if len(requested) < 3:
            return None
        grounded: set[str] = set()
        for item in conversation:
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                if call.get("name") != "search_products":
                    continue
                query = str((call.get("arguments") or {}).get("query", "")).lower()
                result = call.get("result") or {}
                if not isinstance(result, dict) or not result.get("products"):
                    continue
                grounded.update(term for term in requested if term in query)
        missing = requested - grounded
        recommends = re.search(
            r"\b(recommend|best|choose|choice|lean toward|go with|bundle)\b",
            response.text,
            flags=re.IGNORECASE,
        )
        if missing and recommends:
            return (
                "Do not recommend or commit to a partial bundle before grounding every requested item and "
                f"computing the complete total. Missing catalog evidence for: {', '.join(sorted(missing))}. "
                "Continue searching with broader terms and remove an incorrect category filter when needed."
            )
        return None

    @staticmethod
    def _loyalty_cap_feedback(
        response: AgentTurnResponse, conversation: list[dict[str, Any]], phase: str
    ) -> str | None:
        if phase != "pre_final" or not response.text or not re.search(
            r"\b(maximum|max|50% cap|half)\b", response.text, flags=re.IGNORECASE
        ):
            return None
        carts = []
        has_half_cap_policy = False
        for item in conversation:
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                name = str(call.get("name", ""))
                result = call.get("result")
                if name == "get_cart" and isinstance(result, dict):
                    carts.append(result)
                if name == "get_policies":
                    policy_text = json.dumps(result, ensure_ascii=True, default=str).lower()
                    has_half_cap_policy = has_half_cap_policy or "50%" in policy_text
        if not carts or not has_half_cap_policy:
            return None
        cart = carts[-1]
        subtotal = float(cart.get("subtotal", 0) or 0)
        discount = float(cart.get("loyalty_discount", 0) or 0)
        cap = subtotal * 0.5
        if subtotal <= 0 or math.isclose(discount, cap):
            return None
        return (
            f"The numeric loyalty claim is inconsistent with observed state: 50% of the "
            f"${subtotal:g} subtotal is ${cap:g}, while the current loyalty discount is "
            f"${discount:g}. Do not call the current discount the maximum; disclose the gap "
            "and request approval before changing the redemption."
        )

    @staticmethod
    def _redundant_quantity_confirmation_feedback(
        response: AgentTurnResponse, conversation: list[dict[str, Any]], phase: str
    ) -> str | None:
        if phase != "pre_final" or not response.text:
            return None
        latest_user = next(
            (
                str(item.get("content", "")).lower()
                for item in reversed(conversation)
                if item.get("role") == "user" and "[task_done]" not in str(item.get("content", "")).lower()
            ),
            "",
        )
        if not re.search(r"\b(cart|quantity|one|two|second|units?|laptops?)\b", latest_user):
            return None
        if not re.search(r"\b(add|change|increase|reduce|remove|set|update)\b", latest_user):
            return None
        candidate = response.text.lower()
        repeats_quantity_confirmation = (
            ("?" in candidate or re.search(r"\breply (with|to) [\"']?confirm", candidate))
            and re.search(
                r"\b(should i|please confirm|confirm that|do you want me to|reply (with|to) [\"']?confirm)\b",
                candidate,
            )
            and re.search(r"\b(final quantity|quantity|set .*\bto\b|add .*back|remove)\b", candidate)
        )
        unresolved_choice = re.search(r"\b(which|what size|what color|choose|select|prefer)\b", candidate)
        if repeats_quantity_confirmation and not unresolved_choice:
            return (
                "The user already gave an explicit cart quantity instruction and no material choice remains. "
                "Do not ask for the same confirmation again; perform the authorized quantity update using "
                "the live tool schema."
            )
        return None

    @staticmethod
    def _material_write_effect_appendix(
        response: AgentTurnResponse, conversation: list[dict[str, Any]], phase: str
    ) -> str | None:
        if phase != "pre_final" or not response.text:
            return None
        calls: list[tuple[int, str, dict[str, Any], Any]] = []
        order = 0
        for item in conversation:
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                calls.append(
                    (
                        order,
                        str(call.get("name", "")),
                        call.get("arguments") or {},
                        call.get("result"),
                    )
                )
                order += 1
        write_calls = [call for call in calls if call[1] in WRITE_TOOL_NAMES]
        if not write_calls:
            return None
        write_order, write_name, _, write_result = write_calls[-1]
        if not isinstance(write_result, dict):
            return None

        normalized_candidate = re.sub(r"[,$\s]", "", response.text.lower())
        if write_result.get("loyalty_redemption_clamped"):
            values = {
                "previous loyalty discount": write_result.get("previous_loyalty_discount"),
                "new loyalty discount": write_result.get("new_loyalty_discount"),
                "refunded loyalty points": write_result.get("loyalty_points_refunded"),
            }
            missing = [
                label
                for label, value in values.items()
                if value is not None and str(value).replace(",", "") not in normalized_candidate
            ]
            if missing:
                return (
                    f"For clarity, the successful {write_name} call automatically clamped the loyalty "
                    f"discount from ${values['previous loyalty discount']} to "
                    f"${values['new loyalty discount']} and refunded "
                    f"{values['refunded loyalty points']} loyalty points."
                )

        carts_before = [call for call in calls if call[0] < write_order and call[1] == "get_cart"]
        carts_after = [call for call in calls if call[0] > write_order and call[1] == "get_cart"]
        if not carts_before or not carts_after:
            return None
        before = carts_before[-1][3]
        after = carts_after[-1][3]
        if not isinstance(before, dict) or not isinstance(after, dict):
            return None
        subtotal_increased = float(after.get("subtotal", 0) or 0) > float(before.get("subtotal", 0) or 0)
        old_discount = before.get("loyalty_discount")
        new_discount = after.get("loyalty_discount")
        sticky_discount = old_discount and old_discount == new_discount
        if subtotal_increased and sticky_discount:
            explains_non_restore = re.search(
                r"\b(did not|didn't|not automatically|stayed|remained|still)\b",
                response.text.lower(),
            )
            offers_action = re.search(
                r"\b(redeem|re-redeem|apply|use)\b.{0,35}\b(points?|loyalty)\b|"
                r"\b(points?|loyalty)\b.{0,35}\b(redeem|re-redeem|apply|use)\b",
                response.text.lower(),
            )
            if not explains_non_restore or not offers_action:
                return (
                    f"The loyalty redemption did not automatically restore after the cart subtotal increased; "
                    f"the current loyalty discount remains ${new_discount}. If you want, I can redeem "
                    "additional points, but I will not change the redemption without your approval."
                )
        return None

    @classmethod
    def _material_write_effect_feedback(
        cls, response: AgentTurnResponse, conversation: list[dict[str, Any]], phase: str
    ) -> str | None:
        appendix = cls._material_write_effect_appendix(response, conversation, phase)
        if not appendix:
            return None
        return f"Add this material observed side effect to the response: {appendix}"

    def _deterministic_feedback(
        self,
        *,
        phase: str,
        response: AgentTurnResponse,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> str | None:
        calls = self._tool_calls(response)
        events = self._seen_tool_events(conversation)
        last_user_index = max(
            (index for index, item in enumerate(conversation) if item.get("role") == "user"),
            default=-1,
        )
        recent_events = [
            event for event in events if event.message_index > last_user_index
        ]
        last_write_index = max(
            (event.sequence for event in events if event.name in WRITE_TOOL_NAMES),
            default=-1,
        )
        state_events = [event for event in events if event.sequence > last_write_index]

        schema_feedback = self._schema_feedback(response, tools)
        if schema_feedback:
            return schema_feedback

        compatibility_feedback = self._compatibility_evidence_feedback(
            response, conversation, phase
        )
        if compatibility_feedback:
            return compatibility_feedback

        constraint_feedback = self._explicit_user_constraint_feedback(
            response, conversation, phase
        )
        if constraint_feedback:
            return constraint_feedback

        budget_feedback = self._budget_comparison_feedback(response, conversation, phase)
        if budget_feedback:
            return budget_feedback

        bundle_feedback = self._bundle_completeness_feedback(response, conversation, phase)
        if bundle_feedback:
            return bundle_feedback

        loyalty_cap_feedback = self._loyalty_cap_feedback(response, conversation, phase)
        if loyalty_cap_feedback:
            return loyalty_cap_feedback

        quantity_feedback = self._redundant_quantity_confirmation_feedback(
            response, conversation, phase
        )
        if quantity_feedback:
            return quantity_feedback

        write_effect_feedback = self._material_write_effect_feedback(
            response, conversation, phase
        )
        if write_effect_feedback:
            return write_effect_feedback

        if phase == "pre_final" and self._unsupported_error_speculation(
            response, conversation
        ):
            return (
                "State only the exact observed tool failure reason. Do not enumerate or "
                "suggest unobserved possible causes, even inside an uncertainty disclaimer."
            )

        for name, arguments in calls:
            if name not in WRITE_TOOL_NAMES:
                continue
            signature = json.dumps([name, arguments], sort_keys=True, ensure_ascii=True)
            if any(
                json.dumps([old_name, old_arguments], sort_keys=True, ensure_ascii=True)
                == signature
                for old_name, old_arguments in (
                    (event.name, event.arguments) for event in recent_events
                )
            ):
                return f"Do not repeat the already executed {name} call with identical arguments."

        if not self._active_cards:
            return None
        card = self._active_cards[0]
        if int(card.get("support", 0)) < 3 or float(card.get("mean_fitness", 0.0)) < 0.8:
            return None
        rules = [rule for rule in card.get("runtime_rules", []) if isinstance(rule, dict)]
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
            if rule.get("kind") == "require_tool" and required:
                targets = [
                    arguments
                    for name, arguments in calls
                    if not triggers or name in triggers
                ]
                missing = {
                    tool_name
                    for tool_name in required
                    if any(
                        not self._has_required_evidence(
                            tool_name=tool_name,
                            events=state_events,
                            candidate_arguments=arguments,
                        )
                        for arguments in targets
                    )
                }
                if missing:
                    names = ", ".join(sorted(missing))
                    return str(
                        rule.get("feedback")
                        or f"Call the required read tools first: {names}."
                    )
            if phase == "pre_final" and rule.get("kind") == "refresh" and required:
                writes = [event for event in events if event.name in WRITE_TOOL_NAMES]
                if not writes:
                    continue
                last_write = max(writes, key=lambda event: event.sequence)
                after_write = [
                    event for event in events if event.sequence > last_write.sequence
                ]
                missing = {
                    tool_name
                    for tool_name in required
                    if not self._has_required_evidence(
                        tool_name=tool_name,
                        events=after_write,
                        candidate_arguments=last_write.arguments,
                    )
                }
                if missing:
                    names = ", ".join(sorted(missing))
                    return str(
                        rule.get("feedback")
                        or f"Refresh state before replying: {names}."
                    )
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
                max_tokens=500,
                timeout_seconds=float(
                    os.environ.get("STATE_BENCH_VERIFIER_TIMEOUT_SECONDS", "60")
                ),
                max_retries=0,
            )
        except Exception:  # noqa: BLE001 - auxiliary verifier failures must not abort the benchmark
            return True, ""
        usage = getattr(result, "usage", None)
        self.add_token_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            category="other_llm",
        )
        match = re.search(r"\{.*\}", str(getattr(result, "text", "")), flags=re.DOTALL)
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
        issues = [str(value) for value in verdict.get("violations", []) if str(value).strip()]
        if feedback and feedback not in issues:
            issues.append(feedback)
        confirmation_terms = re.compile(
            r"\b(confirm|confirmation|approval|approve|authorization|authorize|consent|permission)\b",
            flags=re.IGNORECASE,
        )
        if (
            issues
            and all(confirmation_terms.search(issue) for issue in issues)
            and self._has_explicit_write_authorization(conversation)
        ):
            return True, ""
        return False, feedback or "Revise the step to satisfy the applicable workflow policy."

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        generation_prompt = system_prompt
        if self.tapm_mode != "off":
            generation_prompt = (
                f"{system_prompt}\n\n{self.transition_memory.prompt_context(conversation)}"
            )
        available_tools = [
            str(tool.get("name") or tool.get("function", {}).get("name", ""))
            for tool in tools
        ]

        def enforce_transition(
            candidate: AgentTurnResponse,
            candidate_context: list[dict[str, Any]],
        ) -> AgentTurnResponse:
            if self.tapm_mode != "enforce":
                return candidate
            for revision in range(self.tapm_max_revisions + 1):
                candidate_calls = self._tool_calls(candidate)
                tapm_feedback = self.transition_memory.feedback(
                    candidate_calls=candidate_calls,
                    conversation=conversation,
                    available_tools=available_tools,
                    final_response=not candidate_calls,
                )
                if not tapm_feedback:
                    return candidate
                if revision >= self.tapm_max_revisions:
                    return AgentTurnResponse(
                        text="Before making that change, I need to refresh the current state.",
                        tool_calls=[],
                    )
                correction = {
                    "role": "system",
                    "content": (
                        "The previous candidate was not executed. Transition-aware state checking found: "
                        f"{tapm_feedback} Generate the minimal corrective next step."
                    ),
                }
                candidate = super(ProcessWorkflowMemoryAgent, self).generate_next_turn(
                    system_prompt=generation_prompt,
                    conversation=[*candidate_context, correction],
                    tools=tools,
                )
            return candidate

        response = super().generate_next_turn(
            system_prompt=generation_prompt,
            conversation=conversation,
            tools=tools,
        )
        response = enforce_transition(response, conversation)
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

            deterministic_feedback = self._deterministic_feedback(
                phase=phase,
                response=response,
                conversation=conversation,
                tools=tools,
            )
            feedback = deterministic_feedback
            allowed = deterministic_feedback is None
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
                    if (
                        deterministic_feedback is None
                        and self._has_explicit_write_authorization(conversation)
                    ):
                        return response
                    return AgentTurnResponse(
                        text=response.text
                        or "Before I make that change, I need to verify the required details or confirmation.",
                        tool_calls=[],
                    )
                appendix = self._material_write_effect_appendix(
                    response, conversation, phase
                )
                if appendix:
                    return AgentTurnResponse(
                        text=f"{response.text.rstrip()}\n\n{appendix}",
                        tool_calls=response.tool_calls,
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
            verifier_context = [*conversation, correction]
            response = super().generate_next_turn(
                system_prompt=generation_prompt,
                conversation=verifier_context,
                tools=tools,
            )
            response = enforce_transition(response, verifier_context)
        return response
