"""Selective, risk-aware retrieval for process workflow memory.

This module is deliberately a sidecar to :mod:`process_workflow_memory_agent`:
the original PWM actor and artifact remain untouched.  When the v2 router is
missing, malformed, disabled for the current domain, or running in ``shadow``
mode, externally visible retrieval is exactly the parent implementation.

The v2 artifact is compiled offline from fixed training trajectories.  Runtime
retrieval is deterministic and makes no model calls.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agents.process_workflow_memory_agent import (
    INTENT_HINTS,
    ProcessWorkflowMemoryAgent as _Parent,
    _char_ngrams,
    _tokens,
)


_DEFAULT_ROUTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "statebench_cross_domain_pwm"
    / "memory"
    / "workflow_router_v2.json"
)

_WEIGHT_DEFAULTS = {
    "field": 0.5,
    "utility": 0.25,
    "risk": 0.5,
    "trace": 0.25,
    "mmr": 0.2,
}

_WEIGHT_GRIDS: dict[str, frozenset[float]] = {
    "field": frozenset({0.25, 0.5, 1.0}),
    "utility": frozenset({0.0, 0.25, 0.5}),
    "risk": frozenset({0.25, 0.5, 1.0}),
    "trace": frozenset({0.0, 0.25}),
    "mmr": frozenset({0.1, 0.2, 0.3}),
}

# The tunable near-tie window controls which candidates may be reranked.  The
# independent hard anchor below still preserves a raw semantic winner whose
# top-1/top-2 margin is greater than 2.0, exactly as required by the protocol.
_NEAR_TIE_GRID = frozenset({0.5, 1.0, 2.0})
_HARD_ANCHOR_MARGIN = 2.0
_RENDER_CHAR_LIMIT = 2200

_THRESHOLD_DEFAULTS: dict[str, float] = {
    "near_tie": 2.0,
    "candidate_pool": 12.0,
    "max_cards": 3.0,
    "default_cards": 1.0,
    "min_relevance": 0.35,
    "min_secondary_score": 0.75,
    "secondary_relative_score": 0.55,
    "same_family_limit": 2.0,
    "duplicate_jaccard": 0.8,
    "utility_cap": 0.75,
    "min_utility_exposures": 5.0,
    "stickiness": 0.2,
}

# The mapping is used only to estimate exposure risk.  It never blocks a tool
# call and therefore cannot turn the retriever into an action policy.
_WRITE_ACTION_PHRASES: dict[str, dict[str, frozenset[str]]] = {
    "shopping_assistant": {
        "add_to_cart": frozenset({"add to cart", "put in cart", "buy it", "buy one", "purchase it"}),
        "remove_from_cart": frozenset({"remove from", "take out", "delete from", "remove it"}),
        "update_cart_item": frozenset({"update quantity", "change quantity", "set quantity"}),
        "apply_promo": frozenset({"apply promo", "apply coupon", "use promo", "use coupon"}),
        "remove_promo": frozenset({"remove promo", "remove coupon"}),
        "redeem_loyalty_points": frozenset({"redeem", "use my points", "use points"}),
        "cancel_loyalty_redemption": frozenset({"cancel redemption", "undo redemption", "restore points"}),
        "set_shipping_option": frozenset({"set shipping", "choose shipping", "select shipping", "switch shipping"}),
    },
    "travel": {
        "update_booking": frozenset({"change my", "move my", "update my", "switch my", "change the booking"}),
        "cancel_booking": frozenset({"cancel my flight", "cancel the booking", "cancel booking"}),
        "create_booking": frozenset({"book a flight", "reserve a flight", "buy a ticket"}),
        "book_hotel": frozenset({"book a hotel", "reserve a hotel", "book the hotel"}),
        "cancel_hotel_reservation": frozenset({"cancel my hotel", "cancel the hotel"}),
        "book_car_rental": frozenset({"book a car", "reserve a car", "book the rental"}),
        "cancel_car_rental": frozenset({"cancel my rental", "cancel the rental"}),
    },
    "customer_support": {
        "process_return": frozenset({"return it", "return this", "send it back", "start a return"}),
        "process_refund": frozenset({"refund me", "issue a refund", "process refund", "price match it"}),
        "process_exchange": frozenset({"exchange it", "replace it", "exchange this", "send a replacement"}),
        "process_warranty_claim": frozenset({"file a warranty", "start a warranty", "repair it"}),
        "cancel_order": frozenset({"cancel my order", "cancel the order"}),
    },
}

_WRITE_ACTION_PATTERNS: dict[str, dict[str, str]] = {
    "shopping_assistant": {
        "add_to_cart": r"\b(?:add|put)\b.{0,30}\b(?:cart|basket)\b|\b(?:buy|purchase)\b",
        "remove_from_cart": r"\b(?:remove|delete|take)\b.{0,40}\b(?:from|cart|basket|item|it)\b",
        "update_cart_item": r"\b(?:update|change|set)\b.{0,30}\b(?:quantity|qty|cart item)\b",
        "apply_promo": r"\b(?:apply|use)\b.{0,30}\b(?:promo|coupon|discount code)\b",
        "remove_promo": r"\b(?:remove|delete)\b.{0,25}\b(?:promo|coupon)\b",
        "redeem_loyalty_points": r"\bredeem\b|\buse\b.{0,25}\bpoints\b",
        "cancel_loyalty_redemption": r"\b(?:cancel|undo|reverse)\b.{0,25}\b(?:redemption|points)\b",
        "set_shipping_option": r"\b(?:set|choose|select|switch|use)\b.{0,30}\b(?:shipping|delivery|express|next.day)\b",
    },
    "travel": {
        "update_booking": r"\b(?:change|move|update|switch)\b.{0,35}\b(?:booking|flight|seat|meal|bag|wifi)\b",
        "cancel_booking": r"\bcancel\b.{0,25}\b(?:booking|flight|trip)\b",
        "create_booking": r"\b(?:book|reserve|buy)\b.{0,25}\b(?:flight|ticket|trip)\b",
        "book_hotel": r"\b(?:book|reserve)\b.{0,20}\bhotel\b",
        "cancel_hotel_reservation": r"\bcancel\b.{0,25}\bhotel\b",
        "book_car_rental": r"\b(?:book|reserve)\b.{0,25}\b(?:car|rental)\b",
        "cancel_car_rental": r"\bcancel\b.{0,25}\b(?:car|rental)\b",
    },
    "customer_support": {
        "process_return": r"^\s*return\b|\b(?:want|need|start|process|initiate|please)\b.{0,25}\breturn\b|\bsend\b.{0,15}\bback\b",
        "process_refund": r"^\s*refund\b|\b(?:want|need|issue|process|please)\b.{0,25}\brefund\b|\brefund me\b",
        "process_exchange": r"^\s*exchange\b|\b(?:want|need|process|please)\b.{0,25}\b(?:exchange|replace)\b",
        "process_warranty_claim": r"\b(?:file|start|open|process)\b.{0,25}\bwarranty\b|\brepair it\b",
        "cancel_order": r"\bcancel\b.{0,25}\border\b",
    },
}

_CONTINUATION_WORDS = frozenset(
    {
        "yes",
        "y",
        "ok",
        "okay",
        "confirm",
        "confirmed",
        "continue",
        "proceed",
        "do",
        "it",
        "that",
        "one",
        "please",
        "sure",
        "no",
        "blue",
        "black",
        "white",
        "red",
        "green",
        "the",
    }
)

_COVERAGE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "before",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "should",
        "that",
        "the",
        "this",
        "to",
        "under",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
    }
)

_RELEVANCE_STOPWORDS = _COVERAGE_STOPWORDS | frozenset(
    {
        "about",
        "answer",
        "asks",
        "help",
        "need",
        "needs",
        "questions",
        "related",
        "requesting",
        "tell",
        "user",
        "want",
        "wants",
    }
)

_OBLIGATION_MARKERS = frozenset(
    {
        "approve",
        "compare",
        "confirm",
        "cost",
        "deadline",
        "disclose",
        "eligibility",
        "explain",
        "fee",
        "must",
        "never",
        "preview",
        "price",
        "quote",
        "show",
        "tell",
        "total",
        "without",
    }
)

_DISCLOSURE_TOPIC_WORDS: dict[str, frozenset[str]] = {
    "amount": frozenset({"charge", "charges", "cost", "costs", "fee", "fees", "price", "quote", "total"}),
    "eligibility": frozenset({"eligible", "eligibility", "qualify", "qualified", "valid"}),
    "timing": frozenset({"deadline", "eta", "timing", "when"}),
    "confirmation": frozenset({"approval", "approve", "confirm", "confirmation", "preview"}),
    "comparison": frozenset({"alternative", "alternatives", "compare", "comparison", "difference", "options"}),
}

_INTENT_WRITE_TOOLS: dict[str, dict[str, frozenset[str]]] = {
    "shopping_assistant": {
        "promo": frozenset({"apply_promo", "remove_promo"}),
        "loyalty": frozenset({"redeem_loyalty_points", "cancel_loyalty_redemption"}),
        "shipping": frozenset({"set_shipping_option"}),
        "remove": frozenset({"remove_from_cart"}),
        "update_cart": frozenset({"update_cart_item"}),
        "add_to_cart": frozenset({"add_to_cart"}),
    },
    "travel": {
        "cancel": frozenset({"cancel_booking", "cancel_hotel_reservation", "cancel_car_rental"}),
        "hotel": frozenset({"book_hotel", "cancel_hotel_reservation"}),
        "car_rental": frozenset({"book_car_rental", "cancel_car_rental"}),
        "baggage": frozenset({"update_booking"}),
        "seat": frozenset({"update_booking"}),
        "ancillary": frozenset({"update_booking"}),
        "change": frozenset({"update_booking"}),
        "book": frozenset({"create_booking", "book_hotel", "book_car_rental"}),
    },
    "customer_support": {
        "price_match": frozenset({"process_refund"}),
        "warranty": frozenset({"process_warranty_claim"}),
        "exchange": frozenset({"process_exchange"}),
        "return": frozenset({"process_return"}),
        "cancel": frozenset({"cancel_order"}),
        "refund": frozenset({"process_refund"}),
    },
}

_NEGATION_PREFIX = re.compile(
    r"(?:\bdo\s+not\b|\bdon't\b|\bnever\b|\bmust\s+not\b|\bwithout\b|"
    r"\bavoid\b|\brefrain\s+from\b|\bno\s+need\s+to\b)"
    r"(?:\W+\w+){0,7}\W*$"
)
_INFORMATION_ONLY_PREFIX = re.compile(
    r"(?:\b(?:only|just)\s+(?:want\s+to\s+)?(?:tell|show|check|explain|describe|verify)\b"
    r"|\b(?:tell|show|check|explain|describe|verify)\b.{0,60}\b(?:whether|if)\b)"
)

_FORBIDDEN_SIDECAR_KEYS = frozenset(
    {
        "task_summary",
        "requirement",
        "requirements",
        "task_requirements",
        "state_requirements",
        "expected_state",
        "oracle",
        "test",
        "test_task",
        "test_task_id",
        "judge",
        "judge_reasoning",
    }
)


def _float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        elif isinstance(item, Mapping):
            name = item.get("name") or item.get("tool") or item.get("text")
            if isinstance(name, str) and name.strip():
                result.append(name.strip())
    return result


def _scope_text_list(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [
            str(value[key]).strip()
            for key in ("title", "family", "domain")
            if isinstance(value.get(key), str) and str(value[key]).strip()
        ]
    return _text_list(value)


def _normalised_words(text: str) -> set[str]:
    return {
        token
        for token in _tokens(text)
        if len(token) > 1 or any("\u4e00" <= character <= "\u9fff" for character in token)
    }


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_SIDECAR_KEYS:
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class _Candidate:
    index: int
    card: dict[str, Any]
    sidecar: dict[str, Any]
    semantic: float
    lexical: float
    character: float
    intent: float
    field_score: float = 0.0
    utility: float = 0.0
    risk: float = 0.0
    trace: float = 0.0
    final: float = 0.0
    extra_writes: tuple[str, ...] = ()
    coverage: frozenset[str] = frozenset()
    adjusted: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def card_id(self) -> str:
        return str(self.card.get("id", ""))

    @property
    def family(self) -> str:
        return str(self.card.get("family", ""))


class RiskAwareProcessWorkflowMemoryAgent(_Parent):
    """PWM with selective decision-aware reranking and typed card packing."""

    router_path = _DEFAULT_ROUTER_PATH

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self._runtime_domain = str(getattr(runtime_context, "domain", "") or "")
        self._router_mode = os.environ.get("STATE_BENCH_WORKFLOW_ROUTER_MODE", "enforce").lower()
        self._router_stage = os.environ.get("STATE_BENCH_WORKFLOW_ROUTER_STAGE", "C").upper()
        self._router_enabled = False
        self._router_reason = "router_not_loaded"
        self._router: dict[str, Any] = {}
        self._router_cards: dict[str, dict[str, Any]] = {}
        self._weights = dict(_WEIGHT_DEFAULTS)
        self._thresholds = dict(_THRESHOLD_DEFAULTS)
        self._field_tokens: list[list[str]] = []
        self._relevance_field_words: list[set[str]] = []
        self._field_df: Counter[str] = Counter()
        self._field_avg_len = 1.0
        self._base_quality_mean = 0.0
        self._base_support_max = 1
        self._state_context: dict[str, Any] = {}
        self._active_intent_signature: tuple[str, ...] = ()
        self._active_tool_signature: tuple[str, ...] = ()
        self._active_card_ids: tuple[str, ...] = ()
        self._active_candidates: tuple[_Candidate, ...] = ()
        self._active_query_key: tuple[Any, ...] | None = None
        self.retrieval_telemetry: deque[dict[str, Any]] = deque(maxlen=64)
        self.last_retrieval_telemetry: dict[str, Any] = {}
        self._load_router(kwargs.get("workflow_router_path"))

    # ------------------------------------------------------------------
    # Artifact loading and fail-open behaviour
    # ------------------------------------------------------------------
    def _fallback(self, reason: str) -> None:
        self._router_enabled = False
        self._router_reason = reason

    def _load_router(self, path_override: Any = None) -> None:
        if self._router_mode not in {"shadow", "enforce"}:
            self._fallback("invalid_router_mode")
            return
        if self._router_stage not in {"A", "B", "C"}:
            self._fallback("invalid_router_stage")
            return
        configured = path_override or os.environ.get("STATE_BENCH_WORKFLOW_ROUTER_PATH")
        path = Path(configured) if configured else Path(self.router_path)
        if not path.is_file():
            self._fallback("router_missing")
            return
        try:
            router = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            self._fallback("router_unreadable")
            return
        if not isinstance(router, dict) or _contains_forbidden_key(router):
            self._fallback("router_invalid_or_forbidden_provenance")
            return
        version = str(router.get("schema_version", router.get("version", "")))
        if not version.startswith("2"):
            self._fallback("unsupported_router_schema")
            return

        expected_hash = router.get("source_memory_sha256")
        provenance = router.get("provenance")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            self._fallback("memory_hash_missing_or_invalid")
            return
        if not isinstance(provenance, Mapping):
            self._fallback("memory_hash_provenance_missing")
            return
        provenance_hash = provenance.get("memory_sha256") or provenance.get("source_memory_sha256")
        if not isinstance(provenance_hash, str) or provenance_hash.lower() != expected_hash.lower():
            self._fallback("memory_hash_declaration_mismatch")
            return
        try:
            actual_hash = hashlib.sha256(Path(self.memory_path).read_bytes()).hexdigest()
        except OSError:
            self._fallback("memory_hash_unavailable")
            return
        if expected_hash.lower() != actual_hash.lower():
            self._fallback("memory_hash_mismatch")
            return

        domain_configs = router.get("domain_configs")
        domain_config: Mapping[str, Any] = {}
        if isinstance(domain_configs, Mapping):
            candidate = domain_configs.get(self._runtime_domain, {})
            if isinstance(candidate, Mapping):
                domain_config = candidate
            promoted = bool(domain_config.get("promoted", False))
        else:
            enabled = router.get("enabled_domains", [])
            if isinstance(enabled, Mapping):
                promoted = bool(enabled.get(self._runtime_domain, False))
            else:
                promoted = self._runtime_domain in _text_list(enabled)
        if not promoted:
            self._fallback("domain_not_promoted")
            return

        raw_cards = router.get("cards")
        if isinstance(raw_cards, Mapping):
            sidecars = {
                str(card_id): dict(value)
                for card_id, value in raw_cards.items()
                if isinstance(value, Mapping)
            }
        elif isinstance(raw_cards, list):
            sidecars = {}
            for value in raw_cards:
                if not isinstance(value, Mapping):
                    continue
                card_id = value.get("id") or value.get("card_id")
                if card_id:
                    sidecars[str(card_id)] = dict(value)
        else:
            self._fallback("router_cards_missing")
            return

        base_ids = {str(card.get("id", "")) for card in self._cards}
        if not base_ids or not base_ids.issubset(sidecars):
            self._fallback("router_card_coverage_incomplete")
            return
        for card in self._cards:
            sidecar = sidecars[str(card.get("id", ""))]
            validation_error = self._sidecar_card_error(sidecar, card)
            if validation_error:
                self._fallback(validation_error)
                return

        defaults = router.get("defaults") if isinstance(router.get("defaults"), Mapping) else None
        if (
            not isinstance(defaults, Mapping)
            or not isinstance(defaults.get("weights"), Mapping)
            or not isinstance(defaults.get("thresholds"), Mapping)
            or not isinstance(domain_config.get("weights"), Mapping)
            or not isinstance(domain_config.get("thresholds"), Mapping)
        ):
            self._fallback("router_config_missing")
            return
        required_weights = set(_WEIGHT_DEFAULTS)
        required_thresholds = set(_THRESHOLD_DEFAULTS) - {"stickiness"}
        if (
            not required_weights.issubset(defaults["weights"])
            or not required_weights.issubset(domain_config["weights"])
            or not required_thresholds.issubset(defaults["thresholds"])
            or not required_thresholds.issubset(domain_config["thresholds"])
        ):
            self._fallback("router_config_missing")
            return
        self._weights = self._merged_numbers(
            _WEIGHT_DEFAULTS,
            defaults.get("weights"),
            router.get("weights"),
            domain_config.get("weights"),
        )
        self._thresholds = self._merged_numbers(
            _THRESHOLD_DEFAULTS,
            defaults.get("thresholds"),
            router.get("thresholds"),
            domain_config.get("thresholds"),
        )
        if not self._configuration_is_safe(self._weights, self._thresholds):
            self._fallback("router_config_invalid")
            return

        self._router = router
        self._router_cards = sidecars
        self._prepare_field_statistics()
        self._prepare_quality_statistics()
        self._router_enabled = True
        self._router_reason = "enforce" if self._router_mode == "enforce" else "shadow"

    @staticmethod
    def _merged_numbers(defaults: Mapping[str, float], *sources: Any) -> dict[str, float]:
        result = dict(defaults)
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            for key in result:
                if key in source:
                    result[key] = _float(source[key], result[key])
        return result

    def _contract(self, sidecar: Mapping[str, Any]) -> Mapping[str, Any]:
        value = sidecar.get("contract")
        return value if isinstance(value, Mapping) else sidecar

    @staticmethod
    def _configuration_is_safe(
        weights: Mapping[str, float], thresholds: Mapping[str, float]
    ) -> bool:
        for name, grid in _WEIGHT_GRIDS.items():
            if not any(math.isclose(weights[name], value, abs_tol=1e-12) for value in grid):
                return False
        if not any(
            math.isclose(thresholds["near_tie"], value, abs_tol=1e-12)
            for value in _NEAR_TIE_GRID
        ):
            return False
        integral_bounds = {
            "candidate_pool": (1, 12),
            "max_cards": (0, 3),
            "default_cards": (0, 3),
            "same_family_limit": (1, 2),
            "min_utility_exposures": (1, 1000),
        }
        for name, (minimum, maximum) in integral_bounds.items():
            value = thresholds[name]
            if not value.is_integer() or not minimum <= int(value) <= maximum:
                return False
        if thresholds["default_cards"] > thresholds["max_cards"]:
            return False
        bounded = {
            "min_relevance": (0.0, 20.0),
            "min_secondary_score": (0.0, 20.0),
            "secondary_relative_score": (0.0, 1.0),
            "duplicate_jaccard": (0.0, 1.0),
            "utility_cap": (0.0, 0.75),
            "stickiness": (0.0, 0.75),
        }
        return all(minimum <= thresholds[name] <= maximum for name, (minimum, maximum) in bounded.items())

    @staticmethod
    def _string_list_is_valid(value: Any, *, allow_empty: bool) -> bool:
        return (
            isinstance(value, list)
            and (allow_empty or bool(value))
            and all(isinstance(item, str) and bool(item.strip()) for item in value)
        )

    def _sidecar_card_error(
        self, sidecar: Mapping[str, Any], card: Mapping[str, Any]
    ) -> str | None:
        declared_id = sidecar.get("id") or sidecar.get("card_id")
        if declared_id and str(declared_id) != str(card.get("id", "")):
            return "router_card_id_mismatch"
        declared_domain = sidecar.get("domain")
        if str(declared_domain or "") != self._runtime_domain:
            return "router_card_domain_mismatch"
        if str(sidecar.get("family", "")) != str(card.get("family", "")):
            return "router_card_family_mismatch"
        source_hash = sidecar.get("source_card_sha256")
        if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_hash):
            return "router_source_card_hash_missing_or_invalid"
        if source_hash.lower() != _canonical_sha256(card):
            return "router_source_card_hash_mismatch"

        contract = sidecar.get("contract")
        if not isinstance(contract, Mapping):
            return "router_contract_invalid"
        trigger = contract.get("trigger")
        if not (
            isinstance(trigger, str)
            and bool(trigger.strip())
            or self._string_list_is_valid(trigger, allow_empty=False)
        ):
            return "router_contract_invalid"
        scope = contract.get("scope")
        if not isinstance(scope, Mapping):
            return "router_contract_invalid"
        expected_scope = {
            "card_id": str(card.get("id", "")),
            "domain": self._runtime_domain,
            "family": str(card.get("family", "")),
        }
        if any(str(scope.get(key, "")) != value for key, value in expected_scope.items()):
            return "router_contract_scope_mismatch"
        if not isinstance(scope.get("title"), str) or not str(scope["title"]).strip():
            return "router_contract_invalid"
        for key, allow_empty in (
            ("required_reads", True),
            ("authorized_writes", True),
            ("decision_rules", False),
            ("verification_rules", False),
            ("required_disclosures", True),
            ("prohibitions", False),
        ):
            if not self._string_list_is_valid(contract.get(key), allow_empty=allow_empty):
                return "router_contract_invalid"
        compiler = sidecar.get("compiler")
        if not isinstance(compiler, Mapping) or not isinstance(compiler.get("valid"), bool):
            return "router_compiler_metadata_invalid"
        valid = bool(compiler["valid"])
        fallback_to_base = compiler.get("fallback_to_base_card")
        if not isinstance(fallback_to_base, bool) or fallback_to_base != (not valid):
            return "router_compiler_metadata_invalid"
        checks = compiler.get("checks")
        if not isinstance(checks, Mapping):
            return "router_compiler_metadata_invalid"
        if valid and (
            any(
                checks.get(key) != "passed"
                for key in (
                    "tool_binding",
                    "write_subset",
                    "disclosure_coverage",
                    "prohibition_coverage",
                    "length_bound",
                )
            )
            or not isinstance(checks.get("variable_binding"), str)
            or not str(checks["variable_binding"]).strip()
        ):
            return "router_compiler_metadata_invalid"
        if valid and (
            not isinstance(sidecar.get("primary_text"), str)
            or not str(sidecar["primary_text"]).strip()
            or not isinstance(sidecar.get("secondary_text"), str)
            or not str(sidecar["secondary_text"]).strip()
        ):
            return "router_compiler_output_invalid"
        utility = sidecar.get("utility")
        if not isinstance(utility, Mapping):
            return "router_utility_invalid"
        for key in ("domain_prior", "card_prior"):
            value = _float(utility.get(key), math.nan)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                return "router_utility_invalid"
        if not isinstance(utility.get("state_priors"), Mapping):
            return "router_utility_invalid"
        return None

    def _valid_sidecar_card(self, sidecar: Mapping[str, Any], card: Mapping[str, Any]) -> bool:
        """Compatibility wrapper retained for focused unit tests."""
        return self._sidecar_card_error(sidecar, card) is None

    def _prepare_field_statistics(self) -> None:
        self._field_tokens = []
        self._relevance_field_words = []
        for card in self._cards:
            sidecar = self._router_cards[str(card.get("id", ""))]
            contract = self._contract(sidecar)
            parts: list[str] = []
            for key in (
                "trigger",
                "triggers",
                "scope",
                "required_reads",
                "authorized_writes",
                "decision_rules",
                "verification_rules",
                "required_disclosures",
                "prohibitions",
                "field_text",
            ):
                values = contract.get(key)
                parts.extend(_scope_text_list(values) if key == "scope" else _text_list(values))
                if contract is not sidecar:
                    sidecar_values = sidecar.get(key)
                    parts.extend(
                        _scope_text_list(sidecar_values)
                        if key == "scope"
                        else _text_list(sidecar_values)
                    )
            tokens = _tokens(" ".join(parts))
            self._field_tokens.append(tokens)
            relevance_parts = [
                *_text_list(contract.get("trigger", contract.get("triggers"))),
                *_scope_text_list(contract.get("scope")),
                *_text_list(card.get("keywords")),
            ]
            self._relevance_field_words.append(
                _normalised_words(" ".join(relevance_parts)) - _RELEVANCE_STOPWORDS
            )
        self._field_df = Counter(token for tokens in self._field_tokens for token in set(tokens))
        self._field_avg_len = sum(map(len, self._field_tokens)) / max(len(self._field_tokens), 1)

    def _prepare_quality_statistics(self) -> None:
        self._base_support_max = max(
            (max(0, int(card.get("support", 0))) for card in self._cards),
            default=1,
        )
        values = [self._base_quality(card) for card in self._cards]
        self._base_quality_mean = sum(values) / max(len(values), 1)

    # ------------------------------------------------------------------
    # State-conditioned query and FlowSwitch-lite cache
    # ------------------------------------------------------------------
    def _query_from_conversation(self, conversation: list[Any]) -> str:
        if (
            not self._router_enabled
            or self._router_mode != "enforce"
            or self._router_stage != "C"
        ):
            return _Parent._query_from_conversation(self, conversation)
        user_messages = [
            str(item.get("content", "")).strip()
            for item in conversation
            if item.get("role") == "user"
            and "[TASK_DONE]" not in str(item.get("content", ""))
            and str(item.get("content", "")).strip()
        ]
        if not user_messages:
            self._state_context = {}
            return ""
        latest = user_messages[-1]
        latest_intents = self._intent_matches(latest)
        short_refinement = len(_normalised_words(latest)) <= 6 and not latest_intents
        previous_effective = str(self._state_context.get("intent_text", "")).strip()
        if (self._is_continuation(latest) or short_refinement) and previous_effective:
            intent_text = f"{previous_effective} {latest}".strip()
        elif len(user_messages) > 1 and (self._is_continuation(latest) or short_refinement):
            prior = next(
                (
                    message
                    for message in reversed(user_messages[:-1])
                    if not self._is_continuation(message)
                ),
                user_messages[-2],
            )
            intent_text = f"{prior} {latest}".strip()
        else:
            intent_text = latest

        observed_tools = [
            str(call.get("name", ""))
            for item in conversation
            if item.get("role") == "assistant"
            for call in (item.get("tool_calls") or [])
            if str(call.get("name", ""))
        ]
        recent_tools = tuple(observed_tools[-2:])
        write_tools = {
            tool
            for sidecar in self._router_cards.values()
            for tool in self._write_tools(sidecar)
        }
        if recent_tools and recent_tools[-1] in write_tools:
            phase = "postwrite"
        elif recent_tools:
            phase = "prewrite"
        else:
            phase = "read"
        intents = tuple(sorted(self._intent_matches(intent_text)))
        signature = intents or tuple(sorted(_normalised_words(intent_text))[:12])
        self._state_context = {
            "intent_text": intent_text,
            "intents": intents,
            "intent_signature": signature,
            "observed_tools": recent_tools,
            "phase": phase,
        }
        return f"{intent_text} {' '.join(recent_tools)}".strip()

    @staticmethod
    def _is_continuation(text: str) -> bool:
        words = _normalised_words(text)
        return len(words) <= 4 and bool(words) and words.issubset(_CONTINUATION_WORDS)

    def _context_for_query(self, query: str) -> dict[str, Any]:
        context = dict(self._state_context)
        state_query = f"{context.get('intent_text', '')} {' '.join(context.get('observed_tools', ())) }".strip()
        if not context or query.strip() != state_query:
            observed = tuple(
                re.findall(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b", query.lower())[-2:]
            )
            intents = tuple(sorted(self._intent_matches(query)))
            signature = intents or tuple(sorted(_normalised_words(query))[:12])
            context = {
                "intent_text": query,
                "intents": intents,
                "intent_signature": signature,
                "observed_tools": observed,
                "phase": "prewrite" if observed else "read",
            }
        return context

    # ------------------------------------------------------------------
    # Separated relevance, utility, exposure-risk and trace components
    # ------------------------------------------------------------------
    def _intent_matches(self, query: str) -> set[str]:
        lowered = query.lower()
        return {
            intent
            for intent, phrases in INTENT_HINTS.get(self._runtime_domain, {}).items()
            if any(phrase in lowered for phrase in phrases)
        }

    @staticmethod
    def _clause_start(text: str, offset: int) -> int:
        boundaries = [text.rfind(marker, 0, offset) for marker in (";", ".", "!", "?", "\n")]
        for marker in (
            " but ",
            " however ",
            " instead ",
            ", but ",
            ", just ",
            ", only ",
            ", then ",
            ", instead ",
        ):
            found = text.rfind(marker, 0, offset)
            if found >= 0:
                boundaries.append(found + len(marker) - 1)
        return max(boundaries, default=-1) + 1

    @classmethod
    def _mention_denies_authorization(
        cls,
        text: str,
        start: int,
        end: int,
        *,
        information_only: bool,
    ) -> bool:
        lowered = text.lower()
        clause_start = cls._clause_start(lowered, start)
        prefix = lowered[clause_start:start]
        if _NEGATION_PREFIX.search(prefix):
            return True
        clause_through_mention = lowered[clause_start:end]
        return information_only and bool(_INFORMATION_ONLY_PREFIX.search(clause_through_mention))

    def _write_mentions(self, text: str, tool: str) -> tuple[tuple[int, int], ...]:
        lowered = text.lower()
        spans: set[tuple[int, int]] = set()
        literal_variants = {tool.lower(), tool.lower().replace("_", " ")}
        phrases = set(_WRITE_ACTION_PHRASES.get(self._runtime_domain, {}).get(tool, ()))
        for intent, tools in _INTENT_WRITE_TOOLS.get(self._runtime_domain, {}).items():
            if tool in tools:
                phrases.update(INTENT_HINTS.get(self._runtime_domain, {}).get(intent, ()))
        for phrase in literal_variants | phrases:
            for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", lowered):
                spans.add(match.span())
        pattern = _WRITE_ACTION_PATTERNS.get(self._runtime_domain, {}).get(tool)
        if pattern:
            spans.update(match.span() for match in re.finditer(pattern, lowered))
        return tuple(sorted(spans))

    def _write_is_authorized(self, text: str, tool: str) -> bool:
        return any(
            not self._mention_denies_authorization(
                text,
                start,
                end,
                information_only=True,
            )
            for start, end in self._write_mentions(text, tool)
        )

    def _positive_intents(self, text: str) -> set[str]:
        lowered = text.lower()
        result: set[str] = set()
        write_intents = _INTENT_WRITE_TOOLS.get(self._runtime_domain, {})
        for intent, phrases in INTENT_HINTS.get(self._runtime_domain, {}).items():
            matches = [
                match
                for phrase in phrases
                for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", lowered)
            ]
            if any(
                not self._mention_denies_authorization(
                    lowered,
                    match.start(),
                    match.end(),
                    information_only=intent in write_intents,
                )
                for match in matches
            ):
                result.add(intent)
        return result

    def _bm25(
        self,
        query_counts: Counter[str],
        document_tokens: Sequence[str],
        document_frequency: Mapping[str, int],
        average_length: float,
        total_documents: int,
    ) -> float:
        document_counts = Counter(document_tokens)
        document_length = sum(document_counts.values())
        result = 0.0
        for token, query_frequency in query_counts.items():
            frequency = document_counts.get(token, 0)
            if not frequency:
                continue
            df = document_frequency.get(token, 0)
            inverse_frequency = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.4 * (
                0.25 + 0.75 * document_length / max(average_length, 1.0)
            )
            result += inverse_frequency * frequency * 2.4 / denominator * min(query_frequency, 2)
        return result

    def _semantic_components(
        self,
        query: str,
        index: int,
        card: Mapping[str, Any],
        intents: set[str],
    ) -> tuple[float, float, float, float]:
        query_counts = Counter(_tokens(query))
        lexical = self._bm25(
            query_counts,
            card.get("tokens", []),
            self._document_frequency,
            self._avg_len,
            len(self._cards),
        )
        query_ngrams = _char_ngrams(query)
        character = len(query_ngrams & self._card_ngrams[index]) / max(len(query_ngrams), 1)
        family_parts = set(str(card.get("family", "")).split("+"))
        intent = float(len(intents & family_parts))
        semantic = lexical + 8.0 * character + 1.8 * intent
        return semantic, lexical, character, intent

    def _field_score(self, query: str, index: int) -> float:
        return self._bm25(
            Counter(_tokens(query)),
            self._field_tokens[index],
            self._field_df,
            self._field_avg_len,
            len(self._cards),
        )

    def _base_quality(self, card: Mapping[str, Any]) -> float:
        support = math.log1p(max(0, int(card.get("support", 0)))) / max(
            math.log1p(max(self._base_support_max, 1)), 1.0
        )
        conformance = max(0.0, min(1.0, _float(card.get("mean_fitness"), 0.0)))
        quality = max(0.0, min(1.0, _float(card.get("quality"), 0.0)))
        return 0.30 * support + 0.40 * conformance + 0.30 * quality

    @staticmethod
    def _prior_value(value: Any) -> tuple[float | None, int | None]:
        if isinstance(value, Mapping):
            raw = value.get("value", value.get("utility", value.get("mean")))
            exposures = value.get("exposures", value.get("count", value.get("support")))
            parsed = _float(raw, math.nan)
            return (parsed if math.isfinite(parsed) else None, int(exposures) if exposures is not None else None)
        parsed = _float(value, math.nan)
        return (parsed if math.isfinite(parsed) else None, None)

    def _state_keys(self, context: Mapping[str, Any], family: str = "") -> list[str]:
        intents = "+".join(context.get("intents", ())) or "unknown"
        family_or_intents = family or intents
        phase = str(context.get("phase", "read"))
        observed_tools = tuple(context.get("observed_tools", ()))
        tools = ">".join(observed_tools) or "none"
        canonical_tools = "+".join(observed_tools) or "none"
        # Runtime phases intentionally require no knowledge of the next action:
        # no tools=read, reads but no writes=prewrite, any write=postwrite.
        # ``read`` is retained after ``prewrite`` only as a compatibility lookup
        # for already-built v2 artifacts whose offline labeller used next-tool
        # information.  Newly built artifacts should use the first key.
        compatible_phases = [phase, "read"] if phase == "prewrite" else [phase]
        keys: list[str] = []
        for candidate_phase in compatible_phases:
            keys.extend(
                [
                    f"{self._runtime_domain}|{family_or_intents}|{canonical_tools}|{candidate_phase}",
                    f"{family_or_intents}|{canonical_tools}|{candidate_phase}",
                    f"{self._runtime_domain}|{intents}|{candidate_phase}|{tools}",
                    f"{intents}|{candidate_phase}|{tools}",
                    f"{intents}|{candidate_phase}",
                    candidate_phase,
                ]
            )
        return list(dict.fromkeys([*keys, "*"]))

    def _utility_score(
        self, card: Mapping[str, Any], sidecar: Mapping[str, Any], context: Mapping[str, Any]
    ) -> float:
        utility = sidecar.get("utility") if isinstance(sidecar.get("utility"), Mapping) else {}
        domain_prior, _ = self._prior_value(utility.get("domain_prior"))
        if domain_prior is None:
            domain_prior = 0.5
        minimum = int(self._thresholds["min_utility_exposures"])
        card_prior, card_exposures = self._prior_value(
            {
                "value": utility.get("card_prior"),
                "exposures": utility.get("scored_exposures"),
            }
        )
        if card_prior is None or card_exposures is None or card_exposures < minimum:
            card_prior = domain_prior
        state_prior = card_prior
        priors = utility.get("state_priors")
        if not isinstance(priors, Mapping):
            priors = sidecar.get("state_priors")
        if not isinstance(priors, Mapping):
            top_priors = self._router.get("state_priors")
            if isinstance(top_priors, Mapping):
                card_priors = top_priors.get(str(card.get("id", "")), top_priors)
                priors = card_priors if isinstance(card_priors, Mapping) else {}
            else:
                priors = {}
        if self._router_stage == "C":
            for key in self._state_keys(context, str(card.get("family", ""))):
                if key not in priors:
                    continue
                value, exposures = self._prior_value(priors[key])
                if value is not None and (exposures is None or exposures >= minimum):
                    state_prior = value
                    break

        quality_delta = self._base_quality(card) - self._base_quality_mean
        card_delta = max(-1.0, min(1.0, card_prior - domain_prior))
        state_delta = max(-1.0, min(1.0, state_prior - card_prior))
        raw = quality_delta + card_delta + state_delta
        cap = abs(self._thresholds["utility_cap"])
        return max(-cap, min(cap, raw))

    def _write_tools(self, sidecar: Mapping[str, Any]) -> tuple[str, ...]:
        contract = self._contract(sidecar)
        values = _text_list(contract.get("authorized_writes"))
        if not values:
            values = _text_list(sidecar.get("write_tools"))
        return tuple(dict.fromkeys(value.split("(", 1)[0].strip() for value in values if value.strip()))

    def _read_tools(self, sidecar: Mapping[str, Any]) -> tuple[str, ...]:
        contract = self._contract(sidecar)
        values = _text_list(contract.get("required_reads"))
        return tuple(dict.fromkeys(value.split("(", 1)[0].strip() for value in values if value.strip()))

    def _risk_score(
        self,
        authorization_text: str,
        card: Mapping[str, Any],
        sidecar: Mapping[str, Any],
        intents: set[str],
    ) -> tuple[float, tuple[str, ...]]:
        # ``authorization_text`` is the latest effective user intent only.  In
        # stage C the retrieval query also carries recent tool names for trace
        # and State-Q lookup; those observations are evidence, never consent.
        _ = card, intents
        writes = self._write_tools(sidecar)
        if not writes:
            return 0.0, ()
        extra = [tool for tool in writes if not self._write_is_authorized(authorization_text, tool)]
        return len(extra) / max(len(writes), 1), tuple(extra)

    def _trace_score(self, sidecar: Mapping[str, Any], context: Mapping[str, Any]) -> float:
        observed = tuple(context.get("observed_tools", ()))
        if not observed:
            return 0.0
        expected = self._read_tools(sidecar) + self._write_tools(sidecar)
        if not expected:
            return 0.0
        overlap = len(set(observed) & set(expected)) / max(len(set(observed)), 1)
        lcs = _lcs_length(observed, expected) / max(len(observed), 1)
        return 0.5 * overlap + 0.5 * lcs

    def _query_obligation_needs(self, text: str) -> set[str]:
        """Return only obligations explicitly grounded in the user text.

        Card prose is intentionally not consulted here.  Otherwise every
        overlapping prohibition in the top-12 pool becomes a synthetic query
        need and turns the adaptive selector back into a fixed three-card
        injector.
        """

        lowered = text.lower()
        words = _normalised_words(lowered) - _COVERAGE_STOPWORDS
        result: set[str] = set()
        disclosure_request = bool(
            words
            & {
                "compare",
                "confirm",
                "disclose",
                "explain",
                "how",
                "quote",
                "show",
                "state",
                "tell",
                "what",
                "whether",
            }
        ) or "?" in text
        strong_disclosure = {"comparison", "deadline", "eligibility", "fee", "fees", "total"}
        for topic, topic_words in _DISCLOSURE_TOPIC_WORDS.items():
            if words & topic_words and (disclosure_request or bool(words & strong_disclosure)):
                result.add(f"required_disclosures:topic:{topic}")

        write_vocabulary = {
            token
            for intent, tools in _INTENT_WRITE_TOOLS.get(self._runtime_domain, {}).items()
            for token in (
                *(_normalised_words(intent.replace("_", " "))),
                *(word for tool in tools for word in _normalised_words(tool.replace("_", " "))),
                *(
                    word
                    for phrase in INTENT_HINTS.get(self._runtime_domain, {}).get(intent, ())
                    for word in _normalised_words(phrase)
                ),
            )
        }
        generic_negative_words = {
            "action",
            "actions",
            "anything",
            "call",
            "else",
            "execute",
            "just",
            "make",
            "must",
            "never",
            "only",
            "perform",
            "use",
            "using",
            "without",
        }
        for clause in re.split(r"[;.!?\n]", lowered):
            match = re.search(
                r"\b(?:do\s+not|don't|never|must\s+not|without|avoid|refrain\s+from)\b(.*)",
                clause,
            )
            if not match:
                continue
            topics = (
                _normalised_words(match.group(1))
                - _COVERAGE_STOPWORDS
                - generic_negative_words
                - write_vocabulary
            )
            if topics:
                signature = "+".join(sorted(topics)[:4])
                result.add(f"prohibitions:clause:{signature}")
        return result

    def _coverage(
        self,
        query: str,
        card: Mapping[str, Any],
        sidecar: Mapping[str, Any],
        intents: set[str],
    ) -> frozenset[str]:
        explicit = _text_list(sidecar.get("coverage"))
        family_parts = set(str(card.get("family", "")).split("+"))
        result = set(explicit) | (family_parts & intents)
        contract = self._contract(sidecar)
        disclosures = _normalised_words(" ".join(_text_list(contract.get("required_disclosures"))))
        prohibitions = _normalised_words(" ".join(_text_list(contract.get("prohibitions"))))
        for need in self._query_obligation_needs(query):
            if need.startswith("required_disclosures:topic:"):
                topic = need.rsplit(":", 1)[-1]
                if disclosures & _DISCLOSURE_TOPIC_WORDS.get(topic, frozenset()):
                    result.add(need)
            elif need.startswith("prohibitions:clause:"):
                topic_words = set(need.rsplit(":", 1)[-1].split("+"))
                if prohibitions & topic_words:
                    result.add(need)
        if not result:
            result.add(f"family:{card.get('family', '')}")
        return frozenset(result)

    def _rank_candidates(self, query: str, context: Mapping[str, Any]) -> list[_Candidate]:
        intents = set(context.get("intents", ())) or self._intent_matches(query)
        authorization_text = str(context.get("intent_text") or query)
        semantic_candidates: list[_Candidate] = []
        for index, card in enumerate(self._cards):
            semantic, lexical, character, intent = self._semantic_components(
                query, index, card, intents
            )
            semantic_candidates.append(
                _Candidate(
                    index=index,
                    card=card,
                    sidecar=self._router_cards[str(card.get("id", ""))],
                    semantic=semantic,
                    lexical=lexical,
                    character=character,
                    intent=intent,
                )
            )
        semantic_candidates.sort(key=lambda item: (-item.semantic, item.card_id))
        pool_size = min(int(self._thresholds["candidate_pool"]), 12, len(semantic_candidates))
        pool = semantic_candidates[:pool_size]
        for candidate in pool:
            candidate.field_score = self._field_score(query, candidate.index)
            candidate.utility = self._utility_score(candidate.card, candidate.sidecar, context)
            candidate.risk, candidate.extra_writes = self._risk_score(
                authorization_text, candidate.card, candidate.sidecar, intents
            )
            candidate.trace = self._trace_score(candidate.sidecar, context)
            sticky = (
                self._thresholds["stickiness"]
                if self._router_stage == "C"
                and candidate.card_id in self._active_card_ids
                and tuple(context.get("intent_signature", ())) == self._active_intent_signature
                else 0.0
            )
            candidate.final = (
                candidate.semantic
                + self._weights["field"] * candidate.field_score
                + self._weights["utility"] * candidate.utility
                - self._weights["risk"] * candidate.risk
                + self._weights["trace"] * candidate.trace
                + sticky
            )
            candidate.coverage = self._coverage(
                authorization_text, candidate.card, candidate.sidecar, intents
            )
            candidate.diagnostics = {"sticky": sticky}
        return pool

    # ------------------------------------------------------------------
    # True greedy MMR and adaptive cardinality
    # ------------------------------------------------------------------
    @staticmethod
    def _candidate_similarity(left: _Candidate, right: _Candidate) -> float:
        text_similarity = _jaccard(
            _normalised_words(str(left.card.get("search_text", ""))),
            _normalised_words(str(right.card.get("search_text", ""))),
        )
        left_tools = set(left.card.get("observed_tools", []))
        right_tools = set(right.card.get("observed_tools", []))
        return max(text_similarity, _jaccard(left_tools, right_tools))

    def _query_needs(
        self,
        query: str,
        context: Mapping[str, Any],
        pool: Sequence[_Candidate],
    ) -> set[str]:
        _ = pool
        intent_text = str(context.get("intent_text") or query)
        return self._positive_intents(intent_text) | self._query_obligation_needs(intent_text)

    def _has_relevance_evidence(
        self,
        query: str,
        context: Mapping[str, Any],
        anchor: _Candidate,
    ) -> bool:
        if context.get("intents") or self._intent_matches(query):
            return True
        query_words = _normalised_words(query) - _RELEVANCE_STOPWORDS
        if not query_words:
            return False
        # Generic procedural constraints often contain words such as "tell",
        # "confirm", and "never".  They are useful for reranking but cannot
        # establish relevance by themselves.  Abstention is grounded only in
        # the card trigger/scope/keywords when no domain intent matched.
        field_words = self._relevance_field_words[anchor.index]
        return bool(query_words & field_words)

    @staticmethod
    def _is_compound_query(query: str, needs: set[str]) -> bool:
        lowered = f" {query.lower()} "
        connectors = (" and ", " also ", " plus ", " then ", ", and ", ";")
        return len(needs) >= 2 or any(connector in lowered for connector in connectors)

    def _eligible_candidate(
        self,
        candidate: _Candidate,
        selected: Sequence[_Candidate],
        family_counts: Counter[str],
    ) -> bool:
        if family_counts[candidate.family] >= int(self._thresholds["same_family_limit"]):
            return False
        duplicate_threshold = self._thresholds["duplicate_jaccard"]
        candidate_words = _normalised_words(str(candidate.card.get("search_text", "")))
        for prior in selected:
            prior_words = _normalised_words(str(prior.card.get("search_text", "")))
            if _jaccard(candidate_words, prior_words) > duplicate_threshold:
                return False
        return True

    def _select_candidates(
        self,
        query: str,
        context: Mapping[str, Any],
        pool: Sequence[_Candidate],
        limit: int,
    ) -> list[_Candidate]:
        if not pool or limit <= 0:
            return []
        if pool[0].semantic < self._thresholds["min_relevance"]:
            return []
        if not self._has_relevance_evidence(query, context, pool[0]):
            return []
        limit = min(limit, int(self._thresholds["max_cards"]), 3)
        needs = self._query_needs(query, context, pool)
        compound = self._is_compound_query(query, needs)
        semantic_margin = (
            pool[0].semantic - pool[1].semantic if len(pool) > 1 else math.inf
        )
        # A decisive raw-semantic winner is immutable.  The calibrated
        # near-tie window may be narrower than this hard 2.0 safety boundary,
        # but no component prior can ever dislodge a winner above it.
        forced_anchor = pool[0] if semantic_margin > _HARD_ANCHOR_MARGIN else None

        selected: list[_Candidate] = []
        remaining = list(pool)
        family_counts: Counter[str] = Counter()
        covered: set[str] = set()
        primary_final: float | None = None
        while remaining and len(selected) < limit:
            if selected and (not compound or needs.issubset(covered)):
                break
            choices: list[tuple[float, _Candidate]] = []
            for candidate in remaining:
                if not self._eligible_candidate(candidate, selected, family_counts):
                    continue
                if (
                    not selected
                    and forced_anchor is None
                    and pool[0].semantic - candidate.semantic
                    > self._thresholds["near_tie"]
                ):
                    # Component priors may reorder only the raw-semantic
                    # near-tie set, never promote a distant top-12 candidate
                    # into the primary slot.
                    continue
                if selected and not ((set(candidate.coverage) - covered) & needs):
                    # Do not let a high-scoring but irrelevant card occupy the
                    # next greedy slot and hide a lower-ranked complementary
                    # workflow.  It remains a candidate for diagnostics only.
                    continue
                redundancy = max(
                    (self._candidate_similarity(candidate, prior) for prior in selected),
                    default=0.0,
                )
                adjusted = candidate.final - self._weights["mmr"] * redundancy
                choices.append((adjusted, candidate))
            if not choices:
                break
            if not selected and forced_anchor is not None:
                chosen = forced_anchor
                adjusted = next(
                    (score for score, item in choices if item.card_id == chosen.card_id),
                    chosen.final,
                )
            else:
                adjusted, chosen = max(choices, key=lambda pair: (pair[0], pair[1].semantic, pair[1].card_id))

            if selected:
                new_coverage = set(chosen.coverage) - covered
                relevant_new = new_coverage & needs
                assert relevant_new
                if adjusted < self._thresholds["min_secondary_score"]:
                    break
                assert primary_final is not None
                # A directly requested, still-uncovered intent earns a slot on
                # absolute evidence.  Relative-to-primary pruning applies only
                # to inferred coverage, otherwise an emphatically worded first
                # action can hide the user's tersely worded second action.
                explicit_intent_coverage = bool(relevant_new & set(context.get("intents", ())))
                if (
                    not explicit_intent_coverage
                    and adjusted < primary_final * self._thresholds["secondary_relative_score"]
                ):
                    break
            chosen.adjusted = adjusted
            selected.append(chosen)
            remaining = [item for item in remaining if item.card_id != chosen.card_id]
            family_counts[chosen.family] += 1
            covered.update(chosen.coverage)
            if primary_final is None:
                primary_final = max(chosen.final, 1e-9)

            # Default cardinality is one.  More slots are earned only by an
            # explicit compound intent and distinct uncovered query need.
            if len(selected) >= int(self._thresholds["default_cards"]) and not compound:
                break
        return selected

    # ------------------------------------------------------------------
    # Typed rendering and public retrieval contract
    # ------------------------------------------------------------------
    def _explicit_write_tools(
        self, query: str, candidate: _Candidate, context: Mapping[str, Any]
    ) -> tuple[str, ...]:
        authorization_text = str(context.get("intent_text") or query)
        return tuple(
            tool
            for tool in self._write_tools(candidate.sidecar)
            if self._write_is_authorized(authorization_text, tool)
        )

    def _explicit_write_requested(
        self, query: str, candidate: _Candidate, context: Mapping[str, Any]
    ) -> bool:
        """Compatibility predicate; authorization is computed per write tool."""

        return bool(self._explicit_write_tools(query, candidate, context))

    @staticmethod
    def _section(name: str, values: Sequence[str]) -> list[str]:
        if not values:
            return []
        return [f"{name}:", *(f"- {value}" for value in values)]

    def _constructed_typed_text(
        self,
        candidate: _Candidate,
        authorized_writes: Sequence[str] | None,
    ) -> str:
        contract = self._contract(candidate.sidecar)
        lines = [f"WORKFLOW MEMORY [{candidate.card_id}]"]
        lines += self._section("WHEN", _text_list(contract.get("trigger", contract.get("triggers"))))
        lines += self._section("SCOPE", _scope_text_list(contract.get("scope")))
        lines += self._section("READ", _text_list(contract.get("required_reads")))
        lines += self._section("DECIDE", _text_list(contract.get("decision_rules")))
        write_values = _text_list(contract.get("authorized_writes"))
        if authorized_writes is not None:
            allowed = set(authorized_writes)
            write_values = [
                value for value in write_values if value.split("(", 1)[0].strip() in allowed
            ]
        lines += self._section("WRITE", write_values)
        lines += self._section("VERIFY", _text_list(contract.get("verification_rules")))
        lines += self._section("SAY", _text_list(contract.get("required_disclosures")))
        lines += self._section("NEVER", _text_list(contract.get("prohibitions")))
        return "\n".join(lines)

    def _secondary_with_writes(
        self,
        candidate: _Candidate,
        supplied: str,
        authorized_writes: Sequence[str],
    ) -> str:
        if not authorized_writes:
            return supplied
        contract = self._contract(candidate.sidecar)
        allowed = set(authorized_writes)
        write_values = [
            value
            for value in _text_list(contract.get("authorized_writes"))
            if value.split("(", 1)[0].strip() in allowed
        ]
        if not write_values:
            return supplied
        lines = supplied.splitlines()
        insertion = next(
            (index for index, line in enumerate(lines) if line.strip() in {"SAY:", "NEVER:"}),
            len(lines),
        )
        packed = "\n".join(
            [*lines[:insertion], *self._section("WRITE", write_values), *lines[insertion:]]
        )
        # The compiler-approved secondary contains every SAY/NEVER constraint.
        # If the optional write rows would exceed the injection cap, retain that
        # complete constraints-only form instead of slicing off its tail.
        return packed if len(packed) <= _RENDER_CHAR_LIMIT else supplied

    def _render_card(
        self,
        candidate: _Candidate,
        *,
        role: str,
        query: str,
        context: Mapping[str, Any],
    ) -> str:
        if self._router_stage == "A":
            text_key = {"hybrid": "text", "awm_only": "awm_text", "process_only": "process_text"}[
                self.mode
            ]
            return str(candidate.card.get(text_key, candidate.card.get("text", "")))[:2200]
        compiler = candidate.sidecar.get("compiler")
        compiler_valid = not isinstance(compiler, Mapping) or compiler.get("valid") is not False
        primary_text = candidate.sidecar.get("primary_text")
        secondary_text = candidate.sidecar.get("secondary_text")
        if compiler_valid and (
            not isinstance(primary_text, str)
            or not isinstance(secondary_text, str)
            or len(primary_text.strip()) > _RENDER_CHAR_LIMIT
            or len(secondary_text.strip()) > _RENDER_CHAR_LIMIT
        ):
            compiler_valid = False
        if not compiler_valid:
            text_key = {"hybrid": "text", "awm_only": "awm_text", "process_only": "process_text"}[
                self.mode
            ]
            return str(candidate.card.get(text_key, candidate.card.get("text", "")))[:_RENDER_CHAR_LIMIT]

        authorized_writes = self._explicit_write_tools(query, candidate, context)
        supplied = candidate.sidecar.get("primary_text" if role == "primary" else "secondary_text")
        if isinstance(supplied, str) and supplied.strip():
            supplied = supplied.strip()
            if role == "primary":
                return supplied
            return self._secondary_with_writes(candidate, supplied, authorized_writes)
        constructed = self._constructed_typed_text(
            candidate,
            authorized_writes=None if role == "primary" else authorized_writes,
        )
        if len(constructed) <= _RENDER_CHAR_LIMIT:
            return constructed
        text_key = {"hybrid": "text", "awm_only": "awm_text", "process_only": "process_text"}[
            self.mode
        ]
        return str(candidate.card.get(text_key, candidate.card.get("text", "")))[:_RENDER_CHAR_LIMIT]

    def _record_telemetry(self, telemetry: dict[str, Any]) -> None:
        self.last_retrieval_telemetry = telemetry
        self.retrieval_telemetry.append(telemetry)

    def _candidate_retrieval(self, query: str, top_k: int) -> list[str]:
        context = self._context_for_query(query)
        limit = min(max(0, int(top_k)), self.retrieve_learnings_top_k, 3)
        query_key = (
            tuple(context.get("intent_signature", ())),
            tuple(context.get("observed_tools", ())),
            context.get("phase"),
        )
        if (
            self._router_stage == "C"
            and query_key == self._active_query_key
            and self._active_candidates
        ):
            selected = list(self._active_candidates[:limit])
            rendered = [
                self._render_card(
                    candidate,
                    role="primary" if index == 0 else "secondary",
                    query=query,
                    context=context,
                )
                for index, candidate in enumerate(selected)
            ]
            self._record_telemetry(
                {
                    "mode": self._router_mode,
                    "stage": self._router_stage,
                    "domain": self._runtime_domain,
                    "phase": context.get("phase"),
                    "fallback_reason": None,
                    "reason": "flowswitch_sticky_reuse",
                    "semantic_margin": None,
                    "final_margin": None,
                    "selected": [
                        {
                            "card_id": candidate.card_id,
                            "family": candidate.family,
                            "slot": index + 1,
                            "role": "primary" if index == 0 else "secondary",
                            "adjusted_score": round(candidate.adjusted, 6),
                            "render_chars": len(rendered[index]),
                        }
                        for index, candidate in enumerate(selected)
                    ],
                    "candidates": [],
                    "injected_chars": sum(map(len, rendered)),
                }
            )
            return rendered
        pool = self._rank_candidates(query, context)
        selected = self._select_candidates(query, context, pool, limit)
        rendered = [
            self._render_card(
                candidate,
                role="primary" if index == 0 else "secondary",
                query=query,
                context=context,
            )
            for index, candidate in enumerate(selected)
        ]
        self._active_intent_signature = tuple(context.get("intent_signature", ()))
        self._active_tool_signature = tuple(context.get("observed_tools", ()))
        self._active_card_ids = tuple(candidate.card_id for candidate in selected)
        self._active_candidates = tuple(selected)
        self._active_query_key = query_key
        selected_details = []
        for index, (candidate, text) in enumerate(zip(selected, rendered)):
            selected_details.append(
                {
                    "card_id": candidate.card_id,
                    "family": candidate.family,
                    "slot": index + 1,
                    "role": "primary" if index == 0 else "secondary",
                    "adjusted_score": round(candidate.adjusted, 6),
                    "render_chars": len(text),
                }
            )
        self._record_telemetry(
            {
                "mode": self._router_mode,
                "stage": self._router_stage,
                "domain": self._runtime_domain,
                "phase": context.get("phase"),
                "fallback_reason": None,
                "reason": (
                    "selected"
                    if selected
                    else (
                        "abstain_low_relevance"
                        if pool and pool[0].semantic < self._thresholds["min_relevance"]
                        else "adaptive_abstain"
                    )
                ),
                "semantic_margin": round(pool[0].semantic - pool[1].semantic, 6)
                if len(pool) > 1
                else None,
                "final_margin": round(pool[0].final - pool[1].final, 6)
                if len(pool) > 1
                else None,
                "selected": selected_details,
                "candidates": [
                    {
                        "card_id": candidate.card_id,
                        "family": candidate.family,
                        "semantic": round(candidate.semantic, 6),
                        "lexical": round(candidate.lexical, 6),
                        "character": round(candidate.character, 6),
                        "intent": round(candidate.intent, 6),
                        "field": round(candidate.field_score, 6),
                        "utility": round(candidate.utility, 6),
                        "risk": round(candidate.risk, 6),
                        "trace": round(candidate.trace, 6),
                        "final": round(candidate.final, 6),
                        "extra_writes": list(candidate.extra_writes),
                    }
                    for candidate in pool
                ],
                "injected_chars": sum(map(len, rendered)),
            }
        )
        return rendered

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        if not self._router_enabled:
            result = _Parent.retrieve_learnings(self, query, top_k=top_k)
            self._record_telemetry(
                {
                    "mode": self._router_mode,
                    "stage": self._router_stage,
                    "domain": self._runtime_domain,
                    "fallback_reason": self._router_reason,
                    "reason": "parent_fallback",
                    "semantic_margin": None,
                    "final_margin": None,
                    "selected": [],
                    "injected_chars": sum(map(len, result)),
                }
            )
            return result
        if not query.strip() or not self._cards:
            self._record_telemetry(
                {
                    "mode": self._router_mode,
                    "stage": self._router_stage,
                    "domain": self._runtime_domain,
                    "fallback_reason": None,
                    "reason": "empty_query",
                    "semantic_margin": None,
                    "final_margin": None,
                    "selected": [],
                    "injected_chars": 0,
                }
            )
            return []
        if self._router_mode == "shadow":
            self._candidate_retrieval(query, top_k)
            candidate_telemetry = dict(self.last_retrieval_telemetry)
            if self.retrieval_telemetry:
                self.retrieval_telemetry.pop()
            result = _Parent.retrieve_learnings(self, query, top_k=top_k)
            telemetry = candidate_telemetry
            telemetry["fallback_reason"] = "shadow_parent_result"
            telemetry["reason"] = "shadow_parent_result"
            telemetry["shadow_parent_chars"] = sum(map(len, result))
            self._record_telemetry(telemetry)
            return result
        return self._candidate_retrieval(query, top_k)

    @staticmethod
    def _safe_telemetry_event(event: Mapping[str, Any]) -> dict[str, Any]:
        """Return an explicit allow-list projection safe for trajectory files."""
        selected = []
        for item in event.get("selected", []):
            if not isinstance(item, Mapping):
                continue
            selected.append(
                {
                    key: item.get(key)
                    for key in (
                        "card_id",
                        "family",
                        "slot",
                        "role",
                        "adjusted_score",
                        "render_chars",
                    )
                }
            )
        candidates = []
        for item in event.get("candidates", []):
            if not isinstance(item, Mapping):
                continue
            candidates.append(
                {
                    key: item.get(key)
                    for key in (
                        "card_id",
                        "family",
                        "semantic",
                        "lexical",
                        "character",
                        "intent",
                        "field",
                        "utility",
                        "risk",
                        "trace",
                        "final",
                        "extra_writes",
                    )
                }
            )
        return {
            "mode": event.get("mode"),
            "stage": event.get("stage"),
            "domain": event.get("domain"),
            "phase": event.get("phase"),
            "reason": event.get("reason"),
            "fallback_reason": event.get("fallback_reason"),
            "semantic_margin": event.get("semantic_margin"),
            "final_margin": event.get("final_margin"),
            "selected": selected,
            "candidates": candidates,
            "injected_chars": event.get("injected_chars", 0),
            "shadow_parent_chars": event.get("shadow_parent_chars"),
        }

    def ingest_trajectory(self, trajectory: Any) -> None:
        """Attach safe retrieval attribution before STATE-Bench writes a run.

        Only IDs, numeric component scores, selection reasons and character
        counts are persisted.  Query/user text and task-or-judge fields are
        intentionally impossible to reach through the allow-list projection.
        """
        _Parent.ingest_trajectory(self, trajectory)
        if not self._router_enabled:
            # A disabled/malformed/unpromoted router is a true parent fallback:
            # do not make the serialized trajectory differ merely because the
            # candidate class was instantiated.
            return
        metadata = getattr(trajectory, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(trajectory, "metadata", metadata)
        events = [self._safe_telemetry_event(event) for event in self.retrieval_telemetry]
        metadata["workflow_router"] = {
            "schema_version": "1.0.0",
            "mode": self._router_mode,
            "stage": self._router_stage,
            "domain": self._runtime_domain,
            "router_enabled": self._router_enabled,
            "router_reason": self._router_reason,
            "events": events,
            "last": events[-1] if events else None,
        }


__all__ = ["RiskAwareProcessWorkflowMemoryAgent"]
