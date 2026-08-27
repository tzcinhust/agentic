"""Domain-agnostic state-transition contracts for tool-using agents."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolContract:
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()
    invalidates: frozenset[str] = frozenset()
    requires_fresh: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ToolObservation:
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    fields: frozenset[str]
    scope: dict[str, frozenset[str]]
    values: dict[str, tuple[Any, ...]]
    preview: bool = False


IDENTIFIER_KEYS = frozenset(
    {
        "booking_id",
        "car_id",
        "customer_id",
        "flight_id",
        "hotel_id",
        "item_id",
        "order_id",
        "product_id",
        "rental_id",
        "reservation_id",
        "user_id",
        "variant_id",
        "warranty_id",
    }
)


def _collect_values(value: Any) -> dict[str, tuple[Any, ...]]:
    collected: dict[str, list[Any]] = {}

    def visit(current: Any, key: str = "") -> None:
        if isinstance(current, dict):
            for child_key, child in current.items():
                visit(child, str(child_key))
        elif isinstance(current, list):
            for child in current:
                visit(child, key)
        elif current is not None and key:
            collected.setdefault(key, []).append(current)

    visit(value)
    return {key: tuple(dict.fromkeys(values)) for key, values in collected.items()}


def _identifier_scope(
    values: dict[str, tuple[Any, ...]],
) -> dict[str, frozenset[str]]:
    scope: dict[str, set[str]] = {}
    for key, items in values.items():
        if key not in IDENTIFIER_KEYS and not key.endswith("_ids"):
            continue
        normalized = key[:-1] if key.endswith("_ids") else key
        scope.setdefault(normalized, set()).update(map(str, items))
    return {key: frozenset(items) for key, items in scope.items()}


def _scope_compatible(
    evidence: dict[str, frozenset[str]], candidate: dict[str, frozenset[str]]
) -> bool:
    if not evidence or not candidate:
        return True
    shared_keys = set(evidence) & set(candidate)
    transaction_keys = shared_keys & {
        "booking_id",
        "order_id",
        "rental_id",
        "reservation_id",
        "warranty_id",
    }
    if transaction_keys:
        return any(evidence[key] & candidate[key] for key in transaction_keys)
    if shared_keys:
        return any(evidence[key] & candidate[key] for key in shared_keys)
    return True


def _mutation_affects(
    observation: ToolObservation,
    mutation_scope: dict[str, frozenset[str]],
) -> bool:
    if not observation.scope or not mutation_scope:
        return True
    if not observation.preview:
        return _scope_compatible(observation.scope, mutation_scope)
    shared_keys = set(observation.scope) & set(mutation_scope)
    if not shared_keys:
        return True
    return all(
        observation.scope[key] & mutation_scope[key] for key in shared_keys
    )


@dataclass
class TransitionLedger:
    fresh: set[str]
    stale: set[str]
    failed: set[str]
    observations: list[ToolObservation] = field(default_factory=list)
    last_write: str | None = None
    last_write_result: Any = None

    @classmethod
    def empty(cls) -> TransitionLedger:
        return cls(fresh=set(), stale=set(), failed=set())

    def observe(
        self,
        tool_name: str,
        contract: ToolContract,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        *,
        preview: bool = False,
    ) -> None:
        arguments = arguments or {}
        mutation_scope = _identifier_scope(_collect_values(arguments))
        fields = contract.reads | contract.writes
        if contract.invalidates:
            retained = []
            for item in self.observations:
                affected = _mutation_affects(item, mutation_scope)
                if not affected or (item.preview and item.tool_name != tool_name):
                    retained.append(item)
                    continue
                if item.preview:
                    continue
                valid_fields = item.fields - contract.invalidates
                if valid_fields:
                    retained.append(replace(item, fields=valid_fields))
            self.observations = retained
        self.fresh.update(fields)
        self.fresh.difference_update(contract.invalidates)
        self.stale.update(contract.invalidates)
        self.stale.difference_update(fields)
        self.failed.difference_update(fields)
        values = _collect_values({"arguments": arguments, "result": result})
        self.observations.append(
            ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                fields=fields,
                scope=_identifier_scope(values),
                values=values,
                preview=preview,
            )
        )
        if contract.writes:
            self.last_write = tool_name
            self.last_write_result = result

    def observe_failure(self, tool_name: str, contract: ToolContract) -> None:
        """Invalidate failed reads without pretending a failed write happened."""
        if contract.reads:
            retained = []
            for item in self.observations:
                valid_fields = item.fields - contract.reads
                if valid_fields:
                    retained.append(replace(item, fields=valid_fields))
            self.observations = retained
        self.fresh.difference_update(contract.reads)
        self.stale.update(contract.reads)
        self.failed.update(contract.reads)

    def missing(
        self,
        contract: ToolContract,
        candidate_arguments: dict[str, Any] | None = None,
        *,
        entity_aware: bool = True,
    ) -> set[str]:
        missing = set(contract.requires_fresh) - self.fresh
        if not entity_aware or not candidate_arguments:
            return missing
        candidate_scope = _identifier_scope(_collect_values(candidate_arguments))
        if not candidate_scope:
            return missing
        missing = set()
        for required in contract.requires_fresh:
            evidence = [item for item in self.observations if required in item.fields]
            if not evidence or not any(
                _scope_compatible(item.scope, candidate_scope) for item in evidence
            ):
                missing.add(required)
        return missing

    def has_tool_evidence(
        self, tool_name: str, candidate_arguments: dict[str, Any], *, entity_aware: bool
    ) -> bool:
        candidate_scope = _identifier_scope(_collect_values(candidate_arguments))
        matches = [item for item in self.observations if item.tool_name == tool_name]
        if not matches:
            return False
        if not entity_aware or not candidate_scope:
            return True
        return any(_scope_compatible(item.scope, candidate_scope) for item in matches)

    def latest_preview(
        self, tool_name: str, candidate_arguments: dict[str, Any], *, entity_aware: bool
    ) -> ToolObservation | None:
        candidate_scope = _identifier_scope(_collect_values(candidate_arguments))
        for item in reversed(self.observations):
            if item.tool_name != tool_name or not item.preview:
                continue
            if not entity_aware or not candidate_scope or _scope_compatible(
                item.scope, candidate_scope
            ):
                return item
        return None


# Tool names are only mapped to abstract state fields.  The checking logic
# below is shared across travel, shopping, and customer-support workflows.
TOOL_CONTRACTS: dict[str, ToolContract] = {
    "search_products": ToolContract(reads=frozenset({"catalog"})),
    "get_product_details": ToolContract(reads=frozenset({"catalog", "product_identity"})),
    "get_variants": ToolContract(reads=frozenset({"catalog", "product_identity", "variant"})),
    "check_compatibility": ToolContract(reads=frozenset({"compatibility"})),
    "get_cart": ToolContract(
        reads=frozenset({"cart", "cart_total", "promotion", "shipping", "loyalty"})
    ),
    "get_customer_account": ToolContract(reads=frozenset({"customer_account", "loyalty"})),
    "get_promotions": ToolContract(reads=frozenset({"promotion"})),
    "get_policies": ToolContract(reads=frozenset({"policy"})),
    "get_shipping_options": ToolContract(reads=frozenset({"shipping"})),
    "validate_promo": ToolContract(
        reads=frozenset({"promotion", "cart", "cart_total"}),
        requires_fresh=frozenset({"cart", "cart_total"}),
    ),
    "add_to_cart": ToolContract(
        writes=frozenset({"cart"}),
        invalidates=frozenset({"cart_total", "promotion", "shipping", "loyalty"}),
    ),
    "remove_from_cart": ToolContract(
        writes=frozenset({"cart"}),
        invalidates=frozenset({"cart_total", "promotion", "shipping", "loyalty"}),
        requires_fresh=frozenset({"cart"}),
    ),
    "update_cart_item": ToolContract(
        writes=frozenset({"cart"}),
        invalidates=frozenset({"cart_total", "promotion", "shipping", "loyalty"}),
        requires_fresh=frozenset({"cart"}),
    ),
    "apply_promo": ToolContract(
        writes=frozenset({"promotion"}),
        invalidates=frozenset({"cart_total", "shipping", "loyalty"}),
    ),
    "remove_promo": ToolContract(
        writes=frozenset({"promotion"}),
        invalidates=frozenset({"cart_total", "shipping", "loyalty"}),
        requires_fresh=frozenset({"cart", "promotion"}),
    ),
    "redeem_loyalty_points": ToolContract(
        writes=frozenset({"loyalty"}),
        invalidates=frozenset({"cart_total", "shipping"}),
    ),
    "cancel_loyalty_redemption": ToolContract(
        writes=frozenset({"loyalty"}),
        invalidates=frozenset({"cart_total", "shipping"}),
    ),
    "set_shipping_option": ToolContract(
        writes=frozenset({"shipping"}),
        invalidates=frozenset({"cart_total"}),
    ),
    "search_flights": ToolContract(reads=frozenset({"flight_options"})),
    "get_user_details": ToolContract(reads=frozenset({"customer_account", "loyalty"})),
    "get_user_reservations": ToolContract(reads=frozenset({"booking", "hotel_booking", "car_booking"})),
    "get_flight_details": ToolContract(reads=frozenset({"flight_options", "flight"})),
    "get_booking": ToolContract(reads=frozenset({"booking", "seat", "ancillary", "price"})),
    "get_flight_status": ToolContract(reads=frozenset({"flight_status"})),
    "get_itinerary": ToolContract(reads=frozenset({"booking", "itinerary", "price"})),
    "get_seat_map": ToolContract(reads=frozenset({"seat"})),
    "get_ancillaries": ToolContract(reads=frozenset({"ancillary"})),
    "create_booking": ToolContract(
        writes=frozenset({"booking"}),
        invalidates=frozenset({"itinerary", "seat", "ancillary", "price"}),
        requires_fresh=frozenset({"flight_options"}),
    ),
    "update_booking": ToolContract(
        writes=frozenset({"booking"}),
        invalidates=frozenset({"itinerary", "seat", "ancillary", "price"}),
    ),
    "cancel_booking": ToolContract(
        writes=frozenset({"booking"}),
        invalidates=frozenset({"itinerary", "seat", "ancillary", "price"}),
        requires_fresh=frozenset({"booking"}),
    ),
    "book_hotel": ToolContract(
        writes=frozenset({"hotel_booking"}),
        invalidates=frozenset({"hotel_price", "itinerary"}),
        requires_fresh=frozenset({"hotel_options"}),
    ),
    "search_hotels": ToolContract(reads=frozenset({"hotel_options"})),
    "get_hotel_reservation": ToolContract(reads=frozenset({"hotel_booking", "hotel_price"})),
    "cancel_hotel_reservation": ToolContract(
        writes=frozenset({"hotel_booking"}),
        invalidates=frozenset({"hotel_price", "itinerary"}),
        requires_fresh=frozenset({"hotel_booking"}),
    ),
    "book_car_rental": ToolContract(
        writes=frozenset({"car_booking"}),
        invalidates=frozenset({"car_price", "itinerary"}),
        requires_fresh=frozenset({"car_options"}),
    ),
    "search_car_rentals": ToolContract(reads=frozenset({"car_options"})),
    "get_car_rental": ToolContract(reads=frozenset({"car_booking", "car_price"})),
    "cancel_car_rental": ToolContract(
        writes=frozenset({"car_booking"}),
        invalidates=frozenset({"car_price", "itinerary"}),
        requires_fresh=frozenset({"car_booking"}),
    ),
    "get_order": ToolContract(reads=frozenset({"order", "order_status", "eligibility", "inventory"})),
    "get_customer": ToolContract(reads=frozenset({"customer_account", "account_balance"})),
    "get_return_policy": ToolContract(reads=frozenset({"policy", "eligibility"})),
    "get_warranty_status": ToolContract(reads=frozenset({"warranty", "eligibility"})),
    "get_shipping_status": ToolContract(reads=frozenset({"shipping_status", "order_status"})),
    "cancel_order": ToolContract(
        writes=frozenset({"order_status"}),
        invalidates=frozenset({"eligibility", "refund_status", "shipping_status"}),
        requires_fresh=frozenset({"order", "policy"}),
    ),
    "process_return": ToolContract(
        writes=frozenset({"return_status"}),
        invalidates=frozenset({"order_status", "refund_status", "eligibility"}),
        requires_fresh=frozenset({"order", "eligibility", "policy"}),
    ),
    "process_refund": ToolContract(
        writes=frozenset({"refund_status"}),
        invalidates=frozenset({"order_status", "eligibility", "account_balance"}),
        requires_fresh=frozenset({"order", "eligibility", "policy"}),
    ),
    "process_exchange": ToolContract(
        writes=frozenset({"exchange_status"}),
        invalidates=frozenset({"order_status", "eligibility", "inventory"}),
        requires_fresh=frozenset({"order", "eligibility", "policy"}),
    ),
    "process_warranty_claim": ToolContract(
        writes=frozenset({"warranty_claim"}),
        invalidates=frozenset({"order_status", "eligibility", "repair_status"}),
        requires_fresh=frozenset({"order", "warranty", "policy"}),
    ),
}


TWO_STEP_TOOLS = frozenset(
    {
        "update_booking",
        "cancel_booking",
        "cancel_hotel_reservation",
        "cancel_car_rental",
        "process_return",
        "process_refund",
        "cancel_order",
        "process_exchange",
        "process_warranty_claim",
    }
)

POLICY_TOPICS_BY_TOOL = {
    "cancel_order": "cancellation",
    "process_exchange": "exchange",
    "process_refund": "refund",
    "process_return": "return",
    "process_warranty_claim": "warranty",
}

POLICY_TOOL_ALIASES = {
    "refund": frozenset({"get_return_policy"}),
    "return": frozenset({"get_return_policy"}),
}

PREVIEW_VALUE_ALIASES = {
    "amount": ("amount", "refund_amount", "net_refund", "refund"),
    "cash_amount": ("cash_amount", "remaining_cash_payment"),
    "points_used": ("points_used",),
}


class TransitionAwareMemory:
    """Reconstruct and check an entity-scoped ledger from visible tool events."""

    def __init__(
        self,
        contracts: dict[str, ToolContract] | None = None,
        *,
        domain: str | None = None,
        artifact_path: str | Path | None = None,
        learned: bool = True,
        entity_aware: bool = True,
        value_aware: bool = True,
    ) -> None:
        self.contracts = contracts or TOOL_CONTRACTS
        self.domain = domain
        self.entity_aware = entity_aware
        self.value_aware = value_aware
        self.learned_contracts: dict[str, dict[str, Any]] = {}
        configured_path = artifact_path or os.environ.get(
            "STATE_BENCH_TAPM_PATH", "configs/tapm_transition_contracts.json"
        )
        if learned and configured_path and Path(configured_path).exists():
            artifact = json.loads(Path(configured_path).read_text(encoding="utf-8"))
            self.learned_contracts = (
                artifact.get("domains", {}).get(domain or "", {}).get("contracts", {})
            )

    def contract_for(self, tool_name: str) -> ToolContract:
        base = self.contracts.get(tool_name, ToolContract())
        learned = self.learned_contracts.get(tool_name, {})
        required_fields: set[str] = set()
        for item in learned.get("required_tools", []):
            required_tool = str(item.get("tool", ""))
            required_fields.update(
                self.contracts.get(required_tool, ToolContract()).reads
                & base.requires_fresh
            )
        return ToolContract(
            reads=base.reads,
            writes=base.writes,
            invalidates=base.invalidates,
            requires_fresh=base.requires_fresh | frozenset(required_fields),
        )

    def _learned_required_tools(self, tool_name: str) -> list[str]:
        return [
            str(item.get("tool", ""))
            for item in self.learned_contracts.get(tool_name, {}).get("required_tools", [])
            if item.get("tool")
        ]

    def _policy_topic(self, tool_name: str) -> str | None:
        if tool_name in POLICY_TOPICS_BY_TOOL:
            return POLICY_TOPICS_BY_TOOL[tool_name]
        topics = self.learned_contracts.get(tool_name, {}).get("policy_topics", [])
        return str(topics[0]) if len(topics) == 1 else None

    def _preview_required(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        if tool_name == "update_booking":
            return bool(arguments.get("flight_id"))
        if self.learned_contracts.get(tool_name, {}).get("preview_required") is True:
            return True
        return tool_name in TWO_STEP_TOOLS

    @staticmethod
    def _is_preview(
        tool_name: str, arguments: dict[str, Any], result: Any = None
    ) -> bool:
        if isinstance(result, dict):
            status = str(result.get("status", "")).lower()
            if status in {"preview", "quoted", "pending_confirmation"}:
                return True
            if status in {
                "cancelled",
                "completed",
                "exchanged",
                "processed",
                "refunded",
                "returned",
                "success",
                "updated",
            }:
                return False
        if tool_name not in TWO_STEP_TOOLS:
            return False
        if tool_name == "update_booking" and not arguments.get("flight_id"):
            return False
        return not bool(arguments.get("confirm"))

    def observed_contract(
        self, tool_name: str, arguments: dict[str, Any], result: Any = None
    ) -> ToolContract:
        contract = self.contract_for(tool_name)
        if not self._is_preview(tool_name, arguments, result):
            return contract
        return ToolContract(
            reads=contract.reads | contract.requires_fresh,
            requires_fresh=contract.requires_fresh,
        )

    @staticmethod
    def observation_succeeded(result: Any) -> bool:
        if not isinstance(result, dict):
            return True
        if result.get("error") or result.get("success") is False:
            return False
        status = str(result.get("status", "")).lower()
        return status not in {"error", "failed", "rejected"}

    @staticmethod
    def scope_matches(
        evidence_arguments: dict[str, Any],
        evidence_result: Any,
        candidate_arguments: dict[str, Any],
    ) -> bool:
        evidence = _identifier_scope(
            _collect_values(
                {"arguments": evidence_arguments, "result": evidence_result}
            )
        )
        candidate = _identifier_scope(_collect_values(candidate_arguments))
        return _scope_compatible(evidence, candidate)

    def ledger_from_conversation(self, conversation: Iterable[dict[str, Any]]) -> TransitionLedger:
        ledger = TransitionLedger.empty()
        for item in conversation:
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                name = str(call.get("name", ""))
                arguments = call.get("arguments") or {}
                result = call.get("result")
                contract = self.observed_contract(name, arguments, result)
                if self.observation_succeeded(result):
                    ledger.observe(
                        name,
                        contract,
                        arguments,
                        result,
                        preview=self._is_preview(name, arguments, result),
                    )
                else:
                    if not self.contract_for(name).writes:
                        ledger.observe_failure(name, contract)
        return ledger

    def refresh_options(self, missing: set[str], available_tools: Iterable[str]) -> list[str]:
        options: list[tuple[int, str]] = []
        for name in available_tools:
            coverage = len(missing & self.contract_for(name).reads)
            if coverage:
                options.append((-coverage, name))
        return [name for _, name in sorted(options)[:3]]

    def feedback(
        self,
        *,
        candidate_calls: Iterable[tuple[str, dict[str, Any]]],
        conversation: list[dict[str, Any]],
        available_tools: Iterable[str],
        final_response: bool = False,
    ) -> str | None:
        ledger = self.ledger_from_conversation(conversation)
        tool_names = list(available_tools)
        for name, arguments in candidate_calls:
            contract = self.contract_for(name)
            if not contract.writes:
                continue
            if self._is_preview(name, arguments):
                continue
            missing = ledger.missing(
                contract, arguments, entity_aware=self.entity_aware
            )
            missing_tools: list[str] = []
            policy_topic = self._policy_topic(name)
            has_policy = any(
                (
                    item.tool_name == "get_policies"
                    and policy_topic in set(map(str, item.values.get("topic", ())))
                )
                or (
                    item.tool_name in POLICY_TOOL_ALIASES.get(policy_topic or "", ())
                    and (
                        not self.entity_aware
                        or self.scope_matches(
                            item.arguments, item.result, arguments
                        )
                    )
                )
                for item in ledger.observations
            )
            if policy_topic and not has_policy:
                missing.add("policy")
                if "get_policies" not in missing_tools:
                    missing_tools.append("get_policies")
            if missing or missing_tools:
                refresh_tools = self.refresh_options(missing, tool_names)
                refresh_tools = list(dict.fromkeys([*missing_tools, *refresh_tools]))[:3]
                fields = ", ".join(sorted(missing)) or "learned process evidence"
                options = ", ".join(refresh_tools) or "the applicable read-only tool"
                return (
                    f"Do not execute {name} yet. The candidate depends on stale, mismatched, or "
                    f"unobserved state: {fields}. Perform the minimal read-only refresh using "
                    f"{options}, consume its live result, then retry {name}."
                )

            if arguments.get("confirm") and self._preview_required(name, arguments):
                preview = ledger.latest_preview(
                    name, arguments, entity_aware=self.entity_aware
                )
                if preview is None:
                    return (
                        f"Do not confirm {name} yet. Generate and consume a preview for the same "
                        "entity before committing the state change."
                    )
                value_feedback = self._preview_value_feedback(name, arguments, preview)
                if value_feedback:
                    return value_feedback

            ledger.observe(name, self.observed_contract(name, arguments), arguments)
        write_result = ledger.last_write_result
        authoritative_result = isinstance(write_result, dict) and any(
            key not in {"message", "status", "success"} for key in write_result
        )
        if final_response and ledger.last_write and not authoritative_result:
            refresh_tools = self.refresh_options(ledger.stale, tool_names)
            if refresh_tools:
                return (
                    "Do not finalize from invalidated pre-mutation facts. Reconcile the changed "
                    f"state with {', '.join(refresh_tools)} and ground the answer in its live result."
                )
        return None

    def _preview_value_feedback(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        preview: ToolObservation,
    ) -> str | None:
        if not self.value_aware:
            return None
        result_values = _collect_values(preview.result)
        for argument_name, aliases in PREVIEW_VALUE_ALIASES.items():
            supplied = arguments.get(argument_name)
            if not isinstance(supplied, (int, float)) or isinstance(supplied, bool):
                continue
            expected = next(
                (
                    value
                    for alias in aliases
                    for value in result_values.get(alias, ())
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ),
                None,
            )
            if expected is not None and float(supplied) != float(expected):
                return (
                    f"Do not confirm {tool_name}: {argument_name}={supplied} conflicts with "
                    f"the live preview value {expected}. Use the preview-grounded value."
                )
        return None

    def prompt_context(self, conversation: list[dict[str, Any]]) -> str:
        ledger = self.ledger_from_conversation(conversation)
        fresh = ", ".join(sorted(ledger.fresh)) or "none observed"
        stale = ", ".join(sorted(ledger.stale)) or "none"
        failed = ", ".join(sorted(ledger.failed)) or "none"
        last_write = ledger.last_write or "none"
        evidence = []
        for item in ledger.observations[-6:]:
            compact = {
                key: list(values)[:3]
                for key, values in item.values.items()
                if key in IDENTIFIER_KEYS
                or key in {"topic", "status", "amount", "refund_amount", "total"}
            }
            evidence.append({"tool": item.tool_name, "facts": compact})
        learned_paths = {
            name: self._learned_required_tools(name)
            for name in self.learned_contracts
            if self._learned_required_tools(name)
        }
        return (
            "Transition-aware state ledger (derived only from observed tool calls):\n"
            f"- fresh fields: {fresh}\n"
            f"- stale or unobserved dependent fields: {stale}\n"
            f"- failed observations: {failed}\n"
            f"- last state-changing tool: {last_write}\n"
            f"- recent scoped evidence: {json.dumps(evidence, ensure_ascii=True)[:1200]}\n"
            f"- train-supported read paths: {json.dumps(learned_paths, ensure_ascii=True)[:800]}\n"
            "Before a state-changing call, refresh only the fields it requires for the same entity. "
            "After a mutation, do not reuse invalidated facts; reconcile them from a live read."
        )
