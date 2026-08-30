from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


# The repository is copied into a pinned STATE-Bench checkout for official
# execution.  Keep these unit tests runnable in the development repository too;
# only the tiny BaseAgent surface reached during construction is needed here.
try:  # pragma: no cover - the real package is used in the pinned environment
    import state_bench.agents.base  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised only outside STATE-Bench
    state_bench = types.ModuleType("state_bench")
    agents_package = types.ModuleType("state_bench.agents")
    base_module = types.ModuleType("state_bench.agents.base")

    class BaseAgent:
        def __init__(self, runtime_context=None, **_: Any) -> None:
            self.runtime_context = runtime_context

        def add_token_usage(self, **_: Any) -> None:
            return None

        def ingest_trajectory(self, trajectory: Any) -> None:
            return None

        @staticmethod
        def inject_system_message(conversation, content):
            return [{"role": "system", "content": content}, *conversation]

    class AgentToolCallRequest:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class AgentTurnResponse(AgentToolCallRequest):
        pass

    base_module.BaseAgent = BaseAgent
    base_module.AgentToolCallRequest = AgentToolCallRequest
    base_module.AgentTurnResponse = AgentTurnResponse
    state_bench.agents = agents_package
    agents_package.base = base_module
    sys.modules.update(
        {
            "state_bench": state_bench,
            "state_bench.agents": agents_package,
            "state_bench.agents.base": base_module,
        }
    )


from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent
from agents.risk_aware_process_workflow_memory_agent import (
    RiskAwareProcessWorkflowMemoryAgent,
    _Candidate,
)


DOMAIN = "customer_support"


class DummyClient:
    pass


def _base_card(
    card_id: str,
    family: str,
    search_text: str,
    *,
    tools: list[str] | None = None,
    text: str | None = None,
) -> dict[str, Any]:
    rendered = text or f"BASE::{card_id}"
    return {
        "id": card_id,
        "domain": DOMAIN,
        "family": family,
        "support": 8,
        "mean_fitness": 0.9,
        "quality": 0.8,
        "observed_tools": tools or [],
        "search_text": search_text,
        "tokens": search_text.lower().split(),
        "text": rendered,
        "awm_text": f"AWM::{card_id}",
        "process_text": f"PROCESS::{card_id}",
    }


def _sidecar_card(
    card: dict[str, Any],
    *,
    reads: list[str] | None = None,
    writes: list[str] | None = None,
    coverage: list[str] | None = None,
    compiler_valid: bool = True,
) -> dict[str, Any]:
    family = str(card["family"])
    return {
        "domain": DOMAIN,
        "family": family,
        "source_card_sha256": hashlib.sha256(
            json.dumps(
                card,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "contract": {
            "trigger": [family.replace("+", " ").replace("_", " ")],
            "scope": {
                "card_id": str(card["id"]),
                "domain": DOMAIN,
                "family": family,
                "title": f"Handle the {family} workflow only.",
            },
            "required_reads": reads or ["get_order"],
            "authorized_writes": writes or [],
            "decision_rules": ["Use current tool state."],
            "verification_rules": ["Verify the current order before a mutation."],
            "required_disclosures": ["Explain the selected branch."],
            "prohibitions": ["Do not use identifiers from memory."],
        },
        "compiler": {
            "valid": compiler_valid,
            "fallback_to_base_card": not compiler_valid,
            "reasons": [] if compiler_valid else ["synthetic_rejection"],
            "checks": {
                "tool_binding": "passed",
                "write_subset": "passed",
                "disclosure_coverage": "passed",
                "prohibition_coverage": "passed",
                "length_bound": "passed",
                "variable_binding": "not_applicable_no_free_variables",
            },
        },
        "primary_text": f"PRIMARY::{card['id']}",
        "secondary_text": f"SECONDARY::{card['id']}",
        "coverage": coverage or family.split("+"),
        "utility": {
            "domain_prior": 0.5,
            "card_prior": 0.5,
            "state_priors": {},
            "state_prior_support": {},
            "exposures": 0,
            "scored_exposures": 0,
        },
    }


def _router(
    cards: list[dict[str, Any]],
    memory_bytes: bytes,
    *,
    promoted: bool = True,
    min_relevance: float = 0.0,
    mode_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = {
        "weights": {
            "field": 0.25,
            "utility": 0.0,
            "risk": 1.0,
            "trace": 0.0,
            "mmr": 0.3,
        },
        "thresholds": {
            "near_tie": 2.0,
            "candidate_pool": 12,
            "max_cards": 3,
            "default_cards": 1,
            "min_relevance": min_relevance,
            "min_secondary_score": 0.0,
            "secondary_relative_score": 0.0,
            "same_family_limit": 2,
            "duplicate_jaccard": 0.8,
            "utility_cap": 0.75,
            "min_utility_exposures": 5,
            "stickiness": 0.4,
        },
    }
    if mode_defaults:
        for section, values in mode_defaults.items():
            defaults.setdefault(section, {}).update(values)
    digest = hashlib.sha256(memory_bytes).hexdigest()
    return {
        "schema_version": "2.0.0",
        "source_memory_sha256": digest,
        "provenance": {"memory_sha256": digest, "api_calls": 0},
        "defaults": defaults,
        "domain_configs": {
            DOMAIN: {
                "promoted": promoted,
                "weights": dict(defaults["weights"]),
                "thresholds": dict(defaults["thresholds"]),
            }
        },
        "cards": {str(card["id"]): _sidecar_card(card) for card in cards},
    }


def _write_artifacts(
    tmp_path: Path,
    cards: list[dict[str, Any]],
    *,
    router: dict[str, Any] | str | None = None,
    promoted: bool = True,
    min_relevance: float = 0.0,
) -> tuple[Path, Path]:
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": cards}, sort_keys=True), encoding="utf-8")
    memory_bytes = memory_path.read_bytes()
    if router is None:
        router = _router(cards, memory_bytes, promoted=promoted, min_relevance=min_relevance)
    router_path = tmp_path / "router.json"
    if isinstance(router, str):
        router_path.write_text(router, encoding="utf-8")
    else:
        router_path.write_text(json.dumps(router, sort_keys=True), encoding="utf-8")
    return memory_path, router_path


def _new_agent(
    monkeypatch: pytest.MonkeyPatch,
    memory_path: Path,
    router_path: Path,
    *,
    mode: str = "enforce",
    stage: str = "C",
    top_k: int = 3,
    runtime_context: Any | None = None,
) -> RiskAwareProcessWorkflowMemoryAgent:
    monkeypatch.setenv("STATE_BENCH_MEMORY_MODE", "hybrid")
    monkeypatch.setenv("STATE_BENCH_WORKFLOW_ROUTER_MODE", mode)
    monkeypatch.setenv("STATE_BENCH_WORKFLOW_ROUTER_STAGE", stage)
    monkeypatch.setenv("STATE_BENCH_WORKFLOW_ROUTER_PATH", str(router_path))
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", memory_path)
    monkeypatch.setattr(RiskAwareProcessWorkflowMemoryAgent, "memory_path", memory_path)
    return RiskAwareProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=runtime_context or SimpleNamespace(domain=DOMAIN),
        retrieve_learnings_top_k=top_k,
        workflow_router_path=router_path,
    )


def test_staged_ablation_keeps_selector_packing_and_state_changes_separable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _base_card("return", "return", "return defective headphones", text="EXACT BASE")
    memory_path, router_path = _write_artifacts(tmp_path, [card])

    stage_a = _new_agent(monkeypatch, memory_path, router_path, stage="A")
    assert stage_a.retrieve_learnings("return defective headphones", 1) == ["EXACT BASE"]

    stage_b = _new_agent(monkeypatch, memory_path, router_path, stage="B")
    assert stage_b.retrieve_learnings("return defective headphones", 1) == ["PRIMARY::return"]
    history = [
        {"role": "user", "content": "return defective headphones"},
        {"role": "user", "content": "Actually exchange it instead"},
    ]
    assert "return defective headphones" in stage_b._query_from_conversation(history)

    stage_c = _new_agent(monkeypatch, memory_path, router_path, stage="C")
    assert stage_c._query_from_conversation(history) == "Actually exchange it instead"


def _new_parent(
    monkeypatch: pytest.MonkeyPatch, memory_path: Path, *, top_k: int = 3
) -> ProcessWorkflowMemoryAgent:
    monkeypatch.setenv("STATE_BENCH_MEMORY_MODE", "hybrid")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", memory_path)
    return ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain=DOMAIN),
        retrieve_learnings_top_k=top_k,
    )


def _candidate(
    index: int,
    card_id: str,
    family: str,
    *,
    final: float,
    semantic: float,
    coverage: set[str],
    search_text: str,
    tools: list[str] | None = None,
) -> _Candidate:
    return _Candidate(
        index=index,
        card={
            "id": card_id,
            "family": family,
            "search_text": search_text,
            "observed_tools": tools or [],
        },
        sidecar={},
        semantic=semantic,
        lexical=semantic,
        character=0.0,
        intent=0.0,
        final=final,
        coverage=frozenset(coverage),
    )


def test_abstains_when_no_candidate_clears_relevance_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = [_base_card("return", "return", "return defective headphones")]
    memory_path, router_path = _write_artifacts(tmp_path, cards, min_relevance=20.0)
    agent = _new_agent(monkeypatch, memory_path, router_path)

    assert agent.retrieve_learnings("unrelated weather forecast", top_k=3) == []
    assert agent.last_retrieval_telemetry["selected"] == []
    assert agent.last_retrieval_telemetry["injected_chars"] == 0


def test_generic_procedural_word_cannot_make_an_unrelated_query_relevant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = [_base_card("return", "return", "return defective headphones")]
    memory_path, router_path = _write_artifacts(tmp_path, cards, min_relevance=0.0)
    agent = _new_agent(monkeypatch, memory_path, router_path)

    assert agent.retrieve_learnings("tell me a joke about astronomy", top_k=3) == []


def test_selection_is_real_greedy_mmr_not_a_posthoc_penalty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _base_card("fixture", "return", "return")
    memory_path, router_path = _write_artifacts(tmp_path, [card])
    agent = _new_agent(monkeypatch, memory_path, router_path)
    agent._weights["mmr"] = 2.0
    agent._thresholds.update(
        {
            "near_tie": 100.0,
            "default_cards": 3.0,
            "min_secondary_score": -100.0,
            "secondary_relative_score": -100.0,
        }
    )
    anchor = _candidate(
        0,
        "anchor",
        "return",
        final=10.0,
        semantic=10.0,
        coverage={"return"},
        search_text="return order headphones",
        tools=["get_order"],
    )
    redundant = _candidate(
        1,
        "redundant",
        "exchange",
        final=9.9,
        semantic=9.9,
        coverage={"exchange"},
        search_text="return order headphones",
        tools=["get_order"],
    )
    diverse = _candidate(
        2,
        "diverse",
        "exchange",
        final=9.8,
        semantic=9.8,
        coverage={"exchange"},
        search_text="replacement different size",
        tools=["process_exchange"],
    )

    selected = agent._select_candidates(
        "return and exchange",
        {"intents": ("return", "exchange")},
        [anchor, redundant, diverse],
        limit=3,
    )

    assert [item.card_id for item in selected] == ["anchor", "diverse"]
    assert diverse.adjusted > redundant.final - agent._weights["mmr"]


def test_decisive_raw_top1_is_preserved_even_when_soft_risk_lowers_final_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _base_card("fixture", "return", "return")
    memory_path, router_path = _write_artifacts(tmp_path, [fixture])
    agent = _new_agent(monkeypatch, memory_path, router_path)
    # The calibrated reranking window may be narrower, but the independent
    # hard anchor still protects a raw semantic margin greater than 2.0.
    agent._thresholds["near_tie"] = 0.5
    anchor = _candidate(
        0,
        "raw-anchor",
        "return",
        semantic=8.0,
        final=1.0,
        coverage={"return"},
        search_text="return current order",
    )
    anchor.risk = 1.0
    reranker_favorite = _candidate(
        1,
        "reranker-favorite",
        "return",
        semantic=5.0,
        final=20.0,
        coverage={"return"},
        search_text="return another order",
    )

    selected = agent._select_candidates(
        "return this order",
        {"intents": ("return",)},
        [anchor, reranker_favorite],
        limit=1,
    )

    assert [item.card_id for item in selected] == ["raw-anchor"]


def test_primary_reranking_cannot_escape_the_semantic_near_tie_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _base_card("fixture", "return", "return")
    memory_path, router_path = _write_artifacts(tmp_path, [fixture])
    agent = _new_agent(monkeypatch, memory_path, router_path)
    anchor = _candidate(
        0,
        "anchor",
        "return",
        semantic=8.0,
        final=7.0,
        coverage={"return"},
        search_text="return current order",
    )
    near_tie = _candidate(
        1,
        "near-tie",
        "return",
        semantic=7.0,
        final=7.5,
        coverage={"return"},
        search_text="return recent order",
    )
    distant = _candidate(
        2,
        "distant",
        "return",
        semantic=2.0,
        final=100.0,
        coverage={"return"},
        search_text="unrelated card",
    )

    selected = agent._select_candidates(
        "return this order",
        {"intents": ("return",)},
        [anchor, near_tie, distant],
        limit=1,
    )

    assert [item.card_id for item in selected] == ["near-tie"]


def test_same_family_cards_can_fill_complementary_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _base_card("fixture", "return", "return")
    memory_path, router_path = _write_artifacts(tmp_path, [fixture])
    agent = _new_agent(monkeypatch, memory_path, router_path)
    agent._thresholds.update(
        {
            "default_cards": 2.0,
            "same_family_limit": 2.0,
            "min_secondary_score": -100.0,
            "secondary_relative_score": -100.0,
        }
    )
    first = _candidate(
        0,
        "return-preview",
        "return+exchange",
        final=4.0,
        semantic=4.0,
        coverage={"return"},
        search_text="return eligibility preview",
    )
    complement = _candidate(
        1,
        "return-exchange",
        "return+exchange",
        final=3.5,
        semantic=3.5,
        coverage={"exchange"},
        search_text="replacement inventory choice",
    )

    selected = agent._select_candidates(
        "return and exchange",
        {"intents": ("return", "exchange")},
        [first, complement],
        limit=2,
    )

    assert [item.card_id for item in selected] == ["return-preview", "return-exchange"]


def test_unrequested_write_risk_is_a_soft_downweight_not_a_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe = _base_card("safe", "return", "return headphones", tools=["process_return"])
    risky = _base_card("risky", "return", "return headphones", tools=["cancel_order"])
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": [safe, risky]}, sort_keys=True), encoding="utf-8")
    router = _router([safe, risky], memory_path.read_bytes())
    router["cards"]["safe"] = _sidecar_card(safe, writes=["process_return"])
    router["cards"]["risky"] = _sidecar_card(risky, writes=["cancel_order"])
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(router, sort_keys=True), encoding="utf-8")
    agent = _new_agent(monkeypatch, memory_path, router_path)

    context = agent._context_for_query("return headphones")
    ranked = {candidate.card_id: candidate for candidate in agent._rank_candidates("return headphones", context)}

    assert set(ranked) == {"safe", "risky"}
    assert ranked["safe"].risk == 0.0
    assert ranked["risky"].risk == 1.0
    assert ranked["risky"].extra_writes == ("cancel_order",)
    assert ranked["safe"].final > ranked["risky"].final


@pytest.mark.parametrize(
    ("query", "tool"),
    [
        ("Do not add anything; just recommend a laptop.", "add_to_cart"),
        ("Never apply the promo; only tell me whether it is valid.", "apply_promo"),
    ],
)
def test_negated_or_information_only_write_mentions_are_not_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    tool: str,
) -> None:
    card = _base_card("risky", "add_to_cart+promo+search", query)
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": [card]}, sort_keys=True), encoding="utf-8")
    router = _router([card], memory_path.read_bytes())
    router["cards"]["risky"] = _sidecar_card(card, writes=[tool])
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(router, sort_keys=True), encoding="utf-8")
    agent = _new_agent(monkeypatch, memory_path, router_path)
    agent._runtime_domain = "shopping_assistant"

    context = agent._context_for_query(query)
    ranked = agent._rank_candidates(query, context)

    assert ranked[0].risk == 1.0
    assert ranked[0].extra_writes == (tool,)


def test_unrelated_negative_clause_does_not_hide_later_positive_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = "Do not show a breakdown, just add the laptop."
    card = _base_card("add", "add_to_cart", "add laptop to cart")
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": [card]}, sort_keys=True), encoding="utf-8")
    router = _router([card], memory_path.read_bytes())
    router["cards"]["add"] = _sidecar_card(card, writes=["add_to_cart"])
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(router, sort_keys=True), encoding="utf-8")
    agent = _new_agent(monkeypatch, memory_path, router_path)
    agent._runtime_domain = "shopping_assistant"

    context = agent._context_for_query(query)
    candidate = agent._rank_candidates(query, context)[0]

    assert candidate.risk == 0.0
    assert candidate.extra_writes == ()


def test_query_needs_are_user_grounded_and_negated_write_does_not_earn_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = "Do not add anything; just recommend a laptop."
    risky = _base_card("risky", "add_to_cart+search", "add recommend laptop")
    safe = _base_card("safe", "search", "recommend laptop")
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": [risky, safe]}, sort_keys=True), encoding="utf-8")
    router = _router([risky, safe], memory_path.read_bytes())
    router["cards"]["risky"] = _sidecar_card(risky, writes=["add_to_cart"])
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(router, sort_keys=True), encoding="utf-8")
    agent = _new_agent(monkeypatch, memory_path, router_path)
    agent._runtime_domain = "shopping_assistant"

    context = agent._context_for_query(query)
    pool = agent._rank_candidates(query, context)
    needs = agent._query_needs(query, context, pool)
    selected = agent._select_candidates(query, context, pool, limit=3)

    assert needs == {"search"}
    assert len(selected) <= 1


def test_secondary_includes_only_the_explicitly_authorized_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _base_card(
        "composite",
        "add_to_cart+remove_from_cart",
        "add or remove a cart item",
    )
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": [card]}, sort_keys=True), encoding="utf-8")
    router = _router([card], memory_path.read_bytes())
    sidecar = _sidecar_card(
        card,
        writes=["add_to_cart", "remove_from_cart"],
    )
    sidecar["secondary_text"] = (
        "WORKFLOW CONSTRAINTS: cart mutation\n"
        "WHEN:\n- a cart mutation is requested\n"
        "READ:\n- get_cart\n"
        "SAY:\n- Explain the selected branch.\n"
        "NEVER:\n- Do not use identifiers from memory."
    )
    router["cards"]["composite"] = sidecar
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(router, sort_keys=True), encoding="utf-8")
    agent = _new_agent(monkeypatch, memory_path, router_path)
    agent._runtime_domain = "shopping_assistant"
    query = "Add this laptop to my cart"
    context = agent._context_for_query(query)
    candidate = agent._rank_candidates(query, context)[0]

    rendered = agent._render_card(candidate, role="secondary", query=query, context=context)

    assert "WRITE:\n- add_to_cart" in rendered
    assert "remove_from_cart" not in rendered
    assert "- Explain the selected branch." in rendered
    assert "- Do not use identifiers from memory." in rendered
    assert len(rendered) <= 2200


def test_secondary_write_rows_never_truncate_say_or_never_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _base_card("long", "add_to_cart", "add laptop to cart")
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": [card]}, sort_keys=True), encoding="utf-8")
    router = _router([card], memory_path.read_bytes())
    sidecar = _sidecar_card(card, writes=["add_to_cart"])
    padding = "x" * 2090
    supplied = (
        f"WORKFLOW CONSTRAINTS: {padding}\n"
        "SAY:\n- Explain the selected branch.\n"
        "NEVER:\n- Do not use identifiers from memory."
    )
    assert len(supplied) <= 2200
    assert len(supplied + "\nWRITE:\n- add_to_cart") > 2200
    sidecar["secondary_text"] = supplied
    router["cards"]["long"] = sidecar
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(router, sort_keys=True), encoding="utf-8")
    agent = _new_agent(monkeypatch, memory_path, router_path)
    agent._runtime_domain = "shopping_assistant"
    query = "Add this laptop to my cart"
    context = agent._context_for_query(query)
    candidate = agent._rank_candidates(query, context)[0]

    rendered = agent._render_card(candidate, role="secondary", query=query, context=context)

    assert rendered == supplied
    assert "WRITE:" not in rendered
    assert "- Explain the selected branch." in rendered
    assert "- Do not use identifiers from memory." in rendered
    assert len(rendered) <= 2200


def test_observed_tool_name_is_trace_evidence_not_user_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _base_card("add", "add_to_cart", "add an item to cart")
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": [card]}, sort_keys=True), encoding="utf-8")
    router = _router([card], memory_path.read_bytes())
    router["cards"]["add"] = _sidecar_card(card, writes=["add_to_cart"])
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(router, sort_keys=True), encoding="utf-8")
    agent = _new_agent(monkeypatch, memory_path, router_path)
    agent._runtime_domain = "shopping_assistant"
    intent_text = "Now only show shipping options; do not add anything else."
    query = f"{intent_text} add_to_cart"
    context = {
        "intent_text": intent_text,
        "intents": tuple(sorted(agent._intent_matches(intent_text))),
        "intent_signature": ("shipping",),
        "observed_tools": ("add_to_cart",),
        "phase": "postwrite",
    }

    candidate = agent._rank_candidates(query, context)[0]

    assert candidate.risk == 1.0
    assert candidate.extra_writes == ("add_to_cart",)


def test_flowswitch_keeps_continuations_sticky_but_resets_on_intent_shift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    return_card = _base_card("return", "return", "return defective headphones")
    exchange_card = _base_card("exchange", "exchange", "exchange replacement headphones")
    memory_path, router_path = _write_artifacts(tmp_path, [return_card, exchange_card])
    agent = _new_agent(monkeypatch, memory_path, router_path)

    first_conversation = [{"role": "user", "content": "Return my defective headphones"}]
    first_query = agent._query_from_conversation(first_conversation)
    assert agent.retrieve_learnings(first_query, top_k=1) == ["PRIMARY::return"]

    continuation = [*first_conversation, {"role": "user", "content": "Yes, please"}]
    continuation_query = agent._query_from_conversation(continuation)
    continuation_pool = agent._rank_candidates(
        continuation_query, agent._context_for_query(continuation_query)
    )
    return_candidate = next(item for item in continuation_pool if item.card_id == "return")
    assert "return" in agent._state_context["intent_signature"]
    assert return_candidate.diagnostics["sticky"] == pytest.approx(agent._thresholds["stickiness"])

    shifted = [*continuation, {"role": "user", "content": "Actually exchange it instead"}]
    shifted_query = agent._query_from_conversation(shifted)
    shifted_pool = agent._rank_candidates(shifted_query, agent._context_for_query(shifted_query))
    old_candidate = next(item for item in shifted_pool if item.card_id == "return")
    assert agent._state_context["intent_signature"] == ("exchange",)
    assert old_candidate.diagnostics["sticky"] == 0.0
    assert agent.retrieve_learnings(shifted_query, top_k=1) == ["PRIMARY::exchange"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("corrupt", "router_unreadable"),
        ("schema", "unsupported_router_schema"),
        ("hash", "memory_hash_mismatch"),
    ],
)
def test_corrupt_schema_and_hash_fail_open_to_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason: str,
) -> None:
    cards = [_base_card("return", "return", "return defective headphones")]
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": cards}, sort_keys=True), encoding="utf-8")
    router: dict[str, Any] | str = _router(cards, memory_path.read_bytes())
    if mutation == "corrupt":
        router = "{not-json"
    elif mutation == "schema":
        router["schema_version"] = "1.0.0"
    else:
        router["source_memory_sha256"] = "0" * 64
        router["provenance"]["memory_sha256"] = "0" * 64
    router_path = tmp_path / "router.json"
    router_path.write_text(router if isinstance(router, str) else json.dumps(router), encoding="utf-8")

    agent = _new_agent(monkeypatch, memory_path, router_path)
    parent = _new_parent(monkeypatch, memory_path)
    query = "return defective headphones"

    assert agent._router_enabled is False
    assert agent._router_reason == reason
    assert agent.retrieve_learnings(query) == parent.retrieve_learnings(query)
    assert agent.last_retrieval_telemetry["fallback_reason"] == reason


def test_unpromoted_domain_is_byte_equivalent_to_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = [
        _base_card("return", "return", "return defective headphones"),
        _base_card("exchange", "exchange", "exchange replacement"),
    ]
    memory_path, router_path = _write_artifacts(tmp_path, cards, promoted=False)
    agent = _new_agent(monkeypatch, memory_path, router_path)
    parent = _new_parent(monkeypatch, memory_path)
    query = "return defective headphones"

    candidate_bytes = json.dumps(
        agent.retrieve_learnings(query), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    parent_bytes = json.dumps(
        parent.retrieve_learnings(query), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")

    assert candidate_bytes == parent_bytes
    assert agent._router_reason == "domain_not_promoted"


def test_enforced_retrieval_is_deterministic_across_fresh_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = [
        _base_card("return", "return", "return defective headphones"),
        _base_card("exchange", "exchange", "exchange replacement headphones"),
        _base_card("warranty", "warranty", "warranty repair headphones"),
    ]
    memory_path, router_path = _write_artifacts(tmp_path, cards)
    first = _new_agent(monkeypatch, memory_path, router_path)
    first_result = first.retrieve_learnings("return and exchange headphones", top_k=3)
    first_telemetry = first.last_retrieval_telemetry
    second = _new_agent(monkeypatch, memory_path, router_path)
    second_result = second.retrieve_learnings("return and exchange headphones", top_k=3)

    assert first_result == second_result
    assert first_telemetry == second.last_retrieval_telemetry


def test_shadow_mode_returns_parent_output_and_records_candidate_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = [
        _base_card("return", "return", "return defective headphones"),
        _base_card("exchange", "exchange", "exchange replacement"),
    ]
    memory_path, router_path = _write_artifacts(tmp_path, cards)
    shadow = _new_agent(monkeypatch, memory_path, router_path, mode="shadow")
    parent = _new_parent(monkeypatch, memory_path)
    query = "return defective headphones"

    assert shadow.retrieve_learnings(query) == parent.retrieve_learnings(query)
    telemetry = shadow.last_retrieval_telemetry
    assert telemetry["mode"] == "shadow"
    assert telemetry["fallback_reason"] == "shadow_parent_result"
    assert telemetry["selected"]
    assert telemetry["candidates"]
    assert telemetry["shadow_parent_chars"] > 0


def test_public_retrieval_never_returns_more_than_three_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = [
        _base_card("return", "return", "return item"),
        _base_card("exchange", "exchange", "exchange replacement"),
        _base_card("warranty", "warranty", "warranty repair"),
        _base_card("refund", "refund", "refund money back"),
    ]
    memory_path, router_path = _write_artifacts(tmp_path, cards)
    agent = _new_agent(monkeypatch, memory_path, router_path, top_k=99)

    result = agent.retrieve_learnings(
        "return and exchange plus warranty, then refund my item", top_k=99
    )

    assert 1 <= len(result) <= 3
    assert len(agent.last_retrieval_telemetry["selected"]) == len(result)


def test_oracle_fields_disable_router_and_runtime_oracle_is_not_queried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = [_base_card("return", "return", "return item")]
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": cards}, sort_keys=True), encoding="utf-8")
    router = _router(cards, memory_path.read_bytes())
    router["provenance"]["task_summary"] = "oracle-only answer"
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(router), encoding="utf-8")
    agent = _new_agent(
        monkeypatch,
        memory_path,
        router_path,
        runtime_context=SimpleNamespace(domain=DOMAIN, task_summary="another oracle answer"),
    )

    assert agent._router_enabled is False
    assert agent._router_reason == "router_invalid_or_forbidden_provenance"
    query = agent._query_from_conversation([{"role": "user", "content": "return this item"}])
    assert query == "return this item"
    assert "oracle" not in query


def test_invalid_compiler_card_uses_base_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _base_card("return", "return", "return item", text="EXACT BASE CARD")
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": [card]}, sort_keys=True), encoding="utf-8")
    router = _router([card], memory_path.read_bytes())
    router["cards"]["return"] = _sidecar_card(card, compiler_valid=False)
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(router), encoding="utf-8")
    agent = _new_agent(monkeypatch, memory_path, router_path)

    assert agent.retrieve_learnings("return item", top_k=1) == ["EXACT BASE CARD"]


def test_telemetry_has_bounded_diagnostic_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = [
        _base_card("return", "return", "return defective headphones"),
        _base_card("exchange", "exchange", "exchange replacement"),
    ]
    memory_path, router_path = _write_artifacts(tmp_path, cards)
    agent = _new_agent(monkeypatch, memory_path, router_path)
    agent.retrieve_learnings("return defective headphones")
    telemetry = agent.last_retrieval_telemetry

    assert {
        "mode",
        "domain",
        "phase",
        "fallback_reason",
        "selected",
        "candidates",
        "injected_chars",
    } <= telemetry.keys()
    assert len(agent.retrieval_telemetry) <= 64
    assert {"card_id", "family", "slot", "role", "adjusted_score", "render_chars"} <= telemetry[
        "selected"
    ][0].keys()
    assert {
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
    } <= telemetry["candidates"][0].keys()


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing_memory_hash", "memory_hash_missing_or_invalid"),
        ("mismatched_declared_hash", "memory_hash_declaration_mismatch"),
        ("source_card_hash", "router_source_card_hash_mismatch"),
        ("family", "router_card_family_mismatch"),
        ("compiler", "router_compiler_metadata_invalid"),
        ("disclosure_check", "router_compiler_metadata_invalid"),
        ("contract", "router_contract_invalid"),
        ("weight", "router_config_invalid"),
        ("near_tie", "router_config_invalid"),
        ("missing_config_key", "router_config_missing"),
    ],
)
def test_untrusted_sidecar_components_disable_the_whole_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason: str,
) -> None:
    cards = [
        _base_card("return", "return", "return defective headphones"),
        _base_card("exchange", "exchange", "exchange replacement"),
    ]
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": cards}, sort_keys=True), encoding="utf-8")
    router = _router(cards, memory_path.read_bytes())
    if mutation == "missing_memory_hash":
        del router["source_memory_sha256"]
    elif mutation == "mismatched_declared_hash":
        router["provenance"]["memory_sha256"] = "0" * 64
    elif mutation == "source_card_hash":
        router["cards"]["exchange"]["source_card_sha256"] = "0" * 64
    elif mutation == "family":
        router["cards"]["exchange"]["family"] = "return"
    elif mutation == "compiler":
        del router["cards"]["exchange"]["compiler"]["valid"]
    elif mutation == "disclosure_check":
        router["cards"]["exchange"]["compiler"]["checks"]["disclosure_coverage"] = "failed"
    elif mutation == "contract":
        del router["cards"]["exchange"]["contract"]["prohibitions"]
    elif mutation == "weight":
        router["domain_configs"][DOMAIN]["weights"]["risk"] = -1.0
    elif mutation == "near_tie":
        router["domain_configs"][DOMAIN]["thresholds"]["near_tie"] = 0.25
    else:
        del router["domain_configs"][DOMAIN]["weights"]["field"]
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(router, sort_keys=True), encoding="utf-8")

    agent = _new_agent(monkeypatch, memory_path, router_path)
    parent = _new_parent(monkeypatch, memory_path)

    assert agent._router_enabled is False
    assert agent._router_reason == reason
    assert agent.retrieve_learnings("return defective headphones") == parent.retrieve_learnings(
        "return defective headphones"
    )


def test_runtime_phase_keys_use_no_future_information_and_accept_legacy_read_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _base_card(
        "return",
        "return",
        "return defective headphones",
        tools=["get_order", "process_return"],
    )
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({"cards": [card]}, sort_keys=True), encoding="utf-8")
    router = _router([card], memory_path.read_bytes())
    router["cards"]["return"] = _sidecar_card(
        card,
        reads=["get_order"],
        writes=["process_return"],
    )
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(router, sort_keys=True), encoding="utf-8")
    agent = _new_agent(monkeypatch, memory_path, router_path)

    initial = [{"role": "user", "content": "return defective headphones"}]
    assert agent._query_from_conversation(initial) == "return defective headphones"
    assert agent._state_context["phase"] == "read"

    after_read = [
        *initial,
        {"role": "assistant", "tool_calls": [{"name": "get_order"}]},
        {"role": "tool", "content": "{}"},
    ]
    agent._query_from_conversation(after_read)
    assert agent._state_context["phase"] == "prewrite"
    keys = agent._state_keys(agent._state_context, "return")
    assert keys[0] == f"{DOMAIN}|return|get_order|prewrite"
    assert f"{DOMAIN}|return|get_order|read" in keys

    after_write = [
        *after_read,
        {"role": "assistant", "tool_calls": [{"name": "process_return"}]},
        {"role": "tool", "content": "{}"},
    ]
    agent._query_from_conversation(after_write)
    assert agent._state_context["phase"] == "postwrite"


def test_disabled_router_does_not_change_trajectory_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _base_card("return", "return", "return defective headphones")
    memory_path, router_path = _write_artifacts(tmp_path, [card], promoted=False)
    agent = _new_agent(monkeypatch, memory_path, router_path)
    agent.retrieve_learnings("return defective headphones")
    trajectory = SimpleNamespace(metadata={"existing": "unchanged"})

    agent.ingest_trajectory(trajectory)

    assert trajectory.metadata["existing"] == "unchanged"
    assert "workflow_router" not in trajectory.metadata
    assert len(trajectory.metadata["provider_request_audit_id"]) == 32
    assert len(trajectory.metadata["provider_task_key"]) == 64


def test_enabled_router_persists_only_safe_workflow_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _base_card("return", "return", "return defective headphones")
    memory_path, router_path = _write_artifacts(tmp_path, [card])
    agent = _new_agent(monkeypatch, memory_path, router_path)
    agent.retrieve_learnings("return defective headphones")
    trajectory = SimpleNamespace(metadata={})

    agent.ingest_trajectory(trajectory)

    payload = trajectory.metadata["workflow_router"]
    assert payload["router_enabled"] is True
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "return defective headphones" not in encoded
    assert "task_summary" not in encoded
