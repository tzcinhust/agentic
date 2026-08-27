from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent
from agents.transition_aware_memory import ToolContract, TransitionAwareMemory


class DummyClient:
    pass


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_transition_memory_invalidates_dependent_fields_after_write() -> None:
    memory = TransitionAwareMemory(
        {
            "read_state": ToolContract(reads=frozenset({"cart", "total"})),
            "mutate": ToolContract(
                writes=frozenset({"cart"}),
                invalidates=frozenset({"total"}),
                requires_fresh=frozenset({"cart"}),
            ),
        }
    )
    ledger = memory.ledger_from_conversation(
        [
            {"role": "assistant", "tool_calls": [{"name": "read_state", "arguments": {}}]},
            {"role": "assistant", "tool_calls": [{"name": "mutate", "arguments": {}}]},
        ]
    )
    assert ledger.fresh == {"cart"}
    assert ledger.stale == {"total"}
    assert ledger.failed == set()
    assert ledger.last_write == "mutate"


def test_transition_memory_requests_only_minimal_refresh() -> None:
    memory = TransitionAwareMemory(
        {
            "read_cart": ToolContract(reads=frozenset({"cart", "total"})),
            "read_policy": ToolContract(reads=frozenset({"policy"})),
            "mutate": ToolContract(
                writes=frozenset({"cart"}),
                requires_fresh=frozenset({"cart"}),
            ),
        }
    )
    feedback = memory.feedback(
        candidate_calls=[("mutate", {})],
        conversation=[],
        available_tools=["read_cart", "read_policy", "mutate"],
    )
    assert feedback is not None
    assert "cart" in feedback
    assert "read_cart" in feedback
    assert "read_policy" not in feedback


def test_transition_memory_does_not_treat_preview_as_mutation() -> None:
    memory = TransitionAwareMemory(
        {
            "read_order": ToolContract(reads=frozenset({"order", "eligibility"})),
            "process": ToolContract(
                writes=frozenset({"return_status"}),
                invalidates=frozenset({"order", "eligibility"}),
                requires_fresh=frozenset({"order", "eligibility"}),
            ),
        }
    )
    ledger = memory.ledger_from_conversation(
        [
            {"role": "assistant", "tool_calls": [{"name": "read_order", "arguments": {}}]},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "process",
                        "arguments": {},
                        "result": {"status": "preview"},
                    }
                ],
            },
        ]
    )
    assert ledger.fresh == {"order", "eligibility"}
    assert ledger.stale == set()


def test_transition_memory_does_not_trust_failed_read() -> None:
    memory = TransitionAwareMemory(
        {
            "read_state": ToolContract(reads=frozenset({"cart"})),
            "mutate": ToolContract(
                writes=frozenset({"cart"}),
                requires_fresh=frozenset({"cart"}),
            ),
        }
    )
    ledger = memory.ledger_from_conversation(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"name": "read_state", "arguments": {}, "result": {"error": "not found"}}
                ],
            }
        ]
    )
    assert ledger.fresh == set()
    assert ledger.stale == {"cart"}
    assert ledger.failed == {"cart"}
    feedback = memory.feedback(
        candidate_calls=[("mutate", {})],
        conversation=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"name": "read_state", "arguments": {}, "result": {"error": "not found"}}
                ],
            }
        ],
        available_tools=["read_state", "mutate"],
    )
    assert feedback and "stale, mismatched, or unobserved" in feedback


def test_customer_support_writes_require_policy_refresh() -> None:
    memory = TransitionAwareMemory()
    feedback = memory.feedback(
        candidate_calls=[("process_return", {"confirm": True})],
        conversation=[
            {
                "role": "assistant",
                "tool_calls": [{"name": "get_order", "arguments": {}, "result": {"order": "ok"}}],
            }
        ],
        available_tools=["get_order", "get_policies", "process_return"],
    )
    assert feedback and "policy" in feedback
    assert "get_policies" in feedback

    cancel_feedback = memory.feedback(
        candidate_calls=[("cancel_order", {"confirm": True})],
        conversation=[
            {
                "role": "assistant",
                "tool_calls": [{"name": "get_order", "arguments": {}, "result": {"order": "ok"}}],
            }
        ],
        available_tools=["get_order", "get_policies", "cancel_order"],
    )
    assert cancel_feedback and "get_policies" in cancel_feedback


def test_transition_memory_checks_candidates_in_execution_order() -> None:
    memory = TransitionAwareMemory(
        {
            "read_cart": ToolContract(reads=frozenset({"cart", "promo"})),
            "mutate_cart": ToolContract(
                writes=frozenset({"cart"}),
                invalidates=frozenset({"promo"}),
                requires_fresh=frozenset({"cart"}),
            ),
            "apply_promo": ToolContract(
                writes=frozenset({"promo"}),
                requires_fresh=frozenset({"cart", "promo"}),
            ),
        }
    )
    feedback = memory.feedback(
        candidate_calls=[("mutate_cart", {}), ("apply_promo", {})],
        conversation=[
            {"role": "assistant", "tool_calls": [{"name": "read_cart", "arguments": {}}]}
        ],
        available_tools=["read_cart", "mutate_cart", "apply_promo"],
    )
    assert feedback and "apply_promo" in feedback
    assert "promo" in feedback


def test_candidate_read_cannot_ground_write_in_same_response() -> None:
    memory = TransitionAwareMemory(
        {
            "read_cart": ToolContract(reads=frozenset({"cart"})),
            "mutate_cart": ToolContract(
                writes=frozenset({"cart"}),
                requires_fresh=frozenset({"cart"}),
            ),
        },
        learned=False,
    )
    feedback = memory.feedback(
        candidate_calls=[("read_cart", {"customer_id": "C-1"}), ("mutate_cart", {"customer_id": "C-1"})],
        conversation=[],
        available_tools=["read_cart", "mutate_cart"],
    )
    assert feedback and "read_cart" in feedback


def test_transition_memory_requires_same_entity_evidence() -> None:
    memory = TransitionAwareMemory(learned=False)
    conversation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_order",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {"order_id": "ORD-1", "status": "processing"},
                },
                {
                    "name": "get_policies",
                    "arguments": {"topic": "cancellation"},
                    "result": {"topic": "cancellation"},
                },
            ],
        }
    ]
    feedback = memory.feedback(
        candidate_calls=[
            ("cancel_order", {"order_id": "ORD-2", "confirm": True})
        ],
        conversation=conversation,
        available_tools=["get_order", "get_policies", "cancel_order"],
    )
    assert feedback and "get_order" in feedback


def test_transition_memory_requires_exact_policy_topic() -> None:
    memory = TransitionAwareMemory(learned=False)
    conversation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_order",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {"order_id": "ORD-1", "items": [{"item_id": "ITEM-1"}]},
                },
                {
                    "name": "get_policies",
                    "arguments": {"topic": "refund"},
                    "result": {"topic": "refund"},
                },
            ],
        }
    ]
    feedback = memory.feedback(
        candidate_calls=[
            ("process_return", {"item_id": "ITEM-1", "confirm": True})
        ],
        conversation=conversation,
        available_tools=["get_order", "get_policies", "process_return"],
    )
    assert feedback and "get_policies" in feedback


def test_preview_probe_is_allowed_before_fresh_state_exists() -> None:
    memory = TransitionAwareMemory(learned=False)
    feedback = memory.feedback(
        candidate_calls=[("process_return", {"item_id": "ITEM-1", "confirm": False})],
        conversation=[],
        available_tools=["get_order", "get_policies", "process_return"],
    )
    assert feedback is None


def test_final_response_requires_post_write_reconciliation() -> None:
    memory = TransitionAwareMemory(learned=False)
    conversation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "add_to_cart",
                    "arguments": {"customer_id": "C-1", "product_id": "P-1"},
                    "result": {"status": "success"},
                }
            ],
        }
    ]
    feedback = memory.feedback(
        candidate_calls=[],
        conversation=conversation,
        available_tools=["get_cart", "add_to_cart"],
        final_response=True,
    )
    assert feedback and "get_cart" in feedback


def test_authoritative_write_result_does_not_force_reconciliation() -> None:
    memory = TransitionAwareMemory(learned=False)
    conversation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "add_to_cart",
                    "arguments": {"customer_id": "C-1", "product_id": "P-1"},
                    "result": {
                        "status": "success",
                        "cart_subtotal": 99,
                        "cart_total": 99,
                    },
                }
            ],
        }
    ]
    feedback = memory.feedback(
        candidate_calls=[],
        conversation=conversation,
        available_tools=["get_cart", "add_to_cart"],
        final_response=True,
    )
    assert feedback is None


def test_transition_memory_checks_same_entity_preview_value() -> None:
    memory = TransitionAwareMemory(learned=False)
    conversation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_order",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {"order_id": "ORD-1", "items": [{"item_id": "ITEM-1"}]},
                },
                {
                    "name": "get_policies",
                    "arguments": {"topic": "return"},
                    "result": {"topic": "return"},
                },
                {
                    "name": "process_return",
                    "arguments": {"item_id": "ITEM-1", "amount": 0},
                    "result": {
                        "status": "preview",
                        "item_id": "ITEM-1",
                        "refund_amount": 99,
                    },
                },
            ],
        }
    ]
    feedback = memory.feedback(
        candidate_calls=[
            ("process_return", {"item_id": "ITEM-1", "amount": 90, "confirm": True})
        ],
        conversation=conversation,
        available_tools=["get_order", "get_policies", "process_return"],
    )
    assert feedback and "live preview value 99" in feedback


def test_specialized_return_policy_satisfies_policy_evidence() -> None:
    memory = TransitionAwareMemory(learned=False)
    conversation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_order",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {"order_id": "ORD-1", "items": [{"item_id": "ITEM-1"}]},
                },
                {
                    "name": "get_return_policy",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {"order_id": "ORD-1", "eligible": True},
                },
                {
                    "name": "process_return",
                    "arguments": {"order_id": "ORD-1", "confirm": False},
                    "result": {
                        "status": "preview",
                        "order_id": "ORD-1",
                        "refund_amount": 99,
                    },
                },
            ],
        }
    ]
    feedback = memory.feedback(
        candidate_calls=[
            (
                "process_return",
                {"order_id": "ORD-1", "amount": 99, "confirm": True},
            )
        ],
        conversation=conversation,
        available_tools=[
            "get_order",
            "get_return_policy",
            "get_policies",
            "process_return",
        ],
    )
    assert feedback is None


def test_direct_booking_attribute_update_does_not_require_stale_booking() -> None:
    memory = TransitionAwareMemory(learned=False)
    feedback = memory.feedback(
        candidate_calls=[
            ("update_booking", {"booking_id": "BK-1", "seat_type": "window"})
        ],
        conversation=[],
        available_tools=["get_booking", "update_booking"],
    )
    assert feedback is None


def test_booking_attribute_update_with_empty_flight_id_does_not_require_preview() -> None:
    memory = TransitionAwareMemory(learned=False)
    feedback = memory.feedback(
        candidate_calls=[
            (
                "update_booking",
                {
                    "booking_id": "BK-1",
                    "flight_id": "",
                    "seat_type": "window",
                    "confirm": True,
                },
            )
        ],
        conversation=[],
        available_tools=["get_booking", "update_booking"],
    )
    assert feedback is None


def test_invalidated_entity_evidence_cannot_be_reused() -> None:
    memory = TransitionAwareMemory(
        {
            "read_cart": ToolContract(reads=frozenset({"cart", "total"})),
            "mutate_cart": ToolContract(
                writes=frozenset({"cart"}),
                invalidates=frozenset({"total"}),
                requires_fresh=frozenset({"cart"}),
            ),
            "checkout": ToolContract(
                writes=frozenset({"order"}),
                requires_fresh=frozenset({"total"}),
            ),
        },
        learned=False,
    )
    conversation = [
        {
            "role": "assistant",
            "tool_calls": [
                {"name": "read_cart", "arguments": {"customer_id": "C-1"}},
                {"name": "mutate_cart", "arguments": {"customer_id": "C-1"}},
                {"name": "read_cart", "arguments": {"customer_id": "C-2"}},
            ],
        }
    ]
    feedback = memory.feedback(
        candidate_calls=[("checkout", {"customer_id": "C-1"})],
        conversation=conversation,
        available_tools=["read_cart", "checkout"],
    )
    assert feedback and "total" in feedback


def test_mutation_preserves_preview_for_another_entity() -> None:
    memory = TransitionAwareMemory(learned=False)
    conversation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_order",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {
                        "order_id": "ORD-1",
                        "items": [{"item_id": "ITEM-1"}, {"item_id": "ITEM-2"}],
                    },
                },
                {
                    "name": "get_policies",
                    "arguments": {"topic": "return"},
                    "result": {"topic": "return"},
                },
                {
                    "name": "process_return",
                    "arguments": {
                        "item_id": "ITEM-1",
                        "amount": 10,
                        "confirm": False,
                    },
                    "result": {
                        "status": "preview",
                        "item_id": "ITEM-1",
                        "refund_amount": 10,
                    },
                },
                {
                    "name": "process_return",
                    "arguments": {
                        "item_id": "ITEM-2",
                        "amount": 20,
                        "confirm": False,
                    },
                    "result": {
                        "status": "preview",
                        "item_id": "ITEM-2",
                        "refund_amount": 20,
                    },
                },
                {
                    "name": "process_return",
                    "arguments": {
                        "item_id": "ITEM-1",
                        "amount": 10,
                        "confirm": True,
                    },
                    "result": {"status": "returned", "item_id": "ITEM-1"},
                },
            ],
        }
    ]
    feedback = memory.feedback(
        candidate_calls=[
            (
                "process_return",
                {"item_id": "ITEM-2", "amount": 20, "confirm": True},
            )
        ],
        conversation=conversation,
        available_tools=["get_order", "get_policies", "process_return"],
    )
    assert feedback is None


def test_mutation_preserves_another_operation_preview_for_same_entity() -> None:
    memory = TransitionAwareMemory(learned=False)
    conversation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_order",
                    "arguments": {"order_id": "ORD-1"},
                    "result": {
                        "order_id": "ORD-1",
                        "items": [{"item_id": "ITEM-1"}],
                    },
                },
                {
                    "name": "get_policies",
                    "arguments": {"topic": "return"},
                    "result": {"topic": "return"},
                },
                {
                    "name": "get_policies",
                    "arguments": {"topic": "refund"},
                    "result": {"topic": "refund"},
                },
                {
                    "name": "process_refund",
                    "arguments": {
                        "item_id": "ITEM-1",
                        "amount": 20,
                        "confirm": False,
                    },
                    "result": {
                        "status": "preview",
                        "item_id": "ITEM-1",
                        "refund_amount": 20,
                    },
                },
                {
                    "name": "process_return",
                    "arguments": {
                        "item_id": "ITEM-1",
                        "amount": 20,
                        "confirm": True,
                    },
                    "result": {"status": "returned", "item_id": "ITEM-1"},
                },
            ],
        }
    ]
    feedback = memory.feedback(
        candidate_calls=[
            (
                "process_refund",
                {"item_id": "ITEM-1", "amount": 20, "confirm": True},
            )
        ],
        conversation=conversation,
        available_tools=[
            "get_order",
            "get_policies",
            "process_return",
            "process_refund",
        ],
    )
    assert feedback is None


def test_transition_memory_loads_learned_contract_as_soft_path(tmp_path: Path) -> None:
    artifact_path = tmp_path / "contracts.json"
    artifact_path.write_text(
        json.dumps(
            {
                "domains": {
                    "demo": {
                        "contracts": {
                            "mutate": {
                                "required_tools": [
                                    {"tool": "read_state", "support": 3, "confidence": 1.0}
                                ]
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    memory = TransitionAwareMemory(
        {
            "read_state": ToolContract(reads=frozenset({"state"})),
            "mutate": ToolContract(writes=frozenset({"state"})),
        },
        domain="demo",
        artifact_path=artifact_path,
    )
    context = memory.prompt_context([])
    assert '"mutate": ["read_state"]' in context


def test_transition_memory_enforcement_is_domain_agnostic(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"cards": []}), encoding="utf-8")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    monkeypatch.setenv("STATE_BENCH_TAPM_MODE", "enforce")
    client = ScriptedClient(
        [
            completion(
                tool_calls=[
                    {
                        "name": "cancel_booking",
                        "arguments": {"booking_id": "BK-1", "confirm": True},
                    }
                ]
            ),
            completion(tool_calls=[{"name": "get_booking", "arguments": {}}]),
        ]
    )
    agent = ProcessWorkflowMemoryAgent(
        client,
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel"),
    )
    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=[{"role": "user", "content": "Update my booking."}],
        tools=[
            {"name": "get_booking", "parameters": {}},
            {"name": "cancel_booking", "parameters": {}},
        ],
    )
    assert response.tool_calls[0].name == "get_booking"
    assert len(client.calls) == 2


def test_agent_revises_final_answer_to_reconcile_write(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"cards": []}), encoding="utf-8")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    monkeypatch.setenv("STATE_BENCH_TAPM_MODE", "enforce")
    client = ScriptedClient(
        [
            completion(text="The item was added."),
            completion(tool_calls=[{"name": "get_cart", "arguments": {"customer_id": "C-1"}}]),
        ]
    )
    agent = ProcessWorkflowMemoryAgent(
        client,
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="shopping_assistant"),
    )
    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=[
            {"role": "user", "content": "Add it."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "add_to_cart",
                        "arguments": {"customer_id": "C-1", "product_id": "P-1"},
                        "result": {"status": "success"},
                    }
                ],
            },
        ],
        tools=[{"name": "get_cart", "parameters": {}}],
    )
    assert response.tool_calls[0].name == "get_cart"
    assert len(client.calls) == 2


def completion(text="", tool_calls=None):
    return SimpleNamespace(text=text, tool_calls=tool_calls or [], usage=None)


def build_verifier_agent(tmp_path, monkeypatch, client, *, runtime_rules=None, max_revisions=2):
    monkeypatch.setenv("STATE_BENCH_TAPM_MODE", "off")
    card = {
        "id": "shopping_assistant:add_to_cart:0",
        "domain": "shopping_assistant",
        "family": "add_to_cart",
        "support": 8,
        "mean_fitness": 1.0,
        "quality": 0.9,
        "observed_tools": ["get_cart", "add_to_cart"],
        "search_text": "add product to cart after checking current cart",
        "tokens": ["add", "product", "cart"],
        "text": "ADD WORKFLOW",
        "awm_text": "ADD WORKFLOW",
        "process_text": "ADD PROCESS",
        "mandatory_disclosures": ["Report the authoritative updated cart total."],
        "confirmation_gates": ["Require explicit approval before add_to_cart."],
        "refresh_after_mutation": ["Refresh cart state before the final response."],
        "forbidden_actions": ["Do not add an unapproved item."],
        "runtime_rules": runtime_rules or [],
    }
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"cards": [card]}), encoding="utf-8")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    monkeypatch.setenv("STATE_BENCH_VERIFIER_MODE", "full")
    monkeypatch.setenv("STATE_BENCH_VERIFIER_MAX_REVISIONS", str(max_revisions))
    agent = ProcessWorkflowMemoryAgent(
        client,
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="shopping_assistant"),
        retrieve_learnings_top_k=1,
    )
    conversation = [{"role": "user", "content": "Add this product to my cart."}]
    agent.prepare_conversation(conversation)
    return agent, conversation


def test_hybrid_retrieval_prefers_matching_workflow(tmp_path: Path, monkeypatch) -> None:
    cards = [
        {
            "domain": "customer_support",
            "family": "process_return:return",
            "support": 8,
            "mean_fitness": 1.0,
            "quality": 0.9,
            "observed_tools": ["get_order", "process_return"],
            "search_text": "return defective headphones send item back refund",
            "tokens": ["return", "defective", "headphones", "refund"],
            "text": "RETURN WORKFLOW",
            "awm_text": "RETURN AWM",
            "process_text": "RETURN PROCESS",
        },
        {
            "domain": "customer_support",
            "family": "process_warranty_claim:warranty",
            "support": 5,
            "mean_fitness": 0.9,
            "quality": 0.8,
            "observed_tools": ["get_warranty_status", "process_warranty_claim"],
            "search_text": "warranty repair recurring defect claim",
            "tokens": ["warranty", "repair", "claim"],
            "text": "WARRANTY WORKFLOW",
            "awm_text": "WARRANTY AWM",
            "process_text": "WARRANTY PROCESS",
        },
    ]
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"cards": cards}), encoding="utf-8")
    monkeypatch.setenv("STATE_BENCH_MEMORY_PATH", str(path))
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    agent = ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="customer_support"),
        retrieve_learnings_top_k=1,
    )
    assert agent.retrieve_learnings("I need to return defective headphones", top_k=1) == ["RETURN WORKFLOW"]


def test_query_does_not_use_runtime_task_summary(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"cards": []}), encoding="utf-8")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    agent = ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel", task_summary="oracle answer"),
    )
    query = agent._query_from_conversation([{"role": "user", "content": "change my flight"}])
    assert query == "change my flight"
    assert "oracle answer" not in query


def test_account_history_risk_selects_specific_card(tmp_path: Path, monkeypatch) -> None:
    common = {
        "domain": "shopping_assistant",
        "family": "account_history+add_to_cart",
        "support": 1,
        "mean_fitness": 1.0,
        "quality": 0.8,
        "observed_tools": ["get_customer_account", "add_to_cart"],
        "tokens": ["add", "product", "cart"],
        "text": "workflow",
        "awm_text": "workflow",
        "process_text": "workflow",
    }
    cards = [
        {
            **common,
            "id": "history",
            "applies_when": "Prior purchase history may make this a duplicate purchase.",
            "keywords": ["already purchased", "buy again"],
            "search_text": "add product cart duplicate purchase history",
        },
        {
            **common,
            "id": "generic",
            "applies_when": "Resolve product variants before adding a catalog item.",
            "keywords": ["search product", "variant lookup"],
            "search_text": "add product cart search variant",
        },
    ]
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"cards": cards}), encoding="utf-8")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    agent = ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="shopping_assistant"),
        retrieve_learnings_top_k=1,
    )

    selected = agent._rank_cards("I may have bought it before; add it again.", top_k=1)

    assert selected[0]["id"] == "history"


def test_more_of_existing_item_routes_to_update_cart(tmp_path: Path, monkeypatch) -> None:
    cards = [
        {
            "id": "add",
            "domain": "shopping_assistant",
            "family": "add_to_cart",
            "support": 5,
            "mean_fitness": 1.0,
            "quality": 0.8,
            "observed_tools": ["add_to_cart"],
            "search_text": "add a new product",
            "applies_when": "Add a new named product.",
            "keywords": ["add item"],
            "tokens": ["add", "product"],
            "text": "add workflow",
            "awm_text": "add workflow",
            "process_text": "add workflow",
        },
        {
            "id": "update",
            "domain": "shopping_assistant",
            "family": "update_cart",
            "support": 5,
            "mean_fitness": 1.0,
            "quality": 0.8,
            "observed_tools": ["get_cart", "update_cart_item"],
            "search_text": "existing cart quantity add more",
            "applies_when": "Increase an item already in the cart.",
            "keywords": ["add more", "existing quantity"],
            "tokens": ["existing", "cart", "quantity", "more"],
            "text": "update workflow",
            "awm_text": "update workflow",
            "process_text": "update workflow",
        },
    ]
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"cards": cards}), encoding="utf-8")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    agent = ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="shopping_assistant"),
        retrieve_learnings_top_k=1,
    )

    selected = agent._rank_cards("Add 3 more of those docks to my cart.", top_k=1)

    assert selected[0]["id"] == "update"


def test_deterministic_rule_revises_without_llm_verifier(tmp_path: Path, monkeypatch) -> None:
    runtime_rules = [
        {
            "id": "read_cart_first",
            "phase": "pre_write",
            "kind": "require_tool",
            "trigger_tools": ["add_to_cart"],
            "required_tools": ["get_cart"],
            "condition": "Before adding, inspect the current cart.",
            "feedback": "Call get_cart before add_to_cart.",
            "enforcement": "deterministic",
        }
    ]
    client = ScriptedClient(
        [
            completion(tool_calls=[{"name": "add_to_cart", "arguments": {"product_id": "SP-1"}}]),
            completion(tool_calls=[{"name": "get_cart", "arguments": {"customer_id": "shop_1"}}]),
        ]
    )
    agent, conversation = build_verifier_agent(
        tmp_path, monkeypatch, client, runtime_rules=runtime_rules
    )
    response = agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[]
    )
    assert response.tool_calls[0].name == "get_cart"
    assert len(client.calls) == 2
    assert "Runtime verification found: Call get_cart" in client.calls[1]["conversation"][-1]["content"]


def test_required_read_remains_valid_after_user_confirmation(tmp_path: Path, monkeypatch) -> None:
    runtime_rules = [
        {
            "id": "read_cart_first",
            "phase": "pre_write",
            "kind": "require_tool",
            "trigger_tools": ["add_to_cart"],
            "required_tools": ["get_cart"],
            "condition": "Inspect the cart before adding.",
            "feedback": "Call get_cart before add_to_cart.",
            "enforcement": "deterministic",
        }
    ]
    client = ScriptedClient(
        [
            completion(tool_calls=[{"name": "add_to_cart", "arguments": {"product_id": "SP-1"}}]),
            completion(text='{"decision":"allow","confidence":0.95,"workflow_ids":[]}'),
        ]
    )
    agent, conversation = build_verifier_agent(
        tmp_path, monkeypatch, client, runtime_rules=runtime_rules
    )
    conversation[:] = [
        {"role": "user", "content": "Check whether I can add this item."},
        {
            "role": "assistant",
            "content": "It is ready to add.",
            "tool_calls": [
                {
                    "name": "get_cart",
                    "arguments": {"customer_id": "shop_1"},
                    "result": {"items": []},
                }
            ],
        },
        {"role": "user", "content": "Yes, please add it."},
    ]

    response = agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[]
    )

    assert response.tool_calls[0].name == "add_to_cart"
    assert len(client.calls) == 2


def test_semantic_verifier_feedback_revises_write(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient(
        [
            completion(tool_calls=[{"name": "add_to_cart", "arguments": {"product_id": "SP-1"}}]),
            completion(
                text=json.dumps(
                    {
                        "decision": "revise",
                        "confidence": 0.95,
                        "workflow_ids": ["shopping_assistant:add_to_cart:0"],
                        "violations": ["missing confirmation"],
                        "feedback": "Ask which variant the user wants before adding.",
                    }
                )
            ),
            completion(tool_calls=[{"name": "get_cart", "arguments": {"customer_id": "shop_1"}}]),
        ]
    )
    agent, conversation = build_verifier_agent(tmp_path, monkeypatch, client)
    response = agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[]
    )
    assert response.tool_calls[0].name == "get_cart"
    assert len(client.calls) == 3
    assert client.calls[1]["timeout_seconds"] == 60
    assert client.calls[1]["max_retries"] == 0


def test_low_confidence_verdict_does_not_overblock(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient(
        [
            completion(tool_calls=[{"name": "add_to_cart", "arguments": {"product_id": "SP-1"}}]),
            completion(text='{"decision":"revise","confidence":0.4,"workflow_ids":["shopping_assistant:add_to_cart:0"],"feedback":"uncertain"}'),
        ]
    )
    agent, conversation = build_verifier_agent(tmp_path, monkeypatch, client)
    response = agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[]
    )
    assert response.tool_calls[0].name == "add_to_cart"
    assert len(client.calls) == 2


def test_rejected_write_fails_closed_after_retry_limit(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient(
        [
            completion(tool_calls=[{"name": "add_to_cart", "arguments": {"product_id": "SP-1"}}]),
            completion(text='{"decision":"revise","confidence":0.99,"workflow_ids":["shopping_assistant:add_to_cart:0"],"feedback":"confirmation missing"}'),
        ]
    )
    agent, conversation = build_verifier_agent(
        tmp_path, monkeypatch, client, max_revisions=0
    )
    conversation[0]["content"] = "I'm considering this product."
    response = agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[]
    )
    assert response.tool_calls == []
    assert response.text


def test_semantic_rejection_fails_open_for_explicitly_authorized_write(
    tmp_path: Path, monkeypatch
) -> None:
    client = ScriptedClient(
        [
            completion(tool_calls=[{"name": "add_to_cart", "arguments": {"product_id": "SP-1"}}]),
            completion(
                text='{"decision":"revise","confidence":0.99,'
                '"workflow_ids":["shopping_assistant:add_to_cart:0"],'
                '"violations":["workflow conflict"],"feedback":"Use another path."}'
            ),
        ]
    )
    agent, conversation = build_verifier_agent(
        tmp_path, monkeypatch, client, max_revisions=0
    )

    response = agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[]
    )

    assert response.tool_calls[0].name == "add_to_cart"


def test_full_mode_checks_final_completeness(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient(
        [
            completion(text="Done."),
            completion(
                text='{"decision":"revise","confidence":0.9,"workflow_ids":["shopping_assistant:add_to_cart:0"],"feedback":"Report the updated total."}'
            ),
            completion(text="The updated cart total is $42."),
            completion(text='{"decision":"allow","confidence":0.95,"feedback":""}'),
        ]
    )
    agent, conversation = build_verifier_agent(tmp_path, monkeypatch, client)
    response = agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[]
    )
    assert response.text == "The updated cart total is $42."
    assert len(client.calls) == 4


def test_post_write_refresh_is_checked_before_final(tmp_path: Path, monkeypatch) -> None:
    runtime_rules = [
        {
            "id": "refresh_cart",
            "phase": "post_write",
            "kind": "refresh",
            "trigger_tools": ["add_to_cart"],
            "required_tools": ["get_cart"],
            "condition": "Refresh the cart after adding an item.",
            "feedback": "Call get_cart before the final response.",
            "enforcement": "deterministic",
        }
    ]
    client = ScriptedClient(
        [
            completion(text="Done."),
            completion(tool_calls=[{"name": "get_cart", "arguments": {"customer_id": "shop_1"}}]),
        ]
    )
    agent, conversation = build_verifier_agent(
        tmp_path, monkeypatch, client, runtime_rules=runtime_rules
    )
    conversation.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "add_to_cart", "arguments": {"product_id": "SP-1"}}
                ],
            },
            {"role": "tool", "content": [{"result": {"status": "success"}}]},
        ]
    )
    response = agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[]
    )
    assert response.tool_calls[0].name == "get_cart"
    assert len(client.calls) == 2


def test_post_write_refresh_in_same_tool_batch_is_recognized(
    tmp_path: Path, monkeypatch
) -> None:
    runtime_rules = [
        {
            "id": "refresh_cart",
            "phase": "post_write",
            "kind": "refresh",
            "trigger_tools": ["add_to_cart"],
            "required_tools": ["get_cart"],
            "condition": "Refresh the cart after adding an item.",
            "feedback": "Call get_cart before the final response.",
            "enforcement": "deterministic",
        }
    ]
    agent, _ = build_verifier_agent(
        tmp_path, monkeypatch, ScriptedClient([]), runtime_rules=runtime_rules
    )
    feedback = agent._deterministic_feedback(
        phase="pre_final",
        response=completion(text="Done."),
        conversation=[
            {"role": "user", "content": "Add it."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "add_to_cart",
                        "arguments": {"customer_id": "C-1", "product_id": "P-1"},
                        "result": {"status": "success"},
                    },
                    {
                        "name": "get_cart",
                        "arguments": {"customer_id": "C-1"},
                        "result": {"customer_id": "C-1", "items": []},
                    },
                ],
            },
        ],
        tools=[],
    )
    assert feedback is None


def test_required_read_rejects_failed_or_wrong_entity_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    runtime_rules = [
        {
            "id": "read_order",
            "phase": "pre_write",
            "kind": "require_tool",
            "trigger_tools": ["cancel_order"],
            "required_tools": ["get_order"],
            "condition": "Read the matching order before cancellation.",
            "feedback": "Read the matching order.",
            "enforcement": "deterministic",
        }
    ]
    agent, _ = build_verifier_agent(
        tmp_path, monkeypatch, ScriptedClient([]), runtime_rules=runtime_rules
    )
    candidate = completion(
        tool_calls=[{"name": "cancel_order", "arguments": {"order_id": "ORD-2"}}]
    )
    failed_feedback = agent._deterministic_feedback(
        phase="pre_write",
        response=candidate,
        conversation=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "get_order",
                        "arguments": {"order_id": "ORD-2"},
                        "result": {"error": "not found"},
                    }
                ],
            }
        ],
        tools=[],
    )
    wrong_entity_feedback = agent._deterministic_feedback(
        phase="pre_write",
        response=candidate,
        conversation=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "get_order",
                        "arguments": {"order_id": "ORD-1"},
                        "result": {"order_id": "ORD-1"},
                    }
                ],
            }
        ],
        tools=[],
    )
    assert failed_feedback == "Read the matching order."
    assert wrong_entity_feedback == "Read the matching order."


def test_verifier_revision_is_checked_by_tapm(tmp_path: Path, monkeypatch) -> None:
    runtime_rules = [
        {
            "id": "read_cart_first",
            "phase": "pre_write",
            "kind": "require_tool",
            "trigger_tools": ["add_to_cart"],
            "required_tools": ["get_cart"],
            "condition": "Inspect the current cart.",
            "feedback": "Call get_cart before add_to_cart.",
            "enforcement": "deterministic",
        }
    ]
    client = ScriptedClient(
        [
            completion(
                tool_calls=[{"name": "add_to_cart", "arguments": {"product_id": "P-1"}}]
            ),
            completion(
                tool_calls=[
                    {
                        "name": "cancel_order",
                        "arguments": {"order_id": "ORD-2", "confirm": True},
                    }
                ]
            ),
            completion(
                tool_calls=[{"name": "get_order", "arguments": {"order_id": "ORD-2"}}]
            ),
        ]
    )
    agent, conversation = build_verifier_agent(
        tmp_path, monkeypatch, client, runtime_rules=runtime_rules
    )
    agent.tapm_mode = "enforce"
    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation,
        tools=[
            {"name": "get_cart", "parameters": {}},
            {"name": "get_order", "parameters": {}},
            {"name": "get_policies", "parameters": {}},
            {"name": "add_to_cart", "parameters": {}},
            {"name": "cancel_order", "parameters": {}},
        ],
    )
    assert response.tool_calls[0].name == "get_order"
    assert len(client.calls) == 3


def test_tool_error_speculation_is_deterministically_revised(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient(
        [
            completion(text="There is no reason beyond STUDENT20 not found, like expired vs restricted."),
            completion(text="The validator reported only that STUDENT20 was not found."),
            completion(text='{"decision":"allow","confidence":0.95,"workflow_ids":[]}'),
        ]
    )
    agent, conversation = build_verifier_agent(tmp_path, monkeypatch, client)
    conversation.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "validate_promo",
                        "arguments": {"promo_code": "STUDENT20"},
                        "result": {"valid": False, "reason": "Promo code not found."},
                    }
                ],
            }
        ]
    )

    response = agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[]
    )

    assert response.text == "The validator reported only that STUDENT20 was not found."
    assert len(client.calls) == 3


def test_live_tool_schema_rejects_memory_signature(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient(
        [
            completion(tool_calls=[{"name": "update_cart_item", "arguments": {"cart_item_id": "CI-1", "quantity": 1}}]),
            completion(tool_calls=[{"name": "update_cart_item", "arguments": {"customer_id": "shop_1", "product_id": "SP-1", "quantity": 1}}]),
            completion(text='{"decision":"allow","confidence":0.95,"workflow_ids":[]}'),
        ]
    )
    agent, conversation = build_verifier_agent(tmp_path, monkeypatch, client)
    conversation[0]["content"] = "Yes, please update it to 1."
    tools = [
        {
            "type": "function",
            "name": "update_cart_item",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "product_id": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["customer_id", "product_id", "quantity"],
                "additionalProperties": False,
            },
        }
    ]

    response = agent.generate_next_turn(system_prompt="system", conversation=conversation, tools=tools)

    assert response.tool_calls[0].arguments["customer_id"] == "shop_1"
    assert "cart_item_id" not in response.tool_calls[0].arguments


def test_explicit_authorization_overrides_confirmation_only_rejection(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient(
        [
            completion(tool_calls=[{"name": "add_to_cart", "arguments": {"product_id": "SP-1"}}]),
            completion(
                text=json.dumps(
                    {
                        "decision": "revise",
                        "confidence": 0.99,
                        "workflow_ids": ["shopping_assistant:add_to_cart:0"],
                        "violations": ["explicit confirmation is missing"],
                        "feedback": "Ask for confirmation before adding.",
                    }
                )
            ),
        ]
    )
    agent, conversation = build_verifier_agent(tmp_path, monkeypatch, client)
    conversation[0]["content"] = "Yes, please add it to my cart."

    response = agent.generate_next_turn(system_prompt="system", conversation=conversation, tools=[])

    assert response.tool_calls[0].name == "add_to_cart"
    assert len(client.calls) == 2


def test_unknown_compatibility_requires_product_evidence(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient(
        [
            completion(text="The device name is unknown."),
            completion(tool_calls=[{"name": "get_product_details", "arguments": {"product_id": "SP-H110"}}]),
        ]
    )
    agent, conversation = build_verifier_agent(tmp_path, monkeypatch, client)
    conversation.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "check_compatibility",
                        "arguments": {"product_id": "SP-H110", "device_name": "CreatorBook"},
                        "result": {"compatible": False, "reason": "Unknown device name 'CreatorBook'."},
                    }
                ],
            }
        ]
    )

    response = agent.generate_next_turn(system_prompt="system", conversation=conversation, tools=[])

    assert response.tool_calls[0].name == "get_product_details"
    assert response.tool_calls[0].arguments["product_id"] == "SP-H110"


def test_unknown_compatibility_requires_consuming_product_evidence(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient(
        [
            completion(text="The checker returned false because CreatorBook is unknown."),
            completion(
                text=(
                    "SP-H110 has compatible_with [ProBook Laptop 13, ProBook Laptop 15]. "
                    "CreatorBook is not listed, so the actual dock is incompatible based on the product catalog."
                )
            ),
            completion(text='{"decision":"allow","confidence":0.95,"workflow_ids":[]}'),
        ]
    )
    agent, conversation = build_verifier_agent(tmp_path, monkeypatch, client)
    conversation.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_product_details",
                        "arguments": {"product_id": "SP-H110"},
                        "result": {
                            "product_id": "SP-H110",
                            "compatible_with": ["ProBook Laptop 13", "ProBook Laptop 15"],
                        },
                    },
                    {
                        "name": "check_compatibility",
                        "arguments": {"product_id": "SP-H110", "device_name": "CreatorBook"},
                        "result": {"compatible": False, "reason": "Unknown device name 'CreatorBook'."},
                    },
                ],
            }
        ]
    )

    response = agent.generate_next_turn(system_prompt="system", conversation=conversation, tools=[])

    assert "compatible_with" in response.text
    assert "not listed" in response.text
    assert len(client.calls) == 3


def test_explicit_no_substitute_constraint_blocks_suggestion(tmp_path: Path, monkeypatch) -> None:
    agent, _ = build_verifier_agent(tmp_path, monkeypatch, ScriptedClient([]))
    feedback = agent._deterministic_feedback(
        phase="pre_final",
        response=completion(text="I can help compare a different dock."),
        conversation=[
            {"role": "user", "content": "Read my cart and do not suggest substitutes."}
        ],
        tools=[],
    )

    assert feedback and "explicitly prohibited" in feedback


def test_explicit_cart_quantity_does_not_require_duplicate_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    agent, _ = build_verifier_agent(tmp_path, monkeypatch, ScriptedClient([]))
    feedback = agent._deterministic_feedback(
        phase="pre_final",
        response=completion(text="Should I set the final quantity to 1?"),
        conversation=[
            {"role": "user", "content": "I only need one laptop; remove one please."}
        ],
        tools=[],
    )

    assert feedback and "already gave an explicit cart quantity" in feedback


def test_reply_with_confirm_is_duplicate_quantity_confirmation(tmp_path: Path, monkeypatch) -> None:
    agent, _ = build_verifier_agent(tmp_path, monkeypatch, ScriptedClient([]))
    feedback = agent._deterministic_feedback(
        phase="pre_final",
        response=completion(text='Reply with "confirm" and I will set the quantity to 2.'),
        conversation=[
            {"role": "user", "content": "Please add the second laptop back to my cart."}
        ],
        tools=[],
    )

    assert feedback and "already gave an explicit cart quantity" in feedback


def test_loyalty_clamp_requires_exact_side_effect_disclosure(tmp_path: Path, monkeypatch) -> None:
    agent, _ = build_verifier_agent(tmp_path, monkeypatch, ScriptedClient([]))
    feedback = agent._deterministic_feedback(
        phase="pre_final",
        response=completion(text="The loyalty redemption was adjusted."),
        conversation=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "update_cart_item",
                        "arguments": {"customer_id": "shop_1", "product_id": "SP-1", "quantity": 1},
                        "result": {
                            "loyalty_redemption_clamped": True,
                            "previous_loyalty_discount": 998,
                            "new_loyalty_discount": 499,
                            "loyalty_points_refunded": 49900,
                        },
                    }
                ],
            }
        ],
        tools=[],
    )

    assert feedback and "from $998 to $499" in feedback
    assert "49900 loyalty points" in feedback


def test_sticky_loyalty_discount_requires_non_restore_offer(tmp_path: Path, monkeypatch) -> None:
    agent, _ = build_verifier_agent(tmp_path, monkeypatch, ScriptedClient([]))
    feedback = agent._deterministic_feedback(
        phase="pre_final",
        response=completion(text="The cart total is $1,499 and loyalty discount is $499."),
        conversation=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_cart",
                        "arguments": {"customer_id": "shop_1"},
                        "result": {"subtotal": 999, "loyalty_discount": 499},
                    },
                    {
                        "name": "update_cart_item",
                        "arguments": {"customer_id": "shop_1", "product_id": "SP-1", "quantity": 2},
                        "result": {"status": "updated"},
                    },
                    {
                        "name": "get_cart",
                        "arguments": {"customer_id": "shop_1"},
                        "result": {"subtotal": 1998, "loyalty_discount": 499},
                    },
                ],
            }
        ],
        tools=[],
    )

    assert feedback and "did not automatically restore" in feedback


def test_budget_comparison_requires_explicit_overage_math(tmp_path: Path, monkeypatch) -> None:
    agent, _ = build_verifier_agent(tmp_path, monkeypatch, ScriptedClient([]))
    feedback = agent._deterministic_feedback(
        phase="pre_final",
        response=completion(text="The premium blender is $899, above your original $500 budget."),
        conversation=[{"role": "user", "content": "My budget is $500."}],
        tools=[],
    )

    assert feedback and "$899 exceeds the user's $500 budget by $399" in feedback


def test_complete_bundle_is_grounded_before_recommendation(tmp_path: Path, monkeypatch) -> None:
    agent, _ = build_verifier_agent(tmp_path, monkeypatch, ScriptedClient([]))
    feedback = agent._deterministic_feedback(
        phase="pre_final",
        response=completion(text="I recommend the SlimBook laptop for this bundle."),
        conversation=[
            {
                "role": "user",
                "content": "I need a laptop, backpack, headphones, and webcam under $1200.",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "search_products",
                        "arguments": {"query": "laptop"},
                        "result": {"products": [{"product_id": "SP-1"}]},
                    },
                    {
                        "name": "search_products",
                        "arguments": {"query": "backpack", "category": "accessories"},
                        "result": {"products": []},
                    },
                    {
                        "name": "search_products",
                        "arguments": {"query": "headphones"},
                        "result": {"products": [{"product_id": "SP-2"}]},
                    },
                    {
                        "name": "search_products",
                        "arguments": {"query": "webcam"},
                        "result": {"products": [{"product_id": "SP-3"}]},
                    },
                ],
            },
        ],
        tools=[],
    )

    assert feedback and "Missing catalog evidence for: backpack" in feedback


def test_loyalty_cap_claim_is_checked_against_cart_math(tmp_path: Path, monkeypatch) -> None:
    agent, _ = build_verifier_agent(tmp_path, monkeypatch, ScriptedClient([]))
    feedback = agent._deterministic_feedback(
        phase="pre_final",
        response=completion(text="Your $500 discount is already the maximum 50% cap."),
        conversation=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_policies",
                        "arguments": {"topic": "loyalty_redemption"},
                        "result": {"rules": ["Cap: 50% of the cart subtotal."]},
                    },
                    {
                        "name": "get_cart",
                        "arguments": {"customer_id": "shop_x"},
                        "result": {"subtotal": 1400, "loyalty_discount": 500},
                    },
                ],
            }
        ],
        tools=[],
    )

    assert feedback and "50% of the $1400 subtotal is $700" in feedback
