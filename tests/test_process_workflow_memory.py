from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent


class DummyClient:
    pass


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def completion(text="", tool_calls=None):
    return SimpleNamespace(text=text, tool_calls=tool_calls or [], usage=None)


def build_verifier_agent(tmp_path, monkeypatch, client, *, runtime_rules=None, max_revisions=2):
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
