from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_process_workflows.py"
SPEC = importlib.util.spec_from_file_location("build_process_workflows", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
_parse_trace = MODULE._parse_trace
_compile_runtime_rules = MODULE._compile_runtime_rules
_assert_train_data_root = MODULE._assert_train_data_root
_training_instance_literals = MODULE._training_instance_literals
_validate_card = MODULE._validate_card


def test_parse_trace_loads_only_matching_train_task_requirements(tmp_path) -> None:
    data_root = tmp_path / "data" / "shopping_assistant"
    tasks_root = tmp_path / "domains"
    tasks_dir = tasks_root / "shopping_assistant" / "tasks"
    data_root.mkdir(parents=True)
    tasks_dir.mkdir(parents=True)
    trajectory = {
        "conversation": [
            {"role": "user", "content": "Add that phone again."},
            {"role": "assistant", "content": "Done.", "tool_calls": []},
        ]
    }
    task = {
        "task_requirements": [
            {
                "kind": "must",
                "evidence": "tool_calls",
                "requirement": "Check purchase history before adding a duplicate.",
            }
        ]
    }
    path = data_root / "119-hard_repeat_purchase.json"
    path.write_text(json.dumps(trajectory), encoding="utf-8")
    (tasks_dir / path.name).write_text(json.dumps(task), encoding="utf-8")

    record = _parse_trace(path, "shopping_assistant", tasks_root)

    assert record.family == "account_history+add_to_cart"
    assert record.task_requirements == task["task_requirements"]


def test_compile_structured_fields_into_runtime_rules() -> None:
    card = {
        "title": "Update an existing cart item",
        "preconditions": ["Read the latest quantity with get_cart."],
        "steps": ["Call update_cart_item after approval.", "Call get_cart."],
        "branches": [],
        "avoid": [],
        "keywords": [],
        "mandatory_disclosures": ["State the requested final quantity."],
        "confirmation_gates": ["Confirm only when the final quantity is ambiguous."],
        "refresh_after_mutation": ["Call get_cart after update_cart_item."],
        "forbidden_actions": ["Do not overwrite an unverified cart item."],
        "runtime_rules": [],
    }

    rules = _compile_runtime_rules(
        card,
        observed_tools={"get_cart", "update_cart_item"},
        allowed_tools={"get_cart", "update_cart_item", "search_products"},
    )

    assert {rule["kind"] for rule in rules} == {
        "require_tool",
        "require_confirmation",
        "disclose",
        "refresh",
        "forbid",
    }
    require_cart = next(rule for rule in rules if rule["kind"] == "require_tool")
    assert require_cart["required_tools"] == ["get_cart"]
    assert require_cart["trigger_tools"] == ["update_cart_item"]
    assert require_cart["enforcement"] == "deterministic"
    refresh = next(rule for rule in rules if rule["kind"] == "refresh")
    assert refresh["required_tools"] == ["get_cart"]
    assert refresh["enforcement"] == "deterministic"


def test_llm_supplied_runtime_rules_are_recompiled_from_grounded_fields() -> None:
    card = {
        "preconditions": ["Read the latest cart with get_cart."],
        "mandatory_disclosures": [],
        "confirmation_gates": [],
        "refresh_after_mutation": [],
        "forbidden_actions": [],
        "runtime_rules": [
            {
                "id": "hallucinated",
                "phase": "pre_write",
                "kind": "require_tool",
                "trigger_tools": ["update_cart_item"],
                "required_tools": ["search_products"],
                "condition": "Always search first.",
                "feedback": "Search first.",
                "enforcement": "deterministic",
            }
        ],
    }

    rules = _compile_runtime_rules(
        card,
        observed_tools={"get_cart", "update_cart_item"},
        allowed_tools={"get_cart", "search_products", "update_cart_item"},
    )

    assert len(rules) == 1
    assert rules[0]["required_tools"] == ["get_cart"]
    assert rules[0]["id"].startswith("compiled_")


def test_card_with_training_instance_values_is_rejected(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "conversation": [
                    {"role": "user", "content": "Return my order."},
                    {
                        "role": "assistant",
                        "content": "Done.",
                        "tool_calls": [
                            {
                                "name": "get_order",
                                "arguments": {"order_id": "ORD-7512"},
                                "result": {"order_id": "ORD-7512", "total_paid": 999},
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    record = _parse_trace(trace_path, "customer_support")
    literals = _training_instance_literals([record])
    card = {
        "title": "Return ORD-7512",
        "applies_when": "The order total is $999.",
        "preconditions": [],
        "steps": [],
        "branches": [],
        "avoid": [],
        "keywords": [],
    }

    assert not _validate_card(
        card,
        {"get_order"},
        forbidden_literals=literals,
    )


def test_test_trajectory_root_is_rejected() -> None:
    try:
        _assert_train_data_root(Path("datasets/test_task_trajectories"))
    except ValueError as exc:
        assert "train trajectories only" in str(exc)
    else:
        raise AssertionError("test trajectory root should be rejected")
