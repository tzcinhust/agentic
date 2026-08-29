from __future__ import annotations

import pytest

from agents.completion_memory import (
    CompletionItem,
    CompletionMemory,
    has_valid_tool_evidence,
    static_completion_requirements,
)


def assistant_call(name: str, arguments: dict, result: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": name, "arguments": arguments, "result": result}],
    }


def test_completion_item_schema_and_status_validation() -> None:
    item = CompletionItem(
        id="cm_1",
        type="comparison",
        description="Compare the two supported outcomes.",
        source="user:turn0",
    )
    assert item.to_dict() == {
        "id": "cm_1",
        "type": "comparison",
        "description": "Compare the two supported outcomes.",
        "source": "user:turn0",
        "status": "pending",
        "evidence": [],
    }
    with pytest.raises(ValueError):
        CompletionItem("bad", "unknown", "x", "user:turn0")
    with pytest.raises(ValueError):
        CompletionItem("bad", "comparison", "x", "user:turn0", status="done")


def test_workflow_completion_does_not_require_user_lexical_overlap() -> None:
    memory = CompletionMemory()
    memory.ingest_user_messages([{"role": "user", "content": "Please handle my booking."}])
    memory.ingest_workflows(
        [
            """Workflow: policy-backed change
Branches:
- If a penalty applies, explain the reason and report the final fee.
Avoid:
- Do not claim a completed change before confirmation.
"""
        ]
    )
    workflow_items = [item for item in memory.items if item.source.startswith("workflow:")]
    assert {item.type for item in workflow_items} >= {
        "explanation_rationale",
        "cost_amount_reporting",
        "boundary_must_not",
    }
    assert all("fee" not in item.description.lower() for item in memory.items if item.source.startswith("user:"))


def test_successful_and_failed_tool_calls_update_execution_evidence() -> None:
    conversation = [
        {"role": "user", "content": "Please make the requested update."},
        assistant_call(
            "update_booking",
            {"booking_id": "BK-1", "confirm": True},
            {"status": "updated", "booking_id": "BK-1", "total": 120},
        ),
        assistant_call(
            "cancel_hotel_reservation",
            {"reservation_id": "H-1", "confirm": True},
            {"status": "rejected", "error": "not cancellable"},
        ),
    ]
    memory = CompletionMemory()
    memory.sync_evidence(conversation)
    executions = {item.description: item for item in memory.items if item.type == "execution"}
    assert executions["The operation represented by update_booking is complete."].status == "satisfied"
    failed = next(item for item in executions.values() if "not complete" in item.description)
    assert failed.status == "pending"
    assert failed.evidence[0]["success"] is False
    assert has_valid_tool_evidence(conversation)


def test_failed_tool_alone_is_not_valid_evidence() -> None:
    conversation = [
        assistant_call(
            "process_refund",
            {"order_id": "O-1"},
            {"status": "failed", "error": "not eligible"},
        )
    ]
    assert not has_valid_tool_evidence(conversation)


def test_explicit_user_confirmation_satisfies_confirmation_item() -> None:
    memory = CompletionMemory()
    memory.ingest_workflows(
        [
            """Workflow: preview first
Branches:
- Ask the user for confirmation before completing the operation.
"""
        ]
    )
    memory.ingest_user_messages(
        [
            {"role": "user", "content": "Please review it."},
            {"role": "assistant", "content": "Would you like me to proceed?"},
            {"role": "user", "content": "Yes, go ahead."},
        ]
    )
    confirmation = next(item for item in memory.items if item.type == "user_confirmation_choice")
    assert confirmation.status == "satisfied"
    assert confirmation.evidence[-1]["kind"] == "user_confirmation"


def test_tool_state_adds_amount_disclosure_and_final_state_items() -> None:
    memory = CompletionMemory()
    memory.sync_evidence(
        [
            assistant_call(
                "process_return",
                {"order_id": "O-1", "confirm": True},
                {
                    "status": "returned",
                    "refund_amount": 69,
                    "policy_reason": "gift return uses credit",
                },
            )
        ]
    )
    pending_types = {item.type for item in memory.pending()}
    assert pending_types >= {
        "cost_amount_reporting",
        "proactive_disclosure",
        "final_state_reporting",
    }


def test_static_requirements_are_plain_untracked_text() -> None:
    requirements = static_completion_requirements(
        [
            """Workflow: compare paths
Branches:
- Compare the final fee and explain the cheaper option.
"""
        ],
        [{"role": "user", "content": "Which path is cheaper?"}],
    )
    assert requirements
    assert all(isinstance(item, str) for item in requirements)
    assert not any(isinstance(item, CompletionItem) for item in requirements)

