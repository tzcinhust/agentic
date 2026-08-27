from __future__ import annotations

from types import SimpleNamespace

from agents.selective_obligation_guard import build_obligation_prompt, guard_feedback


def response(*, text: str = "", calls: list[dict] | None = None) -> SimpleNamespace:
    return SimpleNamespace(text=text, tool_calls=calls or [])


def assistant_call(name: str, arguments: dict, result: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": name, "arguments": arguments, "result": result}],
    }


def test_obligation_prompt_uses_only_top_workflow() -> None:
    workflows = [
        """Workflow: compare products
Branches:
- If one item is unavailable, explain the missing comparison and do not substitute it.
- If both are available, compare price and compatibility.
Avoid:
- Do not add anything before the user confirms.
""",
        """Workflow: unrelated shipping
Branches:
- Explain the shipping bundle discount.
""",
    ]
    prompt = build_obligation_prompt(
        workflows,
        [{"role": "user", "content": "Compare these products but do not add anything."}],
    )
    assert "missing comparison" in prompt
    assert "shipping bundle" not in prompt
    assert "do not add anything" in prompt.lower()


def test_obligation_prompt_marks_failed_tools_without_calling_them_successful() -> None:
    conversation = [
        {"role": "user", "content": "Explain whether this return is allowed."},
        assistant_call(
            "get_order",
            {"order_id": "ORD-1"},
            {"status": "rejected", "error": "not found"},
        ),
    ]
    prompt = build_obligation_prompt([], conversation)
    assert "Failed tools: get_order" in prompt
    assert "Successful evidence tools: none yet" in prompt


def test_guard_leaves_ordinary_pwm_action_unchanged() -> None:
    candidate = response(
        calls=[
            {
                "name": "add_to_cart",
                "arguments": {"customer_id": "shop_1", "product_id": "P-1"},
            }
        ]
    )
    assert guard_feedback(candidate, []) is None


def test_guard_allows_confirm_without_preview_instead_of_overblocking() -> None:
    candidate = response(
        calls=[
            {
                "name": "cancel_booking",
                "arguments": {"booking_id": "BK-1", "confirm": True},
            }
        ]
    )
    assert guard_feedback(candidate, []) is None


def test_guard_detects_historical_preview_value_conflict() -> None:
    conversation = [
        assistant_call(
            "process_refund",
            {"item_id": "ITEM-1", "amount": 30},
            {"status": "preview", "item_id": "ITEM-1", "refund_amount": 30},
        )
    ]
    candidate = response(
        calls=[
            {
                "name": "process_refund",
                "arguments": {"item_id": "ITEM-1", "amount": 10, "confirm": True},
            }
        ]
    )
    feedback = guard_feedback(candidate, conversation)
    assert feedback is not None
    assert "preview value 30" in feedback


def test_guard_allows_same_batch_preview_and_confirm() -> None:
    conversation = [
        assistant_call(
            "process_refund",
            {"item_id": "ITEM-1", "amount": 30},
            {"status": "preview", "item_id": "ITEM-1", "refund_amount": 30},
        )
    ]
    candidate = response(
        calls=[
            {
                "name": "process_refund",
                "arguments": {"item_id": "ITEM-1", "amount": 10},
            },
            {
                "name": "process_refund",
                "arguments": {"item_id": "ITEM-1", "amount": 10, "confirm": True},
            },
        ]
    )
    assert guard_feedback(candidate, conversation) is None


def test_guard_requires_refresh_only_after_same_entity_mutation() -> None:
    conversation = [
        assistant_call(
            "get_cart",
            {"customer_id": "shop_1"},
            {"customer_id": "shop_1", "items": []},
        ),
        assistant_call(
            "add_to_cart",
            {"customer_id": "shop_1", "product_id": "P-1"},
            {"status": "success", "customer_id": "shop_1", "product_id": "P-1"},
        ),
    ]
    same_customer = response(
        calls=[
            {
                "name": "remove_from_cart",
                "arguments": {"customer_id": "shop_1", "product_id": "P-1"},
            }
        ]
    )
    other_customer = response(
        calls=[
            {
                "name": "remove_from_cart",
                "arguments": {"customer_id": "shop_2", "product_id": "P-1"},
            }
        ]
    )
    assert "Call get_cart" in (guard_feedback(same_customer, conversation) or "")
    assert guard_feedback(other_customer, conversation) is None


def test_guard_accepts_post_mutation_refresh() -> None:
    conversation = [
        assistant_call(
            "add_to_cart",
            {"customer_id": "shop_1", "product_id": "P-1"},
            {"status": "success", "customer_id": "shop_1", "product_id": "P-1"},
        ),
        assistant_call(
            "get_cart",
            {"customer_id": "shop_1"},
            {"customer_id": "shop_1", "items": [{"product_id": "P-1"}]},
        ),
    ]
    candidate = response(
        calls=[
            {
                "name": "remove_from_cart",
                "arguments": {"customer_id": "shop_1", "product_id": "P-1"},
            }
        ]
    )
    assert guard_feedback(candidate, conversation) is None


def test_guard_detects_success_claim_after_failed_call() -> None:
    conversation = [
        assistant_call(
            "cancel_booking",
            {"booking_id": "BK-1", "confirm": True},
            {"status": "rejected", "error": "not cancellable"},
        )
    ]
    assert guard_feedback(response(text="Done — the booking has been cancelled."), conversation)
    assert (
        guard_feedback(
            response(text="I couldn't cancel it because the request was rejected."),
            conversation,
        )
        is None
    )
