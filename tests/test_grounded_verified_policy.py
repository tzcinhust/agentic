from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from state_bench.agents.base import AgentToolCallRequest, AgentTurnResponse

from agents.cross_domain_precommit_guard import CrossDomainPrecommitGuard, GuardAction
from agents.final_response_contract import FinalResponseContract, ground_policy_sensitive_claims
from agents.grounded_verified_policy_agent import GroundedVerifiedPolicyAgent
from agents.policy_activation import ActivationValue, PolicyActivation
from agents.runtime_fact_ledger import FactState, RuntimeFactLedger


def observed(name: str, result, arguments=None, *, call_id: str | None = None):
    call = {"name": name, "arguments": arguments or {}, "result": result}
    if call_id:
        call["call_id"] = call_id
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def proposed(name: str, arguments=None, text: str = "") -> AgentTurnResponse:
    return AgentTurnResponse(text=text, tool_calls=[AgentToolCallRequest(name, arguments or {})])


def product_cart(brands: list[str | None], categories: list[str | None] | None = None, *, wraps=None):
    categories = categories or ["electronics"] * len(brands)
    wraps = wraps or [False] * len(brands)
    items = [
        {
            "product_id": f"P{index}",
            "quantity": 1,
            "unit_price": 100,
            "line_total": 100,
            "gift_wrap": wraps[index],
        }
        for index in range(len(brands))
    ]
    conversation = [observed("get_cart", {"items": items, "subtotal": 100 * len(items), "total": 100 * len(items)}, {"customer_id": "C1"})]
    for index, (brand, category) in enumerate(zip(brands, categories)):
        if brand is not None or category is not None:
            details = {"product_id": f"P{index}", "price": 100}
            if brand is not None:
                details["brand"] = brand
            if category is not None:
                details["category"] = category
            conversation.append(observed("get_product_details", details, {"product_id": f"P{index}"}))
    return conversation


def policy(topic: str, summary: str, rules: list[str]):
    return observed("get_policies", {"topic": topic, "summary": summary, "rules": rules}, {"topic": topic})


def test_travel_normal_flight_rejects_weather_reason():
    conversation = [observed("get_flight_status", {"flight_id": "F1", "status": "scheduled", "weather_disruption": False})]
    result = CrossDomainPrecommitGuard().check(
        "travel", conversation, proposed("update_booking", {"booking_id": "B1", "flight_id": "F2", "change_reason": "weather"}), RuntimeFactLedger(conversation)
    )
    assert result.action == GuardAction.BLOCK


def test_travel_false_schedule_change_is_blocked():
    conversation = [observed("get_flight_status", {"flight_id": "F1", "status": "scheduled", "schedule_changed": False})]
    result = CrossDomainPrecommitGuard().check(
        "travel", conversation, proposed("update_booking", {"booking_id": "B1", "flight_id": "F2", "change_reason": "schedule_change"}), RuntimeFactLedger(conversation)
    )
    assert result.action == GuardAction.BLOCK


def test_travel_canonical_disruption_approves_reason():
    conversation = [observed("get_flight_status", {"flight_id": "F1", "status": "scheduled", "weather_disruption": True})]
    result = CrossDomainPrecommitGuard().check(
        "travel", conversation, proposed("update_booking", {"booking_id": "B1", "flight_id": "F2", "change_reason": "weather"}), RuntimeFactLedger(conversation)
    )
    assert result.action == GuardAction.APPROVE


def test_travel_budget_contract_states_headroom():
    conversation = [{"role": "user", "content": "My budget is $700."}, observed("create_booking", {"booking_id": "B1", "total_price": 420})]
    text = FinalResponseContract().apply("travel", conversation, "Booked.", RuntimeFactLedger(conversation))
    assert "within" in text.lower()
    assert "$280" in text


def test_budget_word_followed_by_comma_is_not_a_number():
    assert FinalResponseContract()._budget("Budget, cabin rules, and timing all matter.") is None


def test_lost_package_cannot_use_cancel_order():
    conversation = [observed("get_order", {"order_id": "O1", "status": "lost"})]
    result = CrossDomainPrecommitGuard().check(
        "customer_support", conversation, proposed("cancel_order", {"order_id": "O1"}), RuntimeFactLedger(conversation)
    )
    assert result.action == GuardAction.BLOCK


def test_refund_and_reship_need_customer_choice():
    conversation = [
        {"role": "user", "content": "My parcel is missing. Help me."},
        observed("get_order", {"order_id": "O1", "status": "lost", "available_remedies": ["refund", "reship"]}),
    ]
    result = CrossDomainPrecommitGuard().check(
        "customer_support", conversation, proposed("process_refund", {"item_id": "I1", "amount": 50}), RuntimeFactLedger(conversation)
    )
    assert result.action == GuardAction.NEED_USER_CHOICE


def test_explicit_refund_choice_allows_refund_tool():
    conversation = [
        {"role": "user", "content": "Please refund it."},
        observed("get_order", {"order_id": "O1", "status": "lost", "available_remedies": ["refund", "reship"]}),
    ]
    result = CrossDomainPrecommitGuard().check(
        "customer_support", conversation, proposed("process_refund", {"item_id": "I1", "amount": 50}), RuntimeFactLedger(conversation)
    )
    assert result.action == GuardAction.APPROVE


def test_investigation_only_blocks_every_write():
    conversation = [observed("get_order", {"order_id": "O1", "investigation_status": "pending"})]
    guard = CrossDomainPrecommitGuard()
    for name in ("process_refund", "process_exchange", "cancel_order", "process_return"):
        assert guard.check("customer_support", conversation, proposed(name), RuntimeFactLedger(conversation)).action == GuardAction.BLOCK


def test_exhausted_warranty_reads_return_before_paid_repair():
    conversation = [observed("get_warranty_status", {"item_id": "I1", "eligible": False, "remaining_warranty_claims": 0})]
    response = AgentTurnResponse(text="I recommend paid repair.", tool_calls=[])
    result = CrossDomainPrecommitGuard().check("customer_support", conversation, response, RuntimeFactLedger(conversation))
    assert result.action == GuardAction.NEED_READ
    assert result.read_only_tool_calls[0].name == "get_policies"


def test_buyers_remorse_does_not_claim_shipping_refund():
    conversation = [
        observed("get_order", {"order_id": "O1", "shipping_cost": 12}),
        policy("refund", "Refund rules", ["Buyer's remorse shipping is not refundable."]),
    ]
    text = ground_policy_sensitive_claims(
        "customer_support", conversation, "The item is refundable. The shipping refund is included.", RuntimeFactLedger(conversation)
    )
    assert "shipping refund" not in text.lower()


def test_same_brand_cart_activates_brand_bundle():
    conversation = product_cart(["Acme", "Acme"]) + [policy("brand_bundle", "2+ items, 3% off", ["Two items sharing a brand qualify."])]
    result = PolicyActivation().evaluate("shopping_assistant", "brand_bundle", RuntimeFactLedger(conversation), conversation)
    assert result.value == ActivationValue.TRUE


def test_different_complete_brands_disable_brand_bundle():
    conversation = product_cart(["Acme", "Beta"]) + [policy("brand_bundle", "2+ items, 3% off", [])]
    result = PolicyActivation().evaluate("shopping_assistant", "brand_bundle", RuntimeFactLedger(conversation), conversation)
    assert result.value == ActivationValue.FALSE


def test_unread_brand_is_unknown_and_requests_details():
    conversation = product_cart([None, None], [None, None]) + [policy("brand_bundle", "2+ items, 3% off", [])]
    result = PolicyActivation().evaluate("shopping_assistant", "brand_bundle", RuntimeFactLedger(conversation), conversation)
    assert result.value == ActivationValue.UNKNOWN
    assert {call.name for call in result.missing_read_tools} == {"get_product_details"}


def test_first_time_true_activates_welcome():
    conversation = [observed("get_customer_account", {"customer_id": "C1", "is_first_time": True})]
    assert PolicyActivation().evaluate("shopping_assistant", "welcome_discount", RuntimeFactLedger(conversation), conversation).value == ActivationValue.TRUE


def test_first_time_false_suppresses_welcome_sentence():
    conversation = [observed("get_customer_account", {"customer_id": "C1", "is_first_time": False})]
    text = ground_policy_sensitive_claims(
        "shopping_assistant", conversation, "The product is in your cart. Your welcome discount is 5%.", RuntimeFactLedger(conversation)
    )
    assert "welcome" not in text.lower()
    assert "product is in your cart" in text.lower()


def test_category_bundle_and_promo_are_compared():
    conversation = product_cart(["A", "B", "C"], ["kitchen", "kitchen", "kitchen"])
    conversation += [
        policy("category_bundle", "3+ items in one category get 5% off", ["Does not stack with promo codes."]),
        observed("validate_promo", {"valid": True, "estimated_discount": 20}, {"customer_id": "C1", "promo_code": "SAVE20"}),
    ]
    text = FinalResponseContract().apply("shopping_assistant", conversation, "Here is the comparison.", RuntimeFactLedger(conversation))
    assert "$15" in text and "$20" in text
    assert "better" in text.lower() and "do not stack" in text.lower()


def test_two_wrapped_items_get_total_policy_fee():
    conversation = product_cart(["A", "B"], wraps=[True, True])
    conversation.append(policy("gift_wrap", "Gift wrapping is $5 per item", ["$5 per wrapped item."]))
    text = FinalResponseContract().apply("shopping_assistant", conversation, "Both are wrapped.", RuntimeFactLedger(conversation))
    assert "$10" in text


def test_loyalty_contract_states_total_points():
    conversation = [
        observed("get_customer_account", {"customer_id": "C1", "tier": "Gold"}),
        observed("get_cart", {"items": [], "subtotal": 200, "total": 180}, {"customer_id": "C1"}),
        policy("loyalty_points", "Rates", ["Gold: 2 points per dollar spent.", "Applied to final total."]),
    ]
    text = FinalResponseContract().apply("shopping_assistant", conversation, "Checkout complete.", RuntimeFactLedger(conversation))
    assert "360 points" in text


def test_false_brand_bundle_is_not_added():
    conversation = product_cart(["A", "B"]) + [policy("brand_bundle", "2+ items get 3% off", [])]
    text = FinalResponseContract().apply("shopping_assistant", conversation, "Cart updated.", RuntimeFactLedger(conversation))
    assert "brand-bundle" not in text.lower()


def test_single_item_cart_write_does_not_read_impossible_bundles():
    conversation = [
        observed("get_cart", {"items": [], "subtotal": 0, "total": 0}, {"customer_id": "C1"}),
        observed("get_customer_account", {"customer_id": "C1", "is_first_time": False, "tier": "Standard"}),
    ]
    response = proposed("add_to_cart", {"customer_id": "C1", "product_id": "P1", "quantity": 1})
    result = CrossDomainPrecommitGuard().check("shopping_assistant", conversation, response, RuntimeFactLedger(conversation))
    assert result.action == GuardAction.APPROVE


def test_cart_write_reads_account_for_welcome_and_loyalty():
    conversation = [observed("get_cart", {"items": [], "subtotal": 0, "total": 0}, {"customer_id": "C1"})]
    response = proposed("add_to_cart", {"customer_id": "C1", "product_id": "P1", "quantity": 1})
    result = CrossDomainPrecommitGuard().check("shopping_assistant", conversation, response, RuntimeFactLedger(conversation))
    assert result.action == GuardAction.NEED_READ
    assert [call.name for call in result.read_only_tool_calls] == ["get_customer_account"]


def test_cart_read_prefetches_account_in_same_mutation_batch():
    conversation = [{"role": "user", "content": "Add this item to my cart."}]
    response = proposed("get_cart", {"customer_id": "C1"})
    result = CrossDomainPrecommitGuard().check("shopping_assistant", conversation, response, RuntimeFactLedger(conversation))
    assert result.action == GuardAction.NEED_READ
    assert result.read_only_tool_calls[0].name == "get_customer_account"


def test_cart_write_does_not_stale_catalog_brand():
    conversation = [
        observed("get_product_details", {"product_id": "P1", "brand": "Acme", "category": "kitchen"}),
        observed("add_to_cart", {"items": [{"product_id": "P1", "quantity": 1}]}, {"customer_id": "C1", "product_id": "P1"}),
    ]
    assert RuntimeFactLedger(conversation).latest("brand", "P1").stale is False


def test_cart_write_does_not_stale_account_identity():
    conversation = [
        observed("get_customer_account", {"customer_id": "C1", "tier": "Gold", "is_first_time": True}),
        observed("add_to_cart", {"items": [{"product_id": "P1", "quantity": 1}]}, {"customer_id": "C1", "product_id": "P1"}),
    ]
    ledger = RuntimeFactLedger(conversation)
    assert ledger.latest("tier").stale is False
    assert ledger.latest("is_first_time").stale is False


def test_null_refund_is_not_zero_and_zero_write_is_blocked():
    conversation = [observed("get_order", {"order_id": "O1", "refund_amount": None})]
    ledger = RuntimeFactLedger(conversation)
    assert ledger.latest("refund_amount").state == FactState.NULL
    result = CrossDomainPrecommitGuard().check(
        "customer_support", conversation, proposed("process_refund", {"item_id": "I1", "amount": 0}), ledger
    )
    assert result.action == GuardAction.BLOCK


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return SimpleNamespace(
            text=response.text,
            tool_calls=[{"name": call.name, "arguments": call.arguments} for call in response.tool_calls],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


def test_twice_blocked_write_never_reaches_final_tool_call(monkeypatch):
    unsafe = proposed("update_booking", {"booking_id": "B1", "flight_id": "F2", "change_reason": "weather"})
    client = SequenceClient([unsafe, unsafe])
    context = SimpleNamespace(domain="travel", user_id="U1")
    monkeypatch.setattr(
        GroundedVerifiedPolicyAgent,
        "memory_path",
        Path("artifacts/statebench_cross_domain_pwm/memory/process_workflows.json"),
    )
    agent = GroundedVerifiedPolicyAgent(client, "system", [], {}, runtime_context=context)
    conversation = [observed("get_flight_status", {"flight_id": "F1", "status": "scheduled", "weather_disruption": False})]
    response = agent.generate_next_turn(system_prompt="system", conversation=conversation, tools=[])
    assert response.tool_calls == []
    assert client.calls == 2


def test_finalizer_is_idempotent():
    conversation = [{"role": "user", "content": "Keep this within a $700 budget."}, observed("create_booking", {"booking_id": "B1", "total_price": 420})]
    ledger = RuntimeFactLedger(conversation)
    contract = FinalResponseContract()
    once = contract.apply("travel", conversation, "Booked.", ledger)
    twice = contract.apply("travel", conversation, once, ledger)
    assert once == twice


def test_eighth_generation_is_text_only(monkeypatch):
    unsafe = proposed("add_to_cart", {"customer_id": "C1", "product_id": "P1"})
    client = SequenceClient([unsafe])
    context = SimpleNamespace(domain="shopping_assistant", user_id="C1")
    monkeypatch.setattr(
        GroundedVerifiedPolicyAgent,
        "memory_path",
        Path("artifacts/statebench_cross_domain_pwm/memory/process_workflows.json"),
    )
    agent = GroundedVerifiedPolicyAgent(client, "system", [], {}, runtime_context=context)
    conversation = [{"role": "user", "content": "Continue."}]
    for index in range(7):
        tool_name = "get_customer_account" if index == 0 else "get_cart"
        tool_result = {"customer_id": "C1", "is_first_time": False, "tier": "Standard"} if index == 0 else {"items": []}
        conversation.extend(
            [
                {"role": "assistant", "content": "", "tool_calls": [{"name": tool_name, "arguments": {"customer_id": "C1"}, "result": tool_result}]},
                {"role": "tool", "content": [{"name": tool_name, "arguments": {"customer_id": "C1"}, "result": tool_result}]},
            ]
        )
    response = agent.generate_next_turn(system_prompt="system", conversation=conversation, tools=[])
    assert response.tool_calls == []
    assert "verification limit" in response.text


def test_responses_api_pairs_preserve_source_and_zero():
    conversation = [
        {"type": "function_call", "call_id": "c1", "name": "get_order", "arguments": '{"order_id":"O1"}'},
        {"type": "function_call_output", "call_id": "c1", "output": '{"order_id":"O1","refund_amount":0}'},
    ]
    fact = RuntimeFactLedger(conversation).latest("refund_amount")
    assert fact.state == FactState.ZERO
    assert fact.source_tool == "get_order" and fact.source_call_id == "c1"
