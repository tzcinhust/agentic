from __future__ import annotations

from agents.completion_lifecycle import CompletionTracker
from agents.completion_templates import CompletionTemplateIndex, completion_query, tool_events


def preview(name: str, arguments: dict, result: dict) -> dict:
    return {"name": name, "arguments": arguments, "result": result}


def test_tool_events_preserve_repeated_calls_and_normalize_tool_role() -> None:
    inline = [
        {"role": "assistant", "tool_calls": [preview("get_cart", {"id": "x"}, {"total": 10})]},
        {"role": "assistant", "tool_calls": [preview("get_cart", {"id": "x"}, {"total": 10})]},
    ]
    assert len(tool_events(inline)) == 2

    explicit = [
        {"role": "assistant", "content": "", "tool_calls": [{"name": "get_cart", "arguments": {"id": "x"}}]},
        {
            "role": "tool",
            "content": [{"name": "get_cart", "arguments": {"id": "x"}, "result": {"total": 10}}],
        },
    ]
    events = tool_events(explicit)
    assert len(events) == 1
    assert events[0].result == {"total": 10}


def test_preview_and_scoped_confirmation_lifecycle() -> None:
    conversation = [
        {"role": "user", "content": "Preview both cancellations."},
        {
            "role": "assistant",
            "content": "Both are preview only. Shall I cancel them?",
            "tool_calls": [
                preview("cancel_booking", {"booking_id": "BK-101", "confirm": False}, {"status": "preview"}),
                preview("cancel_hotel", {"hotel_id": "HR-202", "confirm": False}, {"status": "preview"}),
            ],
        },
        {"role": "user", "content": "Yes, cancel BK-101 only."},
    ]
    tracker = CompletionTracker()
    tracker.sync_execution(conversation)
    executions = {item.scope["tool"]: item for item in tracker.items.values() if item.type == "execution"}
    assert executions["cancel_booking"].status == "approved_pending_execution"
    assert executions["cancel_hotel"].status == "awaiting_confirmation"
    assert not any(item.type == "execution" and item.status == "satisfied" for item in tracker.items.values())


def test_qualified_confirmation_does_not_approve_every_preview() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "I previewed the flight and hotel cancellations. Shall I cancel both?",
            "tool_calls": [
                preview("cancel_booking", {"booking_id": "BK-101", "confirm": False}, {"status": "preview"}),
                preview("cancel_hotel", {"hotel_id": "HR-202", "confirm": False}, {"status": "preview"}),
            ],
        },
        {"role": "user", "content": "Yes, the flight only; keep the hotel."},
    ]
    tracker = CompletionTracker()
    tracker.sync_execution(conversation)
    executions = {item.scope["tool"]: item for item in tracker.items.values() if item.type == "execution"}
    assert executions["cancel_booking"].status == "approved_pending_execution"
    assert executions["cancel_hotel"].status == "awaiting_confirmation"


def test_generic_yes_approves_multiple_only_when_previous_request_is_grouped() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "Shall I proceed with both cancellations?",
            "tool_calls": [
                preview("cancel_booking", {"booking_id": "BK-101", "confirm": False}, {"status": "preview"}),
                preview("cancel_hotel", {"hotel_id": "HR-202", "confirm": False}, {"status": "preview"}),
            ],
        },
        {"role": "user", "content": "Yes, proceed."},
    ]
    tracker = CompletionTracker()
    tracker.sync_execution(conversation)
    executions = [item for item in tracker.items.values() if item.type == "execution"]
    assert {item.status for item in executions} == {"approved_pending_execution"}


def test_confirmed_mutation_is_the_only_execution_discharge() -> None:
    conversation = [
        {"role": "user", "content": "Cancel it."},
        {
            "role": "assistant",
            "content": "This is a preview only. Shall I proceed?",
            "tool_calls": [
                preview("cancel_booking", {"booking_id": "BK-101", "confirm": False}, {"status": "preview"})
            ],
        },
        {"role": "user", "content": "Yes, proceed."},
        {
            "role": "assistant",
            "content": "Cancelled.",
            "tool_calls": [
                preview(
                    "cancel_booking",
                    {"booking_id": "BK-101", "confirm": True},
                    {"status": "cancelled", "booking_id": "BK-101"},
                )
            ],
        },
    ]
    tracker = CompletionTracker()
    tracker.sync_execution(conversation)
    execution = next(item for item in tracker.items.values() if item.type == "execution")
    confirmation = next(
        item for item in tracker.items.values() if item.type == "user_confirmation_choice"
    )
    assert execution.status == "satisfied"
    assert confirmation.status == "satisfied"


def test_repeated_identical_preview_refreshes_one_lifecycle_item() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "Preview only. Shall I proceed?",
            "tool_calls": [
                preview(
                    "cancel_booking",
                    {"booking_id": "BK-101", "confirm": False},
                    {"status": "preview"},
                )
            ],
        },
        {
            "role": "assistant",
            "content": "The same cancellation is still only a preview. Shall I proceed?",
            "tool_calls": [
                preview(
                    "cancel_booking",
                    {"booking_id": "BK-101", "confirm": False},
                    {"status": "pending_confirmation"},
                )
            ],
        },
        {"role": "user", "content": "Yes, proceed."},
        {
            "role": "assistant",
            "content": "Cancelled.",
            "tool_calls": [
                preview(
                    "cancel_booking",
                    {"booking_id": "BK-101", "confirm": True},
                    {"status": "cancelled"},
                )
            ],
        },
    ]
    tracker = CompletionTracker()
    tracker.sync_execution(conversation)

    executions = [item for item in tracker.items.values() if item.type == "execution"]
    confirmations = [
        item for item in tracker.items.values() if item.type == "user_confirmation_choice"
    ]
    assert len(executions) == 1
    assert len(confirmations) == 1
    assert executions[0].status == "satisfied"
    assert confirmations[0].status == "satisfied"


def test_success_cannot_discharge_a_later_preview_with_the_same_arguments() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "Cancelled directly.",
            "tool_calls": [
                preview(
                    "cancel_booking",
                    {"booking_id": "BK-101", "confirm": True},
                    {"status": "cancelled"},
                )
            ],
        },
        {
            "role": "assistant",
            "content": "This later request is only a preview. Shall I proceed?",
            "tool_calls": [
                preview(
                    "cancel_booking",
                    {"booking_id": "BK-101", "confirm": False},
                    {"status": "preview"},
                )
            ],
        },
    ]
    tracker = CompletionTracker()
    tracker.sync_execution(conversation)

    executions = [item for item in tracker.items.values() if item.type == "execution"]
    assert sorted(item.status for item in executions) == ["awaiting_confirmation", "satisfied"]


def test_completed_preview_without_bound_confirmation_records_violation() -> None:
    tracker = CompletionTracker()
    tracker.sync_execution(
        [
            {
                "role": "assistant",
                "content": "Preview only. Shall I proceed?",
                "tool_calls": [
                    preview(
                        "cancel_booking",
                        {"booking_id": "BK-101", "confirm": False},
                        {"status": "preview"},
                    )
                ],
            },
            {
                "role": "assistant",
                "content": "Cancelled.",
                "tool_calls": [
                    preview(
                        "cancel_booking",
                        {"booking_id": "BK-101", "confirm": True},
                        {"status": "cancelled"},
                    )
                ],
            },
        ]
    )
    confirmation = next(
        item for item in tracker.items.values() if item.type == "user_confirmation_choice"
    )
    assert confirmation.status == "violated"


def test_failed_mutation_never_satisfies_execution() -> None:
    tracker = CompletionTracker()
    conversation = [
        {
            "role": "assistant",
            "content": "This is only a preview. Shall I apply it?",
            "tool_calls": [
                preview(
                    "apply_promo",
                    {"promo_code": "SAVE", "confirm": False},
                    {"status": "preview"},
                )
            ],
        },
        {"role": "user", "content": "Yes, apply it."},
        {
            "role": "assistant",
            "tool_calls": [
                preview(
                    "apply_promo",
                    {"promo_code": "SAVE", "confirm": True},
                    {"status": "failed", "error": "expired"},
                )
            ],
        },
    ]
    tracker.sync_execution(conversation)
    execution = next(item for item in tracker.items.values() if item.type == "execution")
    assert execution.status == "approved_pending_execution"


def test_read_tool_money_fields_do_not_accumulate_generic_items() -> None:
    tracker = CompletionTracker()
    tracker.sync_execution(
        [
            {
                "role": "assistant",
                "tool_calls": [preview("get_cart", {}, {"total": 100, "discount": 10})],
            }
        ]
    )
    assert tracker.items == {}


def test_semantic_discharge_and_invariant_violation_are_evidence_based() -> None:
    template = {
        "id": "ct_fee",
        "family": "fee_boundary",
        "title": "Fee boundary",
        "obligations": [
            {
                "id": "explain",
                "phase": "final",
                "kind": "achievement",
                "type": "explanation_rationale",
                "requirement": "Explain the applicable timing tier.",
                "activation": "A timing boundary affects the result.",
                "required_evidence": ["authoritative timestamps"],
                "discharge": "The answer relates the timestamps to the boundary.",
                "priority": 20,
            },
            {
                "id": "no_unsupported_fee",
                "phase": "pre_claim",
                "kind": "invariant",
                "type": "boundary_must_not",
                "requirement": "Do not state a concrete fee before resolving the tier.",
                "activation": "A concrete fee may be stated.",
                "required_evidence": ["authoritative fee tier"],
                "discharge": "No unsupported fee claim occurred.",
                "priority": 0,
            },
        ],
    }
    tracker = CompletionTracker()
    tracker.merge_semantic(
        {
            "items": [
                {
                    "template_id": "ct_fee",
                    "obligation_id": "explain",
                    "scope_key": "booking",
                    "applicable": True,
                    "status": "satisfied",
                    "description": "Explain why the current timestamp falls in the applicable tier.",
                    "evidence": [{"kind": "message", "ref": "M4", "detail": "explicit explanation"}],
                    "missing_evidence": [],
                },
                {
                    "template_id": "ct_fee",
                    "obligation_id": "no_unsupported_fee",
                    "scope_key": "booking",
                    "applicable": True,
                    "status": "violated",
                    "description": "A fee was asserted before the tier was resolved.",
                    "evidence": [{"kind": "message", "ref": "M2", "detail": "unsupported fee"}],
                    "missing_evidence": [],
                },
            ]
        },
        [template],
    )
    statuses = {item.obligation_id: item.status for item in tracker.items.values()}
    assert statuses == {"explain": "satisfied", "no_unsupported_fee": "violated"}


def test_invariant_stays_active_until_final_bookkeeping() -> None:
    template = {
        "id": "ct_guard",
        "family": "guard",
        "title": "Guard",
        "obligations": [
            {
                "id": "must_not",
                "phase": "pre_claim",
                "kind": "invariant",
                "type": "boundary_must_not",
                "requirement": "Do not claim an unsupported amount.",
                "activation": "An amount may be reported.",
                "required_evidence": ["authoritative amount"],
                "discharge": "No unsupported claim occurred.",
                "priority": 0,
            }
        ],
    }
    update = {
        "items": [
            {
                "template_id": "ct_guard",
                "obligation_id": "must_not",
                "scope_key": "default",
                "applicable": True,
                "status": "satisfied",
                "description": "No unsupported amount was claimed.",
                "evidence": [{"kind": "message", "ref": "M1", "detail": "grounded claim"}],
                "missing_evidence": [],
            }
        ]
    }
    tracker = CompletionTracker()
    tracker.merge_semantic(update, [template], valid_evidence_refs={"M1"})
    item = next(iter(tracker.items.values()))
    assert item.status == "pending"
    tracker.merge_semantic(
        update,
        [template],
        valid_evidence_refs={"M1"},
        allow_invariant_satisfaction=True,
    )
    assert item.status == "satisfied"


def test_semantic_satisfaction_without_valid_evidence_stays_pending() -> None:
    template = {
        "id": "ct_explain",
        "family": "explain",
        "title": "Explain",
        "obligations": [
            {
                "id": "why",
                "phase": "final",
                "kind": "achievement",
                "type": "explanation_rationale",
                "requirement": "Explain why the outcome applies.",
                "activation": "An explanation is required.",
                "required_evidence": ["authoritative reason"],
                "discharge": "The assistant explicitly explains it.",
                "priority": 10,
            }
        ],
    }
    tracker = CompletionTracker()
    tracker.merge_semantic(
        {
            "items": [
                {
                    "template_id": "ct_explain",
                    "obligation_id": "why",
                    "scope_key": "default",
                    "applicable": True,
                    "status": "satisfied",
                    "description": "Explain why the outcome applies.",
                    "evidence": [{"kind": "message", "ref": "M99", "detail": "invented ref"}],
                    "missing_evidence": [],
                }
            ]
        },
        [template],
        valid_evidence_refs={"M0"},
    )
    item = next(iter(tracker.items.values()))
    assert item.status == "pending"
    assert item.evidence == []

    tracker.merge_semantic(
        {
            "items": [
                {
                    "template_id": "ct_explain",
                    "obligation_id": "why",
                    "scope_key": "default",
                    "applicable": False,
                    "status": "invalidated",
                    "evidence": [],
                }
            ]
        },
        [template],
        valid_evidence_refs={"M0"},
    )
    assert item.status == "pending"


def test_bookkeeper_cannot_inject_an_answer_literal_into_the_closure_description() -> None:
    template = {
        "id": "ct_amount",
        "family": "amount",
        "title": "Amount",
        "obligations": [
            {
                "id": "report",
                "phase": "final",
                "kind": "achievement",
                "type": "cost_amount_reporting",
                "requirement": "Report the exact amount supported by current authoritative evidence.",
                "activation": "An amount affects the outcome.",
                "required_evidence": ["authoritative amount"],
                "discharge": "The grounded amount is communicated.",
                "priority": 10,
            }
        ],
    }
    tracker = CompletionTracker()
    tracker.merge_semantic(
        {
            "items": [
                {
                    "template_id": "ct_amount",
                    "obligation_id": "report",
                    "scope_key": "default",
                    "applicable": True,
                    "status": "pending",
                    "description": "Tell the user the amount is $999.",
                    "evidence": [],
                    "missing_evidence": [],
                }
            ]
        },
        [template],
    )
    item = next(iter(tracker.items.values()))
    assert item.description == template["obligations"][0]["requirement"]


def test_completion_retrieval_is_independent_and_uses_observable_state() -> None:
    artifact = {
        "version": 2,
        "templates": [
            {
                "id": "shopping_benefit",
                "domain": "shopping_assistant",
                "family": "purchase_proactive_benefits",
                "search_text": "first purchase welcome benefit loyalty points add cart",
                "tokens": ["first", "purchase", "welcome", "benefit", "loyalty", "points", "add", "cart"],
                "observed_tools": ["get_customer_account", "add_to_cart"],
                "support": 5,
                "confidence": 0.9,
                "obligations": [],
            },
            {
                "id": "travel_cancel",
                "domain": "shopping_assistant",
                "family": "travel_cancel",
                "search_text": "cancel hotel flight",
                "tokens": ["cancel", "hotel", "flight"],
                "observed_tools": [],
                "support": 1,
                "confidence": 0.5,
                "obligations": [],
            },
        ],
    }
    index = CompletionTemplateIndex(artifact, domain="shopping_assistant", top_k=1)
    conversation = [
        {"role": "user", "content": "Add this on my first purchase."},
        {
            "role": "assistant",
            "tool_calls": [preview("get_customer_account", {}, {"is_first_time": True})],
        },
    ]
    query = completion_query(conversation)
    assert index.retrieve(query)[0]["id"] == "shopping_benefit"
    assert "oracle" not in query


def test_long_completion_query_preserves_opening_request_and_latest_tool_state() -> None:
    conversation = [
        {"role": "user", "content": "OPENING_PURCHASE_REQUEST " + "x" * 5000},
        {"role": "assistant", "content": "middle " + "y" * 7000},
        {
            "role": "assistant",
            "tool_calls": [
                preview(
                    "get_customer_account",
                    {},
                    {"is_first_time": True, "latest_account_signal": "LATENT_BENEFIT_MARKER"},
                )
            ],
        },
    ]
    query = completion_query(conversation)
    assert "OPENING_PURCHASE_REQUEST" in query
    assert "LATENT_BENEFIT_MARKER" in query
    assert len(query) <= 9000
