from __future__ import annotations

import copy

from agents.effect_matched_contracts import (
    ActionLedger,
    ContractEvaluator,
    EffectMatchedContractIndex,
    EvidenceLedger,
    TruthValue,
    effect_signatures,
    opening_query,
    tool_events,
)


def call(name, arguments, result):
    return {"name": name, "arguments": arguments, "result": result}


def boundary_contract():
    return {
        "id": "contract_boundary",
        "domain": "travel",
        "family": "timing_boundary_consequence",
        "title": "Timing boundary consequence",
        "intent": "cancel a reservation near a fee boundary",
        "keywords": ["cancel", "timing", "fee", "next tier"],
        "search_text": "cancel reservation timing fee boundary next tier",
        "tokens": [
            "cancel",
            "reservation",
            "timing",
            "fee",
            "boundary",
            "next",
            "tier",
        ],
        "support": 4,
        "confidence": 0.9,
        "validation": {"precision": 0.8},
        "applicability": {
            "mode": "all",
            "unknown_policy": "require_resolution",
            "unknown_description": "resolve whether a timing boundary changes the outcome",
            "predicates": [
                {
                    "source": "tool_result",
                    "tool": "preview_cancel",
                    "path": "reason",
                    "operator": "contains_any",
                    "values": ["tier", "before"],
                }
            ],
        },
        "obligations": [
            {
                "id": "next_consequence",
                "deadline": "before_final",
                "type": "cost_amount_reporting",
                "requirement": "Explain the current boundary and disclose the next consequence.",
                "priority": 10,
                "evidence_requirements": [
                    {
                        "description": "authoritative next consequence",
                        "required": True,
                        "any_of": [
                            {
                                "source": "tool_result",
                                "tool": "preview_cancel",
                                "path": "next_fee",
                                "operator": "exists",
                            }
                        ],
                    }
                ],
                "response_requirements": [
                    {
                        "kind": "mention_evidence",
                        "description": "state the evidence-grounded next amount",
                        "selectors": [
                            {
                                "source": "tool_result",
                                "tool": "preview_cancel",
                                "path": "next_fee",
                                "operator": "exists",
                            }
                        ],
                        "value_mode": "numeric",
                        "min_mentions": 1,
                    },
                    {
                        "kind": "causal_explanation",
                        "description": "explain why the boundary changes the result",
                    },
                ],
            }
        ],
    }


def evidence_conversation():
    return [
        {"role": "user", "content": "Cancel my reservation."},
        {
            "role": "assistant",
            "content": "I checked the cancellation preview.",
            "tool_calls": [
                call(
                    "preview_cancel",
                    {"reservation_id": "R-1"},
                    {
                        "status": "preview",
                        "reason": "current fee tier applies before the cutoff",
                        "current_fee": 90,
                        "next_fee": 180,
                    },
                )
            ],
        },
    ]


def test_effect_signature_changes_only_after_realized_mutation() -> None:
    conversation = [
        {"role": "user", "content": "Cancel it."},
        {
            "role": "assistant",
            "content": "Preview only. Shall I proceed?",
            "tool_calls": [
                call(
                    "cancel_booking",
                    {"booking_id": "BK-1", "confirm": False},
                    {"status": "preview"},
                )
            ],
        },
        {"role": "user", "content": "Yes."},
        {
            "role": "assistant",
            "content": "Cancelled.",
            "tool_calls": [
                call(
                    "cancel_booking",
                    {"booking_id": "BK-1", "confirm": True},
                    {"status": "cancelled"},
                )
            ],
        },
        {"role": "user", "content": "Why was there a fee?"},
        {"role": "assistant", "content": "It is due to the timing tier."},
    ]
    signatures = effect_signatures(conversation)
    assert signatures[1] != signatures[3]
    assert signatures[3] == signatures[5]


def test_official_working_conversation_does_not_duplicate_tool_evidence() -> None:
    record = call(
        "get_policy", {"topic": "returns"}, {"policy": "authoritative result"},
    )
    conversation = [
        {"role": "user", "content": "Check the policy."},
        {"role": "assistant", "content": "", "tool_calls": [record]},
        {"role": "tool", "content": [record]},
    ]

    events = tool_events(conversation)

    assert len(events) == 1
    assert events[0].assistant_index == 1
    assert (
        len(
            EvidenceLedger(conversation).satisfying(
                {
                    "source": "tool_result",
                    "tool": "get_policy",
                    "path": "policy",
                    "operator": "exists",
                }
            )
        )
        == 1
    )


def test_hidden_intermediate_tool_text_cannot_discharge_final_communication() -> None:
    record = call(
        "preview_cancel",
        {"reservation_id": "R-1"},
        {
            "status": "preview",
            "reason": "current fee tier applies before the cutoff",
            "next_fee": 180,
        },
    )
    conversation = [
        {"role": "user", "content": "Cancel my reservation."},
        {
            "role": "assistant",
            "content": "Because waiting crosses the tier, the next fee is $180.",
            "tool_calls": [record],
        },
        {"role": "tool", "content": [record]},
    ]
    response = type(
        "Response", (), {"text": "The preview is ready.", "tool_calls": []}
    )()

    assert (
        ContractEvaluator([boundary_contract()], conversation)
        .gate(response)
        .should_recover
    )


def test_preview_and_scoped_yes_never_count_as_execution() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "I previewed the flight and hotel. Shall I cancel either?",
            "tool_calls": [
                call("cancel_booking", {"booking_id": "BK-1"}, {"status": "preview"}),
                call("cancel_hotel", {"hotel_id": "HR-2"}, {"status": "preview"}),
            ],
        },
        {"role": "user", "content": "Yes, BK-1 only."},
    ]
    ledger = ActionLedger(conversation)
    states = {record.tool: record.state for record in ledger.records}
    assert states == {
        "cancel_booking": "approved_pending_execution",
        "cancel_hotel": "invalidated",
    }
    assert "executed" not in states.values()


def test_ambiguous_entity_words_do_not_bind_one_of_several_previews() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "I previewed the flight and hotel cancellations. Shall I proceed?",
            "tool_calls": [
                call("cancel_booking", {"booking_id": "BK-1"}, {"status": "preview"}),
                call("cancel_hotel", {"hotel_id": "HR-2"}, {"status": "preview"}),
            ],
        },
        {"role": "user", "content": "Yes, the flight only; keep the hotel."},
    ]
    ledger = ActionLedger(conversation)
    assert {record.state for record in ledger.records} == {"awaiting_confirmation"}


def test_negative_scope_overrides_a_grouped_confirmation_question() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "I previewed both operations. Shall I execute both?",
            "tool_calls": [
                call("cancel_booking", {"booking_id": "BK-1"}, {"status": "preview"}),
                call("cancel_hotel", {"hotel_id": "HR-2"}, {"status": "preview"}),
            ],
        },
        {"role": "user", "content": "Yes, but keep the second one."},
    ]

    assert {record.state for record in ActionLedger(conversation).records} == {
        "awaiting_confirmation"
    }


def test_yes_binds_the_single_identifier_named_in_the_confirmation_request() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "I previewed both. Shall I cancel BK-1 now?",
            "tool_calls": [
                call("cancel_booking", {"booking_id": "BK-1"}, {"status": "preview"}),
                call("cancel_hotel", {"hotel_id": "HR-2"}, {"status": "preview"}),
            ],
        },
        {"role": "user", "content": "Yes, proceed."},
    ]
    ledger = ActionLedger(conversation)
    states = {record.tool: record.state for record in ledger.records}
    assert states["cancel_booking"] == "approved_pending_execution"
    assert states["cancel_hotel"] == "awaiting_confirmation"


def test_explicit_only_choice_invalidates_the_unselected_preview() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "I previewed both. Shall I proceed?",
            "tool_calls": [
                call("cancel_booking", {"booking_id": "BK-1"}, {"status": "preview"}),
                call("cancel_hotel", {"hotel_id": "HR-2"}, {"status": "preview"}),
            ],
        },
        {"role": "user", "content": "Yes, BK-1 only."},
    ]
    states = {
        record.tool: record.state for record in ActionLedger(conversation).records
    }

    assert states == {
        "cancel_booking": "approved_pending_execution",
        "cancel_hotel": "invalidated",
    }


def test_explicit_rejection_invalidates_a_single_preview() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "The preview is ready. Shall I proceed?",
            "tool_calls": [
                call("cancel_booking", {"booking_id": "BK-1"}, {"status": "preview"})
            ],
        },
        {"role": "user", "content": "No, never mind."},
    ]

    assert ActionLedger(conversation).records[0].state == "invalidated"


def test_preview_and_confirm_match_when_computed_amount_changes() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "Preview only. Shall I process the return?",
            "tool_calls": [
                call(
                    "process_return",
                    {
                        "item_id": "ITEM-1",
                        "reason": "changed_mind",
                        "amount": 0,
                        "confirm": False,
                    },
                    {"status": "preview", "refund_amount": 879},
                )
            ],
        },
        {"role": "user", "content": "Yes, proceed."},
        {
            "role": "assistant",
            "content": "The return is complete.",
            "tool_calls": [
                call(
                    "process_return",
                    {
                        "item_id": "ITEM-1",
                        "reason": "changed_mind",
                        "amount": 879,
                        "confirm": True,
                    },
                    {"status": "returned", "refund_amount": 879},
                )
            ],
        },
    ]
    ledger = ActionLedger(conversation)
    assert len(ledger.records) == 1
    assert ledger.records[0].state == "executed"


def test_natural_if_you_want_confirmation_is_bound() -> None:
    ledger = ActionLedger(
        [
            {
                "role": "assistant",
                "content": "This is a preview. If you want, I can cancel it now.",
                "tool_calls": [
                    call(
                        "cancel_booking",
                        {"booking_id": "BK-1", "confirm": False},
                        {"status": "preview"},
                    )
                ],
            },
            {"role": "user", "content": "That works, please do."},
        ]
    )
    assert ledger.records[0].state == "approved_pending_execution"


def test_changed_preview_invalidates_prior_approval() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "The fee preview is $50. Shall I proceed?",
            "tool_calls": [
                call(
                    "cancel_booking",
                    {"booking_id": "BK-1", "confirm": False},
                    {"status": "preview", "fee": 50},
                )
            ],
        },
        {"role": "user", "content": "Yes, proceed."},
        {
            "role": "assistant",
            "content": "The updated preview is $75. Shall I proceed?",
            "tool_calls": [
                call(
                    "cancel_booking",
                    {"booking_id": "BK-1", "confirm": False},
                    {"status": "preview", "fee": 75},
                )
            ],
        },
    ]
    assert ActionLedger(conversation).records[0].state == "awaiting_confirmation"


def test_identical_repreview_preserves_prior_approval() -> None:
    preview = call(
        "cancel_booking",
        {"booking_id": "BK-1", "confirm": False},
        {"status": "preview", "fee": 50},
    )
    conversation = [
        {
            "role": "assistant",
            "content": "The fee preview is $50. Shall I proceed?",
            "tool_calls": [preview],
        },
        {"role": "user", "content": "Yes, proceed."},
        {
            "role": "assistant",
            "content": "The same preview still applies.",
            "tool_calls": [preview],
        },
    ]
    assert ActionLedger(conversation).records[0].state == "approved_pending_execution"


def test_execution_without_bound_confirmation_is_auditable() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "Preview only. Shall I proceed?",
            "tool_calls": [
                call(
                    "cancel_booking",
                    {"booking_id": "BK-1", "confirm": False},
                    {"status": "preview"},
                )
            ],
        },
        {
            "role": "assistant",
            "content": "Cancelled.",
            "tool_calls": [
                call(
                    "cancel_booking",
                    {"booking_id": "BK-1", "confirm": True},
                    {"status": "cancelled"},
                )
            ],
        },
    ]
    record = ActionLedger(conversation).records[0]
    assert record.state == "executed"
    assert record.confirmation_obtained is False
    assert record.violation == "executed_without_bound_confirmation"


def test_executed_action_preserves_prior_confirmation_satisfaction() -> None:
    conversation = [
        {
            "role": "assistant",
            "content": "Preview only. Shall I proceed?",
            "tool_calls": [
                call(
                    "cancel_booking",
                    {"booking_id": "BK-1", "confirm": False},
                    {"status": "preview"},
                )
            ],
        },
        {"role": "user", "content": "Yes, proceed."},
        {
            "role": "assistant",
            "content": "Cancelled.",
            "tool_calls": [
                call(
                    "cancel_booking",
                    {"booking_id": "BK-1", "confirm": True},
                    {"status": "cancelled"},
                )
            ],
        },
    ]
    truth, records = ActionLedger(conversation).evaluate(
        {
            "tool": "cancel_booking",
            "state": "approved_pending_execution",
            "confirmation_required": True,
        }
    )

    assert truth == TruthValue.TRUE
    assert records[0].state == "executed"


def test_evidence_ledger_distinguishes_false_unknown_and_true() -> None:
    ledger = EvidenceLedger(evidence_conversation())
    true, _ = ledger.evaluate(
        {
            "source": "tool_result",
            "tool": "preview_cancel",
            "path": "next_fee",
            "operator": "equals",
            "value": 180,
        }
    )
    false, _ = ledger.evaluate(
        {
            "source": "tool_result",
            "tool": "preview_cancel",
            "path": "next_fee",
            "operator": "equals",
            "value": 90,
        }
    )
    unknown, _ = ledger.evaluate(
        {
            "source": "tool_result",
            "tool": "preview_cancel",
            "path": "missing_field",
            "operator": "exists",
        }
    )
    assert (true, false, unknown) == (
        TruthValue.TRUE,
        TruthValue.FALSE,
        TruthValue.UNKNOWN,
    )


def test_selector_defaults_to_existential_matching() -> None:
    ledger = EvidenceLedger(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    call(
                        "get_items",
                        {},
                        {"items": [{"eligible": False}, {"eligible": True}]},
                    )
                ],
            }
        ]
    )
    value, _ = ledger.evaluate(
        {
            "source": "tool_result",
            "tool": "get_items",
            "path": "items.*.eligible",
            "operator": "truthy",
        }
    )
    assert value == TruthValue.TRUE


def test_false_predicate_does_not_count_as_required_evidence() -> None:
    contract = copy.deepcopy(boundary_contract())
    selector = contract["obligations"][0]["evidence_requirements"][0]["any_of"][0]
    selector.update({"operator": "equals", "value": 999})
    state = ContractEvaluator([contract], evidence_conversation()).states()[0]
    assert state.status == "pending_evidence"


def test_failed_tool_fields_do_not_ground_completion_by_default() -> None:
    contract = copy.deepcopy(boundary_contract())
    contract["applicability"] = {
        "mode": "all",
        "unknown_policy": "inactive",
        "predicates": [
            {
                "source": "user_text",
                "path": "content",
                "operator": "contains",
                "value": "cancel",
            }
        ],
    }
    conversation = [
        {"role": "user", "content": "Cancel my reservation."},
        {
            "role": "assistant",
            "tool_calls": [
                call(
                    "preview_cancel",
                    {"reservation_id": "R-1"},
                    {"status": "failed", "next_fee": 180},
                )
            ],
        },
    ]
    evaluator = ContractEvaluator([contract], conversation)

    assert evaluator.states()[0].status == "pending_evidence"
    failed_selector = {
        "source": "tool_result",
        "tool": "preview_cancel",
        "path": "next_fee",
        "operator": "exists",
        "outcome": "failure",
    }
    assert evaluator.evidence.satisfying(failed_selector, trustworthy_only=True)


def test_response_discharge_is_scoped_after_relevant_evidence() -> None:
    conversation = [
        {"role": "user", "content": "Cancel my reservation."},
        {
            "role": "assistant",
            "content": "Because this is a cancellation, I will check.",
        },
        *evidence_conversation()[1:],
    ]
    response = type(
        "Response", (), {"text": "The next fee would be $180.", "tool_calls": []}
    )()
    assert (
        ContractEvaluator([boundary_contract()], conversation)
        .gate(response)
        .should_recover
        is True
    )


def test_later_authoritative_evidence_does_not_validate_an_earlier_claim() -> None:
    contract = copy.deepcopy(boundary_contract())
    contract["applicability"] = {
        "mode": "all",
        "unknown_policy": "inactive",
        "predicates": [
            {
                "source": "user_text",
                "path": "content",
                "operator": "contains",
                "value": "cancel",
            }
        ],
    }
    conversation = [
        {"role": "user", "content": "Cancel my reservation."},
        {
            "role": "assistant",
            "content": "Because of the timing tier, the next fee is $180.",
        },
        evidence_conversation()[1],
    ]
    response = type(
        "Response", (), {"text": "The next fee is $180.", "tool_calls": []}
    )()

    assert ContractEvaluator([contract], conversation).gate(response).should_recover


def test_final_gate_disappears_when_proposed_answer_discharges_contract() -> None:
    evaluator = ContractEvaluator([boundary_contract()], evidence_conversation())
    missing = type(
        "Response", (), {"text": "The current fee is $90.", "tool_calls": []}
    )()
    complete = type(
        "Response",
        (),
        {
            "text": "Because waiting crosses the next timing tier, the next fee would be $180.",
            "tool_calls": [],
        },
    )()
    assert evaluator.gate(missing).should_recover is True
    assert evaluator.gate(complete).should_recover is False


def test_also_is_not_mistaken_for_a_causal_explanation() -> None:
    evaluator = ContractEvaluator([boundary_contract()], evidence_conversation())
    response = type(
        "Response",
        (),
        {"text": "I also checked that the next fee is $180.", "tool_calls": []},
    )()

    assert evaluator.gate(response).should_recover is True


def test_states_are_derived_not_accumulated() -> None:
    first = ContractEvaluator([boundary_contract()], evidence_conversation()).states()
    second = ContractEvaluator([boundary_contract()], evidence_conversation()).states()
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == 1


def test_read_only_tool_proposal_is_never_intercepted() -> None:
    evaluator = ContractEvaluator([boundary_contract()], evidence_conversation())
    response = type(
        "Response",
        (),
        {"text": "", "tool_calls": [{"name": "get_policy", "arguments": {}}]},
    )()
    decision = evaluator.gate(response)
    assert decision.should_recover is False
    assert decision.reason == "read_only_tool_proposal"


def test_hidden_text_attached_to_mutation_cannot_satisfy_pre_action_disclosure() -> None:
    contract = copy.deepcopy(boundary_contract())
    contract["obligations"][0]["deadline"] = "before_action"
    evaluator = ContractEvaluator([contract], evidence_conversation())
    mutation = type(
        "Response",
        (),
        {
            "text": "Because waiting crosses the tier, the next fee is $180.",
            "tool_calls": [
                {"name": "cancel_booking", "arguments": {"booking_id": "BK-1"}}
            ],
        },
    )()

    assert evaluator.gate(mutation).should_recover is True


def test_proposed_mutation_arguments_can_activate_pre_action_applicability() -> None:
    contract = copy.deepcopy(boundary_contract())
    contract["applicability"] = {
        "mode": "all",
        "unknown_policy": "inactive",
        "predicates": [
            {
                "source": "tool_argument",
                "tool": "cancel_booking",
                "path": "confirm",
                "operator": "truthy",
            }
        ],
    }
    contract["obligations"][0]["deadline"] = "before_action"
    evaluator = ContractEvaluator([contract], evidence_conversation())
    confirmed_mutation = type(
        "Response",
        (),
        {
            "text": "",
            "tool_calls": [
                {
                    "name": "cancel_booking",
                    "arguments": {"booking_id": "BK-1", "confirm": True},
                }
            ],
        },
    )()
    preview = type(
        "Response",
        (),
        {
            "text": "",
            "tool_calls": [
                {
                    "name": "cancel_booking",
                    "arguments": {"booking_id": "BK-1", "confirm": False},
                }
            ],
        },
    )()

    assert evaluator.gate(confirmed_mutation).should_recover is True
    assert evaluator.gate(preview).should_recover is False


def test_preclaim_gate_uses_each_contracts_own_claim_scope() -> None:
    contract = {
        **boundary_contract(),
        "id": "contract_amount_guard",
        "obligations": [
            {
                "id": "ground_amount",
                "deadline": "before_claim",
                "type": "boundary_must_not",
                "requirement": "Do not state an amount before authoritative evidence is available.",
                "priority": 0,
                "evidence_requirements": [],
                "response_requirements": [
                    {
                        "kind": "claim_requires_evidence",
                        "description": "ground concrete amounts",
                        "claim_types": ["amount"],
                        "evidence_any_of": [
                            {
                                "source": "tool_result",
                                "tool": "get_authoritative_amount",
                                "path": "amount",
                                "operator": "exists",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    evaluator = ContractEvaluator([contract], evidence_conversation())
    status_only = type(
        "Response", (), {"text": "The reservation is cancelled.", "tool_calls": []}
    )()
    amount = type("Response", (), {"text": "The fee is $50.", "tool_calls": []})()
    assert evaluator.gate(status_only).should_recover is False
    assert evaluator.gate(amount).should_recover is True


def test_draft_text_can_activate_a_contract_without_joining_the_trajectory() -> None:
    contract = {
        **boundary_contract(),
        "id": "contract_draft_claim",
        "applicability": {
            "mode": "all",
            "unknown_policy": "inactive",
            "predicates": [
                {
                    "source": "assistant_text",
                    "path": "content",
                    "operator": "contains_any",
                    "values": ["fee"],
                }
            ],
        },
        "obligations": [
            {
                "id": "explain_fee",
                "deadline": "before_final",
                "type": "explanation_rationale",
                "requirement": "Explain the stated fee.",
                "priority": 10,
                "evidence_requirements": [],
                "response_requirements": [
                    {
                        "kind": "causal_explanation",
                        "description": "give a causal explanation",
                    }
                ],
            }
        ],
    }
    evaluator = ContractEvaluator(
        [contract], [{"role": "user", "content": "What happened?"}]
    )
    fee_draft = type("Response", (), {"text": "The fee is $50.", "tool_calls": []})()
    unrelated_draft = type(
        "Response", (), {"text": "The request is complete.", "tool_calls": []}
    )()

    assert evaluator.gate(fee_draft).should_recover is True
    assert evaluator.gate(unrelated_draft).should_recover is False
    assert evaluator.conversation == [{"role": "user", "content": "What happened?"}]


def test_retrieval_requires_observable_semantic_overlap() -> None:
    artifact = {
        "version": 3,
        "kind": "effect_matched_closure_contracts",
        "contracts": [boundary_contract()],
    }
    index = EffectMatchedContractIndex(artifact, domain="travel", top_k=3)
    assert index.retrieve("cancel near a fee timing boundary")
    assert index.retrieve("paint a landscape in watercolor") == []
    assert index.retrieve("Please help me with this current task") == []


def test_runtime_retrieval_query_masks_task_literals() -> None:
    query = opening_query(
        [
            {
                "role": "user",
                "content": "Cancel BK-123 on 2026-08-30; the quoted fee is $50.",
            }
        ]
    )
    assert "BK-123" not in query
    assert "2026-08-30" not in query
    assert "$50" not in query
    assert "entity_id" in query
