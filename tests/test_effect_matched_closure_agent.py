from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import agents.effect_matched_closure_agent as closure_module
from agents.effect_matched_closure_agent import EffectMatchedClosureAgent
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent


class RecordingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, *, system_prompt, conversation, tools):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "conversation": conversation,
                "tools": tools,
            }
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        text, tool_calls = item
        return SimpleNamespace(
            text=text,
            tool_calls=tool_calls,
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def write_artifacts(tmp_path: Path, *, include_contract=True):
    process = tmp_path / "process.json"
    process.write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "domain": "travel",
                        "family": "cancel",
                        "support": 3,
                        "mean_fitness": 1,
                        "quality": 1,
                        "observed_tools": ["preview_cancel"],
                        "search_text": "cancel reservation fee timing",
                        "tokens": ["cancel", "reservation", "fee", "timing"],
                        "text": "FROZEN PWM WORKFLOW",
                        "awm_text": "FROZEN PWM WORKFLOW",
                        "process_text": "FROZEN PWM WORKFLOW",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    contracts = tmp_path / "contracts.json"
    item = {
        "id": "contract_boundary",
        "domain": "travel",
        "family": "timing_boundary_consequence",
        "title": "Timing boundary consequence",
        "intent": "cancel a reservation near a fee boundary",
        "keywords": ["cancel", "reservation", "fee", "timing"],
        "search_text": "cancel reservation fee timing boundary consequence",
        "tokens": ["cancel", "reservation", "fee", "timing", "boundary", "consequence"],
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
    contracts.write_text(
        json.dumps(
            {
                "version": 4,
                "kind": "effect_matched_closure_contracts",
                "contracts": [item] if include_contract else [],
            }
        ),
        encoding="utf-8",
    )
    return process, contracts


def conversation_with_evidence():
    return [
        {"role": "user", "content": "Cancel my reservation and explain the fee."},
        {
            "role": "assistant",
            "content": "I checked the preview.",
            "tool_calls": [
                {
                    "name": "preview_cancel",
                    "arguments": {"reservation_id": "R-1"},
                    "result": {
                        "status": "preview",
                        "reason": "current fee tier applies before the cutoff",
                        "current_fee": 90,
                        "next_fee": 180,
                    },
                }
            ],
        },
    ]


def configure(monkeypatch, process: Path, contracts: Path, mode: str):
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", process)
    monkeypatch.setattr(EffectMatchedClosureAgent, "contract_path", contracts)
    monkeypatch.setenv("STATE_BENCH_CLOSURE_MODE", mode)


def test_module_does_not_export_parent_loader_name() -> None:
    assert "ProcessWorkflowMemoryAgent" not in vars(closure_module)


def test_safe_default_is_pwm_only_and_does_not_require_a_contract_artifact(
    tmp_path, monkeypatch
) -> None:
    process, _ = write_artifacts(tmp_path)
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", process)
    monkeypatch.setattr(
        EffectMatchedClosureAgent, "contract_path", tmp_path / "missing-contracts.json"
    )
    monkeypatch.delenv("STATE_BENCH_CLOSURE_MODE", raising=False)
    agent = EffectMatchedClosureAgent(
        RecordingClient([("answer", [])]),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel"),
    )

    assert agent.closure_mode == "pwm_only"
    assert agent._contract_index is None


def test_pwm_only_is_behaviorally_equivalent_to_frozen_pwm(
    tmp_path, monkeypatch
) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "pwm_only")
    context = SimpleNamespace(domain="travel", task_summary="must never be read")
    old_client = RecordingClient([("answer", [])])
    new_client = RecordingClient([("answer", [])])
    old = ProcessWorkflowMemoryAgent(
        old_client, "system", [], {}, runtime_context=context
    )
    new = EffectMatchedClosureAgent(
        new_client, "system", [], {}, runtime_context=context
    )
    conversation = [{"role": "user", "content": "Cancel my reservation."}]
    old_prepared = old.prepare_conversation(conversation)
    new_prepared = new.prepare_conversation(conversation)
    assert old_prepared == new_prepared
    old_response = old.generate_next_turn(
        system_prompt="system", conversation=old_prepared, tools=[{"name": "x"}]
    )
    new_response = new.generate_next_turn(
        system_prompt="system", conversation=new_prepared, tools=[{"name": "x"}]
    )
    assert old_client.calls == new_client.calls
    assert old_response == new_response
    assert "task_summary" not in json.dumps(new_client.calls)


def test_no_retrieved_contract_is_exact_one_call_fallback(
    tmp_path, monkeypatch
) -> None:
    process, contracts = write_artifacts(tmp_path, include_contract=False)
    configure(monkeypatch, process, contracts, "enforce")
    client = RecordingClient([("plain answer", [])])
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=[{"role": "user", "content": "Cancel my reservation."}],
        tools=[{"name": "get_booking"}],
    )
    assert response.text == "plain answer"
    assert len(client.calls) == 1
    assert agent._generation_log[0]["fallback_reason"] == "no_retrieved_contract"


def test_read_only_tool_path_is_not_changed(tmp_path, monkeypatch) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "enforce")
    client = RecordingClient([("", [{"name": "get_policy", "arguments": {}}])])
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation_with_evidence(),
        tools=[{"name": "get_policy"}],
    )
    assert response.tool_calls[0].name == "get_policy"
    assert len(client.calls) == 1
    assert agent._generation_log[0]["gate"]["reason"] == "read_only_tool_proposal"


def test_missing_final_contract_gets_exactly_one_boundary_recovery(
    tmp_path, monkeypatch
) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "enforce")
    client = RecordingClient(
        [
            ("The current fee is $90.", []),
            (
                "Because waiting crosses the next timing tier, the next fee would be $180.",
                [],
            ),
        ]
    )
    agent = EffectMatchedClosureAgent(
        client,
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel", task_summary="oracle answer"),
    )
    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation_with_evidence(),
        tools=[{"name": "get_policy"}],
    )
    assert response.text.endswith("$180.")
    assert len(client.calls) == 2
    assert "Response-boundary closure contract" not in json.dumps(client.calls[0])
    assert "Response-boundary closure contract" in json.dumps(client.calls[1])
    assert "oracle answer" not in json.dumps(client.calls)
    log = agent._generation_log[0]
    assert log["recovery_calls"] == 1
    assert log["total_model_calls"] == 2
    assert log["post_recovery_gate"]["should_recover"] is False


def test_monitor_mode_never_recovers(tmp_path, monkeypatch) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "monitor")
    client = RecordingClient([("The current fee is $90.", [])])
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation_with_evidence(),
        tools=[{"name": "get_policy"}],
    )
    assert len(client.calls) == 1
    assert agent._contract_index.top_k == 3
    assert agent._generation_log[0]["gate"]["should_recover"] is True
    assert agent._generation_log[0]["fallback_reason"] == "monitor_mode"


def test_failed_recovery_falls_back_to_frozen_pwm_draft(tmp_path, monkeypatch) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "enforce")
    client = RecordingClient(
        [("The current fee is $90.", []), RuntimeError("provider timeout")]
    )
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation_with_evidence(),
        tools=[{"name": "get_policy"}],
    )
    assert response.text == "The current fee is $90."
    assert agent._generation_log[0]["fallback_reason"] == "closure_recovery_error"
    assert "provider timeout" in agent._generation_log[0]["error"]
    assert agent._generation_log[0]["closure_injected"] is True
    assert agent._generation_log[0]["recovery_calls"] == 1
    assert agent._generation_log[0]["total_model_calls"] == 2


def test_non_improving_recovery_is_rejected_in_favor_of_pwm_draft(
    tmp_path, monkeypatch
) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "enforce")
    client = RecordingClient(
        [("The current fee is $90.", []), ("The current fee is still $90.", [])]
    )
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )

    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation_with_evidence(),
        tools=[{"name": "get_policy"}],
    )

    assert response.text == "The current fee is $90."
    log = agent._generation_log[0]
    assert log["recovery_generated"] is True
    assert log["recovery_used"] is False
    assert log["fallback_reason"] == "recovery_did_not_reduce_open_obligations"


def test_final_recovery_cannot_replace_the_draft_with_a_mutation(
    tmp_path, monkeypatch
) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "enforce")
    client = RecordingClient(
        [
            ("The current fee is $90.", []),
            (
                "",
                [
                    {
                        "name": "cancel_booking",
                        "arguments": {"booking_id": "BK-1", "confirm": True},
                    }
                ],
            ),
        ]
    )
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )

    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation_with_evidence(),
        tools=[{"name": "cancel_booking"}],
    )

    assert response.text == "The current fee is $90."
    assert response.tool_calls == []
    assert agent._generation_log[0]["fallback_reason"] == "recovery_mutation_forbidden"


def test_missing_evidence_can_bridge_only_to_the_contracts_read_tool(
    tmp_path, monkeypatch
) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "enforce")
    conversation = conversation_with_evidence()
    del conversation[1]["tool_calls"][0]["result"]["next_fee"]
    client = RecordingClient(
        [
            ("The current fee is $90.", []),
            ("", [{"name": "preview_cancel", "arguments": {"reservation_id": "R-1"}}],),
        ]
    )
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )

    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation,
        tools=[{"name": "preview_cancel"}, {"name": "get_unrelated"}],
    )

    assert response.tool_calls[0].name == "preview_cancel"
    assert agent._generation_log[0]["recovery_used"] is True
    assert (
        agent._generation_log[0]["recovery_acceptance_reason"]
        == "grounded_evidence_bridge"
    )


def test_recovery_budget_is_global_to_the_task_not_each_generation(
    tmp_path, monkeypatch
) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "enforce")
    client = RecordingClient(
        [
            ("The current fee is $90.", []),
            ("Still only the current fee is $90.", []),
            ("The current fee remains $90.", []),
        ]
    )
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )

    for _ in range(2):
        agent.generate_next_turn(
            system_prompt="system",
            conversation=conversation_with_evidence(),
            tools=[{"name": "get_policy"}],
        )

    assert len(client.calls) == 3
    assert (
        agent._generation_log[1]["fallback_reason"] == "task_recovery_budget_exhausted"
    )


def test_pre_action_contract_is_monitor_only_by_default(tmp_path, monkeypatch) -> None:
    process, contracts = write_artifacts(tmp_path)
    payload = json.loads(contracts.read_text(encoding="utf-8"))
    payload["contracts"][0]["obligations"][0]["deadline"] = "before_action"
    contracts.write_text(json.dumps(payload), encoding="utf-8")
    configure(monkeypatch, process, contracts, "enforce")
    client = RecordingClient(
        [
            (
                "",
                [
                    {
                        "name": "cancel_booking",
                        "arguments": {"booking_id": "BK-1", "confirm": True},
                    }
                ],
            )
        ]
    )
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )

    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation_with_evidence(),
        tools=[{"name": "cancel_booking"}],
    )

    assert response.tool_calls
    assert len(client.calls) == 1
    assert agent._generation_log[0]["fallback_reason"] == "pre_action_monitor_only"


def test_completion_retrieval_is_one_shot_across_turns(tmp_path, monkeypatch) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "enforce")
    client = RecordingClient(
        [
            ("", [{"name": "get_policy", "arguments": {}}]),
            ("", [{"name": "get_booking", "arguments": {}}]),
        ]
    )
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    for _ in range(2):
        agent.generate_next_turn(
            system_prompt="system",
            conversation=conversation_with_evidence(),
            tools=[{"name": "get_policy"}, {"name": "get_booking"}],
        )
    assert agent._retrieval_log["calls"] == 1
    assert len(agent._generation_log) == 2


def test_failed_retrieval_is_not_retried_on_later_turns(tmp_path, monkeypatch) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "enforce")
    client = RecordingClient([("first", []), ("second", [])])
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    retrieval_calls = 0

    def fail_retrieval(query):
        nonlocal retrieval_calls
        retrieval_calls += 1
        raise RuntimeError("index failure")

    agent._contract_index.retrieve_with_scores = fail_retrieval
    conversation = [{"role": "user", "content": "Cancel my reservation."}]
    for _ in range(2):
        agent.generate_next_turn(
            system_prompt="system", conversation=conversation, tools=[]
        )

    assert retrieval_calls == 1
    assert agent._retrieval_log["calls"] == 1
    assert agent._generation_log[0]["fallback_reason"] == "closure_retrieval_error"
    assert agent._generation_log[1]["fallback_reason"] == "no_retrieved_contract"


def test_trajectory_metadata_records_bounded_overhead(tmp_path, monkeypatch) -> None:
    process, contracts = write_artifacts(tmp_path)
    configure(monkeypatch, process, contracts, "enforce")
    client = RecordingClient([("", [{"name": "get_policy", "arguments": {}}])])
    agent = EffectMatchedClosureAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    conversation = conversation_with_evidence()
    agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation,
        tools=[{"name": "get_policy"}],
    )
    trajectory = SimpleNamespace(conversation=conversation, metadata={})
    agent.ingest_trajectory(trajectory)
    metadata = trajectory.metadata["effect_matched_closure_memory"]
    assert "final_action_ledger" in metadata
    assert metadata["summary"]["one_shot_retrieval_calls"] == 1
    assert metadata["summary"]["maximum_recoveries_per_generation"] == 0
    assert metadata["summary"]["semantic_bookkeeper_calls"] == 0
    assert metadata["summary"]["unbounded_regeneration_loops"] == 0
