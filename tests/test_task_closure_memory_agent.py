from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import agents.task_closure_memory_agent as task_closure_module
from agents.completion_lifecycle import CompletionItem
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent
from agents.task_closure_memory_agent import TaskClosureMemoryAgent


class RecordingClient:
    def __init__(self, main_tool_calls=None, bookkeeper_status="pending"):
        self.calls = []
        self.main_tool_calls = list(main_tool_calls or [])
        self.bookkeeper_status = bookkeeper_status

    def generate(self, *, system_prompt, conversation, tools):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "conversation": conversation,
                "tools": tools,
            }
        )
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        if not tools:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "items": [
                            {
                                "template_id": "ct_fee",
                                "obligation_id": "explain",
                                "scope_key": "booking",
                                "applicable": True,
                                "status": self.bookkeeper_status,
                                "description": "Explain the applicable timing tier and its supported amount.",
                                "evidence": [],
                                "missing_evidence": ["authoritative timing and fee evidence"],
                            }
                        ]
                    }
                ),
                tool_calls=[],
                usage=usage,
            )
        calls = self.main_tool_calls.pop(0) if self.main_tool_calls else []
        return SimpleNamespace(text="main answer", tool_calls=calls, usage=usage)


def write_artifacts(tmp_path: Path) -> tuple[Path, Path]:
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
                        "observed_tools": ["cancel_booking"],
                        "search_text": "cancel booking fee",
                        "tokens": ["cancel", "booking", "fee"],
                        "text": "FROZEN PWM WORKFLOW",
                        "awm_text": "FROZEN PWM WORKFLOW",
                        "process_text": "FROZEN PWM WORKFLOW",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    completion = tmp_path / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "version": 2,
                "templates": [
                    {
                        "id": "ct_fee",
                        "domain": "travel",
                        "family": "cancel_fee_boundary",
                        "title": "Cancellation timing disclosure",
                        "trigger": {"intent": "cancel", "observable_when": ["a timing tier affects fees"]},
                        "keywords": ["cancel", "fee", "timing"],
                        "observed_tools": ["cancel_booking"],
                        "support": 4,
                        "confidence": 0.9,
                        "search_text": "cancel booking timing fee explain amount",
                        "tokens": ["cancel", "booking", "timing", "fee", "explain", "amount"],
                        "obligations": [
                            {
                                "id": "explain",
                                "phase": "final",
                                "kind": "achievement",
                                "type": "explanation_rationale",
                                "requirement": "Explain the applicable timing tier and supported amount.",
                                "activation": "A cancellation fee depends on timing.",
                                "required_evidence": ["current authoritative timing and fee evidence"],
                                "discharge": "The assistant explains the tier and reports the supported amount.",
                                "priority": 10,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return process, completion


def test_module_does_not_export_the_parent_class_to_statebench_loader() -> None:
    assert "ProcessWorkflowMemoryAgent" not in vars(task_closure_module)


def test_pwm_only_is_behaviorally_equivalent_to_frozen_pwm(tmp_path: Path, monkeypatch) -> None:
    process, completion = write_artifacts(tmp_path)
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", process)
    monkeypatch.setattr(TaskClosureMemoryAgent, "completion_memory_path", completion)
    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "pwm_only")
    context = SimpleNamespace(domain="travel", task_summary="must never be read")
    old_client, new_client = RecordingClient(), RecordingClient()
    old = ProcessWorkflowMemoryAgent(old_client, "system", [], {}, runtime_context=context)
    new = TaskClosureMemoryAgent(new_client, "system", [], {}, runtime_context=context)
    conversation = [{"role": "user", "content": "Cancel my booking and explain the fee."}]

    old_prepared = old.prepare_conversation(conversation)
    new_prepared = new.prepare_conversation(conversation)
    assert old_prepared == new_prepared
    old_response = old.generate_next_turn(system_prompt="system", conversation=old_prepared, tools=[{"name": "x"}])
    new_response = new.generate_next_turn(system_prompt="system", conversation=new_prepared, tools=[{"name": "x"}])
    assert old_client.calls == new_client.calls
    assert old_response == new_response
    assert "task_summary" not in json.dumps(new_client.calls)


def test_full_mode_uses_one_main_call_and_selective_final_closure(tmp_path: Path, monkeypatch) -> None:
    process, completion = write_artifacts(tmp_path)
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", process)
    monkeypatch.setattr(TaskClosureMemoryAgent, "completion_memory_path", completion)
    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "full")
    client = RecordingClient()
    agent = TaskClosureMemoryAgent(
        client,
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel", task_summary="oracle"),
    )
    conversation = [
        {"role": "user", "content": "Cancel my booking and explain the fee."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_booking",
                    "arguments": {"booking_id": "BK-1"},
                    "result": {"status": "confirmed", "cancellation_fee": 50},
                }
            ],
        },
    ]
    response = agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation,
        tools=[{"name": "cancel_booking"}],
    )
    assert response.text == "main answer"
    assert len(client.calls) == 2
    assert client.calls[0]["tools"] == []
    assert client.calls[1]["tools"] == [{"name": "cancel_booking"}]
    serialized = json.dumps(client.calls[1]["conversation"])
    assert "Selective final task-closure memory" in serialized
    assert "If you choose to call tools in this turn" in serialized
    assert agent._generation_log[0]["main_model_calls"] == 1
    assert agent._generation_log[0]["bookkeeper_calls"] == 1
    assert agent._generation_log[0]["regenerations"] == 0
    assert "oracle" not in serialized


def test_identical_final_closure_is_suppressed_after_tool_call(tmp_path: Path, monkeypatch) -> None:
    process, completion = write_artifacts(tmp_path)
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", process)
    monkeypatch.setattr(TaskClosureMemoryAgent, "completion_memory_path", completion)
    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "full")
    client = RecordingClient(
        main_tool_calls=[[{"name": "get_booking", "arguments": {"booking_id": "BK-1"}}], []]
    )
    agent = TaskClosureMemoryAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    conversation = [
        {"role": "user", "content": "Cancel and explain the fee."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_booking",
                    "arguments": {"booking_id": "BK-1"},
                    "result": {"status": "confirmed", "cancellation_fee": 50},
                }
            ],
        },
    ]
    agent.generate_next_turn(system_prompt="system", conversation=conversation, tools=[{"name": "get_booking"}])
    agent.generate_next_turn(system_prompt="system", conversation=conversation, tools=[{"name": "get_booking"}])
    main_calls = [call for call in client.calls if call["tools"]]
    assert "Selective final task-closure memory" in json.dumps(main_calls[0]["conversation"])
    assert "Selective final task-closure memory" not in json.dumps(main_calls[1]["conversation"])
    assert [item["main_model_calls"] for item in agent._generation_log] == [1, 1]


def test_final_closure_can_reappear_after_new_observable_evidence(tmp_path: Path, monkeypatch) -> None:
    process, completion = write_artifacts(tmp_path)
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", process)
    monkeypatch.setattr(TaskClosureMemoryAgent, "completion_memory_path", completion)
    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "full")
    client = RecordingClient(
        main_tool_calls=[[{"name": "get_policy", "arguments": {"topic": "cancel"}}], []]
    )
    agent = TaskClosureMemoryAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    conversation = [
        {"role": "user", "content": "Cancel and explain the fee."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_booking",
                    "arguments": {"booking_id": "BK-1"},
                    "result": {"status": "confirmed", "cancellation_fee": 50},
                }
            ],
        },
    ]
    agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[{"name": "get_policy"}]
    )
    conversation = [
        *conversation,
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_policy",
                    "arguments": {"topic": "cancel"},
                    "result": {"status": "success", "rule": "authoritative"},
                }
            ],
        },
    ]
    agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[{"name": "get_policy"}]
    )

    main_calls = [call for call in client.calls if call["tools"]]
    assert "Selective final task-closure memory" in json.dumps(main_calls[0]["conversation"])
    assert "Selective final task-closure memory" in json.dumps(main_calls[1]["conversation"])


def test_unresolved_preclaim_evidence_blocks_final_closure(tmp_path: Path, monkeypatch) -> None:
    process, completion = write_artifacts(tmp_path)
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", process)
    monkeypatch.setattr(TaskClosureMemoryAgent, "completion_memory_path", completion)
    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "full")
    client = RecordingClient()
    agent = TaskClosureMemoryAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    agent._tracker.items["unresolved"] = CompletionItem(
        id="unresolved",
        template_id="manual_test",
        obligation_id="ground",
        phase="pre_claim",
        kind="achievement",
        type="evidence_grounding",
        description="Resolve the authoritative fee tier before claiming an amount.",
        source="completion_template:manual_test",
        status="pending_evidence",
    )
    conversation = [
        {"role": "user", "content": "Cancel and explain the fee."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_booking",
                    "arguments": {"booking_id": "BK-1"},
                    "result": {"status": "confirmed"},
                }
            ],
        },
    ]
    agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation,
        tools=[{"name": "get_booking"}],
    )
    main_call = [call for call in client.calls if call["tools"]][0]
    serialized = json.dumps(main_call["conversation"])
    assert "Evidence-before-claim guard" in serialized
    assert "Selective final task-closure memory" not in serialized
    assert agent._generation_log[0]["exposure_mode"] == "claim_guard"


def test_ingest_saves_lifecycle_retrieval_exposure_and_overhead_logs(tmp_path: Path, monkeypatch) -> None:
    process, completion = write_artifacts(tmp_path)
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", process)
    monkeypatch.setattr(TaskClosureMemoryAgent, "completion_memory_path", completion)
    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "full")
    client = RecordingClient()
    agent = TaskClosureMemoryAgent(
        client, "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    conversation = [
        {"role": "user", "content": "Cancel and explain the fee."},
        {
            "role": "assistant",
            "content": "The current fee is grounded in the booking.",
            "tool_calls": [
                {
                    "name": "get_booking",
                    "arguments": {"booking_id": "BK-1"},
                    "result": {"status": "confirmed", "cancellation_fee": 50},
                }
            ],
        },
    ]
    agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation,
        tools=[{"name": "get_booking"}],
    )
    trajectory = SimpleNamespace(conversation=conversation, metadata={}, token_usage=None)
    agent.ingest_trajectory(trajectory)
    payload = trajectory.metadata["completion_memory"]
    assert payload["version"] == "task_closure_memory_v2"
    assert payload["completion_retrievals"]
    assert payload["bookkeeper_calls"]
    assert payload["generations"][0]["main_model_calls"] == 1
    assert payload["summary"]["regenerations"] == 0
    assert payload["summary"]["tool_calls"] == 1
    assert payload["summary"]["bookkeeper_total_tokens"] > 0
    assert trajectory.token_usage is agent.token_usage
