from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from agents.opencode_agent import OpenCodeAgent
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent


class DummyClient:
    pass


class CapturingClient:
    def __init__(self, outputs: list[SimpleNamespace]):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def completion(text: str = "", calls: list[dict] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        tool_calls=calls or [],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )


class ArchivedPWMReference(ProcessWorkflowMemoryAgent):
    """Behavioral reference containing the archived prepare/generate methods."""

    def prepare_conversation(self, conversation: list) -> list:
        query = self._query_from_conversation(conversation)
        learnings = self.retrieve_learnings(query, top_k=self.retrieve_learnings_top_k)
        if not learnings:
            return conversation
        memory_prompt = (
            "Process-conformant workflow memory follows. Treat it as procedural guidance, not current-task facts. "
            "Verify identifiers, state, prices, eligibility, and policy with current domain tools. "
            "Before any state-changing call, gather required facts and obtain explicit user approval when the "
            "workflow requires a preview or choice. A valid branch may require no state change.\n\n"
            + "\n\n---\n\n".join(learnings)
        )
        return self.inject_system_message(conversation, memory_prompt)

    generate_next_turn = OpenCodeAgent.generate_next_turn


def write_memory(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "domain": "travel",
                        "family": "change",
                        "support": 4,
                        "mean_fitness": 1.0,
                        "quality": 1.0,
                        "observed_tools": ["get_booking", "update_booking"],
                        "search_text": "compare change fee explain final amount",
                        "tokens": ["compare", "change", "fee", "explain", "final", "amount"],
                        "text": (
                            "Workflow: compare changes\nBranches:\n"
                            "- Compare the fees and explain the final amount.\n"
                            "Avoid:\n- Do not update before user confirmation."
                        ),
                        "awm_text": "AWM",
                        "process_text": "PROCESS",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_archived_pwm_assets_remain_byte_equivalent() -> None:
    memory_candidates = [
        Path("artifacts/statebench_cross_domain_pwm/memory/process_workflows.json"),
        Path("outputs/memory/process_workflows.json"),
    ]
    memory_path = next((path for path in memory_candidates if path.exists()), None)
    assert memory_path is not None
    assert normalized_sha256(memory_path) == (
        "c18782ca68b452436d7a1837e9f11a34154ef415b2da74aec5627e92c4dce17b"
    )
    assert normalized_sha256(Path("agents/opencode_agent.py")) == (
        "6b8fa8fbb6dd3d5218998d06b94b4837fc540a227f3e0e17dd9a13db96111188"
    )
    assert normalized_sha256(Path("clients/opencode_client.py")) == (
        "7384dbe22e331bae169ae280ac715d3926eb405527251747fbd66908614dda9c"
    )


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


def test_pwm_only_matches_archived_prompt_and_generation_behavior(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "memory.json"
    write_memory(path)
    monkeypatch.setenv("STATE_BENCH_MEMORY_PATH", str(path))
    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "pwm_only")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    fixed = completion(
        calls=[{"name": "get_booking", "arguments": {"booking_id": "BK-1"}}]
    )
    new_client = CapturingClient([fixed])
    old_client = CapturingClient([fixed])
    kwargs = {
        "runtime_context": SimpleNamespace(
            domain="travel",
            task_summary="oracle must remain unread",
            task_requirements=[{"answer": "oracle"}],
        ),
        "retrieve_learnings_top_k": 3,
    }
    new_agent = ProcessWorkflowMemoryAgent(new_client, "system", [], {}, **kwargs)
    old_agent = ArchivedPWMReference(old_client, "system", [], {}, **kwargs)
    conversation = [{"role": "user", "content": "Compare my change fee."}]

    new_prepared = new_agent.prepare_conversation(conversation)
    old_prepared = old_agent.prepare_conversation(conversation)
    assert new_prepared == old_prepared
    assert json.dumps(new_prepared, sort_keys=True) == json.dumps(old_prepared, sort_keys=True)
    new_result = new_agent.generate_next_turn(
        system_prompt="system", conversation=new_prepared, tools=[{"name": "get_booking"}]
    )
    old_result = old_agent.generate_next_turn(
        system_prompt="system", conversation=old_prepared, tools=[{"name": "get_booking"}]
    )
    assert new_client.calls == old_client.calls
    assert new_result.text == old_result.text
    assert [(call.name, call.arguments) for call in new_result.tool_calls] == [
        (call.name, call.arguments) for call in old_result.tool_calls
    ]
    assert len(new_client.calls) == 1
    assert "Single-call task-closure gate" not in json.dumps(new_client.calls)


def test_structured_gate_logs_tool_call_without_regeneration(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "memory.json"
    write_memory(path)
    monkeypatch.setenv("STATE_BENCH_MEMORY_PATH", str(path))
    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "structured")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    client = CapturingClient(
        [completion(calls=[{"name": "get_booking", "arguments": {"booking_id": "BK-2"}}])]
    )
    agent = ProcessWorkflowMemoryAgent(
        client,
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel"),
    )
    initial = [{"role": "user", "content": "Explain the final fee."}]
    agent.prepare_conversation(initial)
    conversation = [
        *initial,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "get_booking",
                    "arguments": {"booking_id": "BK-1"},
                    "result": {"status": "confirmed", "total_price": 300},
                }
            ],
        },
        {"role": "tool", "content": []},
    ]
    result = agent.generate_next_turn(
        system_prompt="system", conversation=conversation, tools=[{"name": "get_booking"}]
    )
    assert len(client.calls) == 1
    serialized = json.dumps(client.calls[0]["conversation"])
    assert "If you choose to call tools in this turn, ignore all closure requirements below" in serialized
    assert result.tool_calls[0].name == "get_booking"
    log = agent._generation_log[0]
    assert log["closure_injected"] is True
    assert log["output_type"] == "tool_call"
    assert log["tool_calls_after_closure"] == [
        {"name": "get_booking", "arguments": {"booking_id": "BK-2"}}
    ]
    assert log["model_calls_for_generation"] == 1


def test_gate_does_not_inject_without_valid_tool_evidence(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "memory.json"
    write_memory(path)
    monkeypatch.setenv("STATE_BENCH_MEMORY_PATH", str(path))
    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "structured")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    client = CapturingClient([completion(text="I need to check that first.")])
    agent = ProcessWorkflowMemoryAgent(
        client,
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel"),
    )
    conversation = [{"role": "user", "content": "Compare and explain the final fee."}]
    agent.prepare_conversation(conversation)
    agent.generate_next_turn(system_prompt="system", conversation=conversation, tools=[])
    assert len(client.calls) == 1
    assert "Single-call task-closure gate" not in json.dumps(client.calls[0]["conversation"])
    assert agent._generation_log[0]["closure_injected"] is False


def test_generic_mode_uses_one_untracked_reminder(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "memory.json"
    write_memory(path)
    monkeypatch.setenv("STATE_BENCH_MEMORY_PATH", str(path))
    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "generic")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    client = CapturingClient([completion(text="The verified total is $300.")])
    agent = ProcessWorkflowMemoryAgent(
        client,
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel"),
    )
    conversation = [
        {"role": "user", "content": "Tell me the total."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "get_booking",
                    "arguments": {"booking_id": "BK-1"},
                    "result": {"status": "confirmed", "total_price": 300},
                }
            ],
        },
        {"role": "tool", "content": []},
    ]
    agent.generate_next_turn(system_prompt="system", conversation=conversation, tools=[])
    prompt = json.dumps(client.calls[0]["conversation"])
    assert "Generic completeness reminder" in prompt
    assert "Remaining task-closure requirements" not in prompt
    assert agent.completion_memory is None
    assert len(client.calls) == 1


def test_static_and_structured_modes_use_distinct_state_models(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "memory.json"
    write_memory(path)
    monkeypatch.setenv("STATE_BENCH_MEMORY_PATH", str(path))
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    conversation = [{"role": "user", "content": "Compare and explain the fee."}]

    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "static")
    static_agent = ProcessWorkflowMemoryAgent(
        DummyClient(), "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    static_agent.prepare_conversation(conversation)
    assert static_agent.completion_memory is None
    assert static_agent._static_requirements

    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "structured")
    structured_agent = ProcessWorkflowMemoryAgent(
        DummyClient(), "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    structured_agent.prepare_conversation(conversation)
    assert structured_agent.completion_memory is not None
    assert structured_agent.completion_memory.items
    assert structured_agent._static_requirements == []


def test_ingest_trajectory_saves_interference_log(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "memory.json"
    write_memory(path)
    monkeypatch.setenv("STATE_BENCH_MEMORY_PATH", str(path))
    monkeypatch.setenv("STATE_BENCH_COMPLETION_MODE", "generic")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    agent = ProcessWorkflowMemoryAgent(
        DummyClient(), "system", [], {}, runtime_context=SimpleNamespace(domain="travel")
    )
    agent._generation_log.append(
        {
            "generation_index": 0,
            "model_calls_for_generation": 1,
            "closure_injected": True,
            "output_type": "tool_call",
        }
    )
    trajectory = SimpleNamespace(metadata={})
    agent.ingest_trajectory(trajectory)
    payload = trajectory.metadata["completion_memory"]
    assert payload["mode"] == "generic"
    assert payload["summary"]["model_calls_per_generation"] == [1]
    assert payload["summary"]["regenerations"] == 0
    assert payload["summary"]["closure_injected_tool_calls"] == 1
