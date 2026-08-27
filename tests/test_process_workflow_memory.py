from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent


class DummyClient:
    pass


class ScriptedClient:
    def __init__(self, outputs: list[SimpleNamespace]):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        return self.outputs.pop(0)


def completion(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        tool_calls=[],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
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


def test_prepare_conversation_adds_soft_obligation_ledger(tmp_path: Path, monkeypatch) -> None:
    cards = [
        {
            "domain": "travel",
            "family": "change",
            "support": 4,
            "mean_fitness": 1.0,
            "quality": 1.0,
            "observed_tools": ["get_booking"],
            "search_text": "compare change options",
            "tokens": ["compare", "change", "options"],
            "text": (
                "Workflow: compare changes\nBranches:\n"
                "- If two options exist, compare their fees before recommending one."
            ),
            "awm_text": "AWM",
            "process_text": "PROCESS",
        }
    ]
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"cards": cards}), encoding="utf-8")
    monkeypatch.setenv("STATE_BENCH_MEMORY_PATH", str(path))
    monkeypatch.setenv("STATE_BENCH_OBLIGATION_MODE", "on")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    agent = ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel"),
        retrieve_learnings_top_k=1,
    )
    prepared = agent.prepare_conversation(
        [{"role": "user", "content": "Compare my change options."}]
    )
    serialized = json.dumps(prepared)
    assert "Selective obligation ledger" in serialized
    assert "compare their fees" in serialized


def test_high_confidence_guard_revises_at_most_once(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"cards": []}), encoding="utf-8")
    monkeypatch.setenv("STATE_BENCH_MEMORY_PATH", str(path))
    monkeypatch.setenv("STATE_BENCH_SELECTIVE_GUARD_MODE", "enforce")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    client = ScriptedClient(
        [
            completion("Done — the booking has been cancelled."),
            completion("The cancellation failed because the booking is not cancellable."),
        ]
    )
    agent = ProcessWorkflowMemoryAgent(
        client,
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel"),
    )
    conversation = [
        {"role": "user", "content": "Cancel BK-1."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "cancel_booking",
                    "arguments": {"booking_id": "BK-1", "confirm": True},
                    "result": {"status": "rejected", "error": "not cancellable"},
                }
            ],
        },
    ]
    result = agent.generate_next_turn(
        system_prompt="system",
        conversation=conversation,
        tools=[],
    )
    assert client.calls == 2
    assert "failed" in result.text
    assert agent.guard_audit == [
        {
            "feedback": (
                "The latest cancel_booking call failed. State the observed failure instead of "
                "describing the action as completed."
            ),
            "corrected": True,
        }
    ]
