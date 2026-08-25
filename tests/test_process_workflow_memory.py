from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent


class DummyClient:
    pass


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
