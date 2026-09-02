from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.fixed_rank3_pwm_agent import FixedRank3PWMAgent
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent


class DummyClient:
    pass


def _memory(path: Path, shopping_cards: int = 3) -> Path:
    cards = []
    for domain, count in (("shopping_assistant", shopping_cards), ("travel", 3)):
        for rank, token in enumerate(("alpha", "beta", "gamma")[:count], start=1):
            cards.append(
                {
                    "id": f"{domain}:family_{rank}:0",
                    "domain": domain,
                    "family": f"family_{rank}",
                    "support": 1,
                    "mean_fitness": 0.0,
                    "quality": 0.0,
                    "observed_tools": [f"read_{rank}"],
                    "search_text": token,
                    "tokens": [token],
                    "text": f"{domain}-{token}",
                    "awm_text": f"AWM-{token}",
                    "process_text": f"PROCESS-{token}",
                }
            )
    path.write_text(json.dumps({"cards": cards}), encoding="utf-8")
    return path


def _agent(
    memory: Path,
    monkeypatch: pytest.MonkeyPatch,
    domain: str,
) -> FixedRank3PWMAgent:
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", memory)
    return FixedRank3PWMAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain=domain),
        retrieve_learnings_top_k=3,
    )


def test_shopping_injects_exact_baseline_rank_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _memory(tmp_path / "memory.json")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", memory)
    parent = ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="shopping_assistant"),
        retrieve_learnings_top_k=3,
    )
    candidate = _agent(memory, monkeypatch, "shopping_assistant")
    baseline = parent.retrieve_learnings("alpha beta gamma")
    assert candidate.retrieve_learnings("alpha beta gamma") == [baseline[2]]


def test_nonshopping_is_byte_identical_to_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _memory(tmp_path / "memory.json")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", memory)
    parent = ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel"),
        retrieve_learnings_top_k=3,
    )
    candidate = _agent(memory, monkeypatch, "travel")
    query = "alpha beta gamma"
    assert candidate.retrieve_learnings(query) == parent.retrieve_learnings(query)


def test_shopping_abstains_when_rank_three_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _agent(
        _memory(tmp_path / "memory.json", shopping_cards=2),
        monkeypatch,
        "shopping_assistant",
    )
    assert candidate.retrieve_learnings("alpha beta") == []
