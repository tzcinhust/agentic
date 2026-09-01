from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.lattice_subset_pwm_agent import LatticeSubsetPWMAgent
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent


class DummyClient:
    pass


def _artifact(path: Path) -> Path:
    cards = []
    for rank, token in enumerate(("alpha", "beta", "gamma"), start=1):
        cards.append(
            {
                "id": f"shopping_assistant:family_{rank}:0",
                "domain": "shopping_assistant",
                "family": f"family_{rank}",
                "support": 1,
                "mean_fitness": 0.0,
                "quality": 0.0,
                "observed_tools": [f"read_{rank}"],
                "search_text": token,
                "tokens": [token],
                "text": token.upper(),
                "awm_text": f"AWM-{token}",
                "process_text": f"PROCESS-{token}",
            }
        )
    path.write_text(json.dumps({"cards": cards}), encoding="utf-8")
    return path


def _agent(path: Path, monkeypatch: pytest.MonkeyPatch, mask: str) -> LatticeSubsetPWMAgent:
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    monkeypatch.setenv("STATE_BENCH_LATTICE_MASK", mask)
    return LatticeSubsetPWMAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="shopping_assistant"),
        retrieve_learnings_top_k=3,
    )


def test_mask_zero_abstains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(_artifact(tmp_path / "memory.json"), monkeypatch, "0")
    assert agent.retrieve_learnings("alpha beta gamma") == []


def test_mask_selects_baseline_positions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _artifact(tmp_path / "memory.json")
    full = _agent(path, monkeypatch, "7")
    query = "alpha beta gamma"
    full_text = full.retrieve_learnings(query)
    full_ids = full.candidate_card_ids(query)

    subset = _agent(path, monkeypatch, "0b101")
    assert subset.retrieve_learnings(query) == [full_text[0], full_text[2]]
    assert subset.candidate_card_ids(query) == full_ids
    assert len(full_ids) == 3


def test_full_mask_is_byte_identical_to_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _artifact(tmp_path / "memory.json")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", path)
    parent = ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="shopping_assistant"),
        retrieve_learnings_top_k=3,
    )
    child = _agent(path, monkeypatch, "7")
    query = "alpha beta gamma"
    assert child.retrieve_learnings(query) == parent.retrieve_learnings(query)


@pytest.mark.parametrize("mask", ["-1", "8", "not-a-number"])
def test_invalid_mask_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mask: str
) -> None:
    with pytest.raises(ValueError, match="STATE_BENCH_LATTICE_MASK"):
        _agent(_artifact(tmp_path / "memory.json"), monkeypatch, mask)
