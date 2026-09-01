from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.conformal_lattice_router_agent import ConformalLatticeRouterAgent
from agents.conformal_router_features import (
    choose_from_predictions,
    feature_names,
    feature_vector,
)
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent
from scripts.train_conformal_lattice_router import (
    _solve_ridge_multi,
    conformal_harm_certificate,
    deterministic_split,
)


class DummyClient:
    pass


def _memory(path: Path) -> Path:
    cards = []
    for domain in ("shopping_assistant", "travel"):
        for rank, token in enumerate(("alpha", "beta", "gamma"), start=1):
            cards.append(
                {
                    "id": f"{domain}:family_{rank}:0",
                    "domain": domain,
                    "family": f"family_{rank}",
                    "support": rank,
                    "mean_fitness": 0.8,
                    "quality": 0.8,
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


def _weights(value: float) -> list[float]:
    return [value] + [0.0] * (len(feature_names()) - 1)


def _router_artifact(path: Path, enabled: bool = True) -> Path:
    models = {}
    for mask in range(8):
        value = 0.80 if mask == 1 else (0.50 if mask == 7 else 0.40)
        models[str(mask)] = {
            metric: _weights(value)
            for metric in ("completion", "state", "task", "ux", "utility")
        }
    artifact = {
        "schema_version": "conformal_lattice_router_v1",
        "deployment_enabled": enabled,
        "models": models,
        "policy": {
            "min_predicted_utility_gain": 0.05,
            "minimum_safety_delta": 0.0,
            "ux_delta_tolerance": 0.02,
            "card_count_penalty": 0.005,
            "allowed_masks": list(range(7)),
        },
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _agent(
    memory: Path,
    router: Path,
    monkeypatch: pytest.MonkeyPatch,
    domain: str = "shopping_assistant",
) -> ConformalLatticeRouterAgent:
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", memory)
    monkeypatch.setattr(ConformalLatticeRouterAgent, "router_path", router)
    monkeypatch.delenv("STATE_BENCH_CONFORMAL_ROUTER_PATH", raising=False)
    return ConformalLatticeRouterAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain=domain),
        retrieve_learnings_top_k=3,
    )


def test_feature_vector_is_deterministic_and_schema_sized() -> None:
    ranked = [
        (
            3.5,
            {
                "id": "card-a",
                "support": 3,
                "mean_fitness": 0.9,
                "quality": 0.8,
                "observed_tools": ["get_cart", "update_cart_item"],
            },
        )
    ]
    first = feature_vector("keep 2 items and update shipping?", ranked)
    assert first == feature_vector("keep 2 items and update shipping?", ranked)
    assert len(first) == len(feature_names()) == 66


def test_decision_gate_selects_only_safe_gain() -> None:
    baseline = {metric: 0.5 for metric in ("completion", "state", "task", "ux", "utility")}
    safe = {metric: 0.8 for metric in baseline}
    policy = {
        "min_predicted_utility_gain": 0.1,
        "minimum_safety_delta": 0.0,
        "ux_delta_tolerance": 0.02,
        "card_count_penalty": 0.0,
        "allowed_masks": [1],
    }
    assert choose_from_predictions({1: safe, 7: baseline}, policy) == (
        1,
        "certified_alternative",
    )
    unsafe = dict(safe, task=0.49)
    assert choose_from_predictions({1: unsafe, 7: baseline}, policy)[0] == 7


def test_disabled_artifact_is_exact_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _memory(tmp_path / "memory.json")
    router = _router_artifact(tmp_path / "router.json", enabled=False)
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", memory)
    parent = ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="shopping_assistant"),
        retrieve_learnings_top_k=3,
    )
    child = _agent(memory, router, monkeypatch)
    query = "alpha beta gamma"
    assert child.retrieve_learnings(query) == parent.retrieve_learnings(query)


def test_corrupt_artifact_is_exact_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _memory(tmp_path / "memory.json")
    router = tmp_path / "router.json"
    router.write_text(
        json.dumps(
            {
                "schema_version": "conformal_lattice_router_v1",
                "deployment_enabled": True,
                "models": {"7": {"utility": ["bad-weight"]}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", memory)
    parent = ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="shopping_assistant"),
        retrieve_learnings_top_k=3,
    )
    child = _agent(memory, router, monkeypatch)
    query = "alpha beta gamma"
    assert child.retrieve_learnings(query) == parent.retrieve_learnings(query)


def test_nonshopping_domain_is_exact_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _memory(tmp_path / "memory.json")
    router = _router_artifact(tmp_path / "router.json")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", memory)
    parent = ProcessWorkflowMemoryAgent(
        DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="travel"),
        retrieve_learnings_top_k=3,
    )
    child = _agent(memory, router, monkeypatch, domain="travel")
    query = "alpha beta gamma"
    assert child.retrieve_learnings(query) == parent.retrieve_learnings(query)


def test_selected_mask_is_locked_for_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _memory(tmp_path / "memory.json")
    router = _router_artifact(tmp_path / "router.json")
    agent = _agent(memory, router, monkeypatch)
    first = agent.retrieve_learnings("alpha beta gamma")
    assert len(first) == 1
    assert agent._active_mask == 1
    agent._artifact["deployment_enabled"] = False
    second = agent.retrieve_learnings("gamma beta alpha")
    assert len(second) == 1
    assert agent._active_mask == 1


def test_telemetry_contains_no_query_or_task_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _memory(tmp_path / "memory.json")
    router = _router_artifact(tmp_path / "router.json")
    telemetry = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("STATE_BENCH_CONFORMAL_ROUTER_TELEMETRY_PATH", str(telemetry))
    agent = _agent(memory, router, monkeypatch)
    secret_query = "alpha private-user-text"
    agent.retrieve_learnings(secret_query)
    record = json.loads(telemetry.read_text(encoding="utf-8"))
    assert set(record) == {
        "run_uuid",
        "domain",
        "card_ids",
        "selected_mask",
        "selection_reason",
        "predictions",
    }
    assert secret_query not in telemetry.read_text(encoding="utf-8")


def test_ridge_solver_handles_multiple_targets() -> None:
    rows = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    targets = [[1.0, 2.0], [3.0, 5.0], [5.0, 8.0], [7.0, 11.0]]
    weights = _solve_ridge_multi(rows, targets, ridge=1e-6)
    assert weights[0] == pytest.approx([1.0, 2.0], abs=1e-4)
    assert weights[1] == pytest.approx([2.0, 3.0], abs=1e-4)


def test_split_is_deterministic_and_disjoint() -> None:
    task_ids = [f"task-{index}" for index in range(100)]
    first = deterministic_split(task_ids, "salt")
    second = deterministic_split(reversed(task_ids), "salt")
    assert first == second
    assert {label: len(ids) for label, ids in first.items()} == {
        "fit": 60,
        "calibration": 20,
        "lockbox": 20,
    }
    assert not (set(first["fit"]) & set(first["calibration"]))
    assert not (set(first["fit"]) & set(first["lockbox"]))
    assert not (set(first["calibration"]) & set(first["lockbox"]))


def test_conformal_certificate_rejects_repeated_harm() -> None:
    task_ids = [f"task-{index}" for index in range(20)]
    outcomes = {
        task_id: {
            1: {metric: (0.0 if index < 2 else 1.0) for metric in ("completion", "state", "task", "ux", "utility")},
            7: {metric: 1.0 for metric in ("completion", "state", "task", "ux", "utility")},
        }
        for index, task_id in enumerate(task_ids)
    }
    report = {
        "selected": {task_id: 1 for task_id in task_ids},
        "delta": {"completion": -0.1, "state": -0.1, "task": -0.1, "ux": -0.1, "utility": -0.1},
    }
    certificate = conformal_harm_certificate(report, outcomes)
    assert certificate["positive_harm_count"] == 2
    assert certificate["harm_quantile"] == 1.0
    assert not certificate["passed"]
