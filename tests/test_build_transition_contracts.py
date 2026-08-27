from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_transition_contracts import build_artifact


def test_build_transition_contracts_learns_stable_predecessor(tmp_path: Path) -> None:
    domain_path = tmp_path / "shopping_assistant"
    domain_path.mkdir()
    conversation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "get_cart",
                    "arguments": {"customer_id": "C-1"},
                    "result": {"customer_id": "C-1", "items": []},
                },
                {
                    "name": "add_to_cart",
                    "arguments": {"customer_id": "C-1", "product_id": "P-1"},
                    "result": {"status": "success"},
                },
            ],
        }
    ]
    for index in range(3):
        (domain_path / f"{index}.json").write_text(
            json.dumps({"conversation": conversation}), encoding="utf-8"
        )

    artifact = build_artifact(
        tmp_path,
        domains=["shopping_assistant"],
        min_support=3,
        min_confidence=0.65,
    )

    contract = artifact["domains"]["shopping_assistant"]["contracts"]["add_to_cart"]
    assert contract["required_tools"] == [
        {"tool": "get_cart", "support": 3, "confidence": 1.0}
    ]


def test_build_transition_contracts_rejects_test_data(tmp_path: Path) -> None:
    test_root = tmp_path / "test_task_trajectories"

    with pytest.raises(ValueError, match="train trajectories only"):
        build_artifact(
            test_root,
            domains=["shopping_assistant"],
            min_support=3,
            min_confidence=0.65,
        )
