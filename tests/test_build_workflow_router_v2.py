from __future__ import annotations

import math

from scripts.build_workflow_router_v2 import GRID, MAX_RENDER_CHARS, _compile_card, _grid_configs


def test_frozen_cv_grid_executes_every_declared_configuration() -> None:
    configurations = list(_grid_configs())

    assert len(configurations) == math.prod(len(values) for values in GRID.values()) == 486
    assert {configuration["near_tie"] for configuration in configurations} == {
        0.5,
        1.0,
        2.0,
    }


def test_compiler_rejects_output_that_runtime_would_have_to_truncate() -> None:
    disclosure = "Tell the user " + "x" * (MAX_RENDER_CHARS + 100)
    source = "\n".join(
        [
            "Workflow: oversized",
            "Use when: the user asks for a cart mutation",
            "Verify first:",
            "- Call get_cart before changing anything.",
            "Procedure:",
            "1. Call add_to_cart(item_id).",
            "Branches:",
            f"- {disclosure}",
            "Avoid:",
            "- Never invent an item identifier.",
        ]
    )
    card = {
        "id": "shopping_assistant:add_to_cart:oversized",
        "domain": "shopping_assistant",
        "family": "add_to_cart",
        "awm_text": source,
    }

    compiled = _compile_card(card, {"get_cart", "add_to_cart"})

    assert compiled["compiler"]["valid"] is False
    assert "rendered_primary_too_long" in compiled["compiler"]["reasons"]
    assert compiled["compiler"]["checks"]["length_bound"] == "failed"
    assert compiled["primary_text"] == source
