from __future__ import annotations

from pathlib import Path

from scripts.build_completion_templates import (
    TrainTrace,
    merge_templates,
    missing_signal_coverage,
    normalize_template,
)


def trace() -> TrainTrace:
    return TrainTrace(
        domain="travel",
        task_id="train-example",
        path=Path("train-example.json"),
        conversation=[{"role": "user", "content": "Please cancel my hotel."}],
        source_sha256="abc",
    )


def raw_template(requirement: str) -> dict:
    return {
        "family": "hotel_cancel_deadline",
        "title": "Hotel cancellation near a fee boundary",
        "trigger": {
            "intent": "cancel a hotel near a cancellation boundary",
            "observable_when": ["current evidence shows a nearby fee-tier boundary"],
        },
        "keywords": ["hotel", "cancel", "deadline", "next tier"],
        "confidence": 0.9,
        "obligations": [
            {
                "id": "future_fee",
                "phase": "final",
                "kind": "achievement",
                "type": "cost_amount_reporting",
                "requirement": requirement,
                "activation": "Waiting can cross a fee boundary.",
                "required_evidence": ["authoritative current and next fee tiers"],
                "discharge": "The answer states the exact next-tier fee supported by current evidence.",
                "priority": 10,
            }
        ],
    }


def test_induction_validation_rejects_task_specific_answer_literals() -> None:
    assert normalize_template(raw_template("Tell BK-1000 that the fee is $180."), trace(), 0) is None


def test_induction_keeps_general_relational_completion_semantics() -> None:
    template = normalize_template(
        raw_template("Warn about the next tier and report its exact fee from current evidence."),
        trace(),
        0,
    )
    assert template is not None
    assert template["obligations"][0]["phase"] == "final"
    assert "next tier" in template["obligations"][0]["requirement"]


def test_merge_tracks_support_and_deduplicates_obligations() -> None:
    first = normalize_template(
        raw_template("Warn about the next tier and report its exact fee from current evidence."),
        trace(),
        0,
    )
    second_trace = TrainTrace(
        domain="travel",
        task_id="another-example",
        path=Path("another.json"),
        conversation=[{"role": "user", "content": "Cancel a hotel."}],
        source_sha256="def",
    )
    second = normalize_template(
        raw_template("Report the exact next-tier fee and warn when waiting crosses the boundary."),
        second_trace,
        0,
    )
    merged = merge_templates([first, second])
    assert len(merged) == 1
    assert merged[0]["support"] == 2
    assert len(merged[0]["obligations"]) == 1
    assert merged[0]["source_tasks"] == ["another-example", "train-example"]


def test_latent_signal_detection_is_structural_not_task_id_based() -> None:
    sample = TrainTrace(
        domain="shopping_assistant",
        task_id="opaque-name",
        path=Path("opaque.json"),
        source_sha256="abc",
        conversation=[
            {"role": "user", "content": "Please add both items."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "get_customer_account",
                        "arguments": {},
                        "result": {"is_first_time": True, "loyalty_points": 100},
                    },
                    {
                        "name": "add_to_cart",
                        "arguments": {"product_id": "one"},
                        "result": {"status": "added"},
                    },
                    {
                        "name": "add_to_cart",
                        "arguments": {"product_id": "two"},
                        "result": {"status": "added"},
                    },
                ],
            },
        ],
    )
    assert sample.latent_signals == ["profile_benefit", "multi_entity_relation"]


def test_boundary_signal_requires_next_consequence_not_only_current_tier() -> None:
    current_only = [
        {
            "title": "Explain current cutoff",
            "trigger": {"intent": "cancel near a cutoff"},
            "keywords": ["cutoff"],
            "obligations": [{"requirement": "Explain why the current cutoff applies."}],
        }
    ]
    assert missing_signal_coverage(current_only, ["boundary_transition"]) == [
        "boundary_transition"
    ]
    with_future = [
        {
            "title": "Explain boundary transition",
            "trigger": {"intent": "cancel near a boundary"},
            "keywords": ["next tier"],
            "obligations": [
                {"requirement": "Report the exact next tier if waiting crosses the boundary."}
            ],
        }
    ]
    assert missing_signal_coverage(with_future, ["boundary_transition"]) == []
