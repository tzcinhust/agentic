from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent
from agents.transition_patch_memory import TransitionPatchIndex, build_transition_artifact, tokens
from scripts.build_transition_patches import _fallback_patches, _load_case, _validate_patch


def patch(
    patch_id: str,
    *,
    context: str,
    transition: str,
    phase: str = "pre_final",
) -> dict:
    return {
        "id": patch_id,
        "domain": "shopping_assistant",
        "phase": phase,
        "source_task": patch_id,
        "trigger": context,
        "observed_tools": ["redeem_loyalty_points", "get_cart"],
        "state_cues": ["points_redeemed", "discount_applied", "cart_total"],
        "expected_action": transition,
        "obligations": ["Report the applied points, discount, and final total."],
        "forbidden": ["Do not claim that more points were applied than the tool reports."],
        "keywords": ["loyalty", "final total"],
        "context_text": context,
        "transition_text": transition,
    }


def artifact() -> dict:
    patches = [
        patch(
            "loyalty-final",
            phase="post_write",
            context=(
                "phase post_write user apply loyalty points observed tool redeem_loyalty_points "
                "result points_redeemed discount_applied cart_total"
            ),
            transition=(
                "candidate response applied points redeemed discount final total report exact tool result"
            ),
        ),
        patch(
            "promo-final",
            phase="post_write",
            context="phase post_write user apply promo observed tool apply_promo_code result discount cart_total",
            transition="candidate response promo status discount final total",
        ),
        patch(
            "shipping-final",
            phase="post_write",
            context="phase post_write user shipping observed tool set_shipping_option result shipping_cost cart_total",
            transition="candidate response shipping option cost final total",
        ),
    ]
    result = build_transition_artifact(
        patches, coreset_ratio=1.0, min_per_group=1, max_per_group=10
    )
    result["thresholds"]["shopping_assistant"]["post_write"] = {
        "context_radius": 0.45,
        "transition_radius": 0.45,
    }
    return result


def test_coreset_artifact_is_domain_and_phase_scoped() -> None:
    result = build_transition_artifact(
        [
            patch(
                f"task-{index}",
                context=f"cart context category{chr(97 + index)}",
                transition=f"step action{chr(97 + index)}",
            )
            for index in range(12)
        ],
        coreset_ratio=0.25,
        min_per_group=2,
        max_per_group=4,
    )

    assert len(result["patches"]) == 3
    assert result["stats"]["shopping_assistant"]["pre_final"] == 12
    assert result["stats"]["shopping_assistant"]["pre_final_coreset"] == 3


def test_supported_missing_disclosure_triggers_but_compliant_step_does_not() -> None:
    index = TransitionPatchIndex(artifact(), "shopping_assistant")
    context = (
        "phase post_write user apply loyalty points observed tool redeem_loyalty_points "
        "result points_redeemed discount_applied cart_total"
    )
    compliant = index.nearest(
        phase="post_write",
        context_text=context,
        transition_text="candidate response applied points redeemed discount final total exact tool result",
    )
    missing = index.nearest(
        phase="post_write", context_text=context, transition_text="candidate response done"
    )

    assert not index.should_verify("post_write", compliant)
    assert index.should_verify("post_write", missing)


def test_unseen_context_bypasses_even_when_candidate_is_short() -> None:
    index = TransitionPatchIndex(artifact(), "shopping_assistant")
    matches = index.nearest(
        phase="post_write",
        context_text="phase post_write user cancel an international flight after a weather delay",
        transition_text="candidate response done",
    )

    assert not index.should_verify("post_write", matches)


def test_public_train_trajectory_builds_fallback_transition_patch(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    trace_dir = data_root / "shopping_assistant"
    trace_dir.mkdir(parents=True)
    trace = {
        "conversation": [
            {"role": "user", "content": "Apply 500 loyalty points."},
            {
                "role": "assistant",
                "content": "500 points were applied and the final cart total is $45.",
                "tool_calls": [
                    {
                        "name": "redeem_loyalty_points",
                        "arguments": {"customer_id": "shop_1", "points": 500},
                        "result": {"points_redeemed": 500, "cart_total": 45},
                    }
                ],
            },
        ]
    }
    (trace_dir / "1.json").write_text(json.dumps(trace), encoding="utf-8")

    case = _load_case(trace_dir / "1.json", "shopping_assistant")
    raw = _fallback_patches(case)
    post_write = next(item for item in raw if item["phase"] == "post_write")
    validated = _validate_patch(post_write, {"1"}, "shopping_assistant")

    assert validated is not None
    assert validated["phase"] == "post_write"
    assert "final cart total" in validated["transition_text"]
    assert "copy task-specific values" not in validated["transition_text"]
    assert "500" not in validated["transition_text"]


def test_tool_identifiers_share_natural_language_tokens() -> None:
    assert tokens("process_return") == ["process_return", "process", "return"]


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def completion(text="", tool_calls=None):
    return SimpleNamespace(
        text=text,
        tool_calls=tool_calls or [],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
    )


def build_agent(
    tmp_path: Path,
    monkeypatch,
    client: ScriptedClient,
    *,
    mode: str,
) -> ProcessWorkflowMemoryAgent:
    workflow_path = tmp_path / "workflows.json"
    transition_path = tmp_path / "transitions.json"
    workflow_path.write_text(json.dumps({"cards": []}), encoding="utf-8")
    transition_path.write_text(json.dumps(artifact()), encoding="utf-8")
    monkeypatch.setattr(ProcessWorkflowMemoryAgent, "memory_path", workflow_path)
    monkeypatch.setenv("STATE_BENCH_TRANSITION_PATCH_MODE", mode)
    monkeypatch.setenv("STATE_BENCH_TRANSITION_PATCH_PATH", str(transition_path))
    return ProcessWorkflowMemoryAgent(
        client,
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="shopping_assistant"),
    )


def loyalty_conversation() -> list[dict]:
    return [
        {"role": "user", "content": "Apply my loyalty points and tell me the final total."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "redeem_loyalty_points",
                    "arguments": {"customer_id": "shop_1", "points": 24950},
                    "result": {
                        "points_redeemed": 24900,
                        "discount_applied": 249,
                        "cart_total": 250,
                    },
                }
            ],
        },
    ]


def test_mode_off_preserves_original_single_call(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient([completion(text="Done.")])
    agent = build_agent(tmp_path, monkeypatch, client, mode="off")

    response = agent.generate_next_turn(
        system_prompt="system", conversation=loyalty_conversation(), tools=[]
    )

    assert response.text == "Done."
    assert len(client.calls) == 1


def test_anomalous_transition_is_verified_and_revised(tmp_path: Path, monkeypatch) -> None:
    client = ScriptedClient(
        [
            completion(text="Done."),
            completion(
                text=json.dumps(
                    {
                        "decision": "revise",
                        "confidence": 0.95,
                        "patch_ids": ["loyalty-final"],
                        "feedback": "Report the exact applied points, discount, and final total.",
                    }
                )
            ),
            completion(text="24,900 points were applied for $249 off; the final total is $250."),
        ]
    )
    agent = build_agent(tmp_path, monkeypatch, client, mode="enforce")

    response = agent.generate_next_turn(
        system_prompt="system", conversation=loyalty_conversation(), tools=[]
    )

    assert response.text.startswith("24,900 points")
    assert len(client.calls) == 3
    assert "Local transition verification found" in client.calls[2]["conversation"][-1]["content"]


def test_shadow_mode_never_changes_candidate(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("STATE_BENCH_TRANSITION_PATCH_LOG_PATH", str(log_path))
    client = ScriptedClient([completion(text="Done.")])
    agent = build_agent(tmp_path, monkeypatch, client, mode="shadow")

    response = agent.generate_next_turn(
        system_prompt="system", conversation=loyalty_conversation(), tools=[]
    )

    assert response.text == "Done."
    assert len(client.calls) == 1
    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["domain"] == "shopping_assistant"
    assert record["phase"] == "post_write"
    assert record["triggered"] is True
    assert "text" not in record
