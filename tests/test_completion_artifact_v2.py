from __future__ import annotations

import json
import re
from pathlib import Path

from agents.completion_templates import CompletionTemplateIndex


ARTIFACT = (
    Path(__file__).parents[1]
    / "artifacts"
    / "task_closure_memory_v2"
    / "memory"
    / "completion_templates.json"
)
EXPOSED_NUMERIC_LITERAL = re.compile(
    r"(?:[$£€]\s*\d|\b\d+(?:\.\d+)?\s*(?:USD|dollars?|euros?|pounds?)\b)|"
    r"(?<![A-Za-z_])\d+(?:\.\d+)?(?:\s*%|\s*(?:hours?|days?|nights?|items?|points?)\b)",
    re.IGNORECASE,
)
EXPOSED_ENTITY_ID = re.compile(r"\b[A-Z]{1,8}[-_][A-Z0-9]{2,}\b")
EXPOSED_PLANNING_DIRECTIVE = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior)|system prompt|developer message|"
    r"(?:call|invoke|run) (?:the )?(?:tool|[a-z][a-z0-9]+_[a-z0-9_]+)|next tool",
    re.IGNORECASE,
)


def test_committed_completion_artifact_has_no_oracle_or_exposed_answer_literals() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["kind"] == "independent_task_completion_templates"
    assert payload["source"] == {
        "trajectory_count": 300,
        "domains": ["customer_support", "shopping_assistant", "travel"],
        "conversation_only": True,
        "uses_task_summary": False,
        "uses_task_requirements": False,
    }

    templates = payload["templates"]
    identifiers = [item["id"] for item in templates]
    assert len(identifiers) == len(set(identifiers))
    assert all(item["obligations"] for item in templates)

    # These are the only fields shown to the semantic bookkeeper. Retrieval
    # metadata may contain train-language cues, but cannot leak numeric answers
    # into the runtime completion requirements.
    for item in templates:
        exposed = json.dumps(
            {
                "title": item["title"],
                "trigger": item["trigger"],
                "obligations": item["obligations"],
            },
            ensure_ascii=False,
        )
        assert not EXPOSED_NUMERIC_LITERAL.search(exposed), item["id"]
        assert not EXPOSED_ENTITY_ID.search(exposed), item["id"]
        assert not EXPOSED_PLANNING_DIRECTIVE.search(exposed), item["id"]


def test_independent_retrieval_preserves_latent_completion_categories() -> None:
    shopping = CompletionTemplateIndex.from_path(
        ARTIFACT, domain="shopping_assistant", top_k=8
    ).retrieve("add an item to my cart and report the resulting cart total add_to_cart cart_total")
    travel = CompletionTemplateIndex.from_path(ARTIFACT, domain="travel", top_k=8).retrieve(
        "cancel a reservation near a timing deadline; waiting may cross a fee boundary"
    )
    support = CompletionTemplateIndex.from_path(
        ARTIFACT, domain="customer_support", top_k=8
    ).retrieve("exchange several items but keep the protected order unchanged")

    assert any(
        "profile_benefit" in item.get("latent_signal_categories", []) for item in shopping
    )
    assert any(
        "boundary_transition" in item.get("latent_signal_categories", []) for item in travel
    )
    assert any("protected_state" in item.get("latent_signal_categories", []) for item in support)
