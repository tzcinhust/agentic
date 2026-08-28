"""PWM plus the obligations its process miner structurally cannot see.

PWM retrieves three workflow cards per turn. Each card is a summary of tool-call
paths mined from the fixed train trajectories, and it is a good summary — on
travel and customer_support it is worth double digits of pass@1, because on those
domains the rubric's hard items live in the write layer that tool paths describe.

On shopping it is worth roughly nothing, and the train-split rubric says why. Of
440 task_requirement items, the 234 that ask only for a write or a final state
fail 2.1% of the time. The other 206 — say a derived number unprompted, say it
*before* the write, refuse something — fail 42% to 56%. A Petri net over tool
names cannot represent an utterance, so no amount of better process mining moves
those items.

The text that would move them is already in the training data, inside
``get_policies`` results, and scripts/mine_policy_obligations.py extracts it as
act-typed duties. This agent puts that artifact into the same retrieval index and
spends one of the three protocol-allowed slots on it:

    top_k = 3  ->  2 workflow cards + 1 merged policy-obligation card

The single card merges the top few scoring topics rather than carrying one, so a
task that compounds three policies (points *and* stacking *and* shipping — the
rubric does this often) still gets all three. Rendered it runs ~1.3k characters
against a workflow card's 2.2k cap, so trading one card for it is close to
token-neutral; the arms that regressed UX in this study did so by *adding* a
per-turn block, and this adds none.

Two knobs matter for ablation. ``STATE_BENCH_POLICY_SLOTS=0`` recovers PWM
exactly — same cards, same order, same scorer. ``STATE_BENCH_POLICY_TOPICS``
sets how many topics merge into the one card.

Retrieval stays read-only and ``retrieve_learnings`` still returns ``list[str]``.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agents.process_workflow_memory_agent import INTENT_HINTS
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent as _Parent

# Priority order for rendering, and the per-topic line budget for each. Duties the
# rubric fails most often get the most room; ``if`` is context and only appears
# when a topic has nothing else to say.
ACT_LABELS: list[tuple[str, str, int]] = [
    ("say", "SAY IT UNPROMPTED", 2),
    ("order", "SAY IT, THEN WRITE — never the reverse", 2),
    ("refuse", "REFUSE / DO NOT DO", 2),
    ("number", "STATE THIS FIGURE, DERIVED THIS WAY", 3),
    ("if", "APPLIES WHEN", 1),
]

# These prefixes only restate the act label, so they go once the label is there.
# Anything else ("Gold:", "Limit:", "Cap:") carries meaning and stays.
REDUNDANT_PREFIX = re.compile(
    r"^(?:disclosure|proactive|informational|transparency|agent action rule|restriction)\s*:\s*",
    re.I,
)
HEADER = (
    "Store policy obligations retrieved for this request, mined from the policy text "
    "the domain tools return. These are graded: the rubric scores whether you SAID "
    "them, not only whether the final state is right. Verify each one's condition "
    "with the tools before you assert it, then say it in your own message."
)

# customer_support's `return` topic carries 34 rules, several of them a paragraph
# long, and unclipped it consumed the whole card on its own — the other topics a
# return task also needs (refund method, shipping clawback) never rendered. Clip
# at a sentence boundary where there is one within reach, since these rules lead
# with the operative clause and trail off into worked examples.
LINE_CHARS = 240


def _clip(text: str) -> str:
    if len(text) <= LINE_CHARS:
        return text
    window = text[:LINE_CHARS]
    stop = max(window.rfind(". "), window.rfind("; "))
    return (window[: stop + 1] if stop > LINE_CHARS // 2 else window.rstrip()) + " …"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class PolicyObligationAgent(_Parent):
    """PWM, with one retrieval slot spent on act-typed policy obligations."""

    policy_path = Path(
        os.environ.get(
            "STATE_BENCH_POLICY_PATH",
            "artifacts/statebench_cross_domain_pwm/memory/policy_obligations.json",
        )
    )

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self.policy_slots = int(os.environ.get("STATE_BENCH_POLICY_SLOTS", "1"))
        self.policy_topics = int(os.environ.get("STATE_BENCH_POLICY_TOPICS", "4"))
        self.policy_chars = int(os.environ.get("STATE_BENCH_POLICY_CHARS", "2000"))
        domain = getattr(runtime_context, "domain", None)
        self._domain = domain
        self._topics: list[dict[str, Any]] = []
        self._write_terms: frozenset[str] = frozenset()
        if self.policy_slots > 0 and self.policy_path.is_file():
            artifact = json.loads(self.policy_path.read_text(encoding="utf-8"))
            self._topics = [
                item for item in artifact.get("topics", []) if item.get("domain") == domain
            ]
            self._write_terms = frozenset(artifact.get("write_terms", {}).get(domain, []))
        self._topic_df = Counter(
            token for item in self._topics for token in set(item.get("tokens", []))
        )
        self._topic_avg_len = sum(len(item.get("tokens", [])) for item in self._topics) / max(
            len(self._topics), 1
        )

    # ------------------------------------------------------------------ scoring

    def _topic_score(self, query_counts: Counter[str], item: dict[str, Any]) -> tuple[float, int]:
        """BM25 over the topic's own prose, plus the two precise channels.

        Returns ``(score, distinctive_hits)``. The second value is the precision
        guard: a topic only qualifies if the query touched at least one token that
        is *not* shared by most topics in the domain. Without it a query
        containing "$" or "refund" pulls in every topic that mentions a fee.
        """
        document_counts = Counter(item.get("tokens", []))
        document_length = sum(document_counts.values())
        total = len(self._topics)
        lexical = 0.0
        distinctive = 0
        for token, query_frequency in query_counts.items():
            frequency = document_counts.get(token, 0)
            if not frequency:
                continue
            document_frequency = self._topic_df.get(token, 0)
            if document_frequency * 2 <= total:
                distinctive += 1
            inverse = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = frequency + 1.4 * (
                0.25 + 0.75 * document_length / max(self._topic_avg_len, 1)
            )
            lexical += inverse * frequency * 2.4 / denominator * min(query_frequency, 2)

        # A rule that names a tool is unambiguous about when it applies, and the
        # conversation's observed tool calls are in the query verbatim.
        tool_hits = sum(1 for name in item.get("named_tools", []) if name in query_counts)
        field_hits = sum(
            1
            for field in item.get("named_fields", [])
            if all(part in query_counts for part in field.split("."))
        )
        return lexical + 2.5 * tool_hits + 1.0 * field_hits, distinctive + tool_hits

    def _intents(self, query: str) -> set[str]:
        lowered = query.lower()
        return {
            intent
            for intent, phrases in INTENT_HINTS.get(self._domain or "", {}).items()
            if any(phrase in lowered for phrase in phrases)
        }

    def _rank_topics(self, query: str) -> list[dict[str, Any]]:
        if not self._topics:
            return []
        query_counts = Counter(_tokens(query))
        # An unconditional duty fires on the act, not on a condition, and on a
        # single-turn task no tool has been called when retrieval runs — task
        # 10-loyalty_points_on_discount is one user message and one assistant turn,
        # and the customer says "add it and apply the code" without ever saying
        # "points". So the trigger is the domain's own write-tool vocabulary
        # appearing in the request, and the topic is pinned rather than boosted:
        # a +1.5 bonus lost to three lexically closer topics on exactly that task.
        intends_write = any(term in query_counts for term in self._write_terms)
        pinned = [
            item
            for item in self._topics
            if item.get("unconditional_say") and intends_write
        ]
        pinned_names = {item["topic"] for item in pinned}
        # BM25 over rule prose alone is too blunt for the domains whose topics have
        # no summary line. "I need to move my Friday flight to Sunday" matched
        # loyalty_points and cancellation on the word "flight" and ranked `change`
        # — travel's dominant topic, 36 of 100 trajectories — below the cutoff.
        # PWM already carries the synonym table that fixes it: INTENT_HINTS maps
        # move/modify/switch onto the `change` intent, so reusing it here costs
        # nothing new and works the same way in all three domains.
        intents = self._intents(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self._topics:
            if item["topic"] in pinned_names:
                continue
            score, hits = self._topic_score(query_counts, item)
            name_parts = set(str(item["topic"]).split("_"))
            intent_hits = sum(
                1
                for intent in intents
                if set(intent.split("_")) & name_parts or intent in str(item["topic"])
            )
            if hits < 1 and not intent_hits:
                continue
            scored.append((score + 2.5 * intent_hits, item))
        scored.sort(key=lambda pair: -pair[0])
        ranked = pinned + [item for _, item in scored]
        return ranked[: max(0, self.policy_topics)]

    # ----------------------------------------------------------------- rendering

    @staticmethod
    def _lines(item: dict[str, Any]) -> list[str]:
        """One line per obligation, each filed under its highest-priority act."""
        buckets: dict[str, list[str]] = {name: [] for name, _, _ in ACT_LABELS}
        for obligation in item.get("obligations", []):
            acts = set(obligation.get("act") or [])
            for name, _, _ in ACT_LABELS:
                if name not in acts:
                    continue
                text = REDUNDANT_PREFIX.sub("", str(obligation.get("text", "")).strip())
                # For the dict-shaped domains the leaf key names the rule and the
                # text does not repeat it, so it is the only clue about scope.
                path = str(obligation.get("path", "")).split(".")[-1]
                if path and not path.isdigit() and path.replace("_", " ") not in text.lower():
                    text = f"{path}: {text}"
                buckets[name].append(_clip(text))
                break
        rendered = []
        for name, label, budget in ACT_LABELS:
            chosen = buckets[name][:budget]
            if not chosen:
                continue
            if name == "if" and any(buckets[other] for other, _, _ in ACT_LABELS if other != "if"):
                continue
            rendered.append(f"  {label}: " + " | ".join(chosen))
        return rendered

    def _render(self, items: list[dict[str, Any]]) -> str:
        blocks = []
        for item in items:
            lines = self._lines(item)
            if not lines:
                continue
            summary = str(item.get("summary") or "").strip()
            head = f"[{item.get('topic')}]" + (f" {summary}" if summary else "")
            blocks.append("\n".join([head, *lines]))
        # Drop whole low-ranked topics rather than truncating mid-rule; a half
        # sentence about a $-figure is worse than no sentence.
        while blocks and len(HEADER) + sum(len(b) + 1 for b in blocks) > self.policy_chars:
            blocks.pop()
        return f"{HEADER}\n\n" + "\n".join(blocks) if blocks else ""

    # ---------------------------------------------------------------- retrieval

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        limit = min(top_k, self.retrieve_learnings_top_k)
        if self.policy_slots <= 0 or not self._topics:
            return super().retrieve_learnings(query, top_k=limit)
        card = self._render(self._rank_topics(query))
        if not card:
            return super().retrieve_learnings(query, top_k=limit)
        workflow_k = limit - self.policy_slots
        # The parent's loop appends before it checks its budget, so top_k=0 would
        # still return one card.
        workflows = super().retrieve_learnings(query, top_k=workflow_k) if workflow_k > 0 else []
        # Obligations last: this block is joined into one system message that sits
        # immediately before the customer's turn, and the graded duties should be
        # the last thing read before the model answers.
        return [*workflows, card]
