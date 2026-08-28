"""Replay policy-obligation retrieval over a scored run, without any API calls.

Every arm tried in this study cost roughly an hour of relay time per domain per
replicate, and the last three came back null. The reason they came back null was
visible offline in each case: the trigger fired on the wrong turns. So this arm
gets checked offline first.

A scored record carries its full ``conversation``, so the query the agent would
have built at each of its turns can be reconstructed exactly — same
``_query_from_conversation``, same scorer, same renderer, real class. What that
buys is three numbers that decide whether the arm is worth running:

*Fire rate.* How often a card is produced at all. A card on every turn of every
task is the A2 failure mode (35% more tokens, UX down 0.65). A card on almost no
turn is a no-op.

*Coverage.* Whether the retrieved topics actually speak to the items the task
failed. An item counts as being *about* a topic only when it uses a term unique to
that topic within the domain; the first version of this check asked for two merely
uncommon terms and reported 91%, while crediting ``promo_stacking`` for an item
about loyalty points. Items about no topic at all are reported separately, because
they are the artifact's ceiling, not its miss.

*Cost.* Characters added versus the workflow card displaced, since one of the
three slots is spent rather than added.

Coverage here is an upper bound on what the arm can win: it says the obligation
was in front of the model, not that the model acted on it. The anchor terms are
printable with ``--anchors`` because they are approximate and should be audited.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agents.policy_obligation_agent import PolicyObligationAgent
from scripts.rubric_failure_taxonomy import classify


class _Context:
    def __init__(self, domain: str) -> None:
        self.domain = domain


def _stub_agent(domain: str) -> PolicyObligationAgent:
    return PolicyObligationAgent(
        client=None,
        system_prompt="",
        tools=[],
        tool_handlers={},
        runtime_context=_Context(domain),
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _prefixes(conversation: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Every conversation prefix ending just before an assistant turn.

    That is where ``prepare_conversation`` runs, so it is where retrieval sees
    whatever tool calls have already happened.
    """
    out = []
    for index, message in enumerate(conversation):
        if message.get("role") == "assistant":
            out.append(conversation[:index])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--show", type=int, default=0, help="print N failing tasks in full")
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--anchors", action="store_true", help="print the per-topic anchor terms")
    args = parser.parse_args()

    agent = _stub_agent(args.domain)
    if not agent._topics:
        raise SystemExit(f"no policy topics for domain {args.domain}")

    # A rubric item is "about" a topic only if it uses a term that belongs to that
    # topic and to no other in the domain. The first version of this check asked
    # for two merely-uncommon terms and reported 91% coverage while crediting
    # promo_stacking for a loyalty-points item — "cart", "total" and "discount"
    # are enough to match almost anything. Uniqueness is the only bar that fails
    # loudly when the wrong topic is retrieved.
    anchors = {
        topic["topic"]: {
            token
            for token in set(topic["tokens"])
            if agent._topic_df.get(token, 0) == 1 and len(token) > 3
        }
        for topic in agent._topics
    }
    if args.anchors:
        for name, terms in sorted(anchors.items()):
            print(f"{name:24s} {' '.join(sorted(terms))}")
        print()

    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.run.glob("*.json"))]
    records = [r for r in records if r.get("task_completion_pass") is not None]

    turns_total = 0
    turns_fired = 0
    card_chars: list[int] = []
    topic_hits: Counter[str] = Counter()
    covered_by_family: Counter[str] = Counter()
    failed_by_family: Counter[str] = Counter()
    no_topic_by_family: Counter[str] = Counter()
    items_with_topic = 0
    fully_covered_tasks = 0
    reachable_tasks = 0
    failing_tasks = 0
    shown = 0

    for record in records:
        conversation = record.get("conversation") or []
        retrieved: dict[str, list[str]] = {}
        for prefix in _prefixes(conversation):
            turns_total += 1
            query = agent._query_from_conversation(prefix)
            ranked = agent._rank_topics(query)
            card = agent._render(ranked)
            if not card:
                continue
            turns_fired += 1
            card_chars.append(len(card))
            for topic in ranked:
                topic_hits[topic["topic"]] += 1
                retrieved.setdefault(topic["topic"], [])

        failing = [
            str(item.get("requirement") or "")
            for item in record.get("task_requirements_details") or []
            if not item.get("passed")
        ]
        if not failing:
            continue
        failing_tasks += 1
        covered_flags = []
        for requirement in failing:
            family = classify(requirement)
            failed_by_family[family] += 1
            words = set(_tokens(requirement))
            # Which topic the item is about, whether or not it was retrieved.
            about = {name for name, terms in anchors.items() if terms & words}
            if about:
                items_with_topic += 1
            hit = next(iter(sorted(about & set(retrieved))), None)
            covered_flags.append((hit, bool(about)))
            if hit:
                covered_by_family[family] += 1
            elif not about:
                no_topic_by_family[family] += 1
        if all(hit for hit, _ in covered_flags):
            fully_covered_tasks += 1
            # A task only flips if its state requirements already hold; a covered
            # utterance duty cannot rescue a cart that is simply wrong.
            if int(record.get("state_requirements_met") or 0):
                reachable_tasks += 1

        if args.show and shown < args.show and (
            not args.only_failed or not all(hit for hit, _ in covered_flags)
        ):
            shown += 1
            print(f"\n===== {record['task_id']}  retrieved={sorted(retrieved)}")
            for requirement, (hit, about) in zip(failing, covered_flags):
                if hit:
                    mark = f"COVERED by {hit}"
                elif about:
                    mark = "MISSED — a topic exists, retrieval did not surface it"
                else:
                    mark = "OUT OF SCOPE — no policy topic covers this"
                print(f"  [{classify(requirement):20s}] {mark}")
                print(f"      {requirement[:200]}")

    print()
    print(f"tasks {len(records)}   with failing items {failing_tasks}")
    print(
        f"turns {turns_total}   card produced on {turns_fired} "
        f"({100.0 * turns_fired / max(turns_total, 1):.0f}%)"
    )
    if card_chars:
        print(
            f"card chars: mean {sum(card_chars) / len(card_chars):.0f}  "
            f"max {max(card_chars)}  (workflow card cap is 2200, and one is displaced)"
        )
    print()
    print(f"{'family':24s} {'failed':>7s} {'no topic':>9s} {'covered':>8s} {'cov% of reachable':>18s}")
    for family in sorted(failed_by_family, key=lambda f: -failed_by_family[f]):
        failed = failed_by_family[family]
        reachable = failed - no_topic_by_family[family]
        share = 100.0 * covered_by_family[family] / reachable if reachable else 0.0
        print(
            f"{family:24s} {failed:7d} {no_topic_by_family[family]:9d} "
            f"{covered_by_family[family]:8d} {share:17.0f}%"
        )
    total_failed = sum(failed_by_family.values())
    no_topic = sum(no_topic_by_family.values())
    covered = sum(covered_by_family.values())
    reachable = total_failed - no_topic
    print(
        f"{'ALL':24s} {total_failed:7d} {no_topic:9d} {covered:8d} "
        f"{100.0 * covered / max(reachable, 1):17.0f}%"
    )
    print()
    print(
        f"{no_topic}/{total_failed} failing items have no policy topic at all — the rubric "
        f"charges for them but no get_policies result in the training data says so"
    )
    passes = sum(int(r["task_completion_pass"] or 0) for r in records)
    print(
        f"tasks whose every failing item is covered: {fully_covered_tasks}/{failing_tasks}; "
        f"{reachable_tasks} of those also already satisfy their state requirements"
    )
    print(
        f"pass@1 ceiling for this arm: {100.0 * passes / len(records):.1f}% "
        f"-> {100.0 * (passes + reachable_tasks) / len(records):.1f}% "
        f"(every covered duty acted on, which no arm achieves)"
    )
    print()
    print("topic retrieval frequency (turns):")
    for name, count in topic_hits.most_common():
        print(f"  {name:24s} {count:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
