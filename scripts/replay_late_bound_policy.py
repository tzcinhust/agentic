"""Estimate late-bound policy coverage from an existing scored trajectory run.

No API calls are made. A topic is available only when the recorded agent
actually called ``get_policies``; this mirrors LateBoundPolicyAgent's runtime
trigger and therefore gives a stricter ceiling than eager lexical retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from agents.late_bound_policy_agent import LateBoundPolicyAgent
from scripts.replay_policy_retrieval import _Context
from scripts.rubric_failure_taxonomy import classify


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    args = parser.parse_args()

    agent = LateBoundPolicyAgent(
        client=None,
        system_prompt="",
        tools=[],
        tool_handlers={},
        runtime_context=_Context(args.domain),
    )
    anchors = {
        item["topic"]: {
            token
            for token in set(item.get("tokens", []))
            if agent._topic_df.get(token, 0) == 1 and len(token) > 3
        }
        for item in agent._topics
    }

    failed = Counter()
    no_topic = Counter()
    covered = Counter()
    fetched_counts = Counter()
    card_chars: list[int] = []
    fully_covered = 0
    state_reachable = 0
    failing_tasks = 0
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.run.glob("*.json"))]

    for record in records:
        fetched: set[str] = set()
        for message in record.get("conversation") or []:
            for call in message.get("tool_calls") or []:
                if call.get("name") != "get_policies":
                    continue
                arguments = call.get("arguments") or {}
                item = agent._topic_item(str(arguments.get("topic", "")))
                if item is None:
                    continue
                name = str(item["topic"])
                fetched.add(name)
                fetched_counts[name] += 1
                card = agent._render([item])
                if card:
                    card_chars.append(len(card))

        requirements = [
            str(item.get("requirement") or "")
            for item in record.get("task_requirements_details") or []
            if not item.get("passed")
        ]
        if not requirements:
            continue
        failing_tasks += 1
        flags: list[bool] = []
        for requirement in requirements:
            family = classify(requirement)
            failed[family] += 1
            words = _tokens(requirement)
            about = {name for name, terms in anchors.items() if terms & words}
            hit = bool(about & fetched)
            flags.append(hit)
            if hit:
                covered[family] += 1
            elif not about:
                no_topic[family] += 1
        if flags and all(flags):
            fully_covered += 1
            if int(record.get("state_requirements_met") or 0):
                state_reachable += 1

    print(f"tasks {len(records)}   with failing items {failing_tasks}")
    print(f"get_policies calls mapped {sum(fetched_counts.values())}")
    if card_chars:
        print(f"JIT card chars: mean {sum(card_chars) / len(card_chars):.0f}  max {max(card_chars)}")
    print()
    print(f"{'family':24s} {'failed':>7s} {'no topic':>9s} {'covered':>8s} {'cov%':>7s}")
    for family in sorted(failed, key=lambda name: -failed[name]):
        reachable = failed[family] - no_topic[family]
        share = 100.0 * covered[family] / reachable if reachable else 0.0
        print(
            f"{family:24s} {failed[family]:7d} {no_topic[family]:9d} "
            f"{covered[family]:8d} {share:6.0f}%"
        )
    total_failed = sum(failed.values())
    total_no_topic = sum(no_topic.values())
    total_covered = sum(covered.values())
    reachable = total_failed - total_no_topic
    print(
        f"{'ALL':24s} {total_failed:7d} {total_no_topic:9d} {total_covered:8d} "
        f"{100.0 * total_covered / max(reachable, 1):6.0f}%"
    )
    passes = sum(int(record.get("task_completion_pass") or 0) for record in records)
    print()
    print(
        f"tasks whose every failing item is covered: {fully_covered}/{failing_tasks}; "
        f"{state_reachable} also satisfy state requirements"
    )
    print(
        f"pass@1 ceiling: {100.0 * passes / max(len(records), 1):.1f}% -> "
        f"{100.0 * (passes + state_reachable) / max(len(records), 1):.1f}%"
    )
    print("fetched topic counts:")
    for name, count in fetched_counts.most_common():
        print(f"  {name:24s} {count:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
