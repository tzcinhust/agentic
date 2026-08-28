"""Replay the obligation ledger over already-scored trajectories.

Running the agent costs money and takes half an hour; replay costs nothing and
answers the question that decides whether the run is worth it. For each scored
trajectory we walk the conversation prefix by prefix, ask the ledger what it
would have said at that point, and compare the topics it raised against the
rubric items the judge actually failed.

Two numbers matter, and they pull against each other:

*Recall.* Of the failed rubric items whose topic the obligation table covers, on
how many would the ledger have raised that topic *before the conversation
ended*? A covered item the ledger never raises is coverage on paper only.

*Load.* How many lines does the ledger add per turn, and how often does it fire
on tasks that already passed? Every line is prompt the agent has to spend
attention on, and a checklist that fires indiscriminately is one the model
learns to skim. Firing on a passing task is not automatically an error — the
obligation may be live and correctly discharged later in the same turn — but the
rate bounds how much noise the intervention introduces.

Usage:
    uv run python scripts/replay_obligation_ledger.py \
        --domain shopping_assistant \
        --results-dir artifacts/statebench_cross_domain_pwm/runs/shopping_assistant
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agents.obligation_ledger_agent import ObligationLedgerAgent

TOKEN = re.compile(r"[a-z]{4,}")
TOPIC_LINE = re.compile(r"^- ([a-z_]+):")
JOIN_LINE = re.compile(r"decides ([a-z_, ]+)\.")


def raised_topics(line: str) -> set[str]:
    """Topics a ledger line puts in front of the agent.

    The consolidated join line names several topics at once, and it counts for
    each of them: it states the field that decides the rule and the tool that
    supplies it, which is the actionable part.
    """
    join = JOIN_LINE.search(line)
    if join:
        return {topic.strip() for topic in join.group(1).split(",") if topic.strip()}
    match = TOPIC_LINE.match(line)
    return {match.group(1)} if match else set()


def build_ledger(domain: str, obligations: Path | None) -> ObligationLedgerAgent:
    """A ledger with no client, no tools and no model — just the probe logic."""
    agent = ObligationLedgerAgent.__new__(ObligationLedgerAgent)
    if obligations is not None:
        agent.obligations_dir = obligations
    agent._load_obligations(domain)
    return agent


def prefixes(conversation: list[Any]) -> list[list[Any]]:
    """Conversation states the ledger is actually computed from.

    The hook fires once per *tool round*, not once per turn, so the replay has to
    reproduce the intra-turn sequence. A stored trajectory collapses a whole turn
    into one assistant message with all its tool calls nested; the orchestrator
    instead appends them one group at a time. Expanding each assistant message
    into its successive tool-call prefixes reproduces what the live agent sees,
    and it matters: replaying only the pre-turn states makes the ledger look
    inert on every single-turn task, which is most of them.
    """
    states: list[list[Any]] = []
    for index, message in enumerate(conversation):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        head = conversation[:index]
        states.append(head)
        calls = message.get("tool_calls") or []
        for count in range(1, len(calls) + 1):
            # Blank the prose: at tool-round time the assistant has not written
            # it yet, and leaving it in would let the discharge test see the
            # disclosure the ledger is supposed to be prompting for.
            states.append([*head, {**message, "tool_calls": calls[:count], "content": ""}])
    return states


def failed_topics(data: dict[str, Any], probes: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Failed rubric items paired with the probe topic they belong to."""
    pairs: list[tuple[str, str]] = []
    for item in data.get("task_requirements_details") or []:
        if item.get("passed"):
            continue
        text = f"{item.get('id', '')} {item.get('requirement', '')}"
        tokens = set(TOKEN.findall(text.lower()))
        best: tuple[int, str] | None = None
        for probe in probes:
            terms = {part for part in probe["topic"].split("_") if len(part) > 3}
            terms |= set(probe["discharge_check"]["distinctive_terms"])
            overlap = len(terms & tokens)
            if overlap and (best is None or overlap > best[0]):
                best = (overlap, probe["topic"])
        if best:
            pairs.append((best[1], str(item.get("id", ""))))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline replay of the obligation ledger")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--run", type=str, default="run1")
    parser.add_argument("--obligations", type=str, default=None)
    parser.add_argument("--show", type=int, default=6, help="Example ledger blocks to print")
    args = parser.parse_args()

    ledger = build_ledger(args.domain, Path(args.obligations) if args.obligations else None)
    probes = ledger._pool
    files = sorted((Path(args.results_dir) / args.run).glob("*.json"))
    if not files:
        parser.error(f"No trajectories in {Path(args.results_dir) / args.run}")

    hit: Counter[str] = Counter()
    missed: Counter[str] = Counter()
    fired_on_pass: Counter[str] = Counter()
    lines_per_turn: list[int] = []
    tasks = 0
    examples: list[tuple[str, str, list[str]]] = []

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("task_completion_pass") is None:
            continue
        tasks += 1
        conversation = data.get("conversation") or []
        raised: set[str] = set()
        blocks: list[list[str]] = []
        for prefix in prefixes(conversation):
            lines = ledger._ledger(prefix)
            lines_per_turn.append(len(lines))
            if lines:
                blocks.append(lines)
            for line in lines:
                raised |= raised_topics(line)

        pairs = failed_topics(data, probes)
        for topic, item_id in pairs:
            (hit if topic in raised else missed)[topic] += 1
            if topic not in raised and len(examples) < args.show:
                examples.append((path.stem, f"MISS {topic}/{item_id}", blocks[-1] if blocks else []))
        if not pairs:
            for topic in raised:
                fired_on_pass[topic] += 1
        if blocks and pairs and len(examples) < args.show:
            examples.append((path.stem, f"HIT  {pairs[0][0]}/{pairs[0][1]}", blocks[-1]))

    total = sum(hit.values()) + sum(missed.values())
    turns = len(lines_per_turn)
    firing_turns = sum(1 for count in lines_per_turn if count)
    print(f"domain               : {args.domain} ({args.run})")
    print(f"scored tasks         : {tasks}")
    print(f"agent turns replayed : {turns}")
    print(f"turns with a ledger  : {firing_turns} ({firing_turns / max(turns, 1):.0%})")
    print(f"mean lines per turn  : {sum(lines_per_turn) / max(turns, 1):.2f}")
    print()
    print(f"covered failed items : {total}")
    print(f"  raised before end  : {sum(hit.values())} ({sum(hit.values()) / max(total, 1):.0%})")
    print(f"  never raised       : {sum(missed.values())}")
    print()
    print(f"{'topic':22s} {'raised':>7s} {'missed':>7s}   fired on tasks with no covered failure")
    print("-" * 88)
    for probe in probes:
        topic = probe["topic"]
        if not (hit[topic] or missed[topic] or fired_on_pass[topic]):
            continue
        print(f"{topic:22s} {hit[topic]:7d} {missed[topic]:7d}   {fired_on_pass[topic]:d}")
    if examples:
        print()
        for name, label, block in examples:
            print(f"--- {name[:34]:34s} {label}")
            for line in block:
                print(f"    {line}")


if __name__ == "__main__":
    main()
