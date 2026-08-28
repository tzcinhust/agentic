"""Score a ledger against reference behaviour, using no judge labels.

The test split is spent: three design decisions in the parent agent were fitted
to its labels, so it cannot be used again to choose between variants. Train
trajectories are the alternative, but they are reference behaviour rather than
scored attempts — there are no failed rubric items to recall. So the question has
to be reframed into one the reference answers on its own.

*Gold agreement.* Every ledger line points at an action, and the action depends on
what kind of line it is. An unread-policy line points at ``get_policies(topic)``;
a latent-field line points at the detail lookups that supply the field; an
imperative disclosure line points at an *utterance* — the topic appearing in the
agent's own prose beside a figure. Ask whether the reference trajectory, from that
same point onward, actually performs it. If the reference does, the line
anticipated behaviour that the task genuinely required. If the reference never
touches it for the rest of the conversation, the line is noise — not merely
unhelpful but actively costly, since the live agent will obey and spend a tool
call on it.

Scoring the utterance separately is not a refinement, it is the difference between
measuring the imperative arms and libelling them. An imperative line whose topic
the agent has already fetched carries no lookup at all, and there are many of
them — 116 of 164 on shopping, 383 of 520 on customer_support. Charged against a
fetch that by construction lies in the past, every one is an automatic miss, and
the arm's precision collapses for a reason that has nothing to do with its
behaviour. So a disclosure line is satisfied when the reference's later prose
discharges the topic, by the ledger's own ``_discharged`` test.

This is precision, not recall, and it is the metric that matters for the
regression being chased: the parent's damage came from firing too much, not from
firing too little. Load is reported beside it because a ledger can buy agreement
by staying silent.

Anticipation is measured strictly forward. A line only fires while the gap is
open, and it extinguishes the moment the reference closes it, so an action the
reference performed *before* the prefix can never be credited or charged.

Usage:
    uv run python scripts/replay_gold_agreement.py --domain shopping_assistant \
        --agent-class LatentStateAgent
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

POLICY_CALL = re.compile(r"get_policies\('([a-z_]+)'\)")
DECIDES = re.compile(r"decides ([a-z_, ]+)\.")
# Shared by both imperative arms, so one pattern identifies a disclosure demand
# without having to ask which agent produced it.
IMPERATIVE = re.compile(r"- ([a-z_]+): bears on this case")
# The line agrees with its subject, so a single named tool reads "supplies". An
# earlier version of this pattern matched only the plural and silently scored
# every one-tool line as pointing at nothing.
SUPPLY = re.compile(r"([a-z_]+(?: / [a-z_]+)*) suppl(?:y|ies) them per entry")


def half_of(stem: str) -> str:
    """Stable, order-independent assignment of a trajectory to fit or eval.

    Lives here rather than beside the fitting script because that script has to
    disable the gate at import time, and importing it merely to reach this
    function would disable the gate in whatever process asked.

    Hashed rather than striped by index because task ids carry structure —
    customer_support numbers its ``hard_`` variants in runs — and an
    every-other-file split would load one half with them.
    """
    return "fit" if zlib.crc32(stem.encode("utf-8")) % 2 == 0 else "eval"


def load_agent(class_name: str, domain: str, obligations: Path | None) -> Any:
    """Build a ledger with no client, no tools and no model."""
    for module_name in (
        "agents.gated_ledger_agent",
        "agents.latent_state_agent",
        "agents.obligation_ledger_agent",
    ):
        module = importlib.import_module(module_name)
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            break
    else:
        raise SystemExit(f"unknown agent class {class_name}")
    agent = cls.__new__(cls)
    if obligations is not None:
        agent.obligations_dir = obligations
    agent._load_obligations(domain)
    return agent


def future_calls(messages: list[Any]) -> list[dict[str, Any]]:
    return [
        call
        for message in messages
        if isinstance(message, dict)
        for call in (message.get("tool_calls") or [])
        if isinstance(call, dict)
    ]


def walk(conversation: list[Any]) -> Iterator[tuple[list[Any], list[dict[str, Any]], list[Any]]]:
    """Yield each state the live hook would see, paired with what follows it.

    The hook runs once per tool round, so a stored turn — one assistant message
    with all its calls nested — has to be expanded into its successive prefixes.
    The prose is blanked for the same reason it is in the other replay: at tool
    round time it has not been written.

    Three values: the state, the reference's future tool calls, and its future
    messages. The messages are needed because a disclosure line is answered by
    prose rather than by a call, and the current assistant message counts as
    future for every prefix of itself — its content is exactly what has not been
    written yet.
    """
    for index, message in enumerate(conversation):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        head = conversation[:index]
        later = conversation[index:]
        yield head, future_calls(later), later
        calls = [call for call in (message.get("tool_calls") or []) if isinstance(call, dict)]
        rest = future_calls(conversation[index + 1 :])
        for count in range(1, len(calls) + 1):
            yield (
                [*head, {**message, "tool_calls": calls[:count], "content": ""}],
                calls[count:] + rest,
                later,
            )


def pointed_at(line: str) -> list[tuple[str, str]]:
    """The concrete actions a ledger line asks for."""
    actions: list[tuple[str, str]] = [("policy", topic) for topic in POLICY_CALL.findall(line)]
    supply = SUPPLY.search(line)
    if supply:
        actions += [("tool", tool.strip()) for tool in supply.group(1).split("/") if tool.strip()]
    # An imperative line demands an utterance, and it demands it whether or not it
    # also carries a lookup. Scoring only the lookup would let the prefixed half
    # of the arm answer for the bare half, which is the larger half.
    head = IMPERATIVE.match(line)
    if head:
        actions.append(("say", head.group(1)))
    return actions


def satisfied(
    action: tuple[str, str],
    tail: list[dict[str, Any]],
    discharged: Callable[[str], bool] | None = None,
) -> bool:
    kind, target = action
    if kind == "say":
        return bool(discharged and discharged(target))
    for call in tail:
        name = str(call.get("name", ""))
        if kind == "tool" and name == target:
            return True
        if kind == "policy" and name == "get_policies":
            argument = (call.get("arguments") or {}).get("topic")
            if str(argument) == target:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold-agreement replay on train trajectories")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--agent-class", type=str, default="LatentStateAgent")
    parser.add_argument("--trajectories-dir", type=str, default="datasets/train_task_trajectories")
    parser.add_argument("--obligations", type=str, default=None)
    parser.add_argument(
        "--half",
        choices=["fit", "eval"],
        default=None,
        help="Restrict to one hash-half, matching scripts/fit_ledger_priors.py. "
        "Score a gate fitted on --holdout fit against --half eval and the number "
        "means something; score it against the half it was fitted on and it does not.",
    )
    parser.add_argument("--priors-suffix", type=str, default=None,
                        help="e.g. 'fit' to load <domain>.priors.fit.json instead of the "
                             "production prior.")
    parser.add_argument("--show", type=int, default=4)
    args = parser.parse_args()

    if args.priors_suffix:
        os.environ["STATE_BENCH_LEDGER_PRIORS_SUFFIX"] = args.priors_suffix

    agent = load_agent(
        args.agent_class, args.domain, Path(args.obligations) if args.obligations else None
    )
    files = sorted((Path(args.trajectories_dir) / args.domain).glob("*.json"))
    if not files:
        parser.error(f"no trajectories under {Path(args.trajectories_dir) / args.domain}")
    if args.half:
        files = [path for path in files if half_of(path.stem) == args.half]

    agreed: Counter[str] = Counter()
    missed: Counter[str] = Counter()
    lines_per_turn: list[int] = []
    examples: list[tuple[str, str, bool]] = []
    by_topic = {probe["topic"]: probe for probe in agent._pool}

    for path in files:
        conversation = json.loads(path.read_text(encoding="utf-8")).get("conversation") or []
        for state, tail, later in walk(conversation):
            lines = agent._ledger(state)
            lines_per_turn.append(len(lines))
            if not lines:
                continue
            # Deferred until a line exists: extracting the reference's remaining
            # prose is the expensive part of the loop and most turns are silent.
            prose = agent._role_text(later, "assistant")
            discharged = lambda topic: topic in by_topic and agent._discharged(
                prose, by_topic[topic]
            )
            for line in lines:
                for action in pointed_at(line):
                    hit = satisfied(action, tail, discharged)
                    (agreed if hit else missed)[f"{action[0]}:{action[1]}"] += 1
                    if len(examples) < args.show and not hit:
                        examples.append((path.stem, line, hit))

    total = sum(agreed.values()) + sum(missed.values())
    turns = len(lines_per_turn)
    firing = sum(1 for count in lines_per_turn if count)
    print(f"agent                : {args.agent_class}")
    print(f"domain               : {args.domain}  ({len(files)} reference trajectories)")
    print(f"turns replayed       : {turns}")
    print(f"turns with a ledger  : {firing} ({firing / max(turns, 1):.0%})")
    print(f"mean lines per turn  : {sum(lines_per_turn) / max(turns, 1):.2f}")
    print()
    print(f"actions pointed at   : {total}")
    print(f"  reference performs : {sum(agreed.values())} ({sum(agreed.values()) / max(total, 1):.0%})")
    print(f"  reference ignores  : {sum(missed.values())}")
    print(f"pointed per turn     : {total / max(turns, 1):.2f}   "
          f"wasted per turn: {sum(missed.values()) / max(turns, 1):.2f}")
    print()
    print(f"{'action':40s} {'agreed':>7s} {'ignored':>8s} {'rate':>6s}")
    print("-" * 66)
    for key in sorted(set(agreed) | set(missed), key=lambda k: -(agreed[k] + missed[k]))[:16]:
        pool = agreed[key] + missed[key]
        print(f"{key:40s} {agreed[key]:7d} {missed[key]:8d} {agreed[key] / pool:6.0%}")
    if examples:
        print("\nexamples the reference ignores:")
        for name, line, _ in examples:
            print(f"  {name[:34]:34s} {line[:150]}")


if __name__ == "__main__":
    main()
