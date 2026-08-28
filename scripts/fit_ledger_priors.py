"""Fit the ledger's gate on reference behaviour, opportunity-normalised.

The mined obligation table ranks its probes by ``support`` — how many training
conversations the rule was observed in. That is a raw frequency, and it is the
weakness this whole line of work is aimed at: a policy the customer mentions
constantly earns high support whether or not fetching it ever changes the right
answer. Support cannot tell "important" from "ubiquitous".

The correction is to normalise by opportunity. For each gap the ledger can
report, the denominator is the number of reference conversations in which that
gap *arose at all* — the customer named the topic, or the field was missing from
the view — and the numerator is the number of those in which the reference then
went and closed it. A rate near 1 means the gap is one competent behaviour
reliably closes, and reporting it anticipates real work. A rate near 0 means the
reference sees the same gap and correctly ignores it, so reporting it is a
standing instruction to spend a tool call on nothing.

The two gaps are fitted separately because they are different claims:

*topic* — the gap arose on some turn where ``_gaps`` listed the topic as unread;
it was closed if the reference ever calls ``get_policies(topic)``.

*field* — the gap arose where ``_gaps`` reported the field pending; it was closed
if any reference tool result ever carried that field.

*disclose* — the topic was live at some point; it was closed if the reference's
own prose ends up carrying that topic beside a figure, by the ledger's own
``_discharged`` test. This is the prior an imperative ledger needs, and it is a
different question from *topic*: fetching a policy is a precondition for quoting
it, not the same act. A topic the reference reads and then declines to volunteer
is one an imperative line would push the agent into over-disclosing.

Fitting has to run against the *ungated* ledger, or the estimate would be
conditioned on its own output, so this script forces ``STATE_BENCH_LEDGER_PRIORS
=0`` before importing the agent.

Two modes, and the distinction matters for honesty:

``--holdout`` splits the 100 train trajectories by a stable hash, fits on one
half and writes the prior tagged as held-out, so that
``replay_gold_agreement.py --half eval`` scores a gate that has never seen the
conversations it is scored on. This is what a threshold may be chosen against.

Default fits on all 100 and writes the production prior. Run it only after the
threshold is settled; its train agreement is circular by construction and must
not be quoted as evidence.

Usage:
    uv run python scripts/fit_ledger_priors.py --domain travel --holdout fit
    uv run python scripts/fit_ledger_priors.py --domain travel
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

# Must precede the agent import: a gate fitted through itself is not an estimate
# of anything, so the ungated ledger is what supplies the denominators.
os.environ["STATE_BENCH_LEDGER_PRIORS"] = "0"

from scripts.replay_gold_agreement import half_of, load_agent, walk  # noqa: E402


def closed_topics(conversation: list[Any]) -> set[str]:
    """Policy topics the reference fetches anywhere in the conversation."""
    return {
        str((call.get("arguments") or {}).get("topic"))
        for message in conversation
        if isinstance(message, dict)
        for call in (message.get("tool_calls") or [])
        if isinstance(call, dict) and call.get("name") == "get_policies"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the ledger gate on reference behaviour")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--agent-class", type=str, default="LatentStateAgent")
    parser.add_argument("--trajectories-dir", type=str, default="datasets/train_task_trajectories")
    parser.add_argument("--obligations", type=str, default="artifacts/obligations")
    parser.add_argument(
        "--holdout",
        choices=["fit", "eval"],
        default=None,
        help="Fit on one hash-half only, and tag the output as held-out.",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    obligations = Path(args.obligations)
    agent = load_agent(args.agent_class, args.domain, obligations)
    files = sorted((Path(args.trajectories_dir) / args.domain).glob("*.json"))
    if not files:
        parser.error(f"no trajectories under {Path(args.trajectories_dir) / args.domain}")
    if args.holdout:
        files = [path for path in files if half_of(path.stem) == args.holdout]

    # Per trajectory, not per turn: a gap open across eighteen turns of one
    # conversation is one opportunity, not eighteen. Turn-weighting would let the
    # longest conversations set the rate.
    topic_seen: dict[str, int] = defaultdict(int)
    topic_closed: dict[str, int] = defaultdict(int)
    field_seen: dict[str, int] = defaultdict(int)
    field_closed: dict[str, int] = defaultdict(int)
    disclose_seen: dict[str, int] = defaultdict(int)
    disclose_closed: dict[str, int] = defaultdict(int)

    for path in files:
        conversation = json.loads(path.read_text(encoding="utf-8")).get("conversation") or []
        arose_topics: set[str] = set()
        arose_fields: set[str] = set()
        live: set[str] = set()
        for state, _, _ in walk(conversation):
            unread, pending = agent._gaps(state)
            arose_topics.update(unread)
            arose_fields.update(field for entry in pending for field in entry["fields"])
            # Liveness for the disclosure prior is broader than "unread" — a topic
            # already fetched still gets an imperative line — but narrower than
            # "not None": a probe whose fields are still unresolved renders as the
            # join line, which asks for a lookup rather than an utterance. Only a
            # "holds" verdict becomes a disclosure demand.
            observed = agent._observe(state)
            asked = agent._role_text(state, "user").lower()
            live.update(
                probe["topic"]
                for probe in agent._pool
                if agent._liveness(probe, observed, asked) == "holds"
            )
        fetched = closed_topics(conversation)
        resolved = agent._observe(conversation)["resolved"]
        said = agent._role_text(conversation, "assistant")
        for topic in arose_topics:
            topic_seen[topic] += 1
            topic_closed[topic] += topic in fetched
        for field in arose_fields:
            field_seen[field] += 1
            field_closed[field] += field in resolved
        by_topic = {probe["topic"]: probe for probe in agent._pool}
        for topic in live:
            disclose_seen[topic] += 1
            disclose_closed[topic] += agent._discharged(said, by_topic[topic])

    priors = {
        "domain": args.domain,
        "agent_class": args.agent_class,
        "trajectories": len(files),
        "holdout": args.holdout,
        "topic_rate": {t: topic_closed[t] / topic_seen[t] for t in sorted(topic_seen)},
        "topic_opportunities": dict(sorted(topic_seen.items())),
        "field_rate": {f: field_closed[f] / field_seen[f] for f in sorted(field_seen)},
        "field_opportunities": dict(sorted(field_seen.items())),
        "disclose_rate": {t: disclose_closed[t] / disclose_seen[t] for t in sorted(disclose_seen)},
        "disclose_opportunities": dict(sorted(disclose_seen.items())),
    }

    suffix = f".priors.{args.holdout}.json" if args.holdout else ".priors.json"
    out = Path(args.out) if args.out else obligations / f"{args.domain}{suffix}"
    out.write_text(json.dumps(priors, indent=2) + "\n", encoding="utf-8")

    print(f"domain        : {args.domain}")
    print(f"fitted on     : {len(files)} trajectories"
          f"{f' (hash-half {args.holdout})' if args.holdout else ' (all train)'}")
    print(f"written to    : {out}")
    for kind in ("topic", "field", "disclose"):
        rates = priors[f"{kind}_rate"]
        seen = priors[f"{kind}_opportunities"]
        if not rates:
            continue
        print(f"\n{kind:22s} {'arose in':>9s} {'closed':>7s}")
        print("-" * 42)
        for key in sorted(rates, key=lambda k: rates[k]):
            print(f"{key:22s} {seen[key]:6d}/{len(files):<3d} {rates[key]:6.0%}")


if __name__ == "__main__":
    main()
