"""Measure how much of the observed rubric failure the obligation table covers.

The obligation probes are only worth evaluating if they address the items the
judge actually failed. This script reads scored trajectories, keeps the rubric
items whose ``passed`` is false, sorts them onto the four obligation channels
(say / tool / ask / refuse), and asks whether some probe in
``artifacts/obligations/<domain>.json`` names the same topic.

Coverage is an upper bound on the gain, not a prediction of it: a covered item
still has to be discharged at runtime. An *un*covered item is a hard miss — no
amount of runtime discipline will reach it.

Usage:
    uv run python scripts/score_obligation_coverage.py \
        --domain shopping_assistant \
        --results-dir artifacts/statebench_cross_domain_pwm/runs/shopping_assistant
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# How a rubric item's own wording reveals which channel it binds.
CHANNEL_CUES: dict[str, tuple[str, ...]] = {
    "say": (
        r"\bmention",
        r"\bdisclos",
        r"\binform",
        r"\bexplain",
        r"\bstate[sd]?\b",
        r"\bquantif",
        r"\btell",
        r"\bsurface",
        r"\bcommunicat",
        r"\bproactiv",
        r"\bsummar",
        r"\backnowledg",
    ),
    "tool": (
        r"\bcall(?:s|ed|ing)?\b",
        r"\btool\b",
        r"\b(?:set|process|apply|add|update|cancel|remove|redeem|book|create)_\w+",
        r"\bsubmit",
        r"\bamount\b",
        r"\bwrite[s]?\b",
    ),
    "ask": (
        r"\bask",
        r"\bconfirm",
        r"\bconsent",
        r"\bpermission",
        r"\bbefore (?:calling|writing|applying)",
        r"\bexplicit",
        r"\bclarif",
    ),
    "refuse": (
        r"\brefus",
        r"\bdeclin",
        r"\bdeny\b",
        r"\bdenial\b",
        r"\bmust not\b",
        r"\bdoes not\b",
        r"\bdid not\b",
        r"\bwithout\b",
        r"\bnot eligible",
        r"\bineligible",
        r"\bavoid",
        r"\bnot (?:invent|reveal|classify|apply|treat|process|grant|use|leave|move|concede)\b",
    ),
}
TOKEN = re.compile(r"[a-z]{4,}")


def classify(text: str) -> list[str]:
    return sorted(
        channel
        for channel, cues in CHANNEL_CUES.items()
        if any(re.search(cue, text, re.I) for cue in cues)
    )


def load_probes(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["probes"]


def probe_terms(probe: dict[str, Any]) -> set[str]:
    """Words that identify this probe's topic inside a rubric requirement."""
    terms = {part for part in probe["topic"].split("_") if len(part) > 3}
    terms |= set(probe["discharge_check"]["distinctive_terms"])
    return terms


def match_probe(text: str, probes: list[dict[str, Any]]) -> dict[str, Any] | None:
    tokens = set(TOKEN.findall(text.lower()))
    best: tuple[int, dict[str, Any]] | None = None
    for probe in probes:
        overlap = len(probe_terms(probe) & tokens)
        if overlap and (best is None or overlap > best[0]):
            best = (overlap, probe)
    return best[1] if best else None


def match_resolver(text: str, probes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match items that fail on a *precondition* of a duty rather than the duty.

    ``checked_customer_profile`` ("Agent called get_customer_account ... to
    determine tier before disclosing") is not a topic mention — it is the lookup
    that makes a tier-dependent duty evaluable. The obligation table already
    carries it as a latent field with a named resolver, so it counts as
    addressed, but by a different mechanism and worth reporting separately.
    """
    lowered = text.lower()
    for probe in probes:
        for field, tools in probe["resolvers"].items():
            if field in lowered or any(tool in lowered for tool in tools):
                return probe
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Rubric-failure coverage of the obligation table")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--obligations", type=str, default=None)
    parser.add_argument("--run", type=str, default="run1", help="Run subdirectory to read")
    parser.add_argument("--show-uncovered", type=int, default=15)
    args = parser.parse_args()

    probes_path = (
        Path(args.obligations) if args.obligations else Path("artifacts/obligations") / f"{args.domain}.json"
    )
    probes = load_probes(probes_path)
    run_dir = Path(args.results_dir) / args.run
    files = sorted(run_dir.glob("*.json"))
    if not files:
        parser.error(f"No trajectories in {run_dir}")

    by_channel: Counter[str] = Counter()
    covered_by_channel: Counter[str] = Counter()
    resolver_by_channel: Counter[str] = Counter()
    per_topic: Counter[str] = Counter()
    uncovered: list[tuple[str, str, str]] = []
    unclassified: list[tuple[str, str]] = []
    tasks_total = 0
    tasks_state_ok_task_fail = 0
    failed_items = 0
    resolver_only = 0

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("task_completion_pass") is None:
            continue
        tasks_total += 1
        if data.get("state_requirements_met") and not data.get("task_requirements_met"):
            tasks_state_ok_task_fail += 1
        for item in data.get("task_requirements_details") or []:
            if item.get("passed"):
                continue
            failed_items += 1
            text = f"{item.get('id', '')} {item.get('requirement', '')}"
            channels = classify(text)
            if not channels:
                unclassified.append((path.stem, text.strip()[:110]))
            probe = match_probe(text, probes)
            resolver = None if probe else match_resolver(text, probes)
            if resolver:
                resolver_only += 1
            for channel in channels or ["<none>"]:
                by_channel[channel] += 1
                if probe:
                    covered_by_channel[channel] += 1
                elif resolver:
                    resolver_by_channel[channel] += 1
            if probe or resolver:
                per_topic[(probe or resolver)["topic"]] += 1
            else:
                uncovered.append((path.stem, ",".join(channels) or "-", text.strip()[:110]))

    print(f"domain              : {args.domain}  ({args.run})")
    print(f"scored tasks        : {tasks_total}")
    print(f"state ok, task fail : {tasks_state_ok_task_fail}")
    print(f"failed rubric items : {failed_items}")
    print(f"probes available    : {len(probes)}")
    print()
    print(f"{'channel':10s} {'failed':>7s} {'topic':>7s} {'resolv':>7s} {'total':>7s}")
    print("-" * 42)
    for channel in ("say", "tool", "ask", "refuse", "<none>"):
        total = by_channel.get(channel, 0)
        if not total:
            continue
        covered = covered_by_channel.get(channel, 0)
        resolved = resolver_by_channel.get(channel, 0)
        print(f"{channel:10s} {total:7d} {covered:7d} {resolved:7d} {(covered + resolved) / total:6.0%}")
    matched = sum(per_topic.values())
    print()
    print(
        f"items addressed: {matched}/{failed_items} ({matched / max(failed_items, 1):.0%}) "
        f"— {matched - resolver_only} by topic, {resolver_only} by latent-field resolver"
    )
    print()
    print("failures per probe topic:")
    for topic, count in per_topic.most_common():
        print(f"  {count:3d}  {topic}")
    if uncovered:
        print()
        print(f"uncovered failures ({len(uncovered)}), first {args.show_uncovered}:")
        for task, channels, text in uncovered[: args.show_uncovered]:
            print(f"  [{channels or '-':16s}] {task[:28]:28s} {text}")
    if unclassified:
        print()
        print(f"unclassified by channel ({len(unclassified)}), first 5:")
        for task, text in unclassified[:5]:
            print(f"  {task[:28]:28s} {text}")


if __name__ == "__main__":
    main()
