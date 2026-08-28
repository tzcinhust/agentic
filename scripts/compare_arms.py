"""Paired comparison of two arms on the same tasks.

The archive establishes that this benchmark's run-to-run noise is large enough to
swamp the effect we are looking for: the same method scored 46.9% and 55.1% on
consecutive shopping runs, and an unpaired design would need roughly +18 points
to call a difference at 50 tasks x 5 runs. So everything here is paired — the two
arms ran the same 50 task definitions, and only the discordant tasks carry
information.

Four views, in increasing order of power and decreasing order of directness:

*Task rate.* What the leaderboard reports. Quoted with its standard error so the
reader can see how little a two-run point estimate constrains.

*Paired sign test on per-task pass counts.* Each task contributes its change in
number of passing runs; ties drop out. Assumption-light and reads directly as
"the intervention helped on more tasks than it hurt".

*Per-run McNemar.* Run 1 against run 1, run 2 against run 2. Two independent
looks at the same question, which is a cheap check that a single lucky run is not
carrying the result.

*Rubric-item McNemar.* Every rubric item is a paired binary observation, so this
has several times the power of the task-level test. It answers a slightly
different question — did the agent satisfy more requirements — and a method can
win here while losing at task level, because task completion is conjunctive over
all items.

The last section is the safety check, and it is the reason this script exists
rather than a one-line rate comparison. A ledger that reminds the agent to
disclose can push it into disclosing things nobody asked about, which shows up as
``no_unsolicited_*`` and ``no_random_*`` items flipping pass to fail. Net gain is
the only number that matters, so gains and regressions are reported separately
and by item id.

Usage:
    uv run python scripts/compare_arms.py \
        --baseline artifacts/statebench_cross_domain_pwm/runs/shopping_assistant \
        --treatment outputs/a2_shopping --runs 2
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def binomial_p(successes: int, trials: int) -> float:
    """Two-sided exact binomial p at p=0.5, which is McNemar's exact test."""
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    cumulative = sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return min(1.0, 2 * cumulative)


def load_arm(root: Path, runs: int) -> dict[int, dict[str, dict[str, Any]]]:
    """Scored trajectories as ``{run_index: {task_id: trajectory}}``.

    Unscored trajectories are silently useless rather than loudly missing — the
    judge writes its verdict back into the same file, so an unscored run looks
    like a populated directory whose every ``task_completion_pass`` is null.
    Counting them is reported so a scoring gap cannot masquerade as a zero.
    """
    return load_runs([root / f"run{index}" for index in range(1, runs + 1)])


def load_runs(dirs: list[Path]) -> dict[int, dict[str, dict[str, Any]]]:
    """Same, but from an explicit list of run directories.

    Replicates of one arm do not always live side by side: the archived
    customer_support baseline has run1 under artifacts/ and its only scored
    second replicate under outputs/variance_cust/, and shopping has a third
    under outputs/variance_shop/. Forcing them into a common parent would mean
    copying trajectories, which is how provenance gets lost.
    """
    arm: dict[int, dict[str, dict[str, Any]]] = {}
    for index, run_dir in enumerate(dirs, start=1):
        tasks: dict[str, dict[str, Any]] = {}
        unscored = 0
        for path in sorted(run_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("task_completion_pass") is None:
                unscored += 1
                continue
            tasks[path.stem] = data
        if unscored:
            print(f"  warning: {run_dir} has {unscored} unscored trajectories — run score.py on it")
        if not run_dir.is_dir():
            print(f"  warning: {run_dir} does not exist")
        arm[index] = tasks
    return arm


def items(trajectory: dict[str, Any]) -> dict[str, bool]:
    return {
        str(item.get("id")): bool(item.get("passed"))
        for item in trajectory.get("task_requirements_details") or []
        if item.get("id")
    }


def rate(values: list[int]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    return mean, math.sqrt(max(mean * (1 - mean), 0.0) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired two-arm comparison")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Arm root containing run1..runN.")
    parser.add_argument("--treatment", type=str, default=None)
    parser.add_argument("--baseline-runs", type=str, nargs="+", default=None,
                        help="Explicit run directories for the baseline arm, in order. "
                             "Use when an arm's replicates are not siblings.")
    parser.add_argument("--treatment-runs", type=str, nargs="+", default=None)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    if not (args.baseline or args.baseline_runs) or not (args.treatment or args.treatment_runs):
        parser.error("each arm needs either --<arm> or --<arm>-runs")

    if args.baseline_runs:
        base = load_runs([Path(p) for p in args.baseline_runs])
        args.runs = len(args.baseline_runs)
    else:
        base = load_arm(Path(args.baseline), args.runs)
    if args.treatment_runs:
        treat = load_runs([Path(p) for p in args.treatment_runs])
    else:
        treat = load_arm(Path(args.treatment), args.runs)
    if len(treat) != len(base):
        parser.error(
            f"arms have {len(base)} and {len(treat)} replicates; pairing needs the same count"
        )

    print(f"{'':22s} {'baseline':>18s} {'treatment':>18s}")
    print("-" * 60)
    for index in range(1, args.runs + 1):
        b_rate, b_se = rate([int(t["task_completion_pass"]) for t in base[index].values()])
        t_rate, t_se = rate([int(t["task_completion_pass"]) for t in treat[index].values()])
        print(
            f"run{index} pass rate         "
            f"{b_rate:9.1%} +-{b_se:5.1%} ({len(base[index]):2d})"
            f"{t_rate:9.1%} +-{t_se:5.1%} ({len(treat[index]):2d})"
        )

    # Tasks scored in every run of both arms — the only ones that can be paired.
    shared = set.intersection(
        *[set(base[i]) for i in range(1, args.runs + 1)],
        *[set(treat[i]) for i in range(1, args.runs + 1)],
    )
    print(f"\npaired tasks           {len(shared)}")
    if not shared:
        print(
            "\nNothing to pair. Every task must be scored in every run of both arms; "
            "the warnings above say which run to score first."
        )
        return

    b_counts = {
        task: sum(int(base[i][task]["task_completion_pass"]) for i in range(1, args.runs + 1))
        for task in shared
    }
    t_counts = {
        task: sum(int(treat[i][task]["task_completion_pass"]) for i in range(1, args.runs + 1))
        for task in shared
    }
    b_pooled = sum(b_counts.values()) / (len(shared) * args.runs)
    t_pooled = sum(t_counts.values()) / (len(shared) * args.runs)
    print(f"pooled pass rate       {b_pooled:9.1%}          {t_pooled:9.1%}   "
          f"(delta {t_pooled - b_pooled:+.1%})")

    up = [task for task in shared if t_counts[task] > b_counts[task]]
    down = [task for task in shared if t_counts[task] < b_counts[task]]
    print(
        f"\nsign test on per-task pass counts: {len(up)} improved, {len(down)} worsened, "
        f"{len(shared) - len(up) - len(down)} tied  ->  p = {binomial_p(len(up), len(up) + len(down)):.4f}"
    )

    for index in range(1, args.runs + 1):
        gained = [
            task
            for task in shared
            if treat[index][task]["task_completion_pass"] and not base[index][task]["task_completion_pass"]
        ]
        lost = [
            task
            for task in shared
            if base[index][task]["task_completion_pass"] and not treat[index][task]["task_completion_pass"]
        ]
        print(
            f"  run{index} McNemar: +{len(gained)} / -{len(lost)}  "
            f"p = {binomial_p(len(gained), len(gained) + len(lost)):.4f}"
        )

    # Rubric-item level, pooled over runs.
    gained_items: Counter[str] = Counter()
    lost_items: Counter[str] = Counter()
    for index in range(1, args.runs + 1):
        for task in shared:
            b_items = items(base[index][task])
            t_items = items(treat[index][task])
            for item_id in set(b_items) & set(t_items):
                if t_items[item_id] and not b_items[item_id]:
                    gained_items[item_id] += 1
                elif b_items[item_id] and not t_items[item_id]:
                    lost_items[item_id] += 1
    total_gained = sum(gained_items.values())
    total_lost = sum(lost_items.values())
    print(
        f"\nrubric items           +{total_gained} / -{total_lost}  "
        f"net {total_gained - total_lost:+d}  "
        f"p = {binomial_p(total_gained, total_gained + total_lost):.4f}"
    )

    print(f"\ntop item gains (first {args.show}):")
    for item_id, count in gained_items.most_common(args.show):
        print(f"  +{count:2d}  {item_id}")
    print(f"\ntop item regressions (first {args.show}):")
    for item_id, count in lost_items.most_common(args.show):
        print(f"  -{count:2d}  {item_id}")

    # The specific hazard: a disclosure prompt provoking unsolicited action.
    guard = lambda name: name.startswith(("no_", "not_")) or "unsolicited" in name or "random" in name
    guard_gained = sum(count for name, count in gained_items.items() if guard(name))
    guard_lost = sum(count for name, count in lost_items.items() if guard(name))
    print(
        f"\nrestraint items (no_*/unsolicited/random): +{guard_gained} / -{guard_lost}  "
        f"net {guard_gained - guard_lost:+d}"
    )

    for label, arm in (("baseline", base), ("treatment", treat)):
        ux = [t["ux_score"] for i in range(1, args.runs + 1) for t in arm[i].values() if t.get("ux_score")]
        # Trajectories record ``token_usage``, not a dollar figure — the price
        # depends on the provider, so tokens are the portable quantity and the
        # only honest one to compare arms on.
        cost = [
            (t.get("token_usage") or {}).get("total_tokens") or 0
            for i in range(1, args.runs + 1)
            for t in arm[i].values()
        ]
        state_ok_task_fail = sum(
            1
            for i in range(1, args.runs + 1)
            for t in arm[i].values()
            if t.get("state_requirements_met") and not t.get("task_requirements_met")
        )
        reverse = sum(
            1
            for i in range(1, args.runs + 1)
            for t in arm[i].values()
            if t.get("task_requirements_met") and not t.get("state_requirements_met")
        )
        # A task passes only if both halves pass, so the state half is a
        # first-class metric rather than a diagnostic: a method can lift task
        # requirements and still lose by breaking state.
        every = [t for i in range(1, args.runs + 1) for t in arm[i].values()]
        state_rate = sum(1 for t in every if t.get("state_requirements_met")) / max(len(every), 1)
        print(
            f"\n{label:9s} mean UX {sum(ux) / max(len(ux), 1):.2f}   "
            f"mean tokens {sum(cost) / max(len(cost), 1):,.0f}   "
            f"state req met {state_rate:.1%}   "
            f"state-ok/task-fail {state_ok_task_fail}   task-ok/state-fail {reverse}"
        )


if __name__ == "__main__":
    main()
