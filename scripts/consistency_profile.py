"""Where a domain's pass rate actually sits, and how much of the gap is variance.

The leaderboard reports pass@1 and pass^5 side by side, and on shopping the two
leaders sit at 52.0/38.0 and 57.0/42.0 — a 14-to-15-point spread between "passes
once" and "passes five times out of five". That spread is the part of the score
that item-level prompting cannot reach: a task that flips between runs is not
failing because a rubric item is unknown, it is failing because the agent's
behaviour is not reproducible on it.

So this script splits the test set three ways instead of reporting one mean:

*solid*   passes every replicate — the part pass^k keeps.
*flaky*   passes some and fails others — worth twice its weight, because
          converting one flaky task lifts pass@1 by half a task and pass^k by a
          whole one.
*hard*    fails every replicate — the part that needs new capability, not
          consistency.

Runs are matched on task id, and only ids present in every replicate are
counted, so a partial run cannot inflate a rate by dropping its own failures.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_run(run_dir: Path) -> dict[str, dict[str, Any]]:
    scored: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("task_completion_pass") is None:
            continue
        scored[str(record.get("task_id") or path.stem)] = record
    return scored


def summarize(label: str, runs: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    shared = set.intersection(*(set(run) for run in runs)) if runs else set()
    passes = {
        task: [int(run[task]["task_completion_pass"] or 0) for run in runs] for task in shared
    }
    solid = [task for task, votes in passes.items() if all(votes)]
    hard = [task for task, votes in passes.items() if not any(votes)]
    flaky = [task for task, votes in passes.items() if any(votes) and not all(votes)]
    total_votes = sum(sum(votes) for votes in passes.values())
    n_votes = sum(len(votes) for votes in passes.values())
    ux = [
        float(run[task]["ux_score"])
        for run in runs
        for task in shared
        if run[task].get("ux_score") is not None
    ]
    # Both requirement families have to hold, so a failure is only informative
    # once you know which one broke.
    state_fail = sum(
        1 for run in runs for task in shared if not int(run[task]["state_requirements_met"] or 0)
    )
    task_fail = sum(
        1 for run in runs for task in shared if not int(run[task]["task_requirements_met"] or 0)
    )
    both_ok_but_fail = sum(
        1
        for run in runs
        for task in shared
        if int(run[task]["state_requirements_met"] or 0)
        and int(run[task]["task_requirements_met"] or 0)
        and not int(run[task]["task_completion_pass"] or 0)
    )
    return {
        "domain": label,
        "replicates": len(runs),
        "tasks": len(shared),
        "pass@1": 100.0 * total_votes / n_votes if n_votes else 0.0,
        f"pass^{len(runs)}": 100.0 * len(solid) / len(shared) if shared else 0.0,
        "solid": len(solid),
        "flaky": len(flaky),
        "hard": len(hard),
        "ux": sum(ux) / len(ux) if ux else 0.0,
        "state_fail_rate": 100.0 * state_fail / n_votes if n_votes else 0.0,
        "task_fail_rate": 100.0 * task_fail / n_votes if n_votes else 0.0,
        "inconsistent_pass": both_ok_but_fail,
        "flaky_ids": sorted(flaky),
        "hard_ids": sorted(hard),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="label=dir[,dir...] per domain")
    parser.add_argument("--show-ids", action="store_true")
    args = parser.parse_args()

    grouped: dict[str, list[Path]] = defaultdict(list)
    for spec in args.runs:
        label, _, dirs = spec.partition("=")
        for entry in dirs.split(","):
            grouped[label].append(Path(entry))

    rows = []
    for label, dirs in grouped.items():
        loaded = [load_run(d) for d in dirs]
        sizes = ", ".join(f"{d.parent.name}/{d.name}:{len(r)}" for d, r in zip(dirs, loaded))
        print(f"{label}: {sizes}")
        rows.append(summarize(label, loaded))
    print()

    header = (
        f"{'domain':22s} {'k':>2s} {'n':>3s} {'pass@1':>7s} {'pass^k':>7s} "
        f"{'solid':>6s} {'flaky':>6s} {'hard':>5s} {'UX':>5s} {'state✗':>7s} {'task✗':>6s}"
    )
    print(header)
    for row in rows:
        pk = row[f"pass^{row['replicates']}"]
        print(
            f"{row['domain']:22s} {row['replicates']:2d} {row['tasks']:3d} "
            f"{row['pass@1']:7.1f} {pk:7.1f} {row['solid']:6d} {row['flaky']:6d} "
            f"{row['hard']:5d} {row['ux']:5.2f} {row['state_fail_rate']:7.1f} "
            f"{row['task_fail_rate']:6.1f}"
        )
    print()
    for row in rows:
        k = row["replicates"]
        # Ceiling if every flaky task became reliable: pass@1 absorbs the votes it
        # is currently losing, pass^k absorbs the whole task.
        ceiling = row["pass@1"] + (100.0 * row["flaky"] / row["tasks"]) * (k - 1) / k
        print(
            f"{row['domain']:22s} flaky={row['flaky']}/{row['tasks']} "
            f"({100.0 * row['flaky'] / row['tasks']:.0f}%)  "
            f"pass@1 if all flaky fixed -> {ceiling:.1f}  "
            f"pass^{k} -> {row[f'pass^{k}'] + 100.0 * row['flaky'] / row['tasks']:.1f}"
        )
        if args.show_ids:
            print(f"    flaky: {', '.join(row['flaky_ids'])}")
            print(f"    hard:  {', '.join(row['hard_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
