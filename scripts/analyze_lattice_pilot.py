"""Analyze complete memory-subset interventions from scored trajectories."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

METRICS = ("completion", "state", "task", "ux")


def _metric_vector(record: dict[str, Any]) -> dict[str, float]:
    return {
        "completion": float(bool(record.get("task_completion_pass"))),
        "state": float(bool(record.get("state_requirements_met"))),
        "task": float(bool(record.get("task_requirements_met"))),
        "ux": float(record.get("ux_score") or 0.0) / 5.0,
    }


def _load_arm(root: Path, mask: int) -> dict[tuple[str, int], dict[str, Any]]:
    arm: dict[tuple[str, int], dict[str, Any]] = {}
    arm_root = root / f"mask{mask}"
    for run_dir in sorted(arm_root.glob("run*")):
        try:
            run_idx = int(run_dir.name.removeprefix("run"))
        except ValueError:
            continue
        for path in sorted(run_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            arm[(str(record["task_id"]), run_idx)] = record
    return arm


def _submasks(mask: int):
    current = mask
    while True:
        yield current
        if current == 0:
            return
        current = (current - 1) & mask


def _mobius(gain: dict[int, float], mask: int) -> float:
    order = mask.bit_count()
    return sum(
        ((-1.0) ** (order - submask.bit_count())) * gain[submask]
        for submask in _submasks(mask)
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def analyze(root: Path) -> dict[str, Any]:
    arms = {mask: _load_arm(root, mask) for mask in range(8)}
    common = set.intersection(*(set(arm) for arm in arms.values())) if arms else set()
    missing = {
        str(mask): sorted(f"{task_id}:run{run_idx}" for task_id, run_idx in set.union(*(set(a) for a in arms.values())) - set(arms[mask]))
        for mask in range(8)
    }
    if not common:
        raise ValueError(f"No complete task/run blocks under {root}")

    arm_values: dict[int, dict[str, list[float]]] = {
        mask: {metric: [] for metric in METRICS} for mask in range(8)
    }
    coefficients: dict[str, dict[int, list[float]]] = {
        metric: {mask: [] for mask in range(1, 8)} for metric in METRICS
    }
    negative_transfer = {
        metric: {mask: 0 for mask in range(1, 8)} for metric in METRICS
    }
    best_counts = {mask: 0 for mask in range(8)}
    rows: list[dict[str, Any]] = []
    task_observations: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: {
            mask: {metric: [] for metric in METRICS} for mask in range(8)
        }
    )

    for key in sorted(common):
        vectors = {mask: _metric_vector(arms[mask][key]) for mask in range(8)}
        completion_best = max(vectors[mask]["completion"] for mask in range(8))
        tied_best = [
            mask for mask in range(8) if vectors[mask]["completion"] == completion_best
        ]
        chosen_best = max(
            tied_best,
            key=lambda mask: (
                vectors[mask]["task"],
                vectors[mask]["state"],
                vectors[mask]["ux"],
                -mask.bit_count(),
            ),
        )
        best_counts[chosen_best] += 1

        row: dict[str, Any] = {"task_id": key[0], "run": key[1], "best_mask": chosen_best}
        for metric in METRICS:
            gain = {mask: vectors[mask][metric] - vectors[0][metric] for mask in range(8)}
            row[f"{metric}_gain"] = {str(mask): gain[mask] for mask in range(8)}
            row[f"{metric}_mobius"] = {
                str(mask): _mobius(gain, mask) for mask in range(1, 8)
            }
            for mask in range(8):
                arm_values[mask][metric].append(vectors[mask][metric])
                task_observations[key[0]][mask][metric].append(vectors[mask][metric])
            for mask in range(1, 8):
                coefficient = _mobius(gain, mask)
                coefficients[metric][mask].append(coefficient)
                if gain[mask] < 0:
                    negative_transfer[metric][mask] += 1
        rows.append(row)

    arm_by_task: dict[str, Any] = {}
    task_mean_coefficients = {
        metric: {mask: [] for mask in range(1, 8)} for metric in METRICS
    }
    for task_id, by_mask in sorted(task_observations.items()):
        arm_by_task[task_id] = {}
        for mask in range(8):
            arm_by_task[task_id][str(mask)] = {}
            for metric in METRICS:
                values = by_mask[mask][metric]
                arm_by_task[task_id][str(mask)][metric] = {
                    "mean": mean(values),
                    "population_sd": pstdev(values),
                    "range": max(values) - min(values),
                    "n": len(values),
                }
        for metric in METRICS:
            task_arm_means = {
                mask: mean(by_mask[mask][metric]) for mask in range(8)
            }
            gain = {
                mask: task_arm_means[mask] - task_arm_means[0]
                for mask in range(8)
            }
            for mask in range(1, 8):
                task_mean_coefficients[metric][mask].append(_mobius(gain, mask))

    # Independent-within-arm bootstrap. Run indices are blocking labels, not
    # shared random seeds, so uncertainty must resample each arm separately.
    rng = random.Random(0)
    bootstrap_draws = 4000
    bootstrap_coefficients = {
        metric: {mask: [] for mask in range(1, 8)} for metric in METRICS
    }
    bootstrap_contrasts = {
        metric: {mask: [] for mask in range(1, 8)} for metric in METRICS
    }
    task_ids = sorted(task_observations)
    for _ in range(bootstrap_draws):
        sampled_task_ids = [rng.choice(task_ids) for _ in task_ids]
        for metric in METRICS:
            sampled_arm_means: dict[int, float] = {}
            for mask in range(8):
                sampled_task_means = []
                for task_id in sampled_task_ids:
                    values = task_observations[task_id][mask][metric]
                    sampled = [rng.choice(values) for _ in values]
                    sampled_task_means.append(mean(sampled))
                sampled_arm_means[mask] = mean(sampled_task_means)
            gain = {
                mask: sampled_arm_means[mask] - sampled_arm_means[0]
                for mask in range(8)
            }
            for mask in range(1, 8):
                bootstrap_coefficients[metric][mask].append(_mobius(gain, mask))
                bootstrap_contrasts[metric][mask].append(gain[mask])

    count = len(common)
    summary = {
        "complete_blocks": count,
        "complete_tasks": len(task_observations),
        "runs_per_task": sorted(
            {len(by_mask[0]["completion"]) for by_mask in task_observations.values()}
        ),
        "estimator_note": (
            "Run indices are not deterministic shared seeds. Per-block coefficients "
            "are descriptive only; task-mean coefficients and independent-within-arm "
            "bootstrap intervals are the primary stochastic audit."
        ),
        "missing_blocks": missing,
        "arms": {
            str(mask): {
                **{metric: mean(arm_values[mask][metric]) for metric in METRICS},
                "mean_total_tokens": mean(
                    float(arms[mask][key].get("total_tokens") or 0) for key in common
                ),
            }
            for mask in range(8)
        },
        "negative_transfer_rate_vs_empty": {
            metric: {
                str(mask): negative_transfer[metric][mask] / count for mask in range(1, 8)
            }
            for metric in METRICS
        },
        "contrast_vs_empty": {
            metric: {
                str(mask): {
                    "mean": mean(arm_values[mask][metric])
                    - mean(arm_values[0][metric]),
                    "wins": sum(
                        mean(task_observations[task_id][mask][metric])
                        > mean(task_observations[task_id][0][metric])
                        for task_id in task_ids
                    ),
                    "losses": sum(
                        mean(task_observations[task_id][mask][metric])
                        < mean(task_observations[task_id][0][metric])
                        for task_id in task_ids
                    ),
                    "ties": sum(
                        math.isclose(
                            mean(task_observations[task_id][mask][metric]),
                            mean(task_observations[task_id][0][metric]),
                        )
                        for task_id in task_ids
                    ),
                    "hierarchical_bootstrap_90pct": [
                        _percentile(bootstrap_contrasts[metric][mask], 0.05),
                        _percentile(bootstrap_contrasts[metric][mask], 0.95),
                    ],
                }
                for mask in range(1, 8)
            }
            for metric in METRICS
        },
        "mobius": {
            metric: {
                str(mask): {
                    "order": mask.bit_count(),
                    "mean": mean(coefficients[metric][mask]),
                    "mean_abs": mean(abs(value) for value in coefficients[metric][mask]),
                    "nonzero_rate": sum(not math.isclose(value, 0.0) for value in coefficients[metric][mask]) / count,
                }
                for mask in range(1, 8)
            }
            for metric in METRICS
        },
        "mobius_on_task_means": {
            metric: {
                str(mask): {
                    "order": mask.bit_count(),
                    "mean": mean(task_mean_coefficients[metric][mask]),
                    "mean_abs_across_tasks": mean(
                        abs(value) for value in task_mean_coefficients[metric][mask]
                    ),
                    "task_values": task_mean_coefficients[metric][mask],
                    "bootstrap_90pct": [
                        _percentile(bootstrap_coefficients[metric][mask], 0.05),
                        _percentile(bootstrap_coefficients[metric][mask], 0.95),
                    ],
                }
                for mask in range(1, 8)
            }
            for metric in METRICS
        },
        "within_arm_stochasticity": {
            metric: {
                str(mask): {
                    "mean_task_population_sd": mean(
                        pstdev(task_observations[task_id][mask][metric])
                        for task_id in task_ids
                    ),
                    "max_task_range": max(
                        max(task_observations[task_id][mask][metric])
                        - min(task_observations[task_id][mask][metric])
                        for task_id in task_ids
                    ),
                }
                for mask in range(8)
            }
            for metric in METRICS
        },
        "interaction_mass": {
            metric: (
                sum(
                    mean(abs(value) for value in coefficients[metric][mask])
                    for mask in range(1, 8)
                    if mask.bit_count() >= 2
                )
                / max(
                    sum(
                        mean(abs(value) for value in coefficients[metric][mask])
                        for mask in range(1, 8)
                    ),
                    1e-12,
                )
            )
            for metric in METRICS
        },
        "lexicographic_best_mask_counts": {str(mask): best_counts[mask] for mask in range(8)},
        "arm_by_task": arm_by_task,
        "per_block": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.results_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
