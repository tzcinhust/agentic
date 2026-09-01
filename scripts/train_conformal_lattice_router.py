"""Fit and certify a train-only selector over the frozen PWM subset lattice.

The script consumes only scored *train* trajectories.  It writes aggregate
split/certification statistics and model weights; task IDs and user text are
never persisted in the deployment artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Iterable

from agents.conformal_router_features import (
    baseline_ranked_items,
    choose_from_predictions,
    feature_names,
    feature_vector,
    predict,
)
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent


METRICS = ("completion", "state", "task", "ux", "utility")
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)
THRESHOLD_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
UTILITY_WEIGHTS = {
    "completion": 0.60,
    "state": 0.20,
    "task": 0.15,
    "ux": 0.05,
}


class _DummyClient:
    pass


def metric_vector(record: dict[str, Any]) -> dict[str, float]:
    values = {
        "completion": float(bool(record["task_completion_pass"])),
        "state": float(bool(record["state_requirements_met"])),
        "task": float(bool(record["task_requirements_met"])),
        "ux": float(record["ux_score"]) / 5.0,
    }
    values["utility"] = sum(
        UTILITY_WEIGHTS[metric] * values[metric]
        for metric in UTILITY_WEIGHTS
    )
    return values


def first_user_query(record: dict[str, Any]) -> str:
    for item in record.get("conversation", []):
        if item.get("role") == "user":
            content = str(item.get("content", "")).strip()
            if content and "[TASK_DONE]" not in content:
                return content
    raise ValueError(f"trajectory {record.get('task_id', '<unknown>')} has no user query")


def _resolve_lattice_root(root: Path) -> Path:
    nested = root / "shopping_assistant"
    return nested if nested.is_dir() else root


def load_lattice(root: Path) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, str]]:
    """Load one scored run for all 100 train tasks and all eight masks."""

    root = _resolve_lattice_root(root)
    by_mask: dict[int, dict[str, dict[str, Any]]] = {}
    for mask in range(8):
        paths = sorted((root / f"mask{mask}" / "run1").glob("*.json"))
        records: dict[str, dict[str, Any]] = {}
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            task_id = str(record.get("task_id", ""))
            if not task_id or task_id in records:
                raise ValueError(f"invalid or duplicate task id in {path}")
            required = (
                "task_completion_pass",
                "state_requirements_met",
                "task_requirements_met",
                "ux_score",
            )
            if any(record.get(key) is None for key in required):
                raise ValueError(f"unscored trajectory: {path}")
            records[task_id] = record
        by_mask[mask] = records

    task_sets = [set(records) for records in by_mask.values()]
    if not task_sets or any(tasks != task_sets[0] for tasks in task_sets[1:]):
        raise ValueError("lattice masks do not contain identical task IDs")
    task_ids = sorted(task_sets[0])
    if len(task_ids) != 100:
        raise ValueError(f"expected 100 train tasks per mask, found {len(task_ids)}")

    outcomes: dict[str, dict[int, dict[str, float]]] = {}
    queries: dict[str, str] = {}
    for task_id in task_ids:
        query = first_user_query(by_mask[7][task_id])
        queries[task_id] = query
        outcomes[task_id] = {
            mask: metric_vector(by_mask[mask][task_id]) for mask in range(8)
        }
    return outcomes, queries


def deterministic_split(task_ids: Iterable[str], salt: str) -> dict[str, list[str]]:
    ordered = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{salt}:shopping_assistant:{task_id}".encode("utf-8")
        ).digest(),
    )
    if len(ordered) != 100:
        raise ValueError(f"expected exactly 100 task IDs, found {len(ordered)}")
    return {"fit": ordered[:60], "calibration": ordered[60:80], "lockbox": ordered[80:]}


def split_manifest_hash(split: dict[str, list[str]]) -> str:
    payload = "\n".join(
        f"{label}:{task_id}"
        for label in ("fit", "calibration", "lockbox")
        for task_id in split[label]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_features(
    memory_path: Path,
    queries: dict[str, str],
) -> dict[str, list[float]]:
    ProcessWorkflowMemoryAgent.memory_path = memory_path
    agent = ProcessWorkflowMemoryAgent(
        _DummyClient(),
        "system",
        [],
        {},
        runtime_context=SimpleNamespace(domain="shopping_assistant"),
        retrieve_learnings_top_k=3,
    )
    return {
        task_id: feature_vector(query, baseline_ranked_items(agent, query, 3))
        for task_id, query in queries.items()
    }


def _target_keys() -> list[tuple[int, str]]:
    return [(mask, metric) for mask in range(8) for metric in METRICS]


def _solve_ridge_multi(
    rows: list[list[float]],
    targets: list[list[float]],
    ridge: float,
) -> list[list[float]]:
    """Solve all target regressions together with pivoted elimination."""

    if not rows or len(rows) != len(targets):
        raise ValueError("rows and targets must be non-empty and aligned")
    dimension = len(rows[0])
    target_count = len(targets[0])
    gram = [[0.0] * dimension for _ in range(dimension)]
    rhs = [[0.0] * target_count for _ in range(dimension)]
    for row, target in zip(rows, targets):
        if len(row) != dimension or len(target) != target_count:
            raise ValueError("ragged regression matrix")
        for left, left_value in enumerate(row):
            for right in range(left, dimension):
                gram[left][right] += left_value * row[right]
            for output, target_value in enumerate(target):
                rhs[left][output] += left_value * target_value
    for left in range(dimension):
        for right in range(left):
            gram[left][right] = gram[right][left]
        # Keep the intercept effectively unpenalized while preserving a
        # nonsingular system in small folds.
        gram[left][left] += ridge if left else ridge * 1e-6

    augmented = [gram[index] + rhs[index] for index in range(dimension)]
    width = dimension + target_count
    for pivot in range(dimension):
        best = max(range(pivot, dimension), key=lambda row: abs(augmented[row][pivot]))
        if abs(augmented[best][pivot]) < 1e-12:
            raise ValueError("singular ridge system")
        augmented[pivot], augmented[best] = augmented[best], augmented[pivot]
        pivot_value = augmented[pivot][pivot]
        for column in range(pivot, width):
            augmented[pivot][column] /= pivot_value
        for row in range(dimension):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            if abs(factor) < 1e-15:
                continue
            for column in range(pivot, width):
                augmented[row][column] -= factor * augmented[pivot][column]
    # Return target-major weights.
    return [
        [augmented[feature][dimension + output] for feature in range(dimension)]
        for output in range(target_count)
    ]


def _targets_for_task(
    outcomes: dict[str, dict[int, dict[str, float]]], task_id: str
) -> list[float]:
    return [outcomes[task_id][mask][metric] for mask, metric in _target_keys()]


def _fit_models(
    task_ids: list[str],
    features: dict[str, list[float]],
    outcomes: dict[str, dict[int, dict[str, float]]],
    ridge: float,
) -> dict[int, dict[str, list[float]]]:
    weights = _solve_ridge_multi(
        [features[task_id] for task_id in task_ids],
        [_targets_for_task(outcomes, task_id) for task_id in task_ids],
        ridge,
    )
    models: dict[int, dict[str, list[float]]] = {mask: {} for mask in range(8)}
    for (mask, metric), values in zip(_target_keys(), weights):
        models[mask][metric] = values
    return models


def _predict_models(
    models: dict[int, dict[str, list[float]]], values: list[float]
) -> dict[int, dict[str, float]]:
    return {
        mask: {
            metric: max(0.0, min(1.0, predict(weights, values)))
            for metric, weights in metrics.items()
        }
        for mask, metrics in models.items()
    }


def _balanced_folds(task_ids: list[str], salt: str, count: int = 5) -> list[list[str]]:
    ordered = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{salt}:ridge-cv:{task_id}".encode("utf-8")
        ).digest(),
    )
    return [ordered[index::count] for index in range(count)]


def select_ridge(
    fit_ids: list[str],
    features: dict[str, list[float]],
    outcomes: dict[str, dict[int, dict[str, float]]],
    salt: str,
) -> tuple[float, dict[str, dict[int, dict[str, float]]], dict[str, float]]:
    metric_weights = {
        "completion": 0.30,
        "state": 0.20,
        "task": 0.25,
        "ux": 0.05,
        "utility": 0.20,
    }
    folds = _balanced_folds(fit_ids, salt)
    errors: dict[str, float] = {}
    predictions_by_ridge: dict[float, dict[str, dict[int, dict[str, float]]]] = {}
    for ridge in RIDGE_GRID:
        oof: dict[str, dict[int, dict[str, float]]] = {}
        weighted_error = 0.0
        weight_total = 0.0
        for held_out in folds:
            held_set = set(held_out)
            train_ids = [task_id for task_id in fit_ids if task_id not in held_set]
            models = _fit_models(train_ids, features, outcomes, ridge)
            for task_id in held_out:
                predicted = _predict_models(models, features[task_id])
                oof[task_id] = predicted
                for mask in range(8):
                    for metric in METRICS:
                        weight = metric_weights[metric]
                        error = predicted[mask][metric] - outcomes[task_id][mask][metric]
                        weighted_error += weight * error * error
                        weight_total += weight
        errors[str(ridge)] = weighted_error / max(weight_total, 1e-12)
        predictions_by_ridge[ridge] = oof
    selected = min(RIDGE_GRID, key=lambda value: (errors[str(value)], -value))
    return selected, predictions_by_ridge[selected], errors


def _policy(threshold: float) -> dict[str, Any]:
    return {
        "min_predicted_utility_gain": threshold,
        "minimum_safety_delta": 0.0,
        "ux_delta_tolerance": 0.02,
        "card_count_penalty": 0.005,
        "allowed_masks": list(range(7)),
        "fallback_mask": 7,
        "trajectory_decision": "first_effective_query_then_lock",
    }


def evaluate_policy(
    task_ids: list[str],
    predictions: dict[str, dict[int, dict[str, float]]],
    outcomes: dict[str, dict[int, dict[str, float]]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    selected: dict[str, int] = {
        task_id: choose_from_predictions(predictions[task_id], policy)[0]
        for task_id in task_ids
    }
    candidate = {
        metric: mean(outcomes[task_id][selected[task_id]][metric] for task_id in task_ids)
        for metric in METRICS
    }
    baseline = {
        metric: mean(outcomes[task_id][7][metric] for task_id in task_ids)
        for metric in METRICS
    }
    mask_counts = {
        str(mask): sum(value == mask for value in selected.values()) for mask in range(8)
    }
    return {
        "n": len(task_ids),
        "candidate": candidate,
        "baseline_mask7": baseline,
        "delta": {metric: candidate[metric] - baseline[metric] for metric in METRICS},
        "mask_counts": mask_counts,
        "coverage": 1.0 - mask_counts["7"] / max(len(task_ids), 1),
        "selected": selected,
    }


def select_threshold(
    fit_ids: list[str],
    oof_predictions: dict[str, dict[int, dict[str, float]]],
    outcomes: dict[str, dict[int, dict[str, float]]],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    eligible: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    for threshold in THRESHOLD_GRID:
        policy = _policy(threshold)
        report = evaluate_policy(fit_ids, oof_predictions, outcomes, policy)
        delta = report["delta"]
        # Fit is used for policy selection.  Require one net completion win and
        # no aggregate state/task loss before touching calibration.
        if (
            delta["completion"] >= 1.0 / len(fit_ids) - 1e-12
            and delta["state"] >= -1e-12
            and delta["task"] >= -1e-12
            and delta["ux"] >= -0.02 - 1e-12
            and delta["utility"] > 0.0
        ):
            score = (
                delta["completion"],
                delta["task"],
                delta["state"],
                delta["utility"],
                delta["ux"],
                -report["coverage"],
                threshold,
            )
            eligible.append((score, policy, report))
    if eligible:
        _score, policy, report = max(eligible, key=lambda item: item[0])
        return policy, report, True
    fallback = _policy(1.0)
    return fallback, evaluate_policy(fit_ids, oof_predictions, outcomes, fallback), False


def conformal_harm_certificate(
    report: dict[str, Any],
    outcomes: dict[str, dict[int, dict[str, float]]],
    alpha: float = 0.10,
) -> dict[str, Any]:
    selected: dict[str, int] = report["selected"]
    harms = []
    for task_id, mask in selected.items():
        baseline = outcomes[task_id][7]
        candidate = outcomes[task_id][mask]
        harms.append(
            max(
                0.0,
                baseline["completion"] - candidate["completion"],
                baseline["state"] - candidate["state"],
                baseline["task"] - candidate["task"],
                baseline["ux"] - candidate["ux"] - 0.10,
            )
        )
    ordered = sorted(harms)
    index = min(len(ordered) - 1, math.ceil((len(ordered) + 1) * (1.0 - alpha)) - 1)
    quantile = ordered[index]
    delta = report["delta"]
    aggregate_safe = (
        delta["completion"] >= -1e-12
        and delta["state"] >= -1e-12
        and delta["task"] >= -1e-12
        and delta["ux"] >= -0.05 - 1e-12
    )
    return {
        "alpha": alpha,
        "quantile_index_zero_based": index,
        "harm_quantile": quantile,
        "positive_harm_count": sum(value > 0.0 for value in harms),
        "aggregate_safe": aggregate_safe,
        "passed": quantile <= 0.0 and aggregate_safe,
    }


def _public_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "selected"}


def train(
    lattice_root: Path,
    memory_path: Path,
    output: Path,
    salt: str,
) -> dict[str, Any]:
    outcomes, queries = load_lattice(lattice_root)
    split = deterministic_split(outcomes, salt)
    features = build_features(memory_path, queries)
    ridge, oof_predictions, ridge_errors = select_ridge(
        split["fit"], features, outcomes, salt
    )
    policy, fit_report, fit_gate = select_threshold(
        split["fit"], oof_predictions, outcomes
    )
    models = _fit_models(split["fit"], features, outcomes, ridge)
    remaining_predictions = {
        task_id: _predict_models(models, features[task_id])
        for label in ("calibration", "lockbox")
        for task_id in split[label]
    }
    calibration_report = evaluate_policy(
        split["calibration"], remaining_predictions, outcomes, policy
    )
    certificate = conformal_harm_certificate(calibration_report, outcomes)
    calibration_passed = fit_gate and certificate["passed"]

    lockbox_report = evaluate_policy(
        split["lockbox"], remaining_predictions, outcomes, policy
    )
    lockbox_delta = lockbox_report["delta"]
    lockbox_passed = (
        calibration_passed
        and lockbox_delta["completion"] >= 1.0 / len(split["lockbox"]) - 1e-12
        and lockbox_delta["state"] >= -1e-12
        and lockbox_delta["task"] >= -1e-12
        and lockbox_delta["ux"] >= -0.05 - 1e-12
    )

    artifact = {
        "schema_version": "conformal_lattice_router_v1",
        "deployment_enabled": bool(lockbox_passed),
        "domain": "shopping_assistant",
        "baseline_mask": 7,
        "training_data_policy": "train_task_trajectories_only",
        "split": {
            "strategy": "salted_sha256_rank_60_20_20",
            "counts": {label: len(task_ids) for label, task_ids in split.items()},
            "manifest_sha256": split_manifest_hash(split),
        },
        "feature_schema": {"names": feature_names(), "count": len(feature_names())},
        "model": {
            "type": "multioutput_ridge",
            "ridge": ridge,
            "ridge_cv_weighted_mse": ridge_errors,
            "target_metrics": list(METRICS),
            "utility_weights": UTILITY_WEIGHTS,
        },
        "policy": policy,
        "certification": {
            "fit_gate_passed": fit_gate,
            "fit_oof": _public_report(fit_report),
            "calibration": _public_report(calibration_report),
            "conformal_harm": certificate,
            "lockbox_gate_passed": lockbox_passed,
            "lockbox": _public_report(lockbox_report),
        },
        "models": {
            str(mask): {
                metric: [round(value, 12) for value in weights]
                for metric, weights in metrics.items()
            }
            for mask, metrics in models.items()
        },
    }
    rendered = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice-root", type=Path, required=True)
    parser.add_argument("--memory-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", default="statebench-conformal-lattice-v1")
    args = parser.parse_args()
    artifact = train(args.lattice_root, args.memory_path, args.output, args.salt)
    summary = {
        "deployment_enabled": artifact["deployment_enabled"],
        "ridge": artifact["model"]["ridge"],
        "policy": artifact["policy"],
        "certification": artifact["certification"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
