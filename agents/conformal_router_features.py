"""Stable, privacy-safe features for the train-only lattice router."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any


TOKEN_HASH_DIM = 16
CARD_HASH_DIM = 4
INTENTS = (
    "promo",
    "loyalty",
    "shipping",
    "compatibility",
    "remove",
    "update_cart",
    "add_to_cart",
    "search",
)
INTENT_PHRASES = {
    "promo": ("promo", "coupon", "discount", "code"),
    "loyalty": ("loyalty", "points", "tier"),
    "shipping": ("shipping", "delivery", "eta", "arrive"),
    "compatibility": ("compatible", "compatibility", "work with"),
    "remove": ("remove", "delete", "take out"),
    "update_cart": ("quantity", "change cart", "update cart", "swap", "replace"),
    "add_to_cart": ("add", "buy", "put in cart"),
    "search": ("find", "recommend", "looking for", "search", "compare"),
}
WRITE_PREFIXES = (
    "add_",
    "apply_",
    "redeem_",
    "remove_",
    "set_",
    "update_",
)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _signed_bucket(value: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % dimension
    sign = 1.0 if digest[4] & 1 else -1.0
    return bucket, sign


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def baseline_ranked_items(agent: Any, query: str, top_k: int = 3) -> list[tuple[float, dict[str, Any]]]:
    """Reproduce the frozen PWM selection while retaining raw retrieval scores."""

    if not query.strip() or not agent._cards:
        return []
    ranked = sorted(
        ((agent._score(query, index, item), item) for index, item in enumerate(agent._cards)),
        key=lambda pair: pair[0],
        reverse=True,
    )
    selected: list[tuple[float, dict[str, Any]]] = []
    families: set[str] = set()
    observed_tools: set[str] = set()
    for score, item in ranked:
        family = str(item.get("family", ""))
        if family in families:
            continue
        item_tools = set(item.get("observed_tools", []))
        adjusted = score - 0.08 * len(item_tools & observed_tools)
        if adjusted <= 0 and selected:
            continue
        selected.append((float(score), item))
        families.add(family)
        observed_tools.update(item_tools)
        if len(selected) >= min(top_k, agent.retrieve_learnings_top_k, 3):
            break
    return selected


def feature_names() -> list[str]:
    names = ["intercept"]
    names.extend(f"token_hash_{index}" for index in range(TOKEN_HASH_DIM))
    names.extend(f"intent_{intent}" for intent in INTENTS)
    names.extend(
        (
            "query_token_count",
            "query_character_count",
            "query_numeric_count",
            "query_question",
            "query_negation",
            "query_keep",
            "query_pivot",
            "query_temporal",
        )
    )
    for rank in range(1, 4):
        names.extend(
            (
                f"rank{rank}_score",
                f"rank{rank}_support",
                f"rank{rank}_conformance",
                f"rank{rank}_quality",
                f"rank{rank}_tool_count",
                f"rank{rank}_write_fraction",
            )
        )
        names.extend(f"rank{rank}_card_hash_{index}" for index in range(CARD_HASH_DIM))
    names.extend(("tool_jaccard_12", "tool_jaccard_13", "tool_jaccard_23"))
    return names


def feature_vector(query: str, ranked_items: list[tuple[float, dict[str, Any]]]) -> list[float]:
    query_tokens = tokens(query)
    lowered = query.lower()
    vector: list[float] = [1.0]

    hashed = [0.0] * TOKEN_HASH_DIM
    for token in query_tokens:
        bucket, sign = _signed_bucket(token, TOKEN_HASH_DIM)
        hashed[bucket] += sign
    norm = math.sqrt(sum(value * value for value in hashed)) or 1.0
    vector.extend(value / norm for value in hashed)

    vector.extend(
        float(any(phrase in lowered for phrase in INTENT_PHRASES[intent]))
        for intent in INTENTS
    )
    vector.extend(
        (
            _clip(len(query_tokens) / 60.0, 0.0, 1.0),
            _clip(len(query) / 500.0, 0.0, 1.0),
            _clip(sum(token.isdigit() for token in query_tokens) / 8.0, 0.0, 1.0),
            float("?" in query),
            float(any(value in lowered for value in ("don't", "do not", "never", "not "))),
            float(any(value in lowered for value in ("keep", "remain", "still"))),
            float(any(value in lowered for value in ("instead", "change my mind", "swap", "replace"))),
            float(any(value in lowered for value in ("after", "before", "then", "once"))),
        )
    )

    tool_sets: list[set[str]] = []
    for rank in range(3):
        if rank < len(ranked_items):
            score, item = ranked_items[rank]
            tools = set(map(str, item.get("observed_tools", [])))
            writes = {tool for tool in tools if tool.startswith(WRITE_PREFIXES)}
            vector.extend(
                (
                    _clip(score / 20.0, -1.0, 1.0),
                    _clip(math.log1p(max(0, int(item.get("support", 0)))) / 5.0, 0.0, 1.0),
                    _clip(float(item.get("mean_fitness", 0.0)), 0.0, 1.0),
                    _clip(float(item.get("quality", 0.0)), 0.0, 1.0),
                    _clip(len(tools) / 16.0, 0.0, 1.0),
                    len(writes) / max(len(tools), 1),
                )
            )
            card_hash = [0.0] * CARD_HASH_DIM
            bucket, sign = _signed_bucket(str(item.get("id", "")), CARD_HASH_DIM)
            card_hash[bucket] = sign
            vector.extend(card_hash)
            tool_sets.append(tools)
        else:
            vector.extend([0.0] * (6 + CARD_HASH_DIM))
            tool_sets.append(set())

    for left, right in ((0, 1), (0, 2), (1, 2)):
        union = tool_sets[left] | tool_sets[right]
        vector.append(len(tool_sets[left] & tool_sets[right]) / max(len(union), 1))
    if len(vector) != len(feature_names()):
        raise AssertionError(f"feature size mismatch: {len(vector)} != {len(feature_names())}")
    return vector


def predict(weights: list[float], features: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, features))


def predict_masks(artifact: dict[str, Any], features: list[float]) -> dict[int, dict[str, float]]:
    predictions: dict[int, dict[str, float]] = {}
    for raw_mask, metric_models in artifact.get("models", {}).items():
        predictions[int(raw_mask)] = {
            metric: max(0.0, min(1.0, predict(weights, features)))
            for metric, weights in metric_models.items()
        }
    return predictions


def choose_mask(
    artifact: dict[str, Any],
    features: list[float],
    *,
    require_deployment: bool = True,
) -> tuple[int, dict[int, dict[str, float]], str]:
    """Apply the frozen safety gate; mask 7 is the fail-closed baseline."""

    if require_deployment and not artifact.get("deployment_enabled", False):
        return 7, {}, "artifact_not_deployable"
    predictions = predict_masks(artifact, features)
    if 7 not in predictions:
        return 7, predictions, "missing_baseline_model"
    policy = artifact.get("policy", {})
    mask, reason = choose_from_predictions(predictions, policy)
    return mask, predictions, reason


def choose_from_predictions(
    predictions: dict[int, dict[str, float]],
    policy: dict[str, Any],
) -> tuple[int, str]:
    """Select from precomputed predictions using the deployment policy."""

    if 7 not in predictions:
        return 7, "missing_baseline_model"
    min_gain = float(policy.get("min_predicted_utility_gain", 1.0))
    safety_margin = float(policy.get("minimum_safety_delta", 0.0))
    ux_tolerance = float(policy.get("ux_delta_tolerance", 0.0))
    card_penalty = float(policy.get("card_count_penalty", 0.0))
    allowed = {int(mask) for mask in policy.get("allowed_masks", range(8))}
    baseline = predictions[7]
    qualified: list[tuple[float, float, int]] = []
    for mask, metrics in predictions.items():
        if mask == 7 or mask not in allowed:
            continue
        utility_gain = metrics["utility"] - baseline["utility"]
        if utility_gain < min_gain:
            continue
        if any(
            metrics[metric] - baseline[metric] < safety_margin
            for metric in ("completion", "state", "task")
        ):
            continue
        if metrics["ux"] - baseline["ux"] < -ux_tolerance:
            continue
        score = metrics["utility"] - card_penalty * mask.bit_count()
        qualified.append((score, utility_gain, mask))
    if not qualified:
        return 7, "no_certified_alternative"
    _score, _gain, mask = max(qualified, key=lambda item: (item[0], item[1], -item[2].bit_count(), -item[2]))
    return mask, "certified_alternative"
