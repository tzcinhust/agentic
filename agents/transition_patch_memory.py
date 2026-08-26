"""PatchCore-style coreset memory for local agent transitions."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+|[\u4e00-\u9fff]", flags=re.IGNORECASE)


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    text = value.lower()
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " <email> ", text)
    text = re.sub(r"\b[a-z]{1,12}[-_]\d{2,}[a-z0-9_-]*\b", " <id> ", text)
    text = re.sub(r"(?<![a-z])[-+]?\d+(?:\.\d+)?", " <number> ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(value: Any) -> list[str]:
    result = []
    for token in TOKEN_RE.findall(normalize_text(value)):
        result.append(token)
        if "_" in token:
            result.extend(part for part in token.split("_") if part)
    return result


def fit_idf(texts: list[str]) -> dict[str, float]:
    document_frequency = Counter(token for text in texts for token in set(tokens(text)))
    size = max(len(texts), 1)
    return {
        token: math.log((size + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }


def vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(token for token in tokens(text) if token in idf)
    if not counts:
        return {}
    vector = {
        token: (1.0 + math.log(frequency)) * idf[token]
        for token, frequency in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in vector.values()))
    return {token: value / norm for token, value in vector.items()} if norm else {}


def cosine_distance(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 1.0
    if len(left) > len(right):
        left, right = right, left
    similarity = sum(value * right.get(token, 0.0) for token, value in left.items())
    return 1.0 - max(0.0, min(1.0, similarity))


def transition_distance(
    left_context: dict[str, float],
    left_step: dict[str, float],
    right_context: dict[str, float],
    right_step: dict[str, float],
) -> float:
    return 0.55 * cosine_distance(left_context, right_context) + 0.45 * cosine_distance(
        left_step, right_step
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[position]


def _greedy_coreset(
    context_vectors: list[dict[str, float]],
    step_vectors: list[dict[str, float]],
    size: int,
) -> list[int]:
    """Deterministic farthest-first approximation of the PatchCore minimax coreset."""
    count = len(context_vectors)
    if size >= count:
        return list(range(count))
    if not count or size <= 0:
        return []

    average_distances = []
    for index in range(count):
        distances = [
            transition_distance(
                context_vectors[index],
                step_vectors[index],
                context_vectors[other],
                step_vectors[other],
            )
            for other in range(count)
            if other != index
        ]
        average_distances.append(sum(distances) / max(len(distances), 1))
    selected = [min(range(count), key=average_distances.__getitem__)]
    nearest = [
        transition_distance(
            context_vectors[index],
            step_vectors[index],
            context_vectors[selected[0]],
            step_vectors[selected[0]],
        )
        for index in range(count)
    ]
    while len(selected) < size:
        candidate = max(
            (index for index in range(count) if index not in selected),
            key=nearest.__getitem__,
        )
        selected.append(candidate)
        for index in range(count):
            distance = transition_distance(
                context_vectors[index],
                step_vectors[index],
                context_vectors[candidate],
                step_vectors[candidate],
            )
            nearest[index] = min(nearest[index], distance)
    return selected


def build_transition_artifact(
    patches: list[dict[str, Any]],
    *,
    coreset_ratio: float = 0.35,
    min_per_group: int = 6,
    max_per_group: int = 48,
) -> dict[str, Any]:
    clean = []
    seen = set()
    for index, patch in enumerate(patches):
        domain = str(patch.get("domain", "")).strip()
        phase = str(patch.get("phase", "")).strip()
        context_text = normalize_text(patch.get("context_text", ""))
        transition_text = normalize_text(patch.get("transition_text", ""))
        if not domain or phase not in {"pre_write", "post_write", "pre_final"}:
            continue
        if not context_text or not transition_text:
            continue
        fingerprint = (domain, phase, context_text, transition_text)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        clean.append(
            {
                **patch,
                "id": str(patch.get("id") or f"transition:{index}"),
                "domain": domain,
                "phase": phase,
                "context_text": context_text,
                "transition_text": transition_text,
            }
        )

    context_idf = fit_idf([patch["context_text"] for patch in clean])
    transition_idf = fit_idf([patch["transition_text"] for patch in clean])
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for patch in clean:
        groups[(patch["domain"], patch["phase"])].append(patch)

    selected_patches = []
    thresholds: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    stats: dict[str, dict[str, int]] = defaultdict(dict)
    for (domain, phase), group in sorted(groups.items()):
        contexts = [vectorize(patch["context_text"], context_idf) for patch in group]
        steps = [vectorize(patch["transition_text"], transition_idf) for patch in group]
        target = min(
            len(group),
            max(min_per_group, min(max_per_group, math.ceil(len(group) * coreset_ratio))),
        )
        selected_indices = _greedy_coreset(contexts, steps, target)
        selected_set = set(selected_indices)
        selected_patches.extend(group[index] for index in selected_indices)

        context_radii = []
        transition_radii = []
        for index in range(len(group)):
            candidates = [item for item in selected_indices if item != index]
            if not candidates:
                candidates = [item for item in range(len(group)) if item != index]
            if not candidates:
                continue
            nearest_context = min(
                candidates,
                key=lambda item: cosine_distance(contexts[index], contexts[item]),
            )
            context_radii.append(cosine_distance(contexts[index], contexts[nearest_context]))
            transition_radii.append(cosine_distance(steps[index], steps[nearest_context]))
        thresholds[domain][phase] = {
            "context_radius": round(min(0.95, _percentile(context_radii, 0.95) + 0.03), 6),
            "transition_radius": round(
                min(0.85, _percentile(transition_radii, 0.90) + 0.03), 6
            ),
        }
        stats[domain][phase] = len(group)
        stats[domain][f"{phase}_coreset"] = len(selected_set)

    return {
        "version": 1,
        "method": "transition-patch-coreset-memory",
        "source": "STATE-Bench public train trajectories only",
        "context_idf": context_idf,
        "transition_idf": transition_idf,
        "thresholds": dict(thresholds),
        "patches": selected_patches,
        "stats": dict(stats),
    }


@dataclass(frozen=True)
class PatchMatch:
    patch: dict[str, Any]
    context_distance: float
    transition_distance: float
    anomaly_distance: float


class TransitionPatchIndex:
    def __init__(self, artifact: dict[str, Any], domain: str):
        self.domain = domain
        self.context_idf = {
            str(key): float(value) for key, value in artifact.get("context_idf", {}).items()
        }
        self.transition_idf = {
            str(key): float(value)
            for key, value in artifact.get("transition_idf", {}).items()
        }
        self.thresholds = artifact.get("thresholds", {}).get(domain, {})
        self.patches = [
            patch for patch in artifact.get("patches", []) if patch.get("domain") == domain
        ]
        self._vectors = [
            (
                vectorize(str(patch.get("context_text", "")), self.context_idf),
                vectorize(str(patch.get("transition_text", "")), self.transition_idf),
            )
            for patch in self.patches
        ]

    @classmethod
    def from_path(cls, path: Path, domain: str) -> "TransitionPatchIndex":
        return cls(json.loads(path.read_text(encoding="utf-8")), domain)

    def nearest(
        self,
        *,
        phase: str,
        context_text: str,
        transition_text: str,
        top_k: int = 3,
    ) -> list[PatchMatch]:
        query_context = vectorize(context_text, self.context_idf)
        query_step = vectorize(transition_text, self.transition_idf)
        matches = []
        for patch, (patch_context, patch_step) in zip(self.patches, self._vectors):
            if patch.get("phase") != phase:
                continue
            context_distance = cosine_distance(query_context, patch_context)
            step_distance = cosine_distance(query_step, patch_step)
            anomaly_distance = transition_distance(
                query_context, query_step, patch_context, patch_step
            )
            matches.append(PatchMatch(patch, context_distance, step_distance, anomaly_distance))
        return sorted(matches, key=lambda item: (item.context_distance, item.anomaly_distance))[
            :top_k
        ]

    def should_verify(self, phase: str, matches: list[PatchMatch]) -> bool:
        if not matches:
            return False
        threshold = self.thresholds.get(phase, {})
        context_radius = float(threshold.get("context_radius", 0.55))
        transition_radius = float(threshold.get("transition_radius", 0.65))
        supported = [match for match in matches if match.context_distance <= context_radius]
        return bool(supported) and min(
            match.transition_distance for match in supported
        ) > transition_radius
