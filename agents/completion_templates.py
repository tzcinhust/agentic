"""Independent retrieval for learned task-completion templates.

This module intentionally does not import or reuse the PWM procedure retriever.
Procedure cards answer how to proceed; these templates answer what must hold at
specific phases before the task may be considered complete.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def char_ngrams(text: str, n: int = 4) -> set[str]:
    compact = re.sub(r"\s+", " ", text.lower()).strip()
    return {compact[index : index + n] for index in range(max(0, len(compact) - n + 1))}


def _compact(value: Any, limit: int = 300) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _result_terms(value: Any, *, depth: int = 0) -> list[str]:
    """Return observable field/value terms without copying large tool payloads."""

    if depth > 3:
        return []
    if isinstance(value, dict):
        output: list[str] = []
        for key, child in value.items():
            output.append(str(key))
            output.extend(_result_terms(child, depth=depth + 1))
        return output
    if isinstance(value, list):
        output = []
        for child in value[:8]:
            output.extend(_result_terms(child, depth=depth + 1))
        return output
    if isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, str) and len(value) <= 80:
        return [value]
    return []


@dataclass(frozen=True)
class ToolEvent:
    sequence: int
    name: str
    arguments: dict[str, Any]
    result: Any
    assistant_index: int

    @property
    def status(self) -> str:
        if not isinstance(self.result, dict):
            return ""
        return str(self.result.get("status", "")).strip().lower()


def tool_events(conversation: list[dict[str, Any]]) -> list[ToolEvent]:
    """Normalize both inline and explicit tool-result conversation layouts."""

    events: list[ToolEvent] = []
    sequence = 0
    pending: tuple[int, list[dict[str, Any]]] | None = None
    for index, item in enumerate(conversation):
        role = item.get("role")
        if role == "assistant":
            calls = [call for call in (item.get("tool_calls") or []) if isinstance(call, dict)]
            unresolved = []
            for call in calls:
                if "result" not in call:
                    unresolved.append(call)
                    continue
                events.append(
                    ToolEvent(
                        sequence=sequence,
                        name=str(call.get("name", "")),
                        arguments=call.get("arguments")
                        if isinstance(call.get("arguments"), dict)
                        else {},
                        result=call.get("result"),
                        assistant_index=index,
                    )
                )
                sequence += 1
            pending = (index, unresolved) if unresolved else None
            continue

        if role != "tool" or not isinstance(item.get("content"), list):
            continue
        assistant_index, calls = pending or (index, [])
        for position, record in enumerate(item.get("content") or []):
            if not isinstance(record, dict):
                continue
            call = calls[position] if position < len(calls) else {}
            events.append(
                ToolEvent(
                    sequence=sequence,
                    name=str(record.get("name") or call.get("name") or ""),
                    arguments=(
                        record.get("arguments")
                        if isinstance(record.get("arguments"), dict)
                        else call.get("arguments")
                        if isinstance(call.get("arguments"), dict)
                        else {}
                    ),
                    result=record.get("result", record),
                    assistant_index=assistant_index,
                )
            )
            sequence += 1
        pending = None
    return events


def completion_query(conversation: list[dict[str, Any]]) -> str:
    """Build a completion-oriented query from observable conversation state only."""

    parts: list[str] = []
    assistant_text: list[str] = []
    for item in conversation:
        role = item.get("role")
        content = str(item.get("content", ""))
        if role == "user" and "[TASK_DONE]" not in content:
            parts.append(content)
        elif role == "assistant" and content:
            assistant_text.append(content)
    parts.extend(assistant_text[-3:])
    for event in tool_events(conversation):
        parts.append(event.name)
        parts.extend(_result_terms(event.result))
    query = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if len(query) <= 9000:
        return query
    return query[:3000] + " ... " + query[-5995:]


class CompletionTemplateIndex:
    """BM25-style index kept entirely separate from PWM procedure memory."""

    def __init__(self, artifact: dict[str, Any], *, domain: str | None, top_k: int = 8):
        self.top_k = max(1, int(top_k))
        self.templates = [
            item
            for item in artifact.get("templates", [])
            if isinstance(item, dict) and (domain is None or item.get("domain") == domain)
        ]
        self.by_id = {str(item.get("id")): item for item in self.templates}
        self.document_frequency = Counter(
            token for item in self.templates for token in set(item.get("tokens", []))
        )
        self.average_length = sum(len(item.get("tokens", [])) for item in self.templates) / max(
            len(self.templates), 1
        )
        self.ngrams = [char_ngrams(str(item.get("search_text", ""))) for item in self.templates]

    @classmethod
    def from_path(
        cls, path: Path | str, *, domain: str | None, top_k: int = 8
    ) -> "CompletionTemplateIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) < 2:
            raise ValueError("completion template artifact must have version >= 2")
        return cls(payload, domain=domain, top_k=top_k)

    def _score(
        self,
        query_counts: Counter[str],
        query_ngrams: set[str],
        observed_tools: set[str],
        index: int,
        item: dict[str, Any],
    ) -> float:
        document_counts = Counter(item.get("tokens", []))
        document_length = sum(document_counts.values())
        total_documents = len(self.templates)
        lexical = 0.0
        for token, query_frequency in query_counts.items():
            frequency = document_counts.get(token, 0)
            if not frequency:
                continue
            document_frequency = self.document_frequency.get(token, 0)
            inverse_frequency = math.log(
                1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + 1.3 * (
                0.25 + 0.75 * document_length / max(self.average_length, 1)
            )
            lexical += inverse_frequency * frequency * 2.3 / denominator * min(query_frequency, 2)

        character_similarity = len(query_ngrams & self.ngrams[index]) / max(len(query_ngrams), 1)
        tool_overlap = len(observed_tools & set(map(str, item.get("observed_tools", []))))
        if lexical == 0 and character_similarity < 0.002 and tool_overlap == 0:
            return 0.0
        confidence = float(item.get("confidence", 0.5))
        support = math.log1p(max(0, int(item.get("support", 1))))
        return lexical + 7.0 * character_similarity + 0.8 * tool_overlap + 0.25 * confidence + 0.1 * support

    @staticmethod
    def _similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
        left_tokens = set(left.get("tokens", []))
        right_tokens = set(right.get("tokens", []))
        union = left_tokens | right_tokens
        lexical = len(left_tokens & right_tokens) / max(len(union), 1)
        left_obligations = {
            (item.get("phase"), item.get("kind"), item.get("type"))
            for item in left.get("obligations", [])
            if isinstance(item, dict)
        }
        right_obligations = {
            (item.get("phase"), item.get("kind"), item.get("type"))
            for item in right.get("obligations", [])
            if isinstance(item, dict)
        }
        obligation_union = left_obligations | right_obligations
        obligation_similarity = len(left_obligations & right_obligations) / max(
            len(obligation_union), 1
        )
        return max(lexical, obligation_similarity)

    def retrieve_with_scores(
        self, query: str, *, top_k: int | None = None
    ) -> list[tuple[float, dict[str, Any]]]:
        if not query.strip() or not self.templates:
            return []
        limit = min(max(1, int(top_k or self.top_k)), self.top_k)
        query_counts = Counter(tokens(query))
        query_ngrams = char_ngrams(query)
        observed_tools = set(re.findall(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b", query.lower()))
        ranked = sorted(
            (
                (self._score(query_counts, query_ngrams, observed_tools, index, item), item)
                for index, item in enumerate(self.templates)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        selected: list[tuple[float, dict[str, Any]]] = []
        remaining = [pair for pair in ranked[: max(80, 10 * limit)] if pair[0] > 0]
        maximum_score = max(ranked[0][0], 1e-9)
        while remaining and len(selected) < limit:
            selected_signals = {
                str(signal)
                for _, prior in selected
                for signal in prior.get("latent_signal_categories", [])
            }
            best_index = max(
                range(len(remaining)),
                key=lambda index: 0.65 * (remaining[index][0] / maximum_score)
                + 0.18
                * bool(selected)
                * bool(
                    set(map(str, remaining[index][1].get("latent_signal_categories", [])))
                    - selected_signals
                )
                - 0.35
                * max(
                    (self._similarity(remaining[index][1], prior[1]) for prior in selected),
                    default=0.0,
                ),
            )
            score, item = remaining.pop(best_index)
            if score <= 0:
                break
            selected.append((score, item))
        return selected

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        return [item for _, item in self.retrieve_with_scores(query, top_k=top_k)]
