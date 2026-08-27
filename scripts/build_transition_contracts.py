"""Learn conservative transition-contract refinements from public train traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.transition_aware_memory import (
    TOOL_CONTRACTS,
    TransitionAwareMemory,
    _collect_values,
    _identifier_scope,
    _scope_compatible,
)


@dataclass
class ContractStats:
    occurrences: int = 0
    predecessors: Counter[str] = field(default_factory=Counter)
    policy_topics: Counter[str] = field(default_factory=Counter)
    commit_count: int = 0
    preview_before_commit: int = 0
    argument_keys: Counter[str] = field(default_factory=Counter)
    result_fields: Counter[str] = field(default_factory=Counter)


def _successful(result: Any) -> bool:
    if not isinstance(result, dict):
        return True
    if result.get("error") or result.get("success") is False:
        return False
    return str(result.get("status", "")).lower() not in {
        "error",
        "failed",
        "rejected",
    }


def _events(conversation: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], Any]]:
    return [
        (str(call.get("name", "")), call.get("arguments") or {}, call.get("result"))
        for message in conversation
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    ]


def _relevant_predecessors(
    write_name: str, observed_reads: dict[str, tuple[dict[str, Any], Any]]
) -> set[str]:
    write_contract = TOOL_CONTRACTS[write_name]
    relevant = set()
    for read_name in observed_reads:
        read_contract = TOOL_CONTRACTS.get(read_name)
        if not read_contract or read_contract.writes:
            continue
        related_fields = (
            write_contract.requires_fresh
            | write_contract.writes
            | write_contract.invalidates
        )
        if read_name == "get_policies" or (read_contract.reads & related_fields):
            relevant.add(read_name)
    return relevant


def learn_domain(path: Path, *, min_support: int, min_confidence: float) -> dict[str, Any]:
    stats: dict[str, ContractStats] = defaultdict(ContractStats)
    trajectory_count = 0
    for trace_path in sorted(path.glob("*.json")):
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        trajectory_count += 1
        observed_reads: dict[str, tuple[dict[str, Any], Any]] = {}
        previews: dict[str, list[tuple[dict[str, frozenset[str]], Any]]] = defaultdict(list)
        memory = TransitionAwareMemory(learned=False)
        for name, arguments, result in _events(payload.get("conversation", [])):
            contract = TOOL_CONTRACTS.get(name)
            if not contract or not _successful(result):
                continue
            if not contract.writes:
                observed_reads[name] = (arguments, result)
                continue

            item = stats[name]
            item.occurrences += 1
            item.argument_keys.update(arguments.keys())
            item.result_fields.update(_collect_values(result).keys())
            relevant = _relevant_predecessors(name, observed_reads)
            item.predecessors.update(relevant)
            if "get_policies" in observed_reads:
                topic = observed_reads["get_policies"][0].get("topic")
                if topic:
                    item.policy_topics[str(topic)] += 1

            scope = _identifier_scope(_collect_values({"arguments": arguments, "result": result}))
            preview = memory._is_preview(name, arguments, result)
            if preview:
                previews[name].append((scope, result))
                continue

            item.commit_count += 1
            if any(_scope_compatible(prior_scope, scope) for prior_scope, _ in previews[name]):
                item.preview_before_commit += 1
            observed_reads.clear()
            previews.clear()

    learned: dict[str, Any] = {}
    for name, item in sorted(stats.items()):
        required_tools = []
        for tool, support in item.predecessors.most_common():
            confidence = support / max(item.occurrences, 1)
            if support >= min_support and confidence >= min_confidence:
                required_tools.append(
                    {"tool": tool, "support": support, "confidence": round(confidence, 4)}
                )
        policy_topics = [
            topic
            for topic, support in item.policy_topics.most_common()
            if support >= min_support and support / max(item.occurrences, 1) >= min_confidence
        ]
        preview_rate = item.preview_before_commit / max(item.commit_count, 1)
        learned[name] = {
            "occurrences": item.occurrences,
            "required_tools": required_tools[:3],
            "policy_topics": policy_topics,
            "preview_required": item.commit_count >= min_support
            and preview_rate >= min_confidence,
            "preview_support": item.preview_before_commit,
            "commit_count": item.commit_count,
            "preview_rate": round(preview_rate, 4),
            "argument_keys": sorted(item.argument_keys),
            "result_fields": sorted(item.result_fields),
        }
    return {"trajectory_count": trajectory_count, "contracts": learned}


def build_artifact(
    data_root: Path,
    *,
    domains: list[str],
    min_support: int,
    min_confidence: float,
) -> dict[str, Any]:
    blocked_parts = {"test", "test_tasks", "test_task_trajectories"}
    if blocked_parts & {part.lower() for part in data_root.parts}:
        raise ValueError("Transition contracts must be built from train trajectories only")
    return {
        "version": 1,
        "source": "public train trajectories only",
        "min_support": min_support,
        "min_confidence": min_confidence,
        "domains": {
            domain: learn_domain(
                data_root / domain,
                min_support=min_support,
                min_confidence=min_confidence,
            )
            for domain in domains
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".statebench_test/datasets/train_task_trajectories"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("configs/tapm_transition_contracts.json")
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["travel", "customer_support", "shopping_assistant"],
    )
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    args = parser.parse_args()
    artifact = build_artifact(
        args.data_root,
        domains=args.domains,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
