"""Build process-conformant workflow memory from STATE-Bench train trajectories."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DOMAIN_TOOLS = {
    "travel": {
        "get_user_reservations",
        "get_booking",
        "search_flights",
        "get_flight_status",
        "get_user_details",
        "get_policies",
        "create_booking",
        "update_booking",
        "cancel_booking",
        "search_hotels",
        "get_hotel_reservation",
        "book_hotel",
        "cancel_hotel_reservation",
        "search_car_rentals",
        "get_car_rental",
        "book_car_rental",
        "cancel_car_rental",
    },
    "customer_support": {
        "get_customer",
        "get_order",
        "get_policies",
        "search_products",
        "get_product_details",
        "cancel_order",
        "process_return",
        "process_refund",
        "process_exchange",
        "get_warranty_status",
        "process_warranty_claim",
    },
    "shopping_assistant": {
        "search_products",
        "get_product_details",
        "get_variants",
        "get_cart",
        "get_customer_account",
        "get_policies",
        "get_promotions",
        "validate_promo",
        "check_compatibility",
        "add_to_cart",
        "remove_from_cart",
        "update_cart_item",
        "apply_promo",
        "remove_promo",
        "redeem_loyalty_points",
        "cancel_loyalty_redemption",
        "get_shipping_options",
        "set_shipping_option",
    },
}

WRITE_TOOLS = {
    "create_booking",
    "update_booking",
    "cancel_booking",
    "book_hotel",
    "cancel_hotel_reservation",
    "book_car_rental",
    "cancel_car_rental",
    "cancel_order",
    "process_return",
    "process_refund",
    "process_exchange",
    "process_warranty_claim",
    "add_to_cart",
    "remove_from_cart",
    "update_cart_item",
    "apply_promo",
    "remove_promo",
    "redeem_loyalty_points",
    "cancel_loyalty_redemption",
    "set_shipping_option",
}

INTENT_RULES = {
    "travel": [
        ("cancel", ("cancel", "取消")),
        ("hotel", ("hotel", "住宿", "酒店")),
        ("car_rental", ("car rental", "rental car", "租车")),
        ("book", ("book", "new flight", "reserve", "预订")),
        ("status", ("status", "delay", "gate", "terminal", "状态", "延误")),
        ("baggage", ("baggage", "bag", "luggage", "行李")),
        ("seat", ("seat", "window", "aisle", "座位")),
        ("ancillary", ("meal", "wifi", "legroom", "insurance", "餐", "保险")),
        ("change", ("change", "update", "modify", "move", "switch", "改签", "修改")),
    ],
    "customer_support": [
        ("price_match", ("price match", "price drop", "cheaper", "价格", "差价")),
        ("warranty", ("warranty", "repair", "defect", "保修", "维修")),
        ("exchange", ("exchange", "replacement", "replace", "换货", "更换")),
        ("shipping_claim", ("missing", "lost", "damaged", "wrong item", "late", "delivery", "丢失", "破损", "漏发")),
        ("return", ("return", "send back", "退货")),
        ("cancel", ("cancel", "取消")),
        ("refund", ("refund", "money back", "退款")),
    ],
    "shopping_assistant": [
        ("promo", ("promo", "coupon", "discount", "促销", "优惠")),
        ("loyalty", ("loyalty", "points", "积分")),
        (
            "account_history",
            (
                "repeat purchase",
                "purchase history",
                "prior purchase",
                "already purchased",
                "customer account",
                "first time",
            ),
        ),
        ("shipping", ("shipping", "delivery option", "expedited", "配送", "运费")),
        ("compatibility", ("compatible", "compatibility", "work with", "兼容")),
        ("remove", ("remove", "delete", "take out", "移除")),
        ("update_cart", ("quantity", "change cart", "update cart", "数量")),
        ("add_to_cart", ("add", "buy", "put in cart", "加入", "购买")),
        ("search", ("find", "recommend", "looking for", "search", "推荐", "寻找")),
    ],
}


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class TraceRecord:
    task_id: str
    domain: str
    opening_request: str
    request: str
    events: list[ToolEvent]
    final_answer: str
    family: str = ""
    fitness: float = 0.0
    quality: float = 0.0
    issues: list[str] = field(default_factory=list)
    task_requirements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sequence(self) -> list[str]:
        return [event.name for event in self.events]


def _compact(value: Any, limit: int = 360) -> str:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def _infer_intent(domain: str, text: str) -> str:
    lowered = text.lower().replace("_", " ").replace("-", " ")
    hits = []
    for intent, phrases in INTENT_RULES[domain]:
        matched = False
        for phrase in phrases:
            if phrase.isascii():
                pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
                matched = bool(re.search(pattern, lowered))
            else:
                matched = phrase in lowered
            if matched:
                break
        if matched:
            hits.append(intent)
    return "+".join(hits[:2]) if hits else "general"


def _result_has_error(result: Any) -> bool:
    text = _compact(result, 1000).lower()
    return any(marker in text for marker in ('"error"', "tool_error", "invalid argument", "not found"))


def _parse_trace(path: Path, domain: str, tasks_root: Path | None = None) -> TraceRecord:
    conversation = json.loads(path.read_text(encoding="utf-8"))["conversation"]
    task_requirements: list[dict[str, Any]] = []
    if tasks_root is not None:
        task_path = tasks_root / domain / "tasks" / path.name
        if task_path.exists():
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task_requirements = [
                requirement
                for requirement in task.get("task_requirements", [])
                if isinstance(requirement, dict)
            ]
    user_messages = [
        str(item.get("content", "")).strip()
        for item in conversation
        if item.get("role") == "user" and "[TASK_DONE]" not in str(item.get("content", ""))
    ]
    assistant_answers = [
        str(item.get("content", "")).strip()
        for item in conversation
        if item.get("role") == "assistant" and item.get("content")
    ]
    events = []
    issues = []
    previous_read = None
    for item in conversation:
        if item.get("role") != "assistant":
            continue
        for call in item.get("tool_calls") or []:
            name = str(call.get("name", ""))
            if name not in DOMAIN_TOOLS[domain]:
                issues.append(f"unknown_tool:{name}")
                continue
            arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            result = call.get("result")
            fingerprint = (name, json.dumps(arguments, sort_keys=True, default=str))
            if name not in WRITE_TOOLS and fingerprint == previous_read:
                issues.append(f"redundant_read:{name}")
                continue
            previous_read = fingerprint if name not in WRITE_TOOLS else None
            if _result_has_error(result):
                issues.append(f"tool_error:{name}")
            events.append(ToolEvent(name=name, arguments=arguments, result=result))

    request = " ".join(user_messages)
    write_names = sorted({event.name for event in events if event.name in WRITE_TOOLS})
    intent = _infer_intent(domain, f"{path.stem} {request}")
    operation = "+".join(write_names) if write_names else "read_only"
    family = intent if intent != "general" else operation
    return TraceRecord(
        task_id=path.stem,
        domain=domain,
        opening_request=user_messages[0] if user_messages else "",
        request=request,
        events=events,
        final_answer=assistant_answers[-1] if assistant_answers else "",
        family=family,
        issues=issues,
        task_requirements=task_requirements,
    )


def _discover_process(records: list[TraceRecord], noise_threshold: float) -> dict[str, Any]:
    import pandas as pd
    import pm4py

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for case_index, record in enumerate(records):
        for step, activity in enumerate(record.sequence):
            rows.append(
                {
                    "case:concept:name": record.task_id,
                    "concept:name": activity,
                    "time:timestamp": base + timedelta(days=case_index, seconds=step),
                }
            )
    if not rows:
        return {"tree": "tau", "variants": [], "transitions": [], "mean_fitness": 0.0}

    log = pm4py.format_dataframe(pd.DataFrame(rows))
    tree = pm4py.discover_process_tree_inductive(log, noise_threshold=noise_threshold)
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
    diagnostics = pm4py.conformance_diagnostics_token_based_replay(
        log,
        net,
        initial_marking,
        final_marking,
        opt_parameters={"show_progress_bar": False},
    )
    ordered_ids = list(dict.fromkeys(map(str, log["case:concept:name"].tolist())))
    for task_id, diagnostic in zip(ordered_ids, diagnostics):
        record = next(item for item in records if item.task_id == task_id)
        record.fitness = float(diagnostic.get("trace_fitness", 0.0))

    variants = Counter(tuple(record.sequence) for record in records)
    transitions = Counter(
        (left, right)
        for record in records
        for left, right in zip(record.sequence, record.sequence[1:])
    )
    return {
        "tree": str(tree),
        "variants": [
            {"count": count, "sequence": list(sequence)}
            for sequence, count in variants.most_common(8)
        ],
        "transitions": [
            {"from": left, "to": right, "count": count}
            for (left, right), count in transitions.most_common(20)
        ],
        "mean_fitness": sum(record.fitness for record in records) / len(records),
    }


def _score_records(records: list[TraceRecord]) -> None:
    for record in records:
        error_count = sum(issue.startswith(("tool_error", "unknown_tool")) for issue in record.issues)
        redundant_count = sum(issue.startswith("redundant_read") for issue in record.issues)
        write_count = sum(event.name in WRITE_TOOLS for event in record.events)
        record.quality = (
            0.55 * record.fitness
            + 0.20 * min(len(set(record.sequence)) / 5, 1)
            + 0.15 * min(write_count, 2) / 2
            + 0.10 * min(len(record.request) / 300, 1)
            - 0.20 * error_count
            - 0.03 * redundant_count
            - 0.002 * max(0, len(record.sequence) - 15)
        )


def _select_representatives(records: list[TraceRecord], limit: int = 5) -> list[TraceRecord]:
    selected = []
    for candidate in sorted(records, key=lambda item: item.quality, reverse=True):
        candidate_set = set(zip(candidate.sequence, candidate.sequence[1:]))
        if selected:
            similarities = []
            for previous in selected:
                previous_set = set(zip(previous.sequence, previous.sequence[1:]))
                union = candidate_set | previous_set
                similarities.append(len(candidate_set & previous_set) / max(len(union), 1))
            if max(similarities) > 0.92 and len(selected) >= min(2, limit):
                continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _format_trace(record: TraceRecord) -> str:
    lines = [f"Task: {record.request[:700]}", "Observed tool trajectory:"]
    for event in record.events[:18]:
        argument_shape = ", ".join(sorted(event.arguments)) or "no arguments"
        lines.append(f"- {event.name}({argument_shape}) -> {_compact(event.result)}")
    lines.append(f"Final response: {record.final_answer[:500]}")
    if record.task_requirements:
        lines.append("Normative public-train requirements (use these to identify mistakes in the trajectory):")
        for requirement in record.task_requirements[:12]:
            lines.append(
                "- "
                + _compact(
                    {
                        "kind": requirement.get("kind"),
                        "evidence": requirement.get("evidence"),
                        "requirement": requirement.get("requirement"),
                    },
                    700,
                )
            )
    return "\n".join(lines)


def _llm_cards(
    *,
    client: Any,
    model: str,
    domain: str,
    family: str,
    process: dict[str, Any],
    representatives: list[TraceRecord],
    structured: bool = False,
) -> list[dict[str, Any]]:
    observed_tools = sorted({event.name for record in representatives for event in record.events})
    allowed_tools = sorted(DOMAIN_TOOLS[domain])
    examples = "\n\n".join(_format_trace(record) for record in representatives)
    structured_fields = """
When structured policy fields are enabled, each workflow must also contain these
four list fields:
- mandatory_disclosures: facts or benefits that must be explained before a write
- confirmation_gates: conditions requiring explicit user approval before a write
- refresh_after_mutation: state or calculations that must be re-read after a write
- forbidden_actions: unsafe, privacy-violating, stale, or unsupported actions
Also emit runtime_rules, a list of executable verifier specifications. Each rule
must contain exactly these keys:
- id: short snake_case identifier
- phase: one of pre_write, post_write, pre_final
- kind: one of require_tool, require_confirmation, forbid, refresh, disclose
- trigger_tools: list of allowed tool names that activate the rule; empty only for pre_final
- required_tools: list of allowed read-only tools whose observed results satisfy a require_tool or refresh rule
- condition: concise natural-language condition for semantic evaluation
- feedback: concrete correction instruction for the agent
- enforcement: deterministic only when required_tools alone make the rule decidable; otherwise semantic
Ground these fields in the representative dialogues and tool results. Do not
invent policies.
""" if structured else ""
    prompt = f"""You are inducing offline procedural memory for a stateful tool-using agent.
Follow the Agent Workflow Memory principle: abstract reusable sub-routines from multiple trajectories,
but ground every step in the process model and observed tools. The trajectories may be noisy or unsuccessful.
Do not treat a one-off action, repeated call, tool error, concrete entity ID, or concrete price as a reusable rule.
Do not invent tools or policy facts. Current task tool outputs are always authoritative.

Domain: {domain}
Workflow family: {family}
Allowed tools: {json.dumps(allowed_tools)}
Tools observed in these trajectories: {json.dumps(observed_tools)}
Inductive-miner process tree: {process['tree']}
Frequent variants: {json.dumps(process['variants'], ensure_ascii=True)}
Frequent transitions: {json.dumps(process['transitions'], ensure_ascii=True)}

Representative fixed training trajectories:
{examples}

The normative public-train requirements are supervision, while the observed
trajectory may violate them. Induce required checks and forbidden actions from
the requirements, and use the trajectory only as evidence of available tools,
state observations, and common failure modes. Never imitate an observed action
that contradicts a normative requirement.

Return one JSON object with key "workflows". It must contain 1-3 non-overlapping workflow objects.
Each workflow must contain these keys:
- title: short reusable task pattern
- applies_when: user intent and observable state conditions
- preconditions: list of facts to verify with read-only tools before mutation
- steps: ordered list using only allowed tool names; use placeholders instead of concrete IDs
- branches: list of important if/then alternatives, including valid no-write outcomes
- avoid: list of common unsafe or unsupported actions
- keywords: list of retrieval phrases and synonyms
{structured_fields}
Keep each workflow concise and executable. Output JSON only."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=3000,
    )
    text = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("workflow model did not return a JSON object")
    payload = json.loads(match.group(0))
    workflows = payload.get("workflows")
    if not isinstance(workflows, list):
        raise TypeError("workflow response is missing a workflows list")
    return [item for item in workflows if isinstance(item, dict)][:3]


def _fallback_card(
    family: str,
    process: dict[str, Any],
    records: list[TraceRecord],
    structured: bool = False,
) -> dict[str, Any]:
    tool_counts = Counter(event.name for record in records for event in record.events)
    required = [name for name, count in tool_counts.most_common() if count / len(records) >= 0.5]
    variants = process["variants"]
    sequence = variants[0]["sequence"] if variants else required
    card = {
        "title": family.replace(":", " / ").replace("_", " "),
        "applies_when": f"The request matches the {family.replace('_', ' ')} workflow family.",
        "preconditions": [f"Verify current facts needed by {name}." for name in required if name not in WRITE_TOOLS],
        "steps": [f"Use {name} with current-task identifiers and verified arguments." for name in sequence],
        "branches": ["If a required condition is not verified, explain the valid alternative and do not mutate state."],
        "avoid": ["Do not copy identifiers, prices, dates, or policy outcomes from training examples."],
        "keywords": _tokens(family.replace(":", " ")),
    }
    if structured:
        card.update(
            {
                "mandatory_disclosures": [],
                "confirmation_gates": [],
                "refresh_after_mutation": [],
                "forbidden_actions": [],
                "runtime_rules": [],
            }
        )
    return card


def _compile_runtime_rules(
    card: dict[str, Any],
    *,
    observed_tools: set[str],
    allowed_tools: set[str],
) -> list[dict[str, Any]]:
    card_text = json.dumps(card, ensure_ascii=True).lower()
    write_triggers = sorted(
        name
        for name in allowed_tools & WRITE_TOOLS
        if name in observed_tools or re.search(rf"\b{re.escape(name)}\b", card_text)
    )
    read_tools = sorted(allowed_tools - WRITE_TOOLS)

    def mentioned_tools(text: str) -> list[str]:
        lowered = text.lower()
        return [
            name
            for name in read_tools
            if name in observed_tools
            and re.search(rf"\b{re.escape(name)}\b", lowered)
        ]

    rules: list[dict[str, Any]] = []

    def add_rules(
        field: str,
        *,
        phase: str,
        kind: str,
        feedback_prefix: str,
        infer_required_tools: bool = False,
    ) -> None:
        for index, value in enumerate(card.get(field, [])):
            condition = str(value).strip()
            if not condition or (phase != "pre_final" and not write_triggers):
                continue
            required_tools = mentioned_tools(condition) if infer_required_tools else []
            enforcement = (
                "deterministic"
                if kind in {"require_tool", "refresh"} and required_tools
                else "semantic"
            )
            rules.append(
                {
                    "id": f"compiled_{field}_{index + 1}",
                    "phase": phase,
                    "kind": kind,
                    "trigger_tools": write_triggers,
                    "required_tools": required_tools,
                    "condition": condition,
                    "feedback": f"{feedback_prefix}: {condition}",
                    "enforcement": enforcement,
                }
            )

    add_rules(
        "preconditions",
        phase="pre_write",
        kind="require_tool",
        feedback_prefix="Satisfy the required precondition before the write",
        infer_required_tools=True,
    )
    add_rules(
        "confirmation_gates",
        phase="pre_write",
        kind="require_confirmation",
        feedback_prefix="Obtain approval only if this material choice is still unresolved",
    )
    add_rules(
        "mandatory_disclosures",
        phase="pre_write",
        kind="disclose",
        feedback_prefix="Make the required disclosure before the write",
    )
    add_rules(
        "refresh_after_mutation",
        phase="post_write",
        kind="refresh",
        feedback_prefix="Refresh authoritative state after the write",
        infer_required_tools=True,
    )
    add_rules(
        "forbidden_actions",
        phase="pre_write",
        kind="forbid",
        feedback_prefix="Do not perform the forbidden action",
    )
    return rules


def _training_instance_literals(records: list[TraceRecord]) -> set[str]:
    literals: set[str] = set()
    amount_terms = ("amount", "cost", "fee", "price", "subtotal", "total")

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key).lower())
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and (key == "id" or key.endswith("_id")):
            if len(value.strip()) >= 4:
                literals.add(value.strip().lower())
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and any(term in key for term in amount_terms)
            and abs(float(value)) >= 2
        ):
            literals.add(f"${float(value):g}".lower())

    for record in records:
        for event in record.events:
            visit(event.arguments)
            visit(event.result)
    return literals


def _validate_card(
    card: dict[str, Any],
    allowed_tools: set[str],
    structured: bool = False,
    forbidden_literals: set[str] | None = None,
) -> bool:
    required = {"title", "applies_when", "preconditions", "steps", "branches", "avoid", "keywords"}
    structured_required = {
        "mandatory_disclosures",
        "confirmation_gates",
        "refresh_after_mutation",
        "forbidden_actions",
    }
    if structured and not structured_required.issubset(card):
        return False
    if not required.issubset(card) or not isinstance(card.get("steps"), list):
        return False
    if structured:
        if any(not isinstance(card.get(key), list) for key in structured_required):
            return False
        if card.get("runtime_rules") is None:
            card["runtime_rules"] = []
        if not isinstance(card.get("runtime_rules"), list):
            return False
        allowed_phases = {"pre_write", "post_write", "pre_final"}
        allowed_kinds = {"require_tool", "require_confirmation", "forbid", "refresh", "disclose"}
        for rule in card.get("runtime_rules", []):
            if not isinstance(rule, dict):
                return False
            if rule.get("phase") not in allowed_phases or rule.get("kind") not in allowed_kinds:
                return False
            if rule.get("enforcement") not in {"deterministic", "semantic"}:
                return False
            trigger_tools = rule.get("trigger_tools", [])
            required_tools = rule.get("required_tools", [])
            if not isinstance(trigger_tools, list) or not isinstance(required_tools, list):
                return False
            if not set(map(str, trigger_tools + required_tools)).issubset(allowed_tools):
                return False
    text = json.dumps(card, ensure_ascii=True).lower()
    if forbidden_literals:
        for literal in forbidden_literals:
            if literal.startswith("$"):
                pattern = rf"{re.escape(literal)}(?![0-9.])"
            else:
                pattern = rf"(?<![a-z0-9_-]){re.escape(literal)}(?![a-z0-9_-])"
            if re.search(pattern, text):
                return False
    mentioned = set(re.findall(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b", text))
    suspicious = {
        token for token in mentioned
        if token not in allowed_tools and token not in {"read_only", "customer_support", "shopping_assistant"}
    }
    return not any(token.endswith(("_booking", "_order", "_cart", "_refund", "_return")) for token in suspicious)


def _render_card(card: dict[str, Any], process: dict[str, Any], mode: str = "hybrid") -> str:
    semantic = [
        f"Workflow: {card['title']}",
        f"Use when: {card['applies_when']}",
        "Verify first:",
        *[f"- {item}" for item in card.get("preconditions", [])],
        "Procedure:",
        *[f"{index + 1}. {item}" for index, item in enumerate(card.get("steps", []))],
        "Branches:",
        *[f"- {item}" for item in card.get("branches", [])],
        "Avoid:",
        *[f"- {item}" for item in card.get("avoid", [])],
    ]
    if any(
        card.get(key)
        for key in (
            "mandatory_disclosures",
            "confirmation_gates",
            "refresh_after_mutation",
            "forbidden_actions",
        )
    ):
        semantic.extend(
            [
                "Mandatory disclosures before writes:",
                *[f"- {item}" for item in card.get("mandatory_disclosures", [])],
                "Confirmation gates:",
                *[f"- {item}" for item in card.get("confirmation_gates", [])],
                "Refresh after mutation:",
                *[f"- {item}" for item in card.get("refresh_after_mutation", [])],
                "Forbidden actions:",
                *[f"- {item}" for item in card.get("forbidden_actions", [])],
            ]
        )
    structural = [
        f"Process support: mean conformance={process['mean_fitness']:.3f}",
        "Observed frequent paths:",
        *[
            f"- {variant['count']}x: {' -> '.join(variant['sequence'])}"
            for variant in process["variants"][:3]
        ],
    ]
    if mode == "awm_only":
        return "\n".join(semantic)
    if mode == "process_only":
        return "\n".join(structural)
    return "\n".join([*semantic, *structural])


def _generate_cards_for_family(
    job: dict[str, Any],
    *,
    client: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    domain = job["domain"]
    family = job["family"]
    records = job["records"]
    process = job["process"]
    representatives = job["representatives"]
    cache_suffix = "__structured_v4" if args.structured else "__plain_v3"
    cache_name = re.sub(r"[^a-z0-9_.+-]+", "_", f"{domain}__{family}".lower()) + cache_suffix + ".json"
    cache_path = args.cache_dir / cache_name
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("model") == args.llm_model and isinstance(cached.get("cards"), list):
            return cached["cards"]

    cards = []
    has_supervision = any(record.task_requirements for record in records)
    if client is not None and (len(records) >= 2 or has_supervision):
        try:
            cards = _llm_cards(
                client=client,
                model=args.llm_model,
                domain=domain,
                family=family,
                process=process,
                representatives=representatives,
                structured=args.structured,
            )
        except Exception as exc:  # noqa: BLE001 - malformed provider output falls back to process mining
            print(json.dumps({"domain": domain, "family": family, "llm_error": str(exc)}, ensure_ascii=True))
    if not cards:
        cards = [_fallback_card(family, process, records, args.structured)]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"model": args.llm_model, "cards": cards}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return cards


def _assert_train_data_root(data_root: Path) -> None:
    blocked_parts = {"test", "test_tasks", "test_task_trajectories"}
    if blocked_parts & {part.lower() for part in data_root.parts}:
        raise ValueError("Workflow memory must be built from train trajectories only")


def build(args: argparse.Namespace) -> dict[str, Any]:
    _assert_train_data_root(args.data_root)
    client = None
    if not args.no_llm:
        from openai import OpenAI

        api_key = os.environ.get(args.llm_api_key_env)
        if not api_key:
            raise ValueError(f"Set {args.llm_api_key_env} or pass --no-llm")
        client = OpenAI(
            base_url=args.llm_base_url.rstrip("/") + "/",
            api_key=api_key,
            timeout=args.llm_timeout,
            max_retries=1,
        )

    all_cards = []
    stats: dict[str, Any] = {}
    jobs = []
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    selected_ids = None
    if args.task_ids_file:
        selected_ids = {
            line.strip()
            for line in args.task_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    for domain in domains:
        if domain not in DOMAIN_TOOLS:
            raise ValueError(f"Unknown domain: {domain}")
        records = [
            _parse_trace(path, domain, args.tasks_root)
            for path in sorted((args.data_root / domain).glob("*.json"))
            if selected_ids is None or path.stem in selected_ids
        ]
        families: dict[str, list[TraceRecord]] = defaultdict(list)
        for record in records:
            families[record.family].append(record)
        stats[domain] = {"trajectories": len(records), "families": len(families), "cards": 0}

        for family, family_records in sorted(families.items()):
            process = _discover_process(family_records, args.noise_threshold)
            _score_records(family_records)
            representatives = _select_representatives(family_records, args.representatives)
            jobs.append(
                {
                    "domain": domain,
                    "family": family,
                    "records": family_records,
                    "process": process,
                    "representatives": representatives,
                }
            )

    cards_by_key = {}
    with ThreadPoolExecutor(max_workers=max(1, args.llm_workers)) as executor:
        future_to_job = {
            executor.submit(_generate_cards_for_family, job, client=client, args=args): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            cards_by_key[(job["domain"], job["family"])] = future.result()
            print(
                json.dumps(
                    {
                        "completed": len(cards_by_key),
                        "total": len(jobs),
                        "domain": job["domain"],
                        "family": job["family"],
                    },
                    ensure_ascii=True,
                )
            )

    for job in jobs:
        domain = job["domain"]
        family = job["family"]
        family_records = job["records"]
        process = job["process"]
        representatives = job["representatives"]
        cards = cards_by_key[(domain, family)]
        allowed_tools = DOMAIN_TOOLS[domain]
        for index, card in enumerate(cards):
            if args.structured:
                card["runtime_rules"] = _compile_runtime_rules(
                    card,
                    observed_tools={
                        event.name for record in family_records for event in record.events
                    },
                    allowed_tools=allowed_tools,
                )
            forbidden_literals = _training_instance_literals(family_records)
            if not _validate_card(
                card,
                allowed_tools,
                args.structured,
                forbidden_literals=forbidden_literals,
            ):
                card = _fallback_card(family, process, family_records, args.structured)
                if args.structured:
                    card["runtime_rules"] = _compile_runtime_rules(
                        card,
                        observed_tools={
                            event.name for record in family_records for event in record.events
                        },
                        allowed_tools=allowed_tools,
                    )
                if not _validate_card(
                    card,
                    allowed_tools,
                    args.structured,
                    forbidden_literals=forbidden_literals,
                ):
                    raise ValueError(f"Invalid fallback workflow card for {domain}:{family}")
            source_tasks = [record.task_id for record in representatives]
            search_text = " ".join(
                [
                    str(card.get("title", "")),
                    str(card.get("applies_when", "")),
                    " ".join(map(str, card.get("keywords", []))),
                    family.replace(":", " ").replace("_", " "),
                    " ".join(record.opening_request for record in representatives),
                ]
            )
            all_cards.append(
                {
                    "id": f"{domain}:{family}:{index}",
                    "domain": domain,
                    "family": family,
                    "support": len(family_records),
                    "mean_fitness": round(process["mean_fitness"], 6),
                    "quality": round(sum(record.quality for record in representatives) / len(representatives), 6),
                    "source_tasks": source_tasks,
                    "observed_tools": sorted({event.name for record in family_records for event in record.events}),
                    "keywords": list(map(str, card.get("keywords", []))),
                    "applies_when": str(card.get("applies_when", "")),
                    "preconditions": list(map(str, card.get("preconditions", []))),
                    "steps": list(map(str, card.get("steps", []))),
                    "branches": list(map(str, card.get("branches", []))),
                    "avoid": list(map(str, card.get("avoid", []))),
                    "mandatory_disclosures": list(map(str, card.get("mandatory_disclosures", []))),
                    "confirmation_gates": list(map(str, card.get("confirmation_gates", []))),
                    "refresh_after_mutation": list(map(str, card.get("refresh_after_mutation", []))),
                    "forbidden_actions": list(map(str, card.get("forbidden_actions", []))),
                    "runtime_rules": [
                        {
                            "id": str(rule.get("id", "rule")),
                            "phase": str(rule.get("phase", "pre_final")),
                            "kind": str(rule.get("kind", "disclose")),
                            "trigger_tools": list(map(str, rule.get("trigger_tools", []))),
                            "required_tools": list(map(str, rule.get("required_tools", []))),
                            "condition": str(rule.get("condition", "")),
                            "feedback": str(rule.get("feedback", "")),
                            "enforcement": str(rule.get("enforcement", "semantic")),
                        }
                        for rule in card.get("runtime_rules", [])
                        if isinstance(rule, dict)
                    ],
                    "search_text": search_text,
                    "tokens": _tokens(search_text),
                    "text": _render_card(card, process, "hybrid"),
                    "awm_text": _render_card(card, process, "awm_only"),
                    "process_text": _render_card(card, process, "process_only"),
                    "process": process,
                }
            )
            stats[domain]["cards"] += 1

    artifact = {
        "version": 1,
        "method": "process-conformant-workflow-memory",
        "source": "STATE-Bench fixed train trajectories only",
        "cards": all_cards,
        "stats": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/train_task_trajectories"))
    parser.add_argument("--output", type=Path, default=Path("outputs/memory/process_workflows.json"))
    parser.add_argument("--noise-threshold", type=float, default=0.2)
    parser.add_argument("--representatives", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-base-url", default=os.environ.get("WORKFLOW_LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("WORKFLOW_LLM_MODEL", "gpt-5.4"))
    parser.add_argument("--llm-api-key-env", default="WORKFLOW_LLM_API_KEY")
    parser.add_argument("--llm-workers", type=int, default=6)
    parser.add_argument("--llm-timeout", type=float, default=240.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/memory/workflow_cache"))
    parser.add_argument("--domains", default=",".join(DOMAIN_TOOLS))
    parser.add_argument("--task-ids-file", type=Path, default=None)
    parser.add_argument("--tasks-root", type=Path, default=Path("state_bench/domains"))
    parser.add_argument("--structured", action="store_true")
    args = parser.parse_args()
    artifact = build(args)
    print(json.dumps({"output": str(args.output), "cards": len(artifact["cards"]), "stats": artifact["stats"]}))


if __name__ == "__main__":
    main()
