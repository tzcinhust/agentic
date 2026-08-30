"""Build the train-only Selective Decision-Aware PWM sidecar.

The builder deliberately has a narrow data boundary.  It learns from exactly:

* ``STATE-Bench/datasets/train_task_trajectories/<domain>/*.json``;
* the v1 ``process_workflows.json`` memory artifact; and
* a manifest containing only the legacy development-panel task IDs.

It never imports task definitions, requirements, judge output, or test results and
does not make model/API calls.  Missing trajectory scores produce an explicitly
neutral utility prior instead of a fabricated score.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "2.0.0"
DOMAINS = ("shopping_assistant", "travel", "customer_support")
SPLIT_SEED = "workflow-router-v2-sha256"
CV_SEED = "workflow-router-v2-five-fold"
NEUTRAL_UTILITY = 0.5
MIN_UTILITY_EXPOSURES = 5
BETA_ALPHA = 2.0
BETA_BETA = 2.0
MAX_RENDER_CHARS = 2200

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

INTENT_RULES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "travel": (
        ("cancel", ("cancel", "取消")),
        ("hotel", ("hotel", "住宿", "酒店")),
        ("car_rental", ("car rental", "rental car", "租车")),
        ("book", ("book", "new flight", "reserve", "预订")),
        ("status", ("status", "delay", "gate", "terminal", "状态", "延误")),
        ("baggage", ("baggage", "bag", "luggage", "行李")),
        ("seat", ("seat", "window", "aisle", "座位")),
        ("ancillary", ("meal", "wifi", "legroom", "insurance", "餐", "保险")),
        ("change", ("change", "update", "modify", "move", "switch", "改签", "修改")),
    ),
    "customer_support": (
        ("price_match", ("price match", "price drop", "cheaper", "价格", "差价")),
        ("warranty", ("warranty", "repair", "defect", "保修", "维修")),
        ("exchange", ("exchange", "replacement", "replace", "换货", "更换")),
        ("shipping_claim", ("missing", "lost", "damaged", "wrong item", "late", "delivery", "丢失", "破损", "漏发")),
        ("return", ("return", "send back", "退货")),
        ("cancel", ("cancel", "取消")),
        ("refund", ("refund", "money back", "退款")),
    ),
    "shopping_assistant": (
        ("promo", ("promo", "coupon", "discount", "促销", "优惠")),
        ("loyalty", ("loyalty", "points", "积分")),
        ("shipping", ("shipping", "delivery option", "expedited", "配送", "运费")),
        ("compatibility", ("compatible", "compatibility", "work with", "兼容")),
        ("remove", ("remove", "delete", "take out", "移除")),
        ("update_cart", ("quantity", "change cart", "update cart", "数量")),
        ("add_to_cart", ("add", "buy", "put in cart", "加入", "购买")),
        ("search", ("find", "recommend", "looking for", "search", "推荐", "寻找")),
    ),
}

GRID = {
    "field": (0.25, 0.5, 1.0),
    "utility": (0.0, 0.25, 0.5),
    "risk": (0.25, 0.5, 1.0),
    "trace": (0.0, 0.25),
    "near_tie": (0.5, 1.0, 2.0),
    "mmr": (0.1, 0.2, 0.3),
}

SECTION_NAMES = {
    "Workflow": "title",
    "Use when": "trigger",
    "Verify first": "verification_rules",
    "Procedure": "procedure",
    "Branches": "decision_rules",
    "Avoid": "prohibitions",
}

DISCLOSURE_MARKERS = (
    "ask",
    "clarify",
    "compare",
    "confirm",
    "disclose",
    "explain",
    "inform",
    "offer",
    "preview",
    "quote",
    "report",
    "show",
    "state",
    "tell",
    "说明",
    "告知",
    "确认",
    "询问",
)

FORBIDDEN_V1_KEYS = {
    "judge",
    "oracle",
    "requirement",
    "requirements",
    "state_requirements",
    "task_requirements",
    "task_summary",
    "task_definition",
    "task_definitions",
    "expected_state",
    "judge_reasoning",
    "test",
    "test_task",
    "test_task_id",
    "test_result",
    "test_results",
}


@dataclass(frozen=True)
class Trajectory:
    domain: str
    task_id: str
    request: str
    tools: tuple[str, ...]
    writes: frozenset[str]
    family: str
    utility: float | None


@dataclass(frozen=True)
class Example:
    task_id: str
    domain: str
    intent_text: str
    query: str
    family: str
    target_writes: frozenset[str]
    observed_suffix: tuple[str, ...]
    phase: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    compact = re.sub(r"\s+", " ", text.lower()).strip()
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _tool_names(text: str, known_tools: set[str]) -> set[str]:
    candidates = set(re.findall(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b", text.lower()))
    return candidates & known_tools


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lstrip("- ").strip())


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_V1_KEYS:
                return str(key)
            found = _contains_forbidden_key(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _contains_forbidden_key(nested)
            if found:
                return found
    return None


def _infer_intents(domain: str, text: str) -> tuple[str, ...]:
    lowered = text.lower().replace("_", " ").replace("-", " ")
    hits: list[str] = []
    for intent, phrases in INTENT_RULES[domain]:
        for phrase in phrases:
            if phrase.isascii():
                pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
                matched = bool(re.search(pattern, lowered))
            else:
                matched = phrase in lowered
            if matched:
                hits.append(intent)
                break
    return tuple(hits[:2])


def _infer_family(domain: str, task_id: str, request: str, writes: Iterable[str]) -> str:
    intents = _infer_intents(domain, f"{task_id} {request}")
    if intents:
        return "+".join(intents)
    operations = sorted(set(writes))
    return "+".join(operations) if operations else "read_only"


def _normalise_metric(value: Any, *, ux: bool = False) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    number = float(value)
    if ux:
        if 0.0 <= number <= 1.0:
            return number
        if 0.0 <= number <= 5.0:
            return number / 5.0
        return None
    if 0.0 <= number <= 1.0:
        return number
    if 0.0 <= number <= 100.0:
        return number / 100.0
    return None


def _embedded_utility(payload: Mapping[str, Any]) -> float | None:
    """Read only explicitly embedded top-level trajectory metrics.

    Tool results and conversation text are intentionally never searched for metric
    names because that would turn incidental content into a fake evaluation label.
    """

    containers = [payload]
    for key in ("metrics", "scores", "evaluation"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            containers.insert(0, nested)

    aliases = {
        "completion": ("completion", "task_completion", "success"),
        "state": ("state", "state_score", "state_accuracy"),
        "task": ("task", "task_score", "task_accuracy"),
        "ux": ("ux", "ux_score", "user_experience"),
    }
    values: dict[str, float] = {}
    for metric, names in aliases.items():
        raw: Any = None
        found = False
        for container in containers:
            for name in names:
                if name in container:
                    raw = container[name]
                    found = True
                    break
            if found:
                break
        normalised = _normalise_metric(raw, ux=metric == "ux") if found else None
        if normalised is None:
            return None
        values[metric] = normalised
    return (
        0.60 * values["completion"]
        + 0.20 * values["state"]
        + 0.15 * values["task"]
        + 0.05 * values["ux"]
    )


def _inventory_train_files(data_root: Path) -> tuple[dict[str, dict[str, Path]], str]:
    expected_suffix = ("datasets", "train_task_trajectories")
    resolved = data_root.resolve()
    if tuple(part.lower() for part in resolved.parts[-2:]) != expected_suffix:
        raise ValueError(
            "Training data path must end exactly in datasets/train_task_trajectories; "
            f"got {resolved}"
        )

    inventory: dict[str, dict[str, Path]] = {}
    logical_names: list[str] = []
    for domain in DOMAINS:
        domain_root = (resolved / domain).resolve()
        if not domain_root.is_dir():
            raise FileNotFoundError(f"Missing train trajectory domain directory: {domain_root}")
        files = sorted(domain_root.glob("*.json"), key=lambda item: item.name)
        if len(files) != 100 or len({path.stem for path in files}) != 100:
            raise ValueError(f"Expected exactly 100 train trajectories for {domain}, found {len(files)}")
        if any(path.resolve().parent != domain_root for path in files):
            raise ValueError(f"Train inventory for {domain} contains an out-of-directory file")
        inventory[domain] = {path.stem: path for path in files}
        logical_names.extend(
            f"datasets/train_task_trajectories/{domain}/{path.name}" for path in files
        )
    return inventory, _sha256_bytes(_canonical_bytes(sorted(logical_names)))


def _load_trajectories(
    inventory: Mapping[str, Mapping[str, Path]],
    known_tools: Mapping[str, set[str]],
    selected_ids: Mapping[str, set[str]],
) -> tuple[dict[str, list[Trajectory]], dict[str, Any]]:
    by_domain: dict[str, list[Trajectory]] = {}
    manifest_entries: list[dict[str, str]] = []
    file_hashes: dict[str, str] = {}
    for domain in DOMAINS:
        missing = selected_ids[domain] - set(inventory[domain])
        if missing:
            raise ValueError(f"Selected train IDs are absent for {domain}: {sorted(missing)}")
        files = [
            inventory[domain][task_id]
            for task_id in sorted(selected_ids[domain])
        ]
        records: list[Trajectory] = []
        seen_ids: set[str] = set()
        for path in files:
            raw = path.read_bytes()
            digest = _sha256_bytes(raw)
            payload = json.loads(raw.decode("utf-8"))
            forbidden = _contains_forbidden_key(payload)
            if forbidden:
                raise ValueError(
                    f"Forbidden oracle-like field in train trajectory {domain}/{path.stem}: "
                    f"{forbidden}"
                )
            if set(payload) - {"conversation", "metrics", "scores", "evaluation", "completion", "task_completion", "success", "state", "state_score", "state_accuracy", "task", "task_score", "task_accuracy", "ux", "ux_score", "user_experience"}:
                # Unknown fields are ignored, but made visible in provenance through
                # the file hash.  The builder still reads no external source.
                pass
            conversation = payload.get("conversation")
            if not isinstance(conversation, list):
                raise ValueError(f"Trajectory lacks a conversation list: {path}")
            task_id = path.stem
            if task_id in seen_ids:
                raise ValueError(f"Duplicate train task ID in {domain}: {task_id}")
            seen_ids.add(task_id)
            user_messages = [
                str(item.get("content", "")).strip()
                for item in conversation
                if isinstance(item, Mapping)
                and item.get("role") == "user"
                and "[TASK_DONE]" not in str(item.get("content", ""))
            ]
            tools: list[str] = []
            for item in conversation:
                if not isinstance(item, Mapping) or item.get("role") != "assistant":
                    continue
                for call in item.get("tool_calls") or []:
                    if not isinstance(call, Mapping):
                        continue
                    name = str(call.get("name", ""))
                    if name not in known_tools[domain]:
                        raise ValueError(f"Unknown tool {name!r} in train trajectory {domain}/{task_id}")
                    tools.append(name)
            writes = frozenset(name for name in tools if name in WRITE_TOOLS)
            request = " ".join(user_messages)
            records.append(
                Trajectory(
                    domain=domain,
                    task_id=task_id,
                    request=request,
                    tools=tuple(tools),
                    writes=writes,
                    family=_infer_family(domain, task_id, request, writes),
                    utility=_embedded_utility(payload),
                )
            )
            logical_path = f"datasets/train_task_trajectories/{domain}/{path.name}"
            manifest_entries.append({"path": logical_path, "sha256": digest})
            file_hashes[logical_path] = digest
        by_domain[domain] = records
    manifest_entries.sort(key=lambda item: item["path"])
    manifest_hash = _sha256_bytes(_canonical_bytes(manifest_entries))
    return by_domain, {
        "manifest_sha256": manifest_hash,
        "files": file_hashes,
        "file_count": len(manifest_entries),
    }


def _load_v1(path: Path) -> tuple[dict[str, Any], str, dict[str, set[str]]]:
    raw = path.read_bytes()
    artifact = json.loads(raw.decode("utf-8"))
    if artifact.get("version") != 1 or not isinstance(artifact.get("cards"), list):
        raise ValueError("The source memory must be a version-1 process_workflows artifact")
    forbidden = _contains_forbidden_key(artifact)
    if forbidden:
        raise ValueError(f"Forbidden oracle-like field in source memory artifact: {forbidden}")
    known_tools: dict[str, set[str]] = {domain: set() for domain in DOMAINS}
    seen: set[str] = set()
    for card in artifact["cards"]:
        if not isinstance(card, Mapping):
            raise ValueError("Every v1 card must be an object")
        card_id = str(card.get("id", ""))
        domain = str(card.get("domain", ""))
        if not card_id or card_id in seen or domain not in known_tools:
            raise ValueError(f"Invalid or duplicate v1 card: {card_id!r}")
        seen.add(card_id)
        known_tools[domain].update(map(str, card.get("observed_tools") or []))
    return artifact, _sha256_bytes(raw), known_tools


def _load_dev_manifest(
    path: Path,
    inventory_ids: Mapping[str, set[str]],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Development manifest must be an object containing task IDs only")
    extra_keys = set(map(str, payload)) - {*DOMAINS, "seed", "method"}
    if extra_keys:
        raise ValueError(f"Development manifest contains unsupported metadata: {sorted(extra_keys)}")
    result: dict[str, list[str]] = {}
    for domain in DOMAINS:
        values = payload.get(domain)
        if (
            not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
            or len(values) != 10
            or len(set(values)) != 10
        ):
            raise ValueError(f"Development manifest must contain 10 unique IDs for {domain}")
        ids = list(values)
        missing = sorted(set(ids) - inventory_ids[domain])
        if missing:
            raise ValueError(f"Development IDs absent from train trajectories for {domain}: {missing}")
        result[domain] = ids
    metadata = {
        "sha256": _sha256_bytes(raw),
        "seed": payload.get("seed"),
        "method": payload.get("method"),
        "role": "split-metadata-only; contains task IDs but no learned content",
    }
    return result, metadata


def _make_splits(
    inventory_ids: Mapping[str, set[str]],
    dev_ids: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, list[str]]]:
    splits: dict[str, dict[str, list[str]]] = {}
    for domain in DOMAINS:
        dev = list(dev_ids[domain])
        remaining = sorted(inventory_ids[domain] - set(dev))
        ordered = sorted(
            remaining,
            key=lambda task_id: (
                hashlib.sha256(f"{SPLIT_SEED}|{domain}|{task_id}".encode("utf-8")).hexdigest(),
                task_id,
            ),
        )
        lockbox = ordered[:10]
        optimizer = ordered[10:]
        if len(optimizer) != 80:
            raise ValueError(f"Split invariant failed for {domain}: {len(dev)}/{len(lockbox)}/{len(optimizer)}")
        splits[domain] = {"dev": dev, "lockbox": lockbox, "optimizer": optimizer}
    return splits


def _parse_sections(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "title": "",
        "trigger": "",
        "verification_rules": [],
        "procedure": [],
        "decision_rules": [],
        "prohibitions": [],
    }
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = False
        for heading, key in SECTION_NAMES.items():
            prefix = f"{heading}:"
            if line.startswith(prefix):
                current = key
                remainder = _normalize_line(line[len(prefix) :])
                if key in {"title", "trigger"}:
                    parsed[key] = remainder
                elif remainder:
                    parsed[key].append(remainder)
                matched = True
                break
        if matched:
            continue
        if line.startswith("Process support:") or line.startswith("Observed frequent paths:"):
            current = None
            continue
        if current in {"verification_rules", "procedure", "decision_rules", "prohibitions"}:
            cleaned = _normalize_line(re.sub(r"^\d+\.\s*", "", line))
            if cleaned:
                parsed[current].append(cleaned)
    return parsed


def _unique_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _render_block(label: str, values: Sequence[str]) -> list[str]:
    rendered = [f"{label}:"]
    rendered.extend(f"- {value}" for value in values) if values else rendered.append("- NONE")
    return rendered


def _render_contract(contract: Mapping[str, Any], *, secondary: bool) -> str:
    title = str(contract["scope"]["title"])
    lines = [f"WORKFLOW{' CONSTRAINTS' if secondary else ''}: {title}"]
    lines.extend(_render_block("WHEN", [str(contract["trigger"])]))
    lines.extend(_render_block("READ", list(map(str, contract["required_reads"]))))
    if not secondary:
        # Communication branches appear under SAY, not twice under DECIDE.
        disclosure_set = set(map(str, contract["required_disclosures"]))
        decision_only = [
            str(rule) for rule in contract["decision_rules"] if str(rule) not in disclosure_set
        ]
        lines.extend(_render_block("DECIDE", decision_only))
        lines.extend(_render_block("WRITE", list(map(str, contract["authorized_writes"]))))
        lines.extend(_render_block("VERIFY", list(map(str, contract["verification_rules"]))))
    lines.extend(_render_block("SAY", list(map(str, contract["required_disclosures"]))))
    lines.extend(_render_block("NEVER", list(map(str, contract["prohibitions"]))))
    return "\n".join(lines)


def _compile_card(card: Mapping[str, Any], known_tools: set[str]) -> dict[str, Any]:
    source_text = str(card.get("awm_text") or card.get("text") or "")
    parsed = _parse_sections(source_text)
    mentioned = _tool_names(source_text, known_tools)
    verification_text = " ".join(parsed["verification_rules"])
    procedure_text = " ".join(parsed["procedure"])
    reads = sorted(_tool_names(f"{verification_text} {procedure_text}", known_tools) - WRITE_TOOLS)
    writes = sorted(_tool_names(procedure_text, known_tools) & WRITE_TOOLS)
    disclosures = _unique_preserving_order(
        rule
        # Procedure rows are tool invocations (for example ``confirm=false``),
        # not user-facing disclosure obligations.  Communication obligations are
        # compiled only from explicit branch prose to avoid duplicating the full
        # procedural program under SAY.
        for rule in parsed["decision_rules"]
        if any(marker in rule.lower() for marker in DISCLOSURE_MARKERS)
    )
    contract = {
        "trigger": parsed["trigger"],
        "required_reads": reads,
        "authorized_writes": writes,
        "verification_rules": _unique_preserving_order(parsed["verification_rules"]),
        "decision_rules": _unique_preserving_order(parsed["decision_rules"]),
        "required_disclosures": disclosures,
        "prohibitions": _unique_preserving_order(parsed["prohibitions"]),
        "scope": {
            "card_id": str(card["id"]),
            "domain": str(card["domain"]),
            "family": str(card.get("family", "")),
            "title": parsed["title"] or str(card.get("family", "workflow")),
        },
    }
    reasons: list[str] = []
    if not contract["trigger"]:
        reasons.append("missing_trigger")
    if not contract["verification_rules"]:
        reasons.append("missing_verification_rules")
    if not contract["prohibitions"]:
        reasons.append("missing_prohibitions")
    if not set(reads + writes).issubset(mentioned):
        reasons.append("unbound_tool")
    original_writes = mentioned & WRITE_TOOLS
    if not set(writes).issubset(original_writes):
        reasons.append("new_write_action")
    if set(contract["prohibitions"]) != set(parsed["prohibitions"]):
        reasons.append("prohibition_coverage")

    valid = not reasons
    primary = _render_contract(contract, secondary=False) if valid else source_text
    secondary = _render_contract(contract, secondary=True) if valid else source_text
    if valid:
        for tool in reads + writes:
            if tool not in primary:
                reasons.append(f"missing_primary_binding:{tool}")
        for prohibition in contract["prohibitions"]:
            if prohibition not in primary or prohibition not in secondary:
                reasons.append("rendered_prohibition_coverage")
                break
        for disclosure in contract["required_disclosures"]:
            if disclosure not in primary or disclosure not in secondary:
                reasons.append("rendered_disclosure_coverage")
                break
        if len(primary) > MAX_RENDER_CHARS:
            reasons.append("rendered_primary_too_long")
        if len(secondary) > MAX_RENDER_CHARS:
            reasons.append("rendered_secondary_too_long")
        if reasons:
            valid = False
            primary = source_text
            secondary = source_text

    return {
        "contract": contract,
        "compiler": {
            "valid": valid,
            "fallback_to_base_card": not valid,
            "reasons": reasons,
            "checks": {
                "tool_binding": "passed" if not any("binding" in reason for reason in reasons) else "failed",
                "write_subset": "passed" if "new_write_action" not in reasons else "failed",
                "disclosure_coverage": "passed" if not any("disclosure" in reason for reason in reasons) else "failed",
                "prohibition_coverage": "passed" if not any("prohibition" in reason for reason in reasons) else "failed",
                "length_bound": "passed" if not any("too_long" in reason for reason in reasons) else "failed",
                "variable_binding": "not_applicable_no_free_variables",
            },
        },
        "primary_text": primary,
        "secondary_text": secondary,
        "packing": {
            "source_chars": len(source_text),
            "primary_chars": len(primary),
            "secondary_chars": len(secondary),
            "primary_reduction": round(1.0 - len(primary) / max(len(source_text), 1), 6),
            "secondary_reduction": round(1.0 - len(secondary) / max(len(source_text), 1), 6),
        },
    }


def _state_phase(tools: Sequence[str], index: int) -> str:
    if index >= len(tools):
        return "final"
    if any(tool in WRITE_TOOLS for tool in tools[:index]):
        return "postwrite"
    if index > 0:
        return "prewrite"
    return "read"


def _state_key(domain: str, family: str, suffix: Sequence[str], phase: str) -> str:
    """Return the canonical, collision-free State-Q lookup key."""

    return f"{domain}|{family}|{'+'.join(suffix) if suffix else 'none'}|{phase}"


def _beta_mean(values: Sequence[float]) -> float:
    return (BETA_ALPHA + sum(values)) / (BETA_ALPHA + BETA_BETA + len(values))


def _build_utility(
    trajectories: Mapping[str, Sequence[Trajectory]],
    cards: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    availability: dict[str, Any] = {}
    cards_by_domain_family: dict[tuple[str, str], list[str]] = defaultdict(list)
    for card in cards:
        cards_by_domain_family[(str(card["domain"]), str(card.get("family", "")))].append(str(card["id"]))

    for domain in DOMAINS:
        records = trajectories[domain]
        scored = [record.utility for record in records if record.utility is not None]
        domain_prior = _beta_mean(scored) if scored else NEUTRAL_UTILITY
        availability[domain] = {
            "trajectories": len(records),
            "scored_trajectories": len(scored),
            "coverage": round(len(scored) / max(len(records), 1), 6),
            "utility_source": "embedded_trajectory_scores" if scored else "unavailable",
            "fallback": None if scored else "neutral_prior_no_embedded_scores",
        }
        matching_records: dict[str, list[Trajectory]] = defaultdict(list)
        state_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        state_support: dict[tuple[str, str], int] = defaultdict(int)
        for record in records:
            card_ids = cards_by_domain_family.get((domain, record.family), [])
            for card_id in card_ids:
                matching_records[card_id].append(record)
                if record.utility is None:
                    continue
                for index in range(len(record.tools) + 1):
                    suffix = record.tools[max(0, index - 2) : index]
                    key = _state_key(domain, record.family, suffix, _state_phase(record.tools, index))
                    state_values[(card_id, key)].append(record.utility)
                    state_support[(card_id, key)] += 1

        for card in cards:
            if card["domain"] != domain:
                continue
            card_id = str(card["id"])
            exposures = matching_records.get(card_id, [])
            card_scores = [record.utility for record in exposures if record.utility is not None]
            if len(card_scores) >= MIN_UTILITY_EXPOSURES:
                card_prior = _beta_mean(card_scores)
                fallback = None
            else:
                card_prior = domain_prior
                fallback = "domain_prior_insufficient_scored_exposures" if scored else "neutral_prior_no_embedded_scores"
            priors: dict[str, float] = {}
            supports: dict[str, int] = {}
            for (candidate_id, key), values in sorted(state_values.items()):
                if candidate_id != card_id or len(values) < MIN_UTILITY_EXPOSURES:
                    continue
                priors[key] = round(_beta_mean(values), 6)
                supports[key] = state_support[(candidate_id, key)]
            result[card_id] = {
                "utility_source": availability[domain]["utility_source"],
                "neutral_prior": NEUTRAL_UTILITY,
                "domain_prior": round(domain_prior, 6),
                "card_prior": round(card_prior, 6),
                "state_priors": priors,
                "state_prior_support": supports,
                "exposures": len(exposures),
                "scored_exposures": len(card_scores),
                "coverage": round(len(card_scores) / max(len(exposures), 1), 6),
                "fallback": fallback,
            }
    return result, availability


def _make_examples(records: Sequence[Trajectory], optimizer_ids: set[str]) -> list[Example]:
    examples: list[Example] = []
    for record in records:
        if record.task_id not in optimizer_ids:
            continue
        examples.append(
            Example(
                task_id=record.task_id,
                domain=record.domain,
                intent_text=record.request,
                query=record.request,
                family=record.family,
                target_writes=record.writes,
                observed_suffix=(),
                phase="read",
            )
        )
        first_write = next(
            (index for index, tool in enumerate(record.tools) if tool in WRITE_TOOLS),
            len(record.tools),
        )
        read_prefix_end = first_write
        if read_prefix_end:
            suffix = record.tools[max(0, read_prefix_end - 2) : read_prefix_end]
            examples.append(
                Example(
                    task_id=record.task_id,
                    domain=record.domain,
                    intent_text=record.request,
                    query=f"{record.request} {' '.join(suffix)}",
                    family=record.family,
                    target_writes=record.writes,
                    observed_suffix=tuple(suffix),
                    phase="prewrite",
                )
            )
        if first_write < len(record.tools):
            postwrite_end = first_write + 1
            suffix = record.tools[max(0, postwrite_end - 2) : postwrite_end]
            examples.append(
                Example(
                    task_id=record.task_id,
                    domain=record.domain,
                    intent_text=record.request,
                    query=f"{record.request} {' '.join(suffix)}",
                    family=record.family,
                    target_writes=record.writes,
                    observed_suffix=tuple(suffix),
                    phase="postwrite",
                )
            )
    return examples


def _runtime_adapter(
    domain: str,
    base_cards: Sequence[Mapping[str, Any]],
    compiled: Mapping[str, Mapping[str, Any]],
    utility: Mapping[str, Mapping[str, Any]],
) -> Any:
    """Construct a no-client runtime scorer for calibration.

    The offline grid deliberately calls the production agent's scoring,
    coverage, MMR, adaptive-cardinality and typed-rendering methods.  This
    prevents the calibration implementation from drifting from online behavior.
    """

    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from agents.risk_aware_process_workflow_memory_agent import (
        RiskAwareProcessWorkflowMemoryAgent as RuntimeRouter,
    )

    router = RuntimeRouter.__new__(RuntimeRouter)
    router._runtime_domain = domain
    router._cards = [dict(card) for card in base_cards if card.get("domain") == domain]
    router._router_cards = {
        str(card["id"]): {
            "domain": domain,
            "family": str(card.get("family", "")),
            **compiled[str(card["id"])],
            "utility": utility[str(card["id"])],
        }
        for card in router._cards
    }
    router._router = {}
    router._router_stage = "C"
    router.mode = "hybrid"
    router._active_card_ids = ()
    router._active_intent_signature = ()
    router._weights = {"field": 0.0, "utility": 0.0, "risk": 0.0, "trace": 0.0, "mmr": 0.0}
    router._thresholds = {
        "near_tie": 2.0,
        "candidate_pool": 12.0,
        "max_cards": 3.0,
        "default_cards": 1.0,
        "min_relevance": 0.20,
        "min_secondary_score": 0.75,
        "secondary_relative_score": 0.55,
        "same_family_limit": 2.0,
        "duplicate_jaccard": 0.80,
        "utility_cap": 0.75,
        "min_utility_exposures": float(MIN_UTILITY_EXPOSURES),
        "stickiness": 0.20,
    }
    router._document_frequency = Counter(
        token for card in router._cards for token in set(card.get("tokens", []))
    )
    router._avg_len = sum(len(card.get("tokens", [])) for card in router._cards) / max(
        len(router._cards), 1
    )
    router._card_ngrams = [_char_ngrams(str(card.get("search_text", ""))) for card in router._cards]
    router._prepare_field_statistics()
    router._prepare_quality_statistics()
    _memoize_runtime_invariants(router)
    return router


def _memoize_runtime_invariants(router: Any) -> None:
    """Memoize production-method results that cannot vary across grid weights.

    Every cache miss delegates to the production router method.  This preserves
    runtime behavior while avoiding millions of repeated token/Jaccard passes
    during five-fold calibration.
    """

    original_similarity = router._candidate_similarity
    similarity_cache: dict[tuple[str, str], float] = {}

    def candidate_similarity(left: Any, right: Any) -> float:
        key = tuple(sorted((left.card_id, right.card_id)))
        if key not in similarity_cache:
            similarity_cache[key] = original_similarity(left, right)
        return similarity_cache[key]

    original_eligible = router._eligible_candidate
    eligible_cache: dict[tuple[str, tuple[str, ...]], bool] = {}

    def eligible(candidate: Any, selected: Sequence[Any], family_counts: Counter[str]) -> bool:
        key = (candidate.card_id, tuple(item.card_id for item in selected))
        if key not in eligible_cache:
            eligible_cache[key] = original_eligible(candidate, selected, family_counts)
        return eligible_cache[key]

    original_needs = router._query_needs
    needs_cache: dict[tuple[str, tuple[str, ...], tuple[str, ...]], set[str]] = {}

    def query_needs(
        query: str,
        context: Mapping[str, Any],
        pool: Sequence[Any],
    ) -> set[str]:
        key = (
            query,
            tuple(context.get("intents", ())),
            tuple(item.card_id for item in pool),
        )
        if key not in needs_cache:
            needs_cache[key] = set(original_needs(query, context, pool))
        return set(needs_cache[key])

    original_evidence = router._has_relevance_evidence
    evidence_cache: dict[tuple[str, tuple[str, ...], str], bool] = {}

    def relevance_evidence(
        query: str,
        context: Mapping[str, Any],
        anchor: Any,
    ) -> bool:
        key = (query, tuple(context.get("intents", ())), anchor.card_id)
        if key not in evidence_cache:
            evidence_cache[key] = original_evidence(query, context, anchor)
        return evidence_cache[key]

    original_render = router._render_card
    render_cache: dict[tuple[Any, ...], str] = {}

    def render_card(
        candidate: Any,
        *,
        role: str,
        query: str,
        context: Mapping[str, Any],
    ) -> str:
        key = (
            candidate.card_id,
            role,
            query,
            tuple(context.get("intents", ())),
            tuple(context.get("observed_tools", ())),
            str(context.get("phase", "read")),
        )
        if key not in render_cache:
            render_cache[key] = original_render(
                candidate,
                role=role,
                query=query,
                context=context,
            )
        return render_cache[key]

    router._candidate_similarity = candidate_similarity
    router._eligible_candidate = eligible
    router._query_needs = query_needs
    router._has_relevance_evidence = relevance_evidence
    router._render_card = render_card


def _runtime_context(router: Any, example: Example) -> dict[str, Any]:
    intents = tuple(sorted(router._intent_matches(example.intent_text)))
    signature = intents or tuple(sorted(set(_tokens(example.intent_text)))[:12])
    return {
        "intent_text": example.intent_text,
        "intents": intents,
        "intent_signature": signature,
        "observed_tools": example.observed_suffix,
        "phase": example.phase,
    }


def _prepare_runtime_cache(
    router: Any,
    examples: Sequence[Example],
) -> list[tuple[Example, dict[str, Any], tuple[Any, ...]]]:
    prepared = []
    for example in examples:
        context = _runtime_context(router, example)
        prepared.append((example, context, tuple(router._rank_candidates(example.query, context))))
    return prepared


def _evaluate_config(
    router: Any,
    prepared: Sequence[tuple[Example, Mapping[str, Any], Sequence[Any]]],
    config: Mapping[str, float],
) -> dict[str, float]:
    router._weights = {
        key: float(config[key]) for key in ("field", "utility", "risk", "trace", "mmr")
    }
    router._thresholds["near_tie"] = float(config["near_tie"])
    totals: Counter[str] = Counter()
    for example, context, cached_pool in prepared:
        pool = [
            replace(
                candidate,
                final=(
                    candidate.semantic
                    + router._weights["field"] * candidate.field_score
                    + router._weights["utility"] * candidate.utility
                    - router._weights["risk"] * candidate.risk
                    + router._weights["trace"] * candidate.trace
                ),
                adjusted=0.0,
            )
            for candidate in cached_pool
        ]
        selected = router._select_candidates(example.query, context, pool, 3)
        rendered = [
            router._render_card(
                candidate,
                role="primary" if index == 0 else "secondary",
                query=example.query,
                context=context,
            )
            for index, candidate in enumerate(selected)
        ]
        totals["examples"] += 1
        totals["abstain"] += int(not selected)
        totals["recall1"] += int(bool(selected) and selected[0].family == example.family)
        totals["recall3"] += int(any(candidate.family == example.family for candidate in selected))
        selected_writes = (
            set().union(*(router._write_tools(candidate.sidecar) for candidate in selected))
            if selected
            else set()
        )
        if example.target_writes:
            totals["write_recall"] += len(selected_writes & set(example.target_writes)) / len(example.target_writes)
        else:
            totals["write_recall"] += float(not selected_writes)
        totals["extra_write"] += len(selected_writes - set(example.target_writes)) / max(len(selected_writes), 1)
        totals["cards"] += len(selected)
        totals["chars"] += sum(map(len, rendered))
    count = max(totals["examples"], 1)
    metrics = {
        "family_recall_at_1": totals["recall1"] / count,
        "family_recall_at_3": totals["recall3"] / count,
        "write_recall": totals["write_recall"] / count,
        "extra_write_exposure": totals["extra_write"] / count,
        "abstain_rate": totals["abstain"] / count,
        "mean_cards": totals["cards"] / count,
        "mean_injected_chars": totals["chars"] / count,
    }
    metrics["objective"] = (
        0.55 * metrics["family_recall_at_1"]
        + 0.20 * metrics["family_recall_at_3"]
        + 0.15 * metrics["write_recall"]
        - 0.08 * metrics["extra_write_exposure"]
        - 0.02 * min(metrics["mean_injected_chars"] / 6600.0, 1.0)
    )
    return {key: round(value, 6) for key, value in metrics.items()}


def _grid_configs() -> Iterable[dict[str, float]]:
    keys = tuple(GRID)
    for values in itertools.product(*(GRID[key] for key in keys)):
        yield dict(zip(keys, values))


def _mean_metrics(metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted(set().union(*(item.keys() for item in metrics)))
    return {
        key: round(sum(float(item.get(key, 0.0)) for item in metrics) / max(len(metrics), 1), 6)
        for key in keys
    }


def _cv_select_runtime(
    domain: str,
    examples: Sequence[Example],
    base_cards: Sequence[Mapping[str, Any]],
    compiled: Mapping[str, Mapping[str, Any]],
    utility: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    router = _runtime_adapter(domain, base_cards, compiled, utility)
    prepared = _prepare_runtime_cache(router, examples)
    folds: list[list[tuple[Example, Mapping[str, Any], Sequence[Any]]]] = [
        [] for _ in range(5)
    ]
    for example, context, pool in prepared:
        digest = hashlib.sha256(f"{CV_SEED}|{domain}|{example.task_id}".encode("utf-8")).digest()
        folds[int.from_bytes(digest[:4], "big") % 5].append((example, context, pool))
    if any(not fold for fold in folds):
        raise ValueError(f"Five-fold hash split produced an empty fold for {domain}")

    best_config: dict[str, float] | None = None
    best_metrics: dict[str, float] | None = None
    best_fold_metrics: list[dict[str, float]] | None = None
    best_key: tuple[float, ...] | None = None
    for config in _grid_configs():
        fold_metrics = [_evaluate_config(router, fold, config) for fold in folds]
        metrics = _mean_metrics(fold_metrics)
        distance = config["field"] + config["utility"] + config["risk"] + config["trace"] + config["mmr"]
        key = (
            metrics["objective"],
            -distance,
            -metrics["mean_cards"],
            -config["utility"],
            -config["field"],
            -config["risk"],
            -config["trace"],
            -config["mmr"],
            -abs(config["near_tie"] - 2.0),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_config = dict(config)
            best_metrics = metrics
            best_fold_metrics = fold_metrics
    assert best_config is not None and best_metrics is not None and best_fold_metrics is not None
    report = {
        "method": "five_fold_sha256_macro_validation_runtime_isomorphic",
        "seed": CV_SEED,
        "grid_size": math.prod(len(values) for values in GRID.values()),
        "evaluated_configs": sum(1 for _ in _grid_configs()),
        "examples": len(examples),
        "tasks": len({example.task_id for example in examples}),
        "objective": "0.55*family_R1 + 0.20*family_R3 + 0.15*write_recall - 0.08*extra_write - 0.02*chars_cap",
        "selected_metrics": best_metrics,
        "fold_metrics": best_fold_metrics,
        "feasibility_constraints": {
            "hard_anchor_margin": 2.0,
            "near_tie_grid": list(GRID["near_tie"]),
            "runtime_invariant_memoization": True,
            "runtime_components": [
                "semantic",
                "field_bm25",
                "utility",
                "write_exposure_risk",
                "tool_trace",
                "greedy_mmr",
                "adaptive_0_to_3",
                "typed_render_chars",
            ],
        },
        "tie_break": "cv_macro_objective_then_closest_to_baseline_then_fewer_cards_then_lower_utility",
    }
    return best_config, report


def _validate_source_tasks(
    source_memory: Mapping[str, Any],
    inventory_ids: Mapping[str, set[str]],
    splits: Mapping[str, Mapping[str, Sequence[str]]],
    memory_training_split: str,
    *,
    dev_manifest_sha256: str,
    read_manifest_sha256: str,
) -> dict[str, dict[str, list[str]]]:
    base_cards = source_memory["cards"]
    overlap: dict[str, dict[str, list[str]]] = {
        domain: {"dev": [], "lockbox": []} for domain in DOMAINS
    }
    expected_count = 100 if memory_training_split == "all" else 80
    stats = source_memory.get("stats")
    if not isinstance(stats, Mapping):
        raise ValueError("Source memory is missing per-domain trajectory-count provenance")
    for domain in DOMAINS:
        domain_stats = stats.get(domain)
        count = domain_stats.get("trajectories") if isinstance(domain_stats, Mapping) else None
        if count != expected_count:
            raise ValueError(
                f"Source memory {memory_training_split!r} scope requires {expected_count} "
                f"trajectories for {domain}, artifact declares {count!r}"
            )
    if memory_training_split == "optimizer":
        provenance = source_memory.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("optimizer memory requires strict train-selection provenance")
        if provenance.get("task_split") != "optimizer":
            raise ValueError("optimizer memory provenance must declare task_split=optimizer")
        if provenance.get("task_manifest_sha256") != dev_manifest_sha256:
            raise ValueError("optimizer memory was built from a different task-ID manifest")
        if provenance.get("trajectory_manifest_sha256") != read_manifest_sha256:
            raise ValueError("optimizer memory trajectory-content manifest does not match optimizer80")
        selected_counts = provenance.get("selected_counts")
        if not isinstance(selected_counts, Mapping) or any(
            selected_counts.get(domain) != 80 for domain in DOMAINS
        ):
            raise ValueError("optimizer memory provenance must declare exactly 80 selected tasks/domain")
        declared_splits = provenance.get("split_summary")
        if not isinstance(declared_splits, Mapping) or any(
            not isinstance(declared_splits.get(domain), Mapping)
            or {
                split: sorted(map(str, declared_splits[domain].get(split) or []))
                for split in ("dev", "lockbox", "optimizer")
            }
            != {
                split: sorted(map(str, splits[domain][split]))
                for split in ("dev", "lockbox", "optimizer")
            }
            for domain in DOMAINS
        ):
            raise ValueError("optimizer memory split provenance does not match router split")
    for card in base_cards:
        domain = str(card["domain"])
        sources = set(map(str, card.get("source_tasks") or []))
        missing = sources - inventory_ids[domain]
        if missing:
            raise ValueError(f"v1 card {card['id']} references non-train source task IDs: {sorted(missing)}")
        if memory_training_split == "optimizer":
            outside_optimizer = sources - set(splits[domain]["optimizer"])
            if outside_optimizer:
                raise ValueError(
                    f"optimizer memory card {card['id']} references held-out train IDs: "
                    f"{sorted(outside_optimizer)}"
                )
        for split in ("dev", "lockbox"):
            overlap[domain][split].extend(sorted(sources & set(splits[domain][split])))
    return {
        domain: {
            split: sorted(set(values))
            for split, values in split_values.items()
        }
        for domain, split_values in overlap.items()
    }


def _validate_artifact(artifact: Mapping[str, Any], source_cards: Sequence[Mapping[str, Any]]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected router schema version")
    if artifact.get("source_memory_sha256") != artifact.get("provenance", {}).get("memory_sha256"):
        raise ValueError("Source-memory hashes disagree")
    expected = {str(card["id"]) for card in source_cards}
    actual = set(artifact.get("cards", {}))
    if expected != actual:
        raise ValueError(f"Router card coverage mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    promoted = {
        domain for domain, config in artifact["domain_configs"].items() if config.get("promoted")
    }
    for domain in promoted:
        expected_domain = {str(card["id"]) for card in source_cards if card["domain"] == domain}
        if not expected_domain.issubset(actual):
            raise ValueError(f"Promoted domain lacks complete sidecar coverage: {domain}")
    provenance = artifact.get("provenance", {})
    if provenance.get("train_manifest_sha256") != provenance.get(
        "read_content_manifest_sha256"
    ):
        raise ValueError("Read-content manifest hashes disagree")
    expected_files = 300 if provenance.get("memory_training_split") == "all" else 240
    if provenance.get("train_file_count") != expected_files:
        raise ValueError(f"Unexpected number of parsed train files: {provenance.get('train_file_count')}")
    if provenance.get("memory_training_split") == "optimizer" and not provenance.get(
        "lockbox_independent"
    ):
        raise ValueError("optimizer router is not independent of dev/lockbox source content")
    source_by_id = {str(card["id"]): card for card in source_cards}
    for card_id, sidecar in artifact["cards"].items():
        source = source_by_id[card_id]
        if sidecar.get("source_card_sha256") != _sha256_bytes(_canonical_bytes(source)):
            raise ValueError(f"Source-card hash mismatch: {card_id}")
        scope = sidecar.get("contract", {}).get("scope", {})
        if scope.get("card_id") != card_id or scope.get("domain") != source.get("domain"):
            raise ValueError(f"Typed-card scope mismatch: {card_id}")
    for domain, config in artifact["domain_configs"].items():
        if config.get("thresholds", {}).get("near_tie") not in GRID["near_tie"]:
            raise ValueError(f"{domain} near_tie is outside the official calibration grid")
        for key, allowed in GRID.items():
            if key == "near_tie":
                continue
            if config.get("weights", {}).get(key) not in allowed:
                raise ValueError(f"{domain} weight {key} is outside the official calibration grid")
    forbidden = _contains_forbidden_key(artifact)
    if forbidden:
        raise ValueError(f"Forbidden oracle-like field in router artifact: {forbidden}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    state_bench_root = args.state_bench_root.resolve()
    state_bench_import_root = str(state_bench_root)
    if state_bench_import_root not in sys.path:
        sys.path.insert(0, state_bench_import_root)
    data_root = state_bench_root / "datasets" / "train_task_trajectories"
    source_memory, memory_hash, known_tools = _load_v1(args.v1_artifact.resolve())
    inventory, inventory_hash = _inventory_train_files(data_root)
    inventory_ids = {domain: set(inventory[domain]) for domain in DOMAINS}
    dev_ids, dev_metadata = _load_dev_manifest(args.dev_manifest.resolve(), inventory_ids)
    splits = _make_splits(inventory_ids, dev_ids)
    selected_ids = {
        domain: (
            inventory_ids[domain]
            if args.memory_training_split == "all"
            else set(splits[domain]["optimizer"])
        )
        for domain in DOMAINS
    }
    trajectories, train_manifest = _load_trajectories(
        inventory,
        known_tools,
        selected_ids,
    )
    source_task_overlap = _validate_source_tasks(
        source_memory,
        inventory_ids,
        splits,
        args.memory_training_split,
        dev_manifest_sha256=dev_metadata["sha256"],
        read_manifest_sha256=train_manifest["manifest_sha256"],
    )

    compiled = {
        str(card["id"]): _compile_card(card, known_tools[str(card["domain"])])
        for card in source_memory["cards"]
    }
    utility_records = {
        domain: [
            record
            for record in trajectories[domain]
            if args.memory_training_split == "all"
            or record.task_id in set(splits[domain]["optimizer"])
        ]
        for domain in DOMAINS
    }
    utility, utility_availability = _build_utility(utility_records, source_memory["cards"])

    selected_configs: dict[str, dict[str, float]] = {}
    cv_report: dict[str, Any] = {}
    for domain in DOMAINS:
        optimizer_ids = set(splits[domain]["optimizer"])
        examples = _make_examples(trajectories[domain], optimizer_ids)
        selected, report = _cv_select_runtime(
            domain,
            examples,
            source_memory["cards"],
            compiled,
            utility,
        )
        selected_configs[domain] = selected
        cv_report[domain] = report

    promoted_domains = set(args.promoted_domains)
    unknown_promoted = promoted_domains - set(DOMAINS)
    if unknown_promoted:
        raise ValueError(f"Unknown promoted domains: {sorted(unknown_promoted)}")
    defaults = {
        "weights": {"field": 0.25, "utility": 0.0, "risk": 0.25, "trace": 0.0, "mmr": 0.1},
        "thresholds": {
            "near_tie": 2.0,
            "candidate_pool": 12,
            "max_cards": 3,
            "default_cards": 1,
            "min_relevance": 0.20,
            "min_secondary_score": 0.75,
            "secondary_relative_score": 0.55,
            "same_family_limit": 2,
            "duplicate_jaccard": 0.80,
            "utility_cap": 0.75,
            "min_utility_exposures": MIN_UTILITY_EXPOSURES,
            "stickiness": 0.20,
        },
    }
    domain_configs: dict[str, Any] = {}
    for domain in DOMAINS:
        selected = selected_configs[domain]
        domain_configs[domain] = {
            "promoted": domain in promoted_domains,
            "weights": {
                "field": selected["field"],
                "utility": selected["utility"],
                "risk": selected["risk"],
                "trace": selected["trace"],
                "mmr": selected["mmr"],
            },
            "thresholds": {**defaults["thresholds"], "near_tie": selected["near_tie"]},
        }

    router_cards: dict[str, Any] = {}
    for card in source_memory["cards"]:
        card_id = str(card["id"])
        router_cards[card_id] = {
            "domain": str(card["domain"]),
            "family": str(card.get("family", "")),
            "source_card_sha256": _sha256_bytes(_canonical_bytes(card)),
            **compiled[card_id],
            "utility": utility[card_id],
        }

    script_hash = _sha256_file(Path(__file__).resolve())
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": "selective-decision-aware-process-workflow-memory",
        "source_memory_sha256": memory_hash,
        "provenance": {
            "learning_inputs": [
                "datasets/train_task_trajectories/<domain>/*.json",
                "v1 process_workflows.json",
            ],
            "split_metadata_input": "legacy development-panel task-ID manifest",
            "excluded_sources": [
                "task_definitions",
                "requirements",
                "test_tasks",
                "test_environments",
                "test_judge_reasoning",
                "test_outputs",
            ],
            "train_manifest_sha256": train_manifest["manifest_sha256"],
            "train_inventory_sha256": inventory_hash,
            "read_content_manifest_sha256": train_manifest["manifest_sha256"],
            "train_file_count": train_manifest["file_count"],
            "memory_sha256": memory_hash,
            "dev_manifest_sha256": dev_metadata["sha256"],
            "builder_sha256": script_hash,
            "split_seed": SPLIT_SEED,
            "cv_seed": CV_SEED,
            "memory_training_split": args.memory_training_split,
            "memory_trajectory_counts": {
                domain: int(source_memory["stats"][domain]["trajectories"])
                for domain in DOMAINS
            },
            "lockbox_independent": args.memory_training_split == "optimizer"
            and all(
                not source_task_overlap[domain]["dev"]
                and not source_task_overlap[domain]["lockbox"]
                for domain in DOMAINS
            ),
            "source_task_overlap": source_task_overlap,
            "utility_availability": utility_availability,
            "api_calls": 0,
        },
        "splits": splits,
        "defaults": defaults,
        "domain_configs": domain_configs,
        "cards": router_cards,
        "cv_report": cv_report,
    }
    _validate_artifact(artifact, source_memory["cards"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-bench-root",
        type=Path,
        default=Path("../STATE-Bench"),
        help="STATE-Bench root; only datasets/train_task_trajectories is read",
    )
    parser.add_argument(
        "--v1-artifact",
        type=Path,
        default=Path("artifacts/statebench_cross_domain_pwm/memory/process_workflows.json"),
    )
    parser.add_argument(
        "--memory-training-split",
        choices=("all", "optimizer"),
        default="all",
        help=(
            "Declared trajectory scope of the v1 memory. Use optimizer with "
            "process_workflows_optimizer80.json; held-out source IDs fail closed."
        ),
    )
    parser.add_argument(
        "--dev-manifest",
        type=Path,
        default=Path("configs/workflow_router_dev_ids.json"),
        help="Legacy 10-per-domain development task-ID manifest (split metadata only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/statebench_cross_domain_pwm/memory/workflow_router_v2.json"),
    )
    parser.add_argument(
        "--promoted-domains",
        nargs="*",
        default=["shopping_assistant"],
        help="Domains enabled in the generated artifact; default: shopping_assistant",
    )
    args = parser.parse_args()
    artifact = build(args)
    valid = sum(1 for card in artifact["cards"].values() if card["compiler"]["valid"])
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema_version": artifact["schema_version"],
                "cards": len(artifact["cards"]),
                "compiler_valid": valid,
                "promoted_domains": [
                    domain for domain, config in artifact["domain_configs"].items() if config["promoted"]
                ],
                "utility_source": {
                    domain: details["utility_source"]
                    for domain, details in artifact["provenance"]["utility_availability"].items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
