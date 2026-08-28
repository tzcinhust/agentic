"""Compile obligation probes from the sanctioned training trajectories.

Motivation
----------
STATE-Bench passes a task only when both ``state_requirements_met`` and
``task_requirements_met`` hold, and ``task_requirements`` is a conjunction of
rubric items. Many of those items are *communication obligations*: disclose a
policy, quantify it, or refuse to write without explicit consent. Memories that
summarise trajectories (AWM / PWM) are support-weighted, so an obligation
witnessed in one or two training traces gets abstracted away — precisely the
ones the conjunctive rubric binds on. ``brand_bundle`` appears in 2/100 shopping
traces; a single missed disclosure fails the whole task.

This script therefore mines by **existence, not frequency**: every distinct
``get_policies`` payload observed anywhere in training becomes a probe, weight
one, regardless of how rare it was.

What a probe contains
---------------------
Only what the agent cannot obtain at runtime:

* a **trigger** over tool-result fields, marked *latent* when the field is not
  exposed by the cart/booking view and needs a join (e.g. ``brand`` is absent
  from ``get_cart`` items and lives only in ``get_product_details``, so a brand
  bundle is undetectable without an agent-initiated lookup);
* the **channel** the duty discharges on — say / tool / ask / refuse;
* whether the duty must be **quantified**, and the signature terms that show it
  was discharged.

Policy prose is deliberately *not* the payload: the agent fetches it at runtime
via ``get_policies``. Memory carries the routing and the discipline, not the
content.

Compliance
----------
Input is restricted to ``datasets/train_task_trajectories/<domain>/``, the only
source permitted for offline learning extraction. Policy text is read from
``get_policies`` *tool results recorded inside those trajectories*, not from the
domain package. Topics training never fetched are reported as gaps rather than
filled in. No test task definition and no test environment is read.

Usage:
    uv run python scripts/build_obligation_probes.py --domain shopping_assistant
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# --- Role classification ---------------------------------------------------
# Policy rules are prose, so roles are read off content cues rather than a
# closed set of labelled prefixes: 37 of the shopping rules carry no label at
# all, and several labels ("Platinum:", "Express:") are quantifier tables.

CHANNEL_CUES: dict[str, tuple[str, ...]] = {
    "say": (
        r"agent surfaces",
        r"agents? should (?:mention|proactively|surface|explain|check|offer)",
        r"should proactively",
        r"\bdisclos",
        r"\bsurface(?:s|d)? (?:this|it) to the customer",
        r"gets the better of the two",
        r"\bmention(?:ed|s)? (?:to )?the customer",
        r"\binform the customer",
        r"\boffer (?:it|the|backorder|an? )",
    ),
    "ask": (
        r"do NOT call",
        r"must come AFTER",
        r"without an explicit",
        r"explicit customer (?:choice|consent|amount|approval)",
        r"only (?:after|once) the customer",
        r"\brequires change_reason\b",
        r"\bask the customer\b",
        r"\bconfirm with the customer\b",
    ),
    "refuse": (
        r"no exceptions",
        r"\bnot eligible\b",
        r"\bnot returnable\b",
        r"does NOT stack",
        r"\bat most one\b",
        r"\bmaximum \d",
        r"\bnot combinable\b",
        r"\bcannot\b",
        r"\bineligible\b",
        r"\bNOT covered\b",
        r"\bvoids? the claim\b",
        r"\bdeny the claim\b",
        r"\bnot (?:changeable|cancellable|refundable)\b",
        r"\bdo not (?:file|process|issue)\b",
    ),
    "tool": (
        r"applied automatically",
        r"auto-appl",
        r"the cart does not auto-apply",
        r"must be (?:called|applied)",
        r"requires the agent to submit",
        r"the env writes",
        r"fails state scoring",
        r"\bcompute it from\b",
        r"cannot be changed back",
    ),
}

TRIGGER_LABELS = ("eligibility", "detection", "window", "scope", "availability", "limit")
# Numeric thresholds that read as an entry condition rather than a price.
THRESHOLD = re.compile(r"\b\d+\+?\s+(?:items?|units?|days?)\b", re.I)
FIELD_REF = re.compile(r"\b(product|customer|cart|order|booking|reservation)\.([a-z_][a-z0-9_]*)\b")
NUMBER = re.compile(r"\d+(?:\.\d+)?\s?%|\$\s?\d[\d,]*(?:\.\d{2})?|\b\d+\s+(?:points?|units?|days?|items?)\b")
LABELLED = re.compile(r"^\s*([A-Za-z][A-Za-z \-]{2,24}?)\s*:\s*(.+)$", re.S)
WORD = re.compile(r"[a-z][a-z_]{3,}")
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# Keys that are response metadata rather than entity state; a rule mentioning
# these words is not naming a field to resolve.
NON_STATE_KEYS = frozenset(
    {
        "reason",
        "note",
        "error",
        "status",
        "summary",
        "rules",
        "topic",
        "description",
        "label",
        "option",
        "options",
        "eligibility",
        "name",
    }
)
# Fields whose value is a magnitude, so a duty that mentions them must be quantified.
MAGNITUDE_HINT = ("price", "total", "fee", "cost", "points", "amount", "discount", "quantity", "days")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _label_of(rule: str) -> tuple[str, str]:
    """Split ``"Eligibility: 2+ items ..."`` into ``("eligibility", "2+ items ...")``."""
    match = LABELLED.match(rule)
    if not match:
        return "", rule.strip()
    return match.group(1).strip().lower(), match.group(2).strip()


def flatten_rules(rules: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Normalise the three payload shapes the domains use into (label, text) pairs.

    ``shopping_assistant`` returns a list of self-labelled strings
    ("Eligibility: 2+ items ..."), while ``travel`` and ``customer_support``
    return a nested dict whose *key* is the discriminator
    ("economy_medical", "void_exclusions") and whose value carries the numbers.
    Nested keys are joined so the provenance of a rule stays readable.
    """
    pairs: list[tuple[str, str]] = []
    if isinstance(rules, dict):
        for key, value in rules.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            pairs.extend(flatten_rules(value, label))
    elif isinstance(rules, list):
        for item in rules:
            pairs.extend(flatten_rules(item, prefix))
    elif rules is not None:
        text = str(rules)
        if prefix:
            pairs.append((prefix.lower(), text))
        else:
            pairs.append(_label_of(text))
    return pairs


def _channels_of(text: str) -> list[str]:
    return sorted(
        channel
        for channel, cues in CHANNEL_CUES.items()
        if any(re.search(cue, text, re.I) for cue in cues)
    )


def _iter_tool_calls(conversation: list[dict[str, Any]]):
    for index, message in enumerate(conversation):
        for call in message.get("tool_calls") or []:
            yield index, call


def _assistant_text(message: dict[str, Any]) -> str:
    if message.get("role") not in (None, "assistant"):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


# --- Observation pass ------------------------------------------------------


READ_PREFIXES = ("get_", "search_", "validate_", "check_", "list_")


def _detect_view_tool(scan: dict[str, Any]) -> str | None:
    """The read tool exposing the mutable working set the task is scored on.

    Not the most-called read tool: in shopping that is ``get_product_details``,
    the catalog, and treating the catalog as the view would make ``brand`` look
    already-visible when in fact it is absent from the cart. The working set is
    instead the read tool whose fields overlap most with what the *write* tools
    return, since writes report the collection they just mutated.
    """
    written: set[str] = set()
    for name, fields in scan["tool_fields"].items():
        if not name.startswith(READ_PREFIXES):
            written |= fields
    best: tuple[int, int, str] | None = None
    for name in scan["tool_fields"]:
        if not name.startswith("get_") or name == "get_policies":
            continue
        fields = scan["tool_fields"][name] | scan["item_fields"].get(name, set())
        score = (len(fields & written), scan["tool_counts"][name], name)
        if best is None or score > best:
            best = score
    return best[2] if best and best[0] else None


def scan_domain(domain_dir: Path) -> dict[str, Any]:
    """One pass over the training traces collecting everything the probes need."""
    payloads: dict[str, dict[str, Any]] = {}
    witnesses: dict[str, set[str]] = defaultdict(set)
    tool_fields: dict[str, set[str]] = defaultdict(set)
    item_fields: dict[str, set[str]] = defaultdict(set)
    field_providers: dict[str, set[str]] = defaultdict(set)
    followups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    tool_counts: Counter[str] = Counter()

    for path in sorted(domain_dir.glob("*.json")):
        conversation = json.loads(path.read_text(encoding="utf-8")).get("conversation") or []
        for index, call in _iter_tool_calls(conversation):
            name = str(call.get("name"))
            tool_counts[name] += 1
            result = call.get("result")
            if isinstance(result, dict):
                tool_fields[name].update(result.keys())
                for key in result:
                    field_providers[key].add(name)
                for value in result.values():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        item_fields[name].update(value[0].keys())
                        for key in value[0]:
                            field_providers[key].add(name)
            if name != "get_policies" or not isinstance(result, dict):
                continue
            # Key the probe by the argument the agent has to pass, not the label
            # the response echoes back. On travel the two disagree: the tool
            # accepts ``cancel`` / ``hotel_cancel`` / ``loyalty`` and answers
            # with ``cancellation`` / ``hotel_cancellation`` / ``loyalty_points``.
            # Keying on the response produced three topics the tool will not
            # accept, so every line naming them asked for a call that could not
            # succeed — 355 of them across the reference trajectories.
            topic = str((call.get("arguments") or {}).get("topic") or result.get("topic") or "")
            if not topic:
                continue
            witnesses[topic].add(path.stem)
            if topic not in payloads or len(json.dumps(result)) > len(json.dumps(payloads[topic])):
                payloads[topic] = result
            # Assistant prose after the fetch: evidence the duty was discharged.
            for message in conversation[index : index + 4]:
                text = _assistant_text(message).strip()
                if text:
                    followups[topic][path.stem].append(text)

    return {
        "payloads": payloads,
        "witnesses": witnesses,
        "tool_fields": tool_fields,
        "item_fields": item_fields,
        "field_providers": field_providers,
        "followups": followups,
        "tool_counts": tool_counts,
        "trace_count": len(list(domain_dir.glob("*.json"))),
    }


# --- Probe construction ----------------------------------------------------


def _discharge_check(
    topic: str, quantifiers: list[str], field_providers: dict[str, set[str]]
) -> dict[str, Any]:
    """How to tell, from agent prose alone, that the duty was discharged.

    A term that is itself a state field is worthless as evidence: the agent
    prints "Brand: NovaShield" in any product listing, so matching on "brand"
    marks every listing as a brand-bundle disclosure. Only topic terms that are
    *not* field names count, and the number must be the policy's own quantifier
    rather than any number on the page.
    """
    distinctive = [
        part for part in topic.split("_") if len(part) > 3 and part not in field_providers
    ]
    exact = sorted({match.group(0).strip() for text in quantifiers for match in NUMBER.finditer(text)})
    return {
        "distinctive_terms": distinctive,
        "quantifier_mode": "exact" if exact else "computed",
        "quantifier_terms": exact,
    }


def _is_discharged(text: str, check: dict[str, Any]) -> bool:
    lowered = text.lower()
    if check["distinctive_terms"] and not any(
        term in lowered for term in check["distinctive_terms"]
    ):
        return False
    if check["quantifier_mode"] == "exact":
        return any(term.lower() in lowered for term in check["quantifier_terms"])
    return bool(NUMBER.search(text))


def _discharge_span(text: str, check: dict[str, Any]) -> str | None:
    """The narrowest sentence window that carries both a term and the number.

    Storing the whole assistant message hides whether the match was real — a
    product listing and a genuine disclosure look identical in the first 150
    characters. The span makes the evidence auditable and doubles as few-shot
    phrasing.
    """
    sentences = [part.strip() for part in SENTENCE.split(text) if part.strip()]
    for width in (1, 2, 3):
        for start in range(len(sentences) - width + 1):
            window = " ".join(sentences[start : start + width])
            if _is_discharged(window, check):
                return window
    return None


def build_probe(
    topic: str,
    payload: dict[str, Any],
    witnesses: set[str],
    view_fields: set[str],
    field_providers: dict[str, set[str]],
    view_tool: str | None,
    followups: dict[str, list[str]],
) -> dict[str, Any]:
    triggers: list[dict[str, Any]] = []
    quantifiers: list[str] = []
    duties: list[dict[str, Any]] = []
    fields: set[str] = set()

    pairs = flatten_rules(payload.get("rules"))
    summary = str(payload.get("summary") or "")
    if summary:
        pairs.append(("summary", summary))
    # travel returns the policy already specialised to the booking, e.g.
    # applicable_cabin_class="basic_economy" — those keys name the state the
    # obligation is conditioned on.
    context = {
        key[len("applicable_") :]: value
        for key, value in payload.items()
        if key.startswith("applicable_")
    }

    for label, body in pairs:
        rule = f"{label}: {body}" if label else body
        # Two ways a rule names state: a dotted reference ("product.in_stock"),
        # or bare prose ("2+ items sharing the same brand") whose noun happens to
        # be a field some tool returns. The second is what makes brand bundles
        # invisible from the cart view alone.
        dotted = {group[1] for group in FIELD_REF.findall(rule)}
        bare = {
            word
            for word in WORD.findall(body.lower())
            if word in field_providers and word not in NON_STATE_KEYS
        }
        refs = dotted | bare
        fields |= refs
        if label != "summary" and (
            dotted or label in TRIGGER_LABELS or THRESHOLD.search(body) or "_" in label
        ):
            triggers.append({"text": body, "fields": sorted(refs), "label": label or None})
        if NUMBER.search(body):
            quantifiers.append(body)
        for channel in _channels_of(rule):
            duties.append({"channel": channel, "text": body, "label": label or None})

    latent = sorted(field for field in fields if field not in view_fields)
    resolvers = {
        field: sorted(field_providers.get(field, set()) - {view_tool})
        for field in latent
        if field_providers.get(field)
    }
    # "2+ items sharing the same brand" gives the ledger a predicate it can
    # evaluate without reading prose: the working set must hold >= 2 entries
    # before the obligation can possibly apply.
    thresholds = [
        int(match.group(1))
        for trigger in triggers
        for match in [re.match(r"(\d+)\+?\s+(?:items?|units?)\b", trigger["text"].strip(), re.I)]
        if match
    ]
    item_threshold = min(thresholds) if thresholds else None
    channels = sorted({duty["channel"] for duty in duties})
    check = _discharge_check(topic, quantifiers, field_providers)
    # Require the term and the number to co-occur inside a narrow sentence
    # window, not merely somewhere in the same message.
    spans_by_trace = {
        trace: [span for span in (_discharge_span(text, check) for text in texts) if span]
        for trace, texts in followups.items()
    }
    discharging = {trace for trace, spans in spans_by_trace.items() if spans}
    exemplars = [span for spans in spans_by_trace.values() for span in spans]

    return {
        "topic": topic,
        "provenance": "train_trajectory_tool_result",
        "support": len(witnesses),
        "witness_traces": sorted(witnesses),
        "applicable_context": context,
        "rule_count": len(pairs),
        "triggers": triggers,
        "trigger_fields": sorted(fields),
        "latent_fields": latent,
        "resolvers": resolvers,
        "item_threshold": item_threshold,
        "duties": duties,
        "channels": channels,
        "requires_number": bool(quantifiers)
        or any(hint in field for field in latent for hint in MAGNITUDE_HINT),
        "discharge_check": check,
        "discharge_exemplars": exemplars[:3],
        "discharging_traces": sorted(discharging),
        "train_discharge_rate": round(len(discharging) / max(len(witnesses), 1), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile obligation probes from training trajectories")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument(
        "--trajectories-dir",
        type=str,
        default="datasets/train_task_trajectories",
        help="Root of the sanctioned training trajectories",
    )
    parser.add_argument(
        "--view-tool",
        type=str,
        default=None,
        help="Read-only collection view whose item fields define what is NOT latent "
        "(default: get_cart / get_reservations / get_tickets if present)",
    )
    parser.add_argument("--output", type=str, default=None, help="Default: artifacts/obligations/<domain>.json")
    args = parser.parse_args()

    domain_dir = Path(args.trajectories_dir) / args.domain
    if not domain_dir.is_dir():
        parser.error(f"No training trajectories at {domain_dir}")

    scan = scan_domain(domain_dir)
    tool_names = set(scan["tool_counts"])
    view_tool = args.view_tool
    if view_tool is None:
        view_tool = _detect_view_tool(scan)
    view_fields = set(scan["item_fields"].get(view_tool, set())) | set(scan["tool_fields"].get(view_tool, set()))

    probes = [
        build_probe(
            topic,
            scan["payloads"][topic],
            scan["witnesses"][topic],
            view_fields,
            scan["field_providers"],
            view_tool,
            scan["followups"][topic],
        )
        for topic in sorted(scan["payloads"])
    ]

    output = Path(args.output) if args.output else Path("artifacts/obligations") / f"{args.domain}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "domain": args.domain,
        "source": str(domain_dir),
        "trace_count": scan["trace_count"],
        "view_tool": view_tool,
        "view_fields": sorted(view_fields),
        "policy_fetch_counts": {
            topic: len(traces) for topic, traces in sorted(scan["witnesses"].items())
        },
        "probes": probes,
    }
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    channel_counts = Counter(channel for probe in probes for channel in probe["channels"])
    print(f"domain             : {args.domain}")
    print(f"training traces    : {scan['trace_count']}")
    print(f"get_policies calls : {scan['tool_counts'].get('get_policies', 0)}")
    print(f"policy topics seen : {len(probes)}")
    print(f"view tool          : {view_tool}  ({len(view_fields)} fields)")
    print(f"duty channels      : {dict(channel_counts)}")
    print(f"written            : {output}")
    print()
    header = f"{'topic':20s} {'sup':>4s} {'trig':>5s} {'latent':>22s} {'num':>4s} {'disch':>6s}  channels"
    print(header)
    print("-" * len(header))
    for probe in probes:
        latent = ",".join(field.split(".")[-1] for field in probe["latent_fields"]) or "-"
        print(
            f"{probe['topic']:20s} {probe['support']:4d} {len(probe['triggers']):5d} "
            f"{latent[:22]:>22s} {'Y' if probe['requires_number'] else 'n':>4s} "
            f"{probe['train_discharge_rate']:6.2f}  {','.join(probe['channels']) or '-'}"
        )


if __name__ == "__main__":
    main()
