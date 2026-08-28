"""Mine act-typed policy obligations out of the fixed train trajectories.

PWM's miner has one alphabet: tool names. It turns each training trajectory into
a sequence of transitions, folds those into a Petri net, and writes cards that
describe which tool follows which. That representation can express a *write* and
nothing else, and the shopping rubric charges for writes least of all — on the
train split, the 234 rubric items that ask only for a write or a state fact fail
2.1% of the time, while the 206 items that ask the agent to *say* a derived
number, to say it *before* the write, or to *refuse* something fail 42-56%.

The obligations those items charge for are not missing from the training data.
They are sitting inside ``get_policies`` tool *results*, in prose:

    Disclosure: agents should mention points earned after any cart completion.
    Agent action rule: do NOT call set_shipping_option without an explicit
      customer choice ... the write call must come AFTER the customer names
      a specific option.
    No exceptions regardless of tier, promo, or reason.

Two independent reasons the process miner cannot reach them. First the alphabet:
a tool *result* is not a transition, so the text never enters a card. Second the
statistics: ``get_policies`` is called 43 times across 100 shopping trajectories
against ``search_products``'s 163, and the ``loyalty_points`` topic appears in
exactly 3 — so a frequency-based miner actively *suppresses* the one tool that
carries the obligations. This script is therefore existence-based: a topic
observed once contributes its full rule set, with no support threshold.

What it emits per topic is the rules split by which act they oblige, because
that is what the rubric grades:

    say      the agent must utter the fact, unprompted
    number   a figure has to be derived, and derived the documented way
    order    a write may only happen after the customer has named a value
    refuse   something must be declined: no stacking, no exceptions, no claim
    if       the applicability condition — context, not a duty

Only ``rules`` is read, never the sibling fields. Travel and customer_support
call ``get_policies`` with parameters (``cabin_class``, ``loyalty_tier``,
``route_type``), and the check below confirms what makes this safe: every topic
returns a byte-identical ``rules`` dict across all its observed parameterisations,
so the prose is invariant and only the flat numeric siblings vary. Those siblings
are dropped, so nothing tier-specific is ever memorised as if universal.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

# A read prefix names a tool that cannot change state. Used to decide whether a
# conversation has reached the point where an unconditional disclosure is due.
READ_PREFIXES = ("get_", "search_", "check_", "validate_", "list_", "find_")

# Ordered by rendering priority, not by match priority — a rule can carry several
# acts and every one that matches is recorded.
ACTS: list[tuple[str, re.Pattern[str], re.Pattern[str]]] = [
    (
        "say",
        re.compile(
            r"^(?:disclosure|proactive|informational|transparency|notice)\b|"
            r"surfaces? this to the customer|"
            r"should (?:proactively )?(?:mention|surface|state|tell|disclose|explain|check and offer)|"
            r"agents? should mention|confirm with (?:the )?customer|for transparency|"
            r"(?:tell|inform|notify) the customer|mention[a-z ]{0,20} to the customer",
            re.I,
        ),
        re.compile(r"disclos|proactive|informational|transparen|confirm_with|surface", re.I),
    ),
    (
        "order",
        re.compile(
            r"agent action rule|do NOT call \w+ without|must come AFTER|must precede|"
            r"before (?:calling|any |setting|committing)|requires the agent to submit|"
            r"pass \w+ parameter|requires? change_reason|use \w+ instead",
            re.I,
        ),
        re.compile(r"_required$|calculation_order|agent_computes|action_rule", re.I),
    ),
    (
        "refuse",
        re.compile(
            r"does NOT stack|not combinable|no exceptions|better of the two|"
            r"^exclusions?\b|not eligible|ineligible|not allowed|not returnable|"
            r"non-refundable|cannot|claim denied|deny the claim|voids? the claim|"
            r"NOT covered|not (?:cancellable|changeable)|do not (?:complete|reverse|file|auto)|"
            r"is not (?:allowed|available)|^limit\b|does not auto-apply|the cart does not|"
            r"^one \w+ (?:code|promo)|\bnot via\b",
            re.I,
        ),
        re.compile(r"exclusion|void|denied|ineligib|_only_constraint|limit|no_", re.I),
    ),
    (
        "number",
        re.compile(
            r"^(?:rate|cap|limit|deposit|timeline|discount|calculation|fee|amount|minimum|"
            r"window|refund|standard|express|next-day|platinum|gold|silver|standard)\b|"
            r"\d\s*%|\$\s*\d|\d+\s*points?|points? per dollar|rounded|multiplier|max\(|"
            r"minus|\bint\(|\d+\s*(?:-|\s)?(?:day|hour|week|business day|min)|"
            r"\d\+|\bgrants\b|qualif|override|\bfree\b|requires the agent to submit|"
            r"compute it from|component breakdown",
            re.I,
        ),
        re.compile(
            r"amount|fee|rate|cap|limit|calculation|multiplier|clawback|surcharge|"
            r"discount|compute|windows_by|percent|tier",
            re.I,
        ),
    ),
    (
        "if",
        re.compile(
            r"^(?:eligibility|detection|scope|enforcement|application|condition|stacking|"
            r"one-time|window)\b|applies (?:when|to)|is True|is False",
            re.I,
        ),
        re.compile(r"eligib|applies_when|detection|scope|condition|window|active$", re.I),
    ),
]

# "on every purchase", "after any cart completion" — a duty attached to an *act*
# rather than to a condition, so nothing the customer says will ever trigger it.
# Deliberately narrow: "regardless of tier" also reads as unconditional but the
# rule it sits in ("5+ items grants free standard shipping regardless of tier") is
# conditional on the cart, and pinning it would assert free shipping on 1-item
# carts.
UNCONDITIONAL = re.compile(r"\bon every\b|\bafter any\b|\beach time\b", re.I)

TOOL_NAME = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
FIELD_NAME = re.compile(r"\b[a-z_]+\.[a-z_]+\b")
STOP = frozenset(
    "the a an and or of to for in on at is are be if not no with by from as it its this that "
    "you your can may must should do does per when after before within any all each".split()
)


def walk(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    """Flatten a nested rules dict to ``dotted.path -> leaf text``."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk(value, path + (str(index),))
    else:
        yield ".".join(path), str(node)


def acts_of(path: str, text: str) -> list[str]:
    matched = [name for name, body, key in ACTS if body.search(text) or key.search(path)]
    # Everything is at least a condition; without the fallback a rule that only
    # states a fact would vanish from the artifact entirely.
    return matched or ["if"]


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOP and len(t) > 1]


def collect(root: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, set[str]]]:
    """domain -> topic -> {rules, summary, calls, trajectories, co_tools, arg_keys}.

    Second return value is every tool name observed per domain, which is where the
    write vocabulary comes from.
    """
    found: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    seen_tools: dict[str, set[str]] = defaultdict(set)
    variants: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in sorted(root.glob("*/*.json")):
        domain = path.parent.name
        document = json.loads(path.read_text(encoding="utf-8"))
        names: list[str] = []
        seen: dict[str, dict[str, Any]] = {}
        for message in document.get("conversation", []):
            for call in message.get("tool_calls") or []:
                names.append(str(call.get("name") or ""))
                if call.get("name") != "get_policies":
                    continue
                result = call.get("result")
                if not isinstance(result, dict) or not result.get("rules"):
                    continue
                topic = str(result.get("topic") or "")
                seen[topic] = result
                variants[(domain, topic)].add(json.dumps(result["rules"], sort_keys=True))
                entry = found[domain].setdefault(
                    topic,
                    {
                        "rules": result["rules"],
                        "summary": str(result.get("summary") or ""),
                        "calls": 0,
                        "trajectories": 0,
                        "co_tools": set(),
                        "arg_keys": set(),
                    },
                )
                entry["calls"] += 1
                entry["arg_keys"].update(str(k) for k in (call.get("arguments") or {}))
        seen_tools[domain].update(name for name in names if name)
        for topic in seen:
            found[domain][topic]["trajectories"] += 1
            found[domain][topic]["co_tools"].update(n for n in names if n and n != "get_policies")
    for (domain, topic), shapes in sorted(variants.items()):
        if len(shapes) > 1:
            # Would mean the prose itself is parameterised, and memorising it
            # would put tier-specific text in front of the wrong customer.
            raise SystemExit(
                f"{domain}/{topic}: rules differ across parameterisations "
                f"({len(shapes)} shapes) — refusing to memorise"
            )
    return found, seen_tools


def build(domain: str, topic: str, entry: dict[str, Any]) -> dict[str, Any]:
    obligations = []
    for path, text in walk(entry["rules"]):
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        obligations.append(
            {
                "act": acts_of(path, text),
                "path": path,
                "text": text,
                "unconditional": bool(UNCONDITIONAL.search(text)),
            }
        )
    prose = " ".join(item["text"] for item in obligations)
    blob = f"{topic} {entry['summary']} {prose}"
    named_tools = sorted(
        {name for name in TOOL_NAME.findall(prose) if name in entry["co_tools"]}
    )
    return {
        "domain": domain,
        "topic": topic,
        "summary": entry["summary"],
        "calls": entry["calls"],
        "trajectories": entry["trajectories"],
        "arg_keys": sorted(entry["arg_keys"] - {"topic"}),
        "obligations": obligations,
        # Tools the prose names *and* the trajectories confirm exist: a precise
        # trigger, unlike raw co-occurrence which in these logs is near-universal.
        "named_tools": named_tools,
        "named_fields": sorted(set(FIELD_NAME.findall(prose))),
        # An unconditional say/number duty has no trigger in the customer's words,
        # so the agent needs it surfaced by the act it attaches to, not the topic.
        "unconditional_say": any(
            item["unconditional"] and ("say" in item["act"] or "number" in item["act"])
            for item in obligations
        ),
        "write_tools": sorted(
            name for name in entry["co_tools"] if not name.startswith(READ_PREFIXES)
        ),
        "tokens": tokens(blob),
        "search_text": blob,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, default=Path("datasets/train_task_trajectories"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/statebench_cross_domain_pwm/memory/policy_obligations.json"),
    )
    parser.add_argument("--show", action="store_true", help="print every mined obligation")
    args = parser.parse_args()

    found, seen_tools = collect(args.trajectories)
    topics = [
        build(domain, topic, entry)
        for domain in sorted(found)
        for topic, entry in sorted(found[domain].items())
    ]
    # An unconditional duty fires on the *act*, and on a single-turn task the only
    # evidence the act is coming is the customer asking for it. So the trigger is
    # the vocabulary of the domain's own write tools, split on underscores:
    # "add it and apply the code" hits add_to_cart and apply_promo without any
    # tool having been called yet.
    write_terms = {
        domain: sorted(
            {
                part
                for name in names
                if not name.startswith(READ_PREFIXES)
                for part in name.split("_")
                if part not in {"to", "from", "the"} and len(part) > 2
            }
        )
        for domain, names in sorted(seen_tools.items())
    }

    per_act: Counter[str] = Counter()
    for card in topics:
        for item in card["obligations"]:
            per_act.update(item["act"])
    artifact = {
        "version": "1",
        "method": "act-typed policy obligations, existence-based (no support threshold)",
        "source": str(args.trajectories).replace("\\", "/"),
        "write_terms": write_terms,
        "topics": topics,
        "stats": {
            "topics": len(topics),
            "obligations": sum(len(card["obligations"]) for card in topics),
            "per_act": dict(per_act),
            "per_domain": {
                domain: sum(1 for card in topics if card["domain"] == domain)
                for domain in sorted({card["domain"] for card in topics})
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(topics)} topics -> {args.out}")
    print(f"acts: {dict(per_act)}")
    print()
    header = f"{'domain':20s} {'topic':22s} {'traj':>4s} {'obl':>4s} {'say':>4s} {'ord':>4s} {'ref':>4s} {'num':>4s} {'uncond':>7s}"
    print(header)
    for card in topics:
        counts = Counter(a for item in card["obligations"] for a in item["act"])
        print(
            f"{card['domain']:20s} {card['topic']:22s} {card['trajectories']:4d} "
            f"{len(card['obligations']):4d} {counts['say']:4d} {counts['order']:4d} "
            f"{counts['refuse']:4d} {counts['number']:4d} "
            f"{'yes' if card['unconditional_say'] else '-':>7s}"
        )
    if args.show:
        for card in topics:
            print(f"\n=== {card['domain']}/{card['topic']}  tools={card['named_tools']}")
            for item in card["obligations"]:
                print(f"  [{'+'.join(item['act']):<18s}] {item['text'][:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
