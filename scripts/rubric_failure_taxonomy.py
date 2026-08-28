"""What shopping's rubric actually charges for, measured on train.

The 2x2 of obligation-injection arms (A2/A3/A4) all came out null on task
completion, and the reason was visible in the item counts rather than the pass
rate: 4.40 conjunctive items per task, gains and losses within a point of each
other per item. What that analysis never did was ask what the failing items *are*.

This script asks. It reads a scored run on the train split and sorts every
failing task_requirement into families by what the requirement text demands. The
families are not a guess at the taxonomy — they were read off the 97 failing
items in the PWM train run, and the classifier is deliberately keyword-shallow so
that a family's membership can be audited by eye against the printed examples.

Two numbers matter more than the family totals:

*Items per failed task.* The rubric is conjunctive, so a family that accounts for
half the failing items still buys nothing if every task it touches also fails an
item outside it. The report therefore counts tasks whose failures lie **entirely**
inside a family set — those are the tasks a fix could actually flip.

*Reachable pass@1.* Clearing one family lifts pass@1 only by the fully-covered
tasks, not by the item share.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Ordered: the first family whose pattern matches wins, so the more specific
# demand is listed before the general one it would otherwise fall into.
#
# The families collapsed to three super-families after the first pass, because
# the 32 items the fine-grained version left unclassified turned out to be the
# same three demands phrased without the keywords the patterns keyed on
# ("proactively", "specific number"). What separates them is not topic but which
# *act* the rubric charges for:
#
#   say    the agent must utter a fact, usually a figure it has to derive
#   order  an utterance and a write have to happen in a particular order, or a
#          write that was agreed to has to actually happen
#   hold   the agent must decline something: a substitute, a cause it cannot
#          support, a demand the customer pushes twice
#
# All three are properties of the say/ask/write triple. A Petri net whose
# transitions are tool names can express the write and nothing else, which is
# the structural claim this taxonomy exists to support.
FAMILIES: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "order_say_then_write",
        "an utterance and a state change have to be sequenced, or an agreed write never happened",
        re.compile(
            r"BEFORE (?:any |setting |committing|acting)|before adding|before the customer|"
            r"before any add_to_cart|before calling|before confirming|before making any|"
            r"waited for (?:the customer's )?explicit|explicit confirmation|received the customer's|"
            r"consent must precede|must NOT call add_to_cart before|Unilateral|"
            r"after (?:the )?customer(?:'s)? (?:confirm|consent|explicitly chose)|"
            r"Merely recommending .{0,40}without executing fails|"
            r"the disclosure must precede|must precede the write",
            re.I,
        ),
    ),
    (
        "hold_the_line",
        "the agent must decline: no fabricated cause, no silent substitute, no capitulation",
        re.compile(
            r"did NOT invent|did not invent|speculate|silent substitute|"
            r"must NOT (?:recommend or add|add or suggest|apply)|fabricat|"
            r"did not (?:apply|add or suggest|continue pushing)|held the line|"
            r"did NOT reverse|capitulat|must NOT call add_to_cart for .{0,12} at any point|"
            r"never named|never specified",
            re.I,
        ),
    ),
    (
        "say_the_number",
        "the agent must volunteer a fact, with a figure it has to derive correctly",
        re.compile(
            r"proactively|without the customer asking|agent-initiated|"
            r"with a specific|Generic .{0,80}fail|quantified amount|"
            r"must be in the message|Silent(?:ly)? (?:completion|adding|reporting|refreshing|"
            r"confirmation|substitution|knowing|picking)|"
            r"stated|told the customer|communicated|explicitly (?:named|explained|disclosed|"
            r"referenced|computed|surfaced|returned|stated)|disclosed|surfaced|mentioned|"
            r"quoted the correct|correct (?:updated |current |final )?total of|"
            r"used the .{0,20}rate|computed loyalty points|did NOT automatically restore|"
            r"provided at least one distinguishing|both facts must be stated|"
            r"all three elements must be in the offer|suggested adding",
            re.I,
        ),
    ),
    (
        "write_only",
        "purely a tool-call or final-state demand, no utterance involved",
        re.compile(
            r"^Agent called|^The agent|refreshed or reset|re-evaluated|Final cart must|"
            r"must equal exactly|Final .{0,40}must (?:show|be|not)|recognized",
            re.I,
        ),
    ),
]


def classify(requirement: str) -> str:
    for name, _, pattern in FAMILIES:
        if pattern.search(requirement):
            return name
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=0, help="print N examples per family")
    args = parser.parse_args()

    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.run.glob("*.json"))]
    records = [r for r in records if r.get("task_completion_pass") is not None]

    item_total: Counter[str] = Counter()
    item_fail: Counter[str] = Counter()
    fail_by_task: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[str]] = defaultdict(list)
    for record in records:
        for item in record.get("task_requirements_details") or []:
            family = classify(str(item.get("requirement") or ""))
            item_total[family] += 1
            if not item.get("passed"):
                item_fail[family] += 1
                fail_by_task[str(record["task_id"])].add(family)
                examples[family].append(
                    f"{record['task_id']}: {str(item.get('requirement'))[:150]}"
                )

    passes = sum(int(r["task_completion_pass"] or 0) for r in records)
    state_fail = [r for r in records if not int(r["state_requirements_met"] or 0)]
    task_fail = [r for r in records if not int(r["task_requirements_met"] or 0)]
    print(f"{len(records)} tasks   pass@1 {100 * passes / len(records):.1f}%   ")
    print(f"state✗ {len(state_fail)}   task✗ {len(task_fail)}   items {sum(item_total.values())}")
    print()

    print(f"{'family':32s} {'items':>6s} {'fail':>5s} {'fail%':>6s} {'tasks':>6s} {'solo':>5s}")
    order = [name for name, _, _ in FAMILIES] + ["other"]
    for family in order:
        if not item_total[family]:
            continue
        touched = [t for t, fams in fail_by_task.items() if family in fams]
        solo = [t for t, fams in fail_by_task.items() if fams == {family}]
        print(
            f"{family:32s} {item_total[family]:6d} {item_fail[family]:5d} "
            f"{100 * item_fail[family] / item_total[family]:6.1f} "
            f"{len(touched):6d} {len(solo):5d}"
        )
    print()

    # A fix is only worth its item share if it clears every failing item on some
    # task. "solo" above is the single-family version of that; this is the
    # cumulative version over the ranked family list.
    print("cumulative reachable pass@1 (fix families in descending item-fail order):")
    ranked = sorted(
        (f for f in order if item_fail[f]), key=lambda f: -item_fail[f]
    )
    fixed: set[str] = set()
    # A task only flips if its state requirements already hold.
    state_ok = {str(r["task_id"]) for r in records if int(r["state_requirements_met"] or 0)}
    for family in ranked:
        fixed.add(family)
        flipped = [
            task
            for task, fams in fail_by_task.items()
            if fams <= fixed and task in state_ok
        ]
        print(
            f"  +{family:32s} -> {100 * (passes + len(flipped)) / len(records):5.1f}%  "
            f"(+{len(flipped)} tasks)"
        )
    print()

    sizes = Counter(len(fams) for fams in fail_by_task.values())
    print("failing families per task:", dict(sorted(sizes.items())))
    counts = Counter()
    for record in records:
        n = sum(1 for i in record.get("task_requirements_details") or [] if not i.get("passed"))
        if n:
            counts[n] += 1
    print("failing items per task:  ", dict(sorted(counts.items())))

    if args.examples:
        for family in order:
            if not examples[family]:
                continue
            print(f"\n--- {family} ---")
            for line in examples[family][: args.examples]:
                print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
