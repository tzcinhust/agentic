"""Recover the official leaderboard table from its inlined JS array.

The leaderboard at https://microsoft.github.io/STATE-Bench/leaderboard/ builds its
table in the browser; there is no data.json to fetch (probed: data.json,
leaderboard.json, data/leaderboard.json, submissions.json — all 404). The numbers
live as ``const entries = [...]`` inside leaderboard/leaderboard.js, so the only
way to read them is to convert that object literal to JSON.

Two things make the conversion less trivial than a regex:

*Finding the end.* Slicing at the first ``\\n];`` cuts the array short — the file
has more than one such terminator — and json.loads then reports "Extra data"
because it parsed a complete value and found code after it. A bracket-depth scan
from the opening ``[`` is the only reading that cannot be fooled.

*Unquoted keys and JS literals.* Keys are bare identifiers, nested and inline
(``travel: { passAt1: ... }`` on one line), and a missing cost is written
``undefined``, which JSON has no word for.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BACKSLASH = chr(92)


def array_span(text: str, marker: str = "const entries = [") -> str:
    """The balanced ``[...]`` following ``marker``, string-aware."""
    head = text.index(marker)
    start = head + text[head:].index("[")
    depth = 0
    in_string = False
    escaped = False
    for offset, char in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif char == BACKSLASH:
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return text[start : offset + 1]
    raise ValueError("unbalanced array")


def to_json(literal: str) -> str:
    literal = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', literal)
    literal = re.sub(r"\bundefined\b", "null", literal)
    return re.sub(r",(\s*[}\]])", r"\1", literal)


def cell(value: object, spec: str, width: int) -> str:
    return format("n/a", f">{width}") if value is None else format(value, spec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--js", type=Path, default=Path("artifacts/leaderboard/leaderboard.js"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/leaderboard/entries.json"))
    parser.add_argument("--domain", default="shoppingAssistant")
    args = parser.parse_args()

    entries = json.loads(to_json(array_span(args.js.read_text(encoding="utf-8"))))
    args.out.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"{len(entries)} entries -> {args.out}")
    versions = sorted({str(entry.get("benchmarkVersion")) for entry in entries})
    print(f"benchmark versions: {', '.join(versions)}")
    print(f"keys: {', '.join(sorted(entries[0]))}\n")

    for track in sorted({str(entry.get("track")) for entry in entries}):
        rows = [entry for entry in entries if str(entry.get("track")) == track]
        print(f"===== track={track}  ({len(rows)}) =====")
        print(
            f"{'model / agent':46s} {'org':14s} {'ovr':>6s} "
            f"{'SHOP@1':>7s} {'p^5':>6s} {'UX':>5s} {'$/task':>8s} {'ver':>6s}"
        )
        for entry in sorted(
            rows,
            key=lambda row: -(row["metrics"]["domains"][args.domain]["passAt1"] or 0),
        ):
            metrics = entry["metrics"]
            shop = metrics["domains"][args.domain]
            label = entry.get("model", "?")
            if entry.get("agent"):
                label += f" + {entry['agent']}"
            print(
                f"{label[:45]:46s} {str(entry.get('organization'))[:13]:14s} "
                f"{cell(metrics.get('overallPassAt1'), '6.1f', 6)} "
                f"{cell(shop.get('passAt1'), '7.1f', 7)} "
                f"{cell(shop.get('passAt5'), '6.1f', 6)} "
                f"{cell(shop.get('meanUxScore'), '5.2f', 5)} "
                f"{cell(shop.get('costPerTask'), '8.4f', 8)} "
                f"{str(entry.get('benchmarkVersion')):>6s}"
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
