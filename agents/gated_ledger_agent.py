"""Imperative disclosure, but only where the reference actually discloses.

This is the fourth cell of a 2x2 that the previous three filled in:

                    ungated                          gated
    imperative      A2: rubric +33/-24, UX -0.65,    this agent
                        tokens +35%, 3 state fails
    epistemic       (not run)                        A3: rubric +15/-23,
                                                         UX -0.26, tokens -4%

A2 told the agent what to say about every live obligation and paid for it: the UX
judge charged 0.70 points across 66 of 98 paired instances, tokens rose 35%, and
three ``ask``-channel lines converted a required write into a request for
permission. A3 removed the telling and kept only the naming; it recovered the
whole cost — tokens fell 4% *below* baseline, UX to -0.26, state failures to
zero — and delivered no task-level gain at all, 51.0% against 51.0%.

The rubric says why. Disclosure items are 15% of shopping's rubric and fail at
four times the rate of everything else: 59.7% for the baseline against 14.6% for
restraint items and 18.0% for action items. A2 is the only arm that moved them,
to 52.9%. A3 did not, and drifted to 64.7%. Reading is not the bottleneck — the
agent already reads enough. Saying is the bottleneck, and A3 removed the only
lever on it.

So the telling comes back, and what changes instead is *when*. A2's cost tracked
volume, not phrasing: 2.58 lines per turn on 91% of turns at 3% gold agreement.
Two things cut that volume without touching the imperative itself. Dropping the
parent's ``allresolved -> holds`` promotion (inherited from LatentStateAgent)
makes a "holds" verdict rare — 1 to 16 conversations in 100 per topic, against
the promotion's 648 of 910 turns for ``price_match`` alone. And a disclosure
prior, fitted on train by scripts/fit_ledger_priors.py, drops the topics the
reference declines to volunteer.

That prior is the one this agent needs and it is not the one A3 uses. A3 gates on
whether the reference *fetches* a policy; an imperative line demands that the
agent *quote* it, and the two diverge — shopping's ``shipping`` is fetched in 27%
of the conversations where it is unread but disclosed in 69% of the conversations
where it holds, so the fetch prior would silence exactly the wrong topic. The
disclosure prior asks the matching question, using the ledger's own
``_discharged`` test against reference prose.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agents.latent_state_agent import LatentStateAgent as _Parent
from agents.obligation_ledger_agent import (
    CHANNEL_INSTRUCTION,
    DEFAULT_INSTRUCTION,
    TOOL_NOTE,
)

# Disclosure is a different question from retrieval, so it gets its own
# threshold. Held at the same 0.30 as the retrieval gates for want of any reason
# to move it — the value was fixed before either was fitted.
MIN_DISCLOSE_RATE = float(os.environ.get("STATE_BENCH_LEDGER_MIN_DISCLOSE_RATE", "0.30"))

# The ``ask`` channel is the one measured harm in A2. It converts a write duty
# into a request for permission, and all three of A2's task-ok/state-fail
# instances were the ``loyalty_redemption`` probe doing exactly that. Reference
# behaviour does not settle it either — ask-then-write ran 6 against
# write-without-asking 6 on train, a coin flip — while the benchmark's state
# requirements need the write. So the channel is dropped and those probes fall
# back to plain disclosure.
DROPPED_CHANNELS = frozenset({"ask"})


class GatedLedgerAgent(_Parent):
    """Tell the agent what to disclose, on the few turns where that is warranted.

    Parent aliased on import for the reason given in latent_state_agent: a
    re-exported class name makes that arm unloadable.
    """

    # A2 allowed three imperative lines plus a join line. Two total, because the
    # thing that cost 0.70 UX points was how much the block asked for at once.
    max_lines = int(os.environ.get("STATE_BENCH_LEDGER_MAX_LINES", "2"))

    def _load_obligations(self, domain: str | None) -> None:
        super()._load_obligations(domain)
        self._quiet_disclosure: set[str] = set()
        config = _disclosure_gate()
        if not config["enabled"]:
            return
        stem = f"{domain}.priors"
        if config["suffix"]:
            stem += f".{config['suffix']}"
        path = self.obligations_dir / f"{stem}.json"
        if not path.is_file():
            return
        priors = json.loads(path.read_text(encoding="utf-8"))
        self._quiet_disclosure = self._below(
            priors, "disclose", MIN_DISCLOSE_RATE, config["min_seen"]
        )

    @staticmethod
    def _instruction(probe: dict[str, Any]) -> str:
        channels = set(probe.get("channels") or []) - DROPPED_CHANNELS
        text = next(
            (phrase for channel, phrase in CHANNEL_INSTRUCTION if channel in channels),
            DEFAULT_INSTRUCTION,
        )
        return text + (TOOL_NOTE if "tool" in channels else "")

    def _ledger(self, conversation: list[Any]) -> list[str]:
        """Imperative lines for decided obligations, then one epistemic line.

        Order matters: the disclosure demand is the part that earns rubric items,
        so it takes the slots first, and the field-gap line only fills a slot the
        imperative half left empty.
        """
        state = self._observe(conversation)
        said = self._role_text(conversation, "assistant")
        asked = self._role_text(conversation, "user").lower()
        decided: list[str] = []
        pending: list[dict[str, Any]] = []
        for probe in self._pool:
            verdict = self._liveness(probe, state, asked)
            if verdict is None or self._discharged(said, probe):
                continue
            unresolved = [
                field for field in (probe.get("resolvers") or {}) if field not in state["resolved"]
            ]
            if verdict == "pending" and unresolved:
                pending.append({"probe": probe, "fields": unresolved})
                continue
            topic = probe["topic"]
            if topic in self._quiet_disclosure:
                continue
            prefix = "" if topic in state["topics_fetched"] else f"call get_policies('{topic}'), then "
            decided.append(
                f"- {topic}: bears on this case and is still unsaid — "
                f"{prefix}{self._instruction(probe)}"
            )
            if len(decided) >= self.max_lines:
                break
        lines = decided[: self.max_lines]
        if pending and state["view_ids"] and len(lines) < self.max_lines:
            lines.append(self._gap_line(pending, self._view_fields(conversation)))
        return lines

    @staticmethod
    def _block(lines: list[str]) -> str:
        # The parent's preamble ran 57 words and was re-injected on every tool
        # round. Its force came from one clause — that an unsaid obligation fails
        # the task on its own — so that clause is what survives the cut.
        return (
            "Still-unsaid obligations for this conversation. Each is a rubric "
            "requirement on its own: meeting the customer's literal request while "
            "leaving one unsaid is an incomplete task. Quantify every figure.\n"
            + "\n".join(lines)
        )


def _disclosure_gate() -> dict[str, Any]:
    return {
        "enabled": os.environ.get("STATE_BENCH_LEDGER_PRIORS", "1") != "0",
        "suffix": os.environ.get("STATE_BENCH_LEDGER_PRIORS_SUFFIX", ""),
        "min_seen": int(os.environ.get("STATE_BENCH_LEDGER_MIN_OPPORTUNITIES", "5")),
    }
