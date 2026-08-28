"""Report epistemic gaps only: what the current view cannot decide.

This is the obligation ledger with its second half removed, and the removal is
the point.

The ledger it derives from did two things. It named latent state — a policy
field the mutable working set does not expose — and it also told the agent what
to *say* about each live obligation ("state it to the customer with the exact
figure, without being asked"). Measured against the same 49 tasks, those two
halves came apart cleanly. The naming half moved the agent's reads:
``get_product_details`` 45 -> 75, ``get_customer_account`` 18 -> 43, while writes
stayed flat. The telling half moved its prose, and the UX judge charged 0.70
points for it across 66 of 98 paired instances, along with 35% more tokens and
three new state failures where an ``ask`` instruction displaced a required write.

So this agent keeps the naming and drops the telling. It reports two kinds of
gap and nothing else:

*Unread policy.* A topic the case demonstrably turns on — the customer named it,
or a counting antecedent is provably satisfied over the joined working set — that
the agent has not fetched.

*Unresolved latent field.* A field some policy quantifies over that no tool
result has yet supplied. The sharp form of this is not omission but renaming:
cart entries carry ``unit_price`` and ``line_total``, never the ``price`` the
policies are written in, so an agent reasoning off the view never encounters the
identifier the rule uses. Where such a near-miss column exists the line names it,
because "you are looking at a different quantity" is more actionable than "you
are missing a field".

Both gaps extinguish on *reading*, never on speaking: a topic leaves the list
when ``get_policies`` returns it, a field when some result carries it. Nothing
here inspects the agent's own prose. That is what bounds the cost — the block
empties itself as the agent informs itself — and it is also the substantive
claim: close the information gap and leave the disclosure decision to the model,
rather than legislating the utterance.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agents.obligation_ledger_agent import ObligationLedgerAgent as _Parent

# Naming every live topic at once is what made the parent expensive: each named
# topic became a ``get_policies`` call, 302 of them across 50 tasks. Two per turn
# still surfaces up to sixteen over a full tool loop.
MAX_NAMED_TOPICS = int(os.environ.get("STATE_BENCH_LEDGER_MAX_TOPICS", "2"))
MAX_NAMED_FIELDS = 4
MAX_NAMED_TOOLS = 2

# A resolver resolves by observing. The miner reads a rule's field list and
# collects every tool whose schema mentions those fields, which sweeps in the
# tools that *write* them: reference behaviour is pointed at ``create_booking``
# 1543 times across travel and ``process_warranty_claim`` 569 times across
# customer_support, agreeing 19% and 2% of the time. Naming a mutating call as
# the way to learn a fact is useless at best, and at worst it is a nudge toward
# an unrequested write — the failure mode that cost three state requirements.
READ_PREFIXES = ("get_", "search_", "list_", "check_", "lookup_", "find_", "view_", "validate_")

# Whether a multi-word topic needs all of its words named before the customer
# counts as having raised it. Env-gated because it is a genuine trade-off rather
# than a bug fix, and the measurement that settles it is in the module docstring
# of scripts/replay_gold_agreement.py.
STRICT_TERMS = os.environ.get("STATE_BENCH_LEDGER_STRICT_TERMS", "1") != "0"

# The mined table says which policies exist and what they quantify over. It does
# not say which gaps are worth reporting, and the miner's support counts cannot
# say either, because support is a raw frequency: a topic the customer mentions
# in every conversation earns high support whether or not resolving it ever
# changes what the agent should do. Measured against reference behaviour the two
# come apart hard. travel's ``cancel`` line is open in most conversations and the
# reference fetches that policy in 10% of them; customer_support's gap line names
# ``cost`` and ``eligible``, which the reference obtains in 9 conversations out of
# 100. Those two domains are exactly the ones where the ledger never stops
# firing, at 93% and 95% of turns, while shopping — whose fields the reference
# resolves 76-80% of the time — fires on 43%.
#
# So the gate is an opportunity-normalised rate, fitted on train trajectories by
# scripts/fit_ledger_priors.py: of the conversations where this gap arose at all,
# in what fraction did the reference go and close it? Reporting a gap the
# reference habitually leaves open is not conservatism, it is a standing
# instruction to spend a tool call on something the task does not need.
#
# Read at load time rather than import time so the fitting script can turn the
# gate off, and the replay harness can point at a held-out prior, without
# depending on which module got imported first.
def _gate_config() -> dict[str, Any]:
    return {
        "enabled": os.environ.get("STATE_BENCH_LEDGER_PRIORS", "1") != "0",
        "suffix": os.environ.get("STATE_BENCH_LEDGER_PRIORS_SUFFIX", ""),
        "min_fetch": float(os.environ.get("STATE_BENCH_LEDGER_MIN_FETCH_RATE", "0.30")),
        "min_resolve": float(os.environ.get("STATE_BENCH_LEDGER_MIN_RESOLVE_RATE", "0.30")),
        # A rate estimated from a handful of conversations is not an estimate.
        # Below this many opportunities the gap keeps its slot rather than being
        # gated out on noise.
        "min_seen": int(os.environ.get("STATE_BENCH_LEDGER_MIN_OPPORTUNITIES", "5")),
    }


class LatentStateAgent(_Parent):
    """Surface unread policies and unresolved latent fields; say nothing else.

    The parent is aliased on import rather than bound under its own name because
    state_bench.agents.loader resolves ``--agent-class`` by importing every file
    under ./agents/ and rejecting a name it finds in more than one of them. A
    plain ``from ... import ObligationLedgerAgent`` makes that name an attribute
    of this module too, and the loader then refuses to load the parent at all —
    which silently costs the ablation arm, since a baseline you cannot re-run is
    not a baseline.
    """

    max_lines = int(os.environ.get("STATE_BENCH_LEDGER_MAX_LINES", "2"))

    # -- table loading --------------------------------------------------------

    def _load_obligations(self, domain: str | None) -> None:
        """Load the mined table, then gate it on the fitted prior.

        The two gaps are gated independently because they are independent claims.
        A topic whose policy the reference rarely fetches loses its unread line
        but keeps its fields; a field the reference rarely obtains is pruned from
        every probe that names it while those probes keep their unread lines.
        """
        super()._load_obligations(domain)
        self._silent_topics: set[str] = set()
        config = _gate_config()
        if not config["enabled"]:
            return
        stem = f"{domain}.priors"
        if config["suffix"]:
            stem += f".{config['suffix']}"
        path = self.obligations_dir / f"{stem}.json"
        if not path.is_file():
            return
        priors = json.loads(path.read_text(encoding="utf-8"))
        self._silent_topics = self._below(priors, "topic", config["min_fetch"], config["min_seen"])
        quiet = self._below(priors, "field", config["min_resolve"], config["min_seen"])
        self._pool = [
            {
                **probe,
                "resolvers": {
                    field: tools
                    for field, tools in (probe.get("resolvers") or {}).items()
                    if field not in quiet
                },
            }
            for probe in self._pool
        ]

    @staticmethod
    def _below(priors: dict[str, Any], kind: str, threshold: float, min_seen: int) -> set[str]:
        rates = priors.get(f"{kind}_rate") or {}
        seen = priors.get(f"{kind}_opportunities") or {}
        return {
            key
            for key, rate in rates.items()
            if rate < threshold and seen.get(key, 0) >= min_seen
        }

    # -- gap detection --------------------------------------------------------

    @staticmethod
    def _named(terms: list[str], asked: str) -> bool:
        """Whether the customer has named this topic specifically.

        A compound topic is more specific than its parts, and its parts may be
        another topic's whole name: travel carries both ``cancel`` and
        ``hotel_cancel``, so "cancel my flight" satisfies ``any`` for both even
        though only one applies. Requiring every term makes the compound topic
        cost what its extra word says it costs.
        """
        if STRICT_TERMS and len(terms) > 1:
            return all(term in asked for term in terms)
        return any(term in asked for term in terms)

    def _liveness(self, probe: dict[str, Any], state: dict[str, Any], asked: str) -> str | None:
        """Live only on positive evidence, never on the absence of ignorance.

        The parent promotes a probe to ``holds`` once every field its rule
        mentions has appeared in some tool result, on the reasoning that a latent
        obligation must not go silent at the moment it becomes evaluable. That
        reasoning was fitted to test recall, and reference behaviour refutes it.
        ``price_match`` resolves on the single field ``price``, which every
        product and cart result carries, so the promotion declared the rule live
        on 648 of 910 reference turns while the reference fetched that policy in
        1% of them; ``brand_bundle`` resolves on ``brand`` and ``category``, both
        returned by ``search_products``, for another 626.

        Evaluable is not applicable. Where the miner recovered a counting
        antecedent the predicate can establish applicability; where it did not,
        resolution establishes nothing, and the only defensible report is the
        *gap* — which extinguishes on its own once the fields arrive.
        """
        if probe.get("item_threshold") is not None and self._predicate(probe, state) == "holds":
            return "holds"
        if probe["topic"] in state["topics_fetched"]:
            return "holds"
        terms = (probe.get("discharge_check") or {}).get("distinctive_terms") or []
        if terms and self._named(terms, asked):
            return "holds"
        resolvers = probe.get("resolvers") or {}
        if resolvers and any(field not in state["resolved"] for field in resolvers):
            return "pending"
        return None

    def _view_fields(self, conversation: list[Any]) -> set[str]:
        """Column names the working-set view has actually returned.

        Needed to tell renaming from omission: without the view's own vocabulary
        there is no way to point at ``unit_price`` as the thing standing in for
        ``price``.
        """
        fields: set[str] = set()
        for name, result in self._tool_results(conversation):
            if name != self._view_tool or not isinstance(result, dict):
                continue
            for value in result.values():
                if isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict):
                            fields |= set(entry)
        return fields

    @staticmethod
    def _near_misses(wanted: list[str], available: set[str]) -> list[str]:
        """View columns that read like a policy field but are not it.

        Substring either way: ``unit_price`` contains ``price``, and a view that
        exposed ``price_cents`` would be caught the same way. An unrelated column
        such as ``line_total`` is correctly left out — it is a missing field, not
        a renamed one.
        """
        return sorted(
            column
            for column in available
            for field in wanted
            if column != field and (field in column or column in field)
        )

    def _gaps(self, conversation: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
        """Every open gap, before any gating or truncation.

        Split out from ``_ledger`` because the prior in
        ``scripts/fit_ledger_priors.py`` has to be fitted against raw detections:
        its denominator is "conversations where this gap arose at all", and a
        gap that arose but lost its slot to ``MAX_NAMED_TOPICS`` still arose.
        """
        state = self._observe(conversation)
        asked = self._role_text(conversation, "user").lower()
        unread: list[str] = []
        pending: list[dict[str, Any]] = []
        for probe in self._pool:
            verdict = self._liveness(probe, state, asked)
            if verdict is None:
                continue
            if verdict == "holds":
                # ``_liveness`` already reports "holds" for a fetched topic, so
                # anything left here is live *and* unread.
                if probe["topic"] not in state["topics_fetched"]:
                    unread.append(probe["topic"])
                continue
            fields = [
                field for field in (probe.get("resolvers") or {}) if field not in state["resolved"]
            ]
            if fields:
                pending.append({"probe": probe, "fields": fields})
        return unread, pending

    def _ledger(self, conversation: list[Any]) -> list[str]:
        """At most two lines, both purely epistemic.

        The parent emitted up to three imperative lines plus a join line and
        tracked, per topic, whether the agent had yet said the thing. None of
        that survives: a topic is either read or unread, a field either resolved
        or not, and both consolidate into a single line so the block cannot grow
        with the number of live obligations.
        """
        unread, pending = self._gaps(conversation)
        unread = [topic for topic in unread if topic not in self._silent_topics]
        lines: list[str] = []
        if unread:
            lines.append(self._unread_line(unread[:MAX_NAMED_TOPICS]))
        if pending and self._observe(conversation)["view_ids"]:
            lines.append(self._gap_line(pending, self._view_fields(conversation)))
        return lines[: self.max_lines]

    # -- line construction ---------------------------------------------------

    @staticmethod
    def _unread_line(topics: list[str]) -> str:
        listed = ", ".join(f"get_policies('{topic}')" for topic in topics)
        return (
            f"- this case turns on {'a policy' if len(topics) == 1 else 'policies'} you have not "
            f"read: {listed}."
        )

    def _gap_line(self, pending: list[dict[str, Any]], view_fields: set[str]) -> str:
        topics = ", ".join(entry["probe"]["topic"] for entry in pending)
        fields = sorted({field for entry in pending for field in entry["fields"]})
        tools = sorted(
            {
                tool
                for entry in pending
                for field in entry["fields"]
                for tool in (entry["probe"].get("resolvers") or {}).get(field, [])
                if tool.startswith(READ_PREFIXES)
            }
        )
        view = self._view_tool or "the working set"
        line = (
            f"- {view} has not supplied {', '.join(fields[:MAX_NAMED_FIELDS])}, which is what "
            f"decides {topics}."
        )
        near = self._near_misses(fields, view_fields)
        if near:
            line += f" Its {', '.join(near[:MAX_NAMED_FIELDS])} {'is' if len(near) == 1 else 'are'} a different quantity."
        if tools:
            named = " / ".join(tools[:MAX_NAMED_TOOLS])
            line += f" {named} {'supply' if len(tools[:MAX_NAMED_TOOLS]) > 1 else 'supplies'} them per entry."
        return line

    @staticmethod
    def _block(lines: list[str]) -> str:
        # Re-injected every tool round, so the preamble is charged once per round
        # and is kept to one sentence. The "not instructions" framing is the part
        # that must survive: it is what distinguishes this from the imperative
        # ledger whose disclosure demands cost 0.70 UX points.
        return (
            "Gaps in what you have read or resolved — your own state of "
            "knowledge, not instructions about what to say.\n" + "\n".join(lines)
        )
