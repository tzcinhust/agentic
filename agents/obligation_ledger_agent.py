"""Runtime obligation ledger on top of process workflow memory.

The archive's failure profile is lopsided: across 150 scored tasks, 39 fail the
rubric while satisfying the database state, and 4 fail the reverse. The agent
already does the right writes; it omits the things it was obliged to *say*. Two
mechanisms produce those omissions, and this agent addresses both without an
extra model call.

**Unresolved latent state.** A brand bundle needs two cart items to share a
brand, but ``get_cart`` items carry no brand — it lives only in
``get_product_details`` / ``search_products``. The working set does not merely
omit policy fields, it renames them: cart entries expose ``unit_price`` and
``line_total``, never the ``price`` that the policies quantify over. An agent
reasoning off the cart therefore never encounters the identifier the rule is
written in, cannot perform the join, and so cannot disclose. The ledger names
the missing field and the tool that supplies it.

**Undischarged duties.** Once the state is resolved, the obligation still has to
be stated *with its number*. The ledger tracks, per policy topic, whether the
agent's own prose has yet carried both the topic term and the quantity, and
keeps the outstanding ones in front of the model every turn.

Deliberately *not* done: blocking or rewriting tool calls. State requirements
are the part that currently works (4 failures against 39), and a gate that
stalls a write to force a disclosure would trade a working channel for a broken
one. The ledger only ever adds a checklist to the turn input, and it hangs off
``generate_next_turn`` rather than ``prepare_conversation`` so that it observes
tool results within the turn that produced them — see that method for why the
pre-turn hook cannot work.

Compliance: the ledger reads the live conversation and an offline artifact mined
from ``datasets/train_task_trajectories/``. It never reads
``runtime_context.task_requirements`` or ``state_requirements``, which the
harness makes available but which are the rubric itself.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent as _Parent

NUMBER = re.compile(r"\d+(?:\.\d+)?\s?%|\$\s?\d[\d,]*(?:\.\d{2})?|\b\d+\s+(?:points?|units?|days?|items?)\b")
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_LEDGER_LINES = 3
MAX_NAMED_FIELDS = 2
MAX_PENDING_TOPICS = 6
# Fields too coarse to carry a "N items share this value" count on their own.
COARSE_FIELDS = frozenset({"category"})

# How the duty is discharged, by channel. A probe whose duty cues were too weak
# to classify still carries a quantified rule, so plain disclosure is the default
# rather than a reason to drop it — travel and customer_support have no
# say-classified probes at all, and dropping the unclassified ones would leave
# the ledger inert on two of the three domains.
CHANNEL_INSTRUCTION: tuple[tuple[str, str], ...] = (
    ("say", "state it to the customer with the exact figure, without being asked"),
    ("refuse", "check the present case against it and tell the customer the determination and the figure"),
    ("ask", "get the customer's explicit go-ahead, quoting the figure, before you write"),
)
DEFAULT_INSTRUCTION = "state it to the customer with the exact figure"
TOOL_NOTE = " The figure must also reach the tool call, not only the prose."


class ObligationLedgerAgent(_Parent):
    """Track which policy obligations are live and which are still unspoken.

    The parent is aliased on import — as it aliases its own — because
    state_bench.agents.loader resolves ``--agent-class`` by scanning every file
    under ./agents/ and refusing a name that appears in more than one. Binding
    ``ProcessWorkflowMemoryAgent`` here made the *baseline* unloadable, which is
    the worst possible thing to break silently: an arm whose control cannot be
    re-run has no control.
    """

    obligations_dir = Path(
        os.environ.get("STATE_BENCH_OBLIGATIONS_PATH", "artifacts/obligations")
    )

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self._load_obligations(getattr(runtime_context, "domain", None))

    def _load_obligations(self, domain: str | None) -> None:
        """Read the mined obligation table. Split out so the offline replay
        harness can exercise the ledger without constructing an LLM client."""
        path = self.obligations_dir / f"{domain}.json"
        artifact = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        self._view_tool = artifact.get("view_tool")
        self._probes = artifact.get("probes", [])
        # A probe is usable only if its discharge test can fire at all: a
        # distinctive term to look for, and a quantity to look for it beside.
        # Ordered rarest-first, which is the hypothesis made operational — the
        # obligations that support-weighted summarisation drops are the ones the
        # ledger has to carry, so they get the scarce slots in the block.
        self._pool = sorted(
            (
                probe
                for probe in self._probes
                if probe.get("requires_number")
                and (probe.get("discharge_check") or {}).get("distinctive_terms")
            ),
            key=lambda probe: probe.get("support", 0),
        )

    # -- conversation reading -------------------------------------------------

    @staticmethod
    def _tool_results(conversation: list[Any]) -> list[tuple[str, Any]]:
        """Pair tool calls with their results across both conversation shapes.

        ``prepare_conversation`` receives Responses API items, where a call and
        its output are separate items joined by ``call_id``. Stored trajectories
        instead nest ``tool_calls`` with an inline ``result``.
        """
        names: dict[str, str] = {}
        paired: list[tuple[str, Any]] = []
        for item in conversation:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "function_call":
                names[str(item.get("call_id"))] = str(item.get("name", ""))
            elif kind == "function_call_output":
                name = names.get(str(item.get("call_id")), "")
                output = item.get("output")
                if isinstance(output, str):
                    try:
                        output = json.loads(output)
                    except json.JSONDecodeError:
                        pass
                paired.append((name, output))
            for call in item.get("tool_calls") or []:
                if isinstance(call, dict) and "result" in call:
                    paired.append((str(call.get("name", "")), call.get("result")))
        return paired

    @staticmethod
    def _role_text(conversation: list[Any], role: str) -> str:
        parts: list[str] = []
        for item in conversation:
            if not isinstance(item, dict) or item.get("role") != role:
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    str(chunk.get("text", "")) for chunk in content if isinstance(chunk, dict)
                )
        return "\n".join(parts)

    def _observe(self, conversation: list[Any]) -> dict[str, Any]:
        """What the conversation so far establishes about the working set.

        The ledger performs the same join it asks the agent to perform. Entity
        rows are keyed by identifier, so a ``brand`` learned from
        ``get_product_details`` attaches to the cart line with the matching
        ``product_id``. Without that, "2+ items sharing a brand" degrades into
        "the cart holds 2+ items", which is true of most carts and makes the
        obligation fire indiscriminately.
        """
        resolved: set[str] = set()
        view_items = 0
        topics_fetched: set[str] = set()
        tool_names: set[str] = set()
        view_ids: set[Any] = set()
        entities: dict[Any, dict[str, Any]] = {}
        rows: list[tuple[str, dict[str, Any]]] = []
        for name, result in self._tool_results(conversation):
            tool_names.add(name)
            if not isinstance(result, dict):
                continue
            if name == "get_policies" and result.get("topic"):
                topics_fetched.add(str(result["topic"]))
            rows.append((name, result))
            for value in result.values():
                if isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict):
                            rows.append((name, entry))
                    if name == self._view_tool and value and isinstance(value[0], dict):
                        view_items = max(view_items, len(value))

        for name, row in rows:
            resolved |= {key for key, value in row.items() if value not in (None, "", [], {})}
            key_field = next(
                (
                    field
                    for field in row
                    if field.endswith("_id") and not field.startswith(("cart_", "order_", "booking_"))
                ),
                None,
            )
            if key_field is None:
                continue
            identifier = row[key_field]
            if not isinstance(identifier, (str, int)):
                continue
            entity = entities.setdefault(identifier, {})
            entity.update(
                {
                    field: value
                    for field, value in row.items()
                    if isinstance(value, (str, int, float, bool)) and value not in (None, "")
                }
            )
            if name == self._view_tool:
                view_ids.add(identifier)

        return {
            "resolved": resolved,
            "view_items": view_items,
            "view_ids": view_ids,
            "entities": entities,
            "topics_fetched": topics_fetched,
            "tool_names": tool_names,
        }

    # -- obligation state ----------------------------------------------------

    @staticmethod
    def _discharged(text: str, probe: dict[str, Any]) -> bool:
        """Whether the agent's own prose already carries this topic and its number.

        Applied over a narrow sentence window: a product listing that prints
        "Brand: NovaShield" and "$29" in different paragraphs is not a bundle
        disclosure, and treating it as one would silently drop the obligation.
        """
        check = probe.get("discharge_check") or {}
        terms = check.get("distinctive_terms") or []
        quantifiers = check.get("quantifier_terms") or []
        exact = check.get("quantifier_mode") == "exact"
        sentences = [part.strip() for part in SENTENCE.split(text) if part.strip()]
        for width in (1, 2, 3):
            for start in range(len(sentences) - width + 1):
                window = " ".join(sentences[start : start + width]).lower()
                if terms and not any(term in window for term in terms):
                    continue
                if exact:
                    if any(term.lower() in window for term in quantifiers):
                        return True
                elif NUMBER.search(window):
                    return True
        return False

    @staticmethod
    def _predicate(probe: dict[str, Any], state: dict[str, Any]) -> str:
        """Evaluate a counting obligation against the joined working set.

        Returns ``"pending"`` when the join is incomplete, ``"holds"`` when some
        value recurs often enough to satisfy the rule's antecedent, and
        ``"fails"`` when the join is complete and it does not.

        A ``"fails"`` verdict is weaker evidence than it looks, and callers must
        not treat it as a veto. The threshold is mined per *rule*, but attaches
        to the whole *topic*: ``shipping`` inherits a 5-item free-shipping
        threshold even though its disclosure duty — re-quote the cost after a
        quantity change — has nothing to do with item counts. Used as a silencer
        it suppressed every real shipping failure in the archive.
        """
        threshold = probe.get("item_threshold")
        if threshold is None:
            return "pending"
        if not state["view_ids"]:
            # The view tool has not been called yet, so an empty count is
            # ignorance rather than evidence.
            return "pending"
        if state["view_items"] < threshold:
            return "fails"
        # "2+ items sharing the same brand" counts brands, not categories, but
        # the miner cannot tell which of a rule's fields the count ranges over
        # and hands back both. Counting categories would fire on any two items
        # from the same aisle, so a coarse field yields to a discriminating one
        # and is only used when it is all there is.
        fields = [
            field
            for field in (probe.get("resolvers") or {})
            if field not in COARSE_FIELDS or len(probe["resolvers"]) == 1
        ]
        if not fields:
            return "holds"
        ids = state["view_ids"]
        entities = state["entities"]
        for field in fields:
            known = [
                entities[identifier][field]
                for identifier in ids
                if field in entities.get(identifier, {})
            ]
            if len(known) < len(ids):
                return "pending"
            if known and max(Counter(known).values()) >= threshold:
                return "holds"
        return "fails"

    def _liveness(self, probe: dict[str, Any], state: dict[str, Any], asked: str) -> str | None:
        """Why this obligation bears on the case, or ``None`` if it does not.

        Four routes, doing distinct work, and the counting one is purely
        additive. A satisfied count is the strongest evidence available — two
        cart lines provably sharing a brand means the bundle rule fires, and
        nobody in the conversation will ever say the word "bundle". An
        unsatisfied count is not evidence of anything, because the threshold was
        mined per rule and applied per topic, so it falls through to the routes
        below rather than vetoing them.
        """
        if probe.get("item_threshold") is not None and self._predicate(probe, state) == "holds":
            return "holds"
        if probe["topic"] in state["topics_fetched"]:
            return "holds"
        terms = (probe.get("discharge_check") or {}).get("distinctive_terms") or []
        if any(term in asked for term in terms):
            return "holds"
        # Nobody says "price alerts" — the customer says the price looks wrong.
        # An obligation whose deciding field the view withholds is therefore
        # pending on the join, not absent, and it belongs in the consolidated
        # join line even though no one has named it. Once the join lands it must
        # be *promoted*, not dropped: the fields being known is precisely when
        # the rule becomes evaluable and so precisely when it must be stated.
        resolvers = probe.get("resolvers") or {}
        if resolvers:
            return "pending" if any(field not in state["resolved"] for field in resolvers) else "holds"
        return None

    @staticmethod
    def _instruction(probe: dict[str, Any]) -> str:
        channels = set(probe.get("channels") or [])
        text = next(
            (phrase for channel, phrase in CHANNEL_INSTRUCTION if channel in channels),
            DEFAULT_INSTRUCTION,
        )
        return text + (TOOL_NOTE if "tool" in channels else "")

    def _ledger(self, conversation: list[Any]) -> list[str]:
        """Two tiers, because the ledger can decide far less than it can notice.

        A counting rule over a completed join is *decided*: the ledger knows the
        bundle applies and says so. Everything else it can only *flag*, and
        flagging topic by topic was what made the first version unusable — five
        lines a turn, each repeating "call get_product_details". Those collapse
        into one line about the view itself, which is both cheaper and closer to
        the actual defect: the working set omits the fields the policies are
        written in, so the summary the agent is about to give is computed over
        the wrong columns.
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
            prefix = "" if topic in state["topics_fetched"] else f"call get_policies('{topic}'), then "
            decided.append(
                f"- {topic}: bears on this case and is still unsaid — {prefix}{self._instruction(probe)}"
            )
            if len(decided) == MAX_LEDGER_LINES:
                break
        lines = decided[:MAX_LEDGER_LINES]
        if pending and state["view_ids"] and len(lines) < MAX_LEDGER_LINES:
            lines.append(self._join_line(pending))
        return lines

    def _join_line(self, pending: list[dict[str, Any]]) -> str:
        topics = ", ".join(entry["probe"]["topic"] for entry in pending[:MAX_PENDING_TOPICS])
        fields = sorted({field for entry in pending for field in entry["fields"]})
        tools = sorted(
            {
                tool
                for entry in pending
                for field in entry["fields"]
                for tool in (entry["probe"].get("resolvers") or {}).get(field, [])[:1]
            }
        )
        return (
            f"- undecidable from {self._view_tool or 'the working set'} alone: it does not expose "
            f"{', '.join(fields[:MAX_NAMED_FIELDS * 2])}, which is what decides {topics}. "
            f"Look these up per entry with {' / '.join(tools[:2]) or 'the detail lookups'} before "
            f"you summarise, then disclose whichever rules turn out to apply — the figures cannot "
            f"be computed from the view's own columns."
        )

    # -- harness hook --------------------------------------------------------

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Recompute the ledger before every model call, including tool rounds.

        This is the hook that has to carry the intervention, and the obvious
        alternative does not work. ``prepare_conversation`` runs once per
        assistant turn, *before* the orchestrator's tool loop, which then appends
        up to eight rounds of results without calling it again. Most tasks finish
        in a single turn, so a ledger built there observes an empty working set,
        suppresses every line, and leaves behaviour bit-identical to the
        baseline — which is exactly what the first smoke test showed on the brand
        bundle task.

        ``generate_next_turn`` is called once per round, so the ledger sees each
        tool result as it lands and can react within the turn that matters. On
        the first round of a fresh task nothing has been observed yet and the
        ledger is empty, so the agent is left untouched.
        """
        lines = self._ledger(conversation)
        if lines:
            conversation = self.inject_system_message(
                conversation, self._block(lines), before_last_user=False
            )
        return super().generate_next_turn(
            system_prompt=system_prompt, conversation=conversation, tools=tools
        )

    @staticmethod
    def _block(lines: list[str]) -> str:
        return (
            "Outstanding disclosure obligations for this conversation, derived from "
            "store policy and the state you have already observed. Each is a rubric "
            "requirement in its own right: satisfying the customer's literal request "
            "while leaving one of these unsaid still counts as an incomplete task. "
            "Resolve and discharge them before you finish, and quantify every figure.\n"
            + "\n".join(lines)
        )
