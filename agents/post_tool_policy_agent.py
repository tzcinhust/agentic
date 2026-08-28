"""PWM with policy obligations introduced after the first tool round.

The initial action decision sees byte-identical three-card PWM retrieval. Once
the harness returns any tool result, the next generation receives the eager
arm's act-typed policy card, ranked from the full request plus observed tool
names. This places obligations after workflow selection but early enough to
fetch missing facts before a write or final response.

Unlike final-response review, the agent can still call read tools needed for a
derived disclosure. Unlike eager injection, the policy card never replaces a
workflow card on the initial decision.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from state_bench.agents.base import AgentTurnResponse

from agents.late_bound_policy_agent import LateBoundPolicyAgent as _LateParent
from agents.policy_obligation_agent import ACT_LABELS
from agents.policy_obligation_agent import REDUNDANT_PREFIX
from agents.policy_obligation_agent import PolicyObligationAgent as _PolicyParent
from agents.policy_obligation_agent import _clip


POST_TOOL_HEADER = (
    "Tool-grounded policy obligations for the active request follow. Use the "
    "returned facts to verify conditions, fetch any still-missing read-only facts, "
    "and satisfy applicable speech/order/refusal duties before the next write or "
    "final answer. When a calculation duty applies and its inputs are known, state "
    "the computed result explicitly, not only the rate or formula."
)

_LEADING_LABEL = re.compile(r"^([a-z][a-z -]{1,40}):", re.I)
_GENERIC_NUMBER_LABELS = {
    "amount",
    "bundle override",
    "calculation",
    "cap",
    "deposit",
    "discount",
    "fee",
    "limit",
    "minimum",
    "rate",
    "refund",
    "timeline",
    "window",
}


class PostToolPolicyAgent(_LateParent):
    """Choose the PWM workflow first, then expose relevant policy obligations."""

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self._post_tool_write_names = frozenset(
            str(name)
            for item in self._topics
            for name in item.get("write_tools", [])
        )

    @staticmethod
    def _lines(item: dict[str, Any]) -> list[str]:
        # The base renderer keeps three numeric leaves. ``loyalty_points`` has
        # exactly four: three tier rates followed by the post-discount
        # calculation rule. Dropping the fourth made the agent state Gold's rate
        # but omit the required total. Keep one extra number in this later,
        # tool-grounded prompt; eager-v1's renderer stays byte-identical.
        budgets = {name: (4 if name == "number" else budget) for name, _, budget in ACT_LABELS}
        buckets: dict[str, list[str]] = {name: [] for name, _, _ in ACT_LABELS}
        for obligation in item.get("obligations", []):
            acts = set(obligation.get("act") or [])
            for name, _, _ in ACT_LABELS:
                if name not in acts:
                    continue
                text = REDUNDANT_PREFIX.sub("", str(obligation.get("text", "")).strip())
                path = str(obligation.get("path", "")).split(".")[-1]
                if path and not path.isdigit() and path.replace("_", " ") not in text.lower():
                    text = f"{path}: {text}"
                buckets[name].append(_clip(text))
                break
        rendered = []
        for name, label, _ in ACT_LABELS:
            chosen = buckets[name][: budgets[name]]
            if not chosen:
                continue
            if name == "if" and any(buckets[other] for other, _, _ in ACT_LABELS if other != "if"):
                continue
            rendered.append(f"  {label}: " + " | ".join(chosen))
        return rendered

    @staticmethod
    def _contextual_items(
        items: list[dict[str, Any]], conversation: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Drop unmatched numeric branch alternatives when tools identify one.

        Policy lists often encode mutually exclusive branches as leading labels
        (Gold/Platinum/Standard, Express/Next-day). Showing all branches makes a
        model repeat a rate table instead of applying the observed branch. If at
        least one non-generic label occurs in the tool-grounded conversation,
        retain only matching alternatives plus generic calculation/override
        rules. The source artifact is copied and remains read-only.
        """
        context = json.dumps(conversation, ensure_ascii=False, default=str).lower()
        contextual = copy.deepcopy(items)
        for item in contextual:
            numbered = [
                obligation
                for obligation in item.get("obligations", [])
                if "number" in set(obligation.get("act") or [])
            ]
            labels: dict[int, str] = {}
            for obligation in numbered:
                match = _LEADING_LABEL.match(str(obligation.get("text", "")).strip())
                if match:
                    labels[id(obligation)] = match.group(1).lower().strip()
            branch_labels = {
                label for label in labels.values() if label not in _GENERIC_NUMBER_LABELS
            }
            matched = {
                label
                for label in branch_labels
                if re.search(rf"\b{re.escape(label)}\b", context, re.I)
            }
            if not matched:
                continue
            item["obligations"] = [
                obligation
                for obligation in item.get("obligations", [])
                if id(obligation) not in labels
                or labels[id(obligation)] in _GENERIC_NUMBER_LABELS
                or labels[id(obligation)] in matched
            ]
        return contextual

    @staticmethod
    def _unresolved_unconditional_branches(
        items: list[dict[str, Any]],
        conversation: list[dict[str, Any]],
        proposed_writes: set[str],
    ) -> dict[str, list[str]]:
        """Return applicable unconditional topics whose numeric branch is unknown."""
        context = json.dumps(conversation, ensure_ascii=False, default=str).lower()
        unresolved: dict[str, list[str]] = {}
        for item in items:
            if not item.get("unconditional_say"):
                continue
            if not (proposed_writes & {str(name) for name in item.get("write_tools", [])}):
                continue
            labels = set()
            for obligation in item.get("obligations", []):
                if "number" not in set(obligation.get("act") or []):
                    continue
                match = _LEADING_LABEL.match(str(obligation.get("text", "")).strip())
                if not match:
                    continue
                label = match.group(1).lower().strip()
                if label not in _GENERIC_NUMBER_LABELS:
                    labels.add(label)
            if labels and not any(
                re.search(rf"\b{re.escape(label)}\b", context, re.I) for label in labels
            ):
                unresolved[str(item.get("topic", ""))] = sorted(labels)
        return unresolved

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        grounded = bool(conversation and conversation[-1].get("role") == "tool")
        ranked_items: list[dict[str, Any]] = []
        if grounded:
            query = self._query_from_conversation(conversation)
            ranked_items = self._rank_topics(query)
            items = self._contextual_items(ranked_items, conversation)
            card = self._render(items)
            if card:
                # Extend the existing system prompt instead of inserting a
                # system-role item between assistant tool calls and tool results.
                system_prompt = f"{system_prompt}\n\n{POST_TOOL_HEADER}\n\n{card}"
        response = _PolicyParent.generate_next_turn(
            self,
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
        proposed_writes = {
            call.name for call in response.tool_calls if call.name in self._post_tool_write_names
        }
        if grounded and proposed_writes:
            unresolved = self._unresolved_unconditional_branches(
                ranked_items,
                conversation,
                proposed_writes,
            )
            if unresolved:
                details = "; ".join(
                    f"{topic}: one of {', '.join(labels)}"
                    for topic, labels in sorted(unresolved.items())
                )
                correction_prompt = (
                    f"{system_prompt}\n\nThe proposed write calls are deferred because an "
                    f"unconditional policy disclosure has an unresolved branch ({details}). "
                    "Request only the read-only domain tools needed to identify the applicable "
                    "branch. Do not write state and do not answer the customer yet."
                )
                corrected = _PolicyParent.generate_next_turn(
                    self,
                    system_prompt=correction_prompt,
                    conversation=conversation,
                    tools=tools,
                )
                reads = [
                    call
                    for call in corrected.tool_calls
                    if call.name not in self._post_tool_write_names
                ]
                if reads:
                    return AgentTurnResponse(text="", tool_calls=reads)
        if grounded or not response.tool_calls:
            return response

        # Chat models often place independent reads and dependent writes in one
        # parallel tool batch. The harness executes that batch atomically, so a
        # policy card injected after it is too late for say-before-write duties
        # or for fetching facts required by a derived disclosure. When the first
        # batch mixes both kinds, execute only its requested reads. The model sees
        # those canonical results on the next round, receives the policy card,
        # and remains responsible for reissuing any valid writes.
        reads = [call for call in response.tool_calls if call.name not in self._post_tool_write_names]
        if reads and len(reads) < len(response.tool_calls):
            return AgentTurnResponse(text="", tool_calls=reads)
        return response
