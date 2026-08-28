"""PWM + policy obligations + loyalty verifier + deterministic grounding guards."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from state_bench.agents.base import AgentTurnResponse

from agents.cross_domain_precommit_guard import (
    CrossDomainPrecommitGuard,
    GuardAction,
    GuardResult,
)
from agents.final_response_contract import FinalResponseContract, ground_policy_sensitive_claims
from agents.loyalty_verified_policy_agent import LoyaltyVerifiedPolicyAgent as _Parent
from agents.policy_activation import ActivationValue, PolicyActivation
from agents.policy_addendum_agent import review_policy_addendum
from agents.runtime_fact_ledger import RuntimeFactLedger, WRITE_TOOLS


LOG = logging.getLogger(__name__)


class GroundedVerifiedPolicyAgent(_Parent):
    """Enable the new policy guards without changing any existing agent arm."""

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self._grounded_domain = str(getattr(runtime_context, "domain", "") or "")
        self._activation = PolicyActivation(self._topics)
        self._guard = CrossDomainPrecommitGuard(self._topics)
        self._contract = FinalResponseContract(self._topics)
        self._ranking_conversation: list[dict[str, Any]] = []
        self.guard_stats: dict[str, Any] = {
            "guard_invocations": 0,
            "blocked_writes": 0,
            "need_read_count": 0,
            "need_user_choice_count": 0,
            "active_policy_topics": {},
            "suppressed_policy_topics": {},
            "deterministic_addendum_count": 0,
            "llm_verifier_count": 0,
            "round_limit_stops": 0,
        }

    def prepare_conversation(self, conversation: list[Any]) -> list[Any]:
        self._ranking_conversation = [item for item in conversation if isinstance(item, dict)]
        return super().prepare_conversation(conversation)

    @staticmethod
    def _bump(mapping: dict[str, int], key: str) -> None:
        mapping[key] = mapping.get(key, 0) + 1

    def _activation_topic(self, topic: str) -> str:
        mapping = {
            "change": "voluntary_change",
            "cancellation": "cancellation",
            "hotel_cancellation": "cancellation",
        }
        return mapping.get(topic, topic)

    def _filter_topics(
        self,
        items: list[dict[str, Any]],
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not conversation:
            return items
        ledger = RuntimeFactLedger(conversation)
        active: list[dict[str, Any]] = []
        for item in items:
            topic = str(item.get("topic", ""))
            verdict = self._activation.evaluate(
                self._grounded_domain,
                self._activation_topic(topic),
                ledger,
                conversation,
            )
            if verdict.value == ActivationValue.TRUE:
                self._bump(self.guard_stats["active_policy_topics"], topic)
                active.append(item)
            else:
                self._bump(self.guard_stats["suppressed_policy_topics"], topic)
        return active

    def _rank_topics(self, query: str) -> list[dict[str, Any]]:
        ranked = super()._rank_topics(query)
        active = self._filter_topics(ranked, self._ranking_conversation)
        # Preserve the parent's relevance order while promoting grounded TRUE
        # topics ahead of any future compatible extension.
        return active

    def _rank_topics_for_conversation(
        self,
        query: str,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ranked = super()._rank_topics(query)
        return self._filter_topics(ranked, conversation)

    def _record_guard(self, result: GuardResult) -> None:
        self.guard_stats["guard_invocations"] += 1
        if result.action == GuardAction.BLOCK:
            self.guard_stats["blocked_writes"] += 1
        elif result.action == GuardAction.NEED_READ:
            self.guard_stats["need_read_count"] += 1
        elif result.action == GuardAction.NEED_USER_CHOICE:
            self.guard_stats["need_user_choice_count"] += 1
        LOG.info("grounded_policy_guard %s", json.dumps(self.guard_stats, sort_keys=True))
        if os.environ.get("STATE_BENCH_GUARD_DEBUG") == "1":
            LOG.warning(
                "grounded_policy_guard_debug action=%s reads=%s correction=%s stats=%s",
                result.action.value,
                [call.name for call in result.read_only_tool_calls],
                result.correction[:240],
                json.dumps(self.guard_stats, sort_keys=True),
            )

    @classmethod
    def _safe_reads(
        cls,
        result: GuardResult,
        proposed: AgentTurnResponse | None = None,
    ) -> AgentTurnResponse:
        calls = [call for call in result.read_only_tool_calls if call.name not in WRITE_TOOLS]
        if proposed is not None:
            calls.extend(call for call in proposed.tool_calls if cls._name(call) not in WRITE_TOOLS)
        unique = []
        seen = set()
        for call in calls:
            key = (cls._name(call), json.dumps(cls._args(call), sort_keys=True, default=str))
            if key in seen:
                continue
            seen.add(key)
            unique.append(call)
        calls = unique
        if calls:
            return AgentTurnResponse(text="", tool_calls=calls)
        return AgentTurnResponse(
            text="I need one more canonical read before I can safely continue; I have not changed anything.",
            tool_calls=[],
        )

    def _guard_result_response(
        self,
        result: GuardResult,
        proposed: AgentTurnResponse | None = None,
    ) -> AgentTurnResponse:
        if result.action == GuardAction.NEED_READ:
            return self._safe_reads(result, proposed)
        if result.action == GuardAction.NEED_USER_CHOICE:
            return AgentTurnResponse(text=result.question, tool_calls=[])
        return AgentTurnResponse(
            text=(
                "I can’t safely execute that state change because the canonical facts conflict with it. "
                "I have not changed anything. " + result.correction
            ).strip(),
            tool_calls=[],
        )

    @staticmethod
    def _tool_rounds_in_current_turn(conversation: list[dict[str, Any]]) -> int:
        last_user = max(
            (index for index, item in enumerate(conversation) if item.get("role") == "user"),
            default=-1,
        )
        return sum(
            1
            for item in conversation[last_user + 1 :]
            if item.get("role") == "assistant" and item.get("tool_calls")
        )

    def _respect_round_limit(
        self,
        conversation: list[dict[str, Any]],
        response: AgentTurnResponse,
    ) -> AgentTurnResponse:
        # The official harness permits eight generations but needs the last one
        # to be text-only; a tool call on generation eight is executed and then
        # raises before a final answer can be requested.
        if response.tool_calls and self._tool_rounds_in_current_turn(conversation) >= 7:
            self.guard_stats["round_limit_stops"] += 1
            return AgentTurnResponse(
                text="I reached the safe verification limit for this turn, so I have not executed any additional change. Please ask me to continue if another step is still needed.",
                tool_calls=[],
            )
        return response

    def _finalize(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        proposed: AgentTurnResponse,
        ledger: RuntimeFactLedger,
    ) -> AgentTurnResponse:
        grounded = ground_policy_sensitive_claims(
            self._grounded_domain,
            conversation,
            proposed.text,
            ledger,
            self._topics,
        )
        if not grounded and proposed.text.strip():
            grounded = "I need to verify the applicable policy before making that commitment."
        completed = self._contract.apply(self._grounded_domain, conversation, grounded, ledger)
        if completed != grounded:
            self.guard_stats["deterministic_addendum_count"] += 1
        draft = AgentTurnResponse(text=completed, tool_calls=[])
        items = self._rank_topics_for_conversation(
            self._query_from_conversation(conversation),
            conversation,
        )
        if not items:
            return draft
        # The reviewer has tools=[] and its provider-reported tokens are added by
        # the shared helper. Relay pricing is unknown, so no fabricated USD cost
        # is reported through add_cost_usd.
        self.guard_stats["llm_verifier_count"] += 1
        return review_policy_addendum(
            self,
            system_prompt=system_prompt,
            conversation=conversation,
            draft=draft,
            items=items,
        )

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        proposed = super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
        )
        ledger = RuntimeFactLedger(conversation)
        if not proposed.tool_calls:
            return self._finalize(
                system_prompt=system_prompt,
                conversation=conversation,
                proposed=proposed,
                ledger=ledger,
            )

        verdict = self._guard.check(self._grounded_domain, conversation, proposed, ledger)
        self._record_guard(verdict)
        if verdict.action == GuardAction.APPROVE:
            return self._respect_round_limit(conversation, proposed)
        if verdict.action in {GuardAction.NEED_READ, GuardAction.NEED_USER_CHOICE}:
            return self._guard_result_response(verdict, proposed)

        corrected = super().generate_next_turn(
            system_prompt=(
                f"{system_prompt}\n\nA deterministic pre-commit guard found this concrete conflict: "
                f"{verdict.correction}\nRegenerate only the next step once. Preserve valid reads and "
                "customer choices; do not repeat the blocked write."
            ),
            conversation=conversation,
            tools=tools,
        )
        second = self._guard.check(self._grounded_domain, conversation, corrected, ledger)
        self._record_guard(second)
        if second.action == GuardAction.APPROVE:
            if corrected.tool_calls:
                return self._respect_round_limit(conversation, corrected)
            return self._finalize(
                system_prompt=system_prompt,
                conversation=conversation,
                proposed=corrected,
                ledger=ledger,
            )
        return self._guard_result_response(second, corrected)

    def ingest_trajectory(self, trajectory: Any) -> None:
        LOG.info("grounded_policy_summary %s", json.dumps(self.guard_stats, sort_keys=True))
        return super().ingest_trajectory(trajectory)
