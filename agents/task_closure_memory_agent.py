"""PWM plus independently learned, lifecycle-aware Task-Closure Memory v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from agents.completion_lifecycle import CompletionItem, CompletionTracker, OPEN_STATUSES, compact
from agents.completion_templates import CompletionTemplateIndex, completion_query, tool_events
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent as _ProcessWorkflowMemoryAgent


CLOSURE_MODES = frozenset({"pwm_only", "full"})
FINAL_CLOSURE_RULE = (
    "If you choose to call tools in this turn, ignore all final-closure requirements below. "
    "They must not affect tool selection, tool arguments, or whether another tool call is needed. "
    "Apply them only if you are otherwise ready to answer the user without tool calls."
)


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("bookkeeper returned no JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict) or not isinstance(payload.get("items", []), list):
        raise ValueError("bookkeeper JSON must contain an items list")
    return payload


def _usage_dict(result: Any) -> dict[str, int]:
    usage = getattr(result, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


class TaskClosureMemoryAgent(_ProcessWorkflowMemoryAgent):
    """Freeze PWM execution and add a separate what-remains-before-done ledger."""

    completion_memory_path = Path(
        os.environ.get(
            "STATE_BENCH_COMPLETION_MEMORY_PATH",
            "artifacts/task_closure_memory_v2/memory/completion_templates.json",
        )
    )

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(client, system_prompt, tools, tool_handlers, runtime_context, **kwargs)
        self.closure_mode = os.environ.get("STATE_BENCH_COMPLETION_MODE", "full")
        if self.closure_mode not in CLOSURE_MODES:
            raise ValueError("STATE_BENCH_COMPLETION_MODE must be pwm_only or full")

        self._tracker: CompletionTracker | None = None
        self._completion_index: CompletionTemplateIndex | None = None
        self._known_templates: dict[str, dict[str, Any]] = {}
        self._retrieval_log: list[dict[str, Any]] = []
        self._bookkeeper_log: list[dict[str, Any]] = []
        self._generation_log: list[dict[str, Any]] = []
        self._initial_items: list[dict[str, Any]] | None = None
        self._last_bookkeeper_fingerprint = ""
        self._suppressed_final_signatures: set[str] = set()

        if self.closure_mode == "full":
            domain = getattr(runtime_context, "domain", None)
            top_k = int(os.environ.get("STATE_BENCH_COMPLETION_TOP_K", "8"))
            self._completion_index = CompletionTemplateIndex.from_path(
                self.completion_memory_path,
                domain=domain,
                top_k=top_k,
            )
            self._tracker = CompletionTracker()

    @staticmethod
    def _observable_conversation(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in conversation if item.get("role") != "system"]

    @staticmethod
    def _transcript(conversation: list[dict[str, Any]]) -> str:
        observable = TaskClosureMemoryAgent._observable_conversation(conversation)
        selected = observable[-20:]
        first_user = next((item for item in observable if item.get("role") == "user"), None)
        if first_user is not None and first_user not in selected:
            selected = [first_user, *selected]

        lines = []
        for index, message in enumerate(selected):
            role = str(message.get("role", "unknown"))
            content = compact(message.get("content", ""), 1200)
            if content:
                lines.append(f"M{index} {role}: {content}")
        for event in tool_events(observable)[-24:]:
            lines.append(
                f"T{event.sequence} tool={event.name} arguments={compact(event.arguments, 500)} "
                f"result={compact(event.result, 1000)}"
            )
        return "\n".join(lines)[-24000:]

    @staticmethod
    def _template_view(template: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": template.get("id"),
            "family": template.get("family"),
            "title": template.get("title"),
            "trigger": template.get("trigger"),
            "latent_signal_categories": template.get("latent_signal_categories", []),
            "obligations": template.get("obligations", []),
        }

    def _retrieve_completion_templates(
        self, conversation: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        assert self._completion_index is not None
        query = completion_query(self._observable_conversation(conversation))
        ranked = self._completion_index.retrieve_with_scores(query)
        templates = [template for _, template in ranked]
        for template in templates:
            self._known_templates[str(template.get("id", ""))] = template
        self._retrieval_log.append(
            {
                "index": len(self._retrieval_log),
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "query_preview": compact(query, 600),
                "candidates": [
                    {
                        "id": template.get("id"),
                        "family": template.get("family"),
                        "score": round(score, 6),
                    }
                    for score, template in ranked
                ],
            }
        )
        return templates

    def _bookkeeper_prompt(
        self,
        conversation: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, str]], set[str]]:
        assert self._tracker is not None
        existing = [
            item
            for item in self._tracker.snapshot()
            if item["template_id"] != "runtime_execution"
            and item["status"] in OPEN_STATUSES | {"violated"}
        ]
        system_prompt = (
            "You are a conservative completion-state bookkeeper, not an agent, planner, verifier, or judge. "
            "Use only the observable transcript and the supplied learned templates. Never propose a tool, "
            "change an action, reject an answer, or invent facts. Decide which template obligations actually "
            "apply to this task, instantiate them for a concrete scope, and update whether they remain pending. "
            "Uncertainty must stay pending or pending_evidence. A communication obligation is satisfied only "
            "when an assistant message actually communicates it with authoritative support. A failed or preview "
            "tool result never proves execution. An invariant becomes violated if any earlier claim/action broke it; "
            "later correction does not erase that historical violation. Return JSON only."
        )
        schema = {
            "items": [
                {
                    "template_id": "one supplied template id",
                    "obligation_id": "one obligation id from that template",
                    "scope_key": "short stable entity/operation scope, or default",
                    "applicable": True,
                    "status": "pending_evidence|pending|satisfied|invalidated|violated",
                    "description": "grounded requirement for this task without invented facts",
                    "evidence": [
                        {"kind": "message|tool_result", "ref": "M# or T#", "detail": "short fact"}
                    ],
                    "missing_evidence": ["specific evidence still needed"],
                }
            ]
        }
        transcript = self._transcript(conversation)
        user_payload = {
            "rules": [
                "Return only applicable items, plus existing items that need a status update.",
                "Use only supplied template_id and obligation_id values.",
                "Do not mark a requirement satisfied merely because a tool returned a related field.",
                "When a latent template's intent matches but the structural signal or eligibility has not yet been resolved, instantiate it as pending_evidence instead of treating missing evidence as proof of inapplicability.",
                "Do not copy training-specific IDs, dates, or amounts into a new task.",
                "Use separate scope_key values only when the same obligation truly applies to multiple entities.",
                "Reuse an existing item's scope_key exactly when updating that item.",
                "Mark an existing item inapplicable/invalidated only with an explicit transcript evidence ref; unknown remains pending.",
            ],
            "output_schema": schema,
            "templates": [self._template_view(item) for item in candidates],
            "existing_items": existing,
            "observable_transcript": transcript,
        }
        valid_refs = {
            line.split(" ", 1)[0]
            for line in transcript.splitlines()
            if line.startswith(("M", "T")) and line.split(" ", 1)[0][1:].isdigit()
        }
        return (
            system_prompt,
            [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
            valid_refs,
        )

    def _sync_bookkeeper(
        self,
        conversation: list[dict[str, Any]],
        *,
        force: bool = False,
        finalizing: bool = False,
    ) -> None:
        assert self._tracker is not None
        self._tracker.sync_execution(self._observable_conversation(conversation))
        current = self._retrieve_completion_templates(conversation)
        referenced_ids = {
            item.template_id
            for item in self._tracker.items.values()
            if item.template_id != "runtime_execution"
            and item.status in OPEN_STATUSES | {"violated"}
        }
        candidate_map = {str(item.get("id", "")): item for item in current}
        candidate_map.update(
            {
                template_id: self._known_templates[template_id]
                for template_id in referenced_ids
                if template_id in self._known_templates
            }
        )
        candidates = list(candidate_map.values())
        fingerprint = _json_hash(
            {
                "conversation": self._observable_conversation(conversation),
                "candidate_ids": [item.get("id") for item in candidates],
                "items": self._tracker.snapshot(),
            }
        )
        if not force and fingerprint == self._last_bookkeeper_fingerprint:
            return

        system_prompt, messages, valid_refs = self._bookkeeper_prompt(conversation, candidates)
        record: dict[str, Any] = {
            "index": len(self._bookkeeper_log),
            "input_fingerprint": fingerprint,
            "candidate_ids": [item.get("id") for item in candidates],
            "parsed": False,
        }
        try:
            result = self.client.generate(
                system_prompt=system_prompt,
                conversation=messages,
                tools=[],
            )
            usage = _usage_dict(result)
            self.add_token_usage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                category="memory_retrieval",
            )
            record["usage"] = usage
            record["response_sha256"] = hashlib.sha256(
                str(getattr(result, "text", "")).encode("utf-8")
            ).hexdigest()
            if getattr(result, "tool_calls", None):
                raise ValueError("bookkeeper unexpectedly returned a tool call")
            payload = _parse_json_object(str(getattr(result, "text", "")))
            self._tracker.merge_semantic(
                payload,
                candidates,
                valid_evidence_refs=valid_refs,
                allow_invariant_satisfaction=finalizing,
            )
            record["parsed"] = True
            record["returned_items"] = len(payload.get("items", []))
            self._last_bookkeeper_fingerprint = _json_hash(
                {
                    "conversation": self._observable_conversation(conversation),
                    "candidate_ids": [item.get("id") for item in candidates],
                    "items": self._tracker.snapshot(),
                }
            )
        except Exception as error:  # A bookkeeping failure must not block the frozen PWM agent.
            record["error"] = f"{type(error).__name__}: {compact(error, 500)}"
        self._bookkeeper_log.append(record)
        if self._initial_items is None:
            self._initial_items = self._tracker.snapshot()

    @staticmethod
    def _dedupe(items: list[CompletionItem], limit: int) -> list[CompletionItem]:
        selected: list[CompletionItem] = []
        descriptions: set[str] = set()
        runtime_scopes: set[str] = set()
        for item in items:
            description_key = " ".join(item.description.casefold().split()).rstrip(".")
            action_scope = str(item.scope.get("action_key", ""))
            if description_key in descriptions:
                continue
            if action_scope and action_scope in runtime_scopes and item.type != "execution":
                continue
            descriptions.add(description_key)
            if action_scope:
                runtime_scopes.add(action_scope)
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _item_signature(items: list[CompletionItem]) -> str:
        return _json_hash(
            [
                {"id": item.id, "status": item.status, "description": item.description}
                for item in items
            ]
        )

    def _final_signature(
        self, items: list[CompletionItem], conversation: list[dict[str, Any]]
    ) -> str:
        return _json_hash(
            {
                "items": [
                    {"id": item.id, "status": item.status, "description": item.description}
                    for item in items
                ],
                "observable_conversation": self._observable_conversation(conversation),
            }
        )

    def _exposure(
        self, conversation: list[dict[str, Any]]
    ) -> tuple[str | None, list[CompletionItem], str, str]:
        assert self._tracker is not None
        actionable = self._dedupe(self._tracker.actionable_items(), 4)
        guards = self._dedupe(self._tracker.guard_items(), 3)
        violated = self._dedupe(self._tracker.violated_items(), 2)
        final = self._dedupe(self._tracker.final_items(), 6)

        if actionable:
            lines = [
                "Task-completion state guard (this is not a procedure or a tool plan):",
                "Procedure Memory still determines how to proceed. Do not claim the task is complete while "
                "the following already-applicable prerequisites remain open:",
            ]
            lines.extend(f"- [{item.status}] {item.description}" for item in actionable)
            prompt = "\n".join(lines)
            return "action_guard", actionable, prompt, self._item_signature(actionable)

        evidence_guards = self._dedupe(
            [
                item
                for item in self._tracker.open_items()
                if item.status == "pending_evidence" and item.phase != "pre_action"
            ],
            4,
        )
        if evidence_guards:
            lines = [
                "Evidence-before-claim guard (not a procedure or a tool plan):",
                "The following applicable facts are still unresolved. Do not assert the corresponding concrete "
                "fee, policy tier, eligibility, benefit, or outcome until current authoritative evidence supports it:",
            ]
            lines.extend(f"- {item.description}" for item in evidence_guards)
            prompt = "\n".join(lines)
            return "claim_guard", evidence_guards, prompt, self._item_signature(evidence_guards)

        if final and self._tracker.has_valid_evidence(self._observable_conversation(conversation)):
            items = self._dedupe([*violated, *guards, *final], 7)
            lines = [
                "Selective final task-closure memory:",
                FINAL_CLOSURE_RULE,
                "If answering without tools, cover only the applicable remaining conditions below using current "
                "authoritative evidence; do not invent facts or completed actions:",
            ]
            lines.extend(f"- [{item.type}] {item.description}" for item in items)
            prompt = "\n".join(lines)
            signature = self._final_signature(items, conversation)
            if signature in self._suppressed_final_signatures:
                return None, [], "", signature
            return "final_closure", items, prompt, signature

        claim_items = self._dedupe([*violated, *guards], 4)
        if claim_items:
            lines = [
                "Evidence-before-claim guard (not a procedure or a tool plan):",
                "Do not assert a concrete fee, policy tier, eligibility, or completed action until it is supported "
                "by current authoritative evidence:",
            ]
            lines.extend(f"- {item.description}" for item in claim_items)
            prompt = "\n".join(lines)
            return "claim_guard", claim_items, prompt, self._item_signature(claim_items)

        return None, [], "", ""

    @staticmethod
    def _response_calls(response: Any) -> list[dict[str, Any]]:
        output = []
        for call in getattr(response, "tool_calls", []) or []:
            if isinstance(call, dict):
                output.append({"name": str(call.get("name", "")), "arguments": call.get("arguments") or {}})
            else:
                output.append(
                    {
                        "name": str(getattr(call, "name", "")),
                        "arguments": getattr(call, "arguments", {}) or {},
                    }
                )
        return output

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ):
        if self.closure_mode == "pwm_only":
            return super().generate_next_turn(
                system_prompt=system_prompt,
                conversation=conversation,
                tools=tools,
            )

        assert self._tracker is not None
        bookkeeper_before = len(self._bookkeeper_log)
        self._sync_bookkeeper(conversation)
        exposure_mode, exposed_items, exposure_prompt, signature = self._exposure(conversation)
        model_conversation = (
            self.inject_system_message(conversation, exposure_prompt, before_last_user=False)
            if exposure_prompt
            else conversation
        )

        # Exactly one main agent generation. There is no reject/rewrite/regeneration loop.
        response = super().generate_next_turn(
            system_prompt=system_prompt,
            conversation=model_conversation,
            tools=tools,
        )
        calls = self._response_calls(response)
        if exposure_mode == "final_closure" and calls:
            self._suppressed_final_signatures.add(signature)
        self._generation_log.append(
            {
                "generation_index": len(self._generation_log),
                "main_model_calls": 1,
                "bookkeeper_calls": len(self._bookkeeper_log) - bookkeeper_before,
                "regenerations": 0,
                "closure_injected": bool(exposure_prompt),
                "exposure_mode": exposure_mode,
                "exposure_prompt_sha256": (
                    hashlib.sha256(exposure_prompt.encode("utf-8")).hexdigest()
                    if exposure_prompt
                    else None
                ),
                "exposure_signature": signature or None,
                "exposed_items": [item.id for item in exposed_items],
                "pending_items": [item.id for item in self._tracker.open_items()],
                "status_counts": self._tracker.status_counts(),
                "output_type": "tool_call" if calls else "final_text",
                "tool_calls_after_exposure": calls if exposure_prompt and calls else [],
            }
        )
        return response

    def ingest_trajectory(self, trajectory: Any) -> None:
        if self.closure_mode == "pwm_only":
            return super().ingest_trajectory(trajectory)

        assert self._tracker is not None
        try:
            self._sync_bookkeeper(
                list(getattr(trajectory, "conversation", []) or []),
                force=True,
                finalizing=True,
            )
        except Exception as error:
            self._bookkeeper_log.append(
                {
                    "index": len(self._bookkeeper_log),
                    "parsed": False,
                    "stage": "post_trajectory_sync",
                    "error": f"{type(error).__name__}: {compact(error, 500)}",
                }
            )
        if hasattr(trajectory, "token_usage"):
            trajectory.token_usage = self.token_usage

        final_items = self._tracker.snapshot()
        final_conversation = list(getattr(trajectory, "conversation", []) or [])
        bookkeeper_input = sum(
            int(item.get("usage", {}).get("input_tokens", 0)) for item in self._bookkeeper_log
        )
        bookkeeper_output = sum(
            int(item.get("usage", {}).get("output_tokens", 0)) for item in self._bookkeeper_log
        )
        metadata = getattr(trajectory, "metadata", None)
        if metadata is None:
            metadata = {}
            trajectory.metadata = metadata
        metadata["completion_memory"] = {
            "version": "task_closure_memory_v2",
            "mode": self.closure_mode,
            "completion_artifact": str(self.completion_memory_path),
            "completion_artifact_sha256": hashlib.sha256(
                self.completion_memory_path.read_bytes()
            ).hexdigest(),
            "initial_items": self._initial_items or [],
            "final_items": final_items,
            "final_open_items": [item["id"] for item in final_items if item["status"] in OPEN_STATUSES],
            "final_satisfied_items": [item["id"] for item in final_items if item["status"] == "satisfied"],
            "final_invalidated_items": [item["id"] for item in final_items if item["status"] == "invalidated"],
            "final_violated_items": [item["id"] for item in final_items if item["status"] == "violated"],
            "status_counts": self._tracker.status_counts(),
            "lifecycle_events": list(self._tracker.events),
            "completion_retrievals": list(self._retrieval_log),
            "bookkeeper_calls": list(self._bookkeeper_log),
            "generations": list(self._generation_log),
            "summary": {
                "main_model_generations": len(self._generation_log),
                "main_calls_per_generation": [1 for _ in self._generation_log],
                "semantic_bookkeeper_calls": len(self._bookkeeper_log),
                "regenerations": 0,
                "exposures": sum(bool(item["closure_injected"]) for item in self._generation_log),
                "final_closure_exposures": sum(
                    item["exposure_mode"] == "final_closure" for item in self._generation_log
                ),
                "exposure_to_tool_call": sum(
                    bool(item["closure_injected"] and item["output_type"] == "tool_call")
                    for item in self._generation_log
                ),
                "final_closure_to_tool_call": sum(
                    bool(item["exposure_mode"] == "final_closure" and item["output_type"] == "tool_call")
                    for item in self._generation_log
                ),
                "assistant_turns": sum(
                    item.get("role") == "assistant" for item in final_conversation
                ),
                "tool_calls": len(tool_events(final_conversation)),
                "bookkeeper_input_tokens": bookkeeper_input,
                "bookkeeper_output_tokens": bookkeeper_output,
                "bookkeeper_total_tokens": bookkeeper_input + bookkeeper_output,
                "scores": "state/task scores are written as sibling top-level fields by STATE-Bench scoring",
            },
        }
