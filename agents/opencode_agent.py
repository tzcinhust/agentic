"""Custom STATE-Bench agent backed by an OpenAI-compatible Chat Completions API."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from state_bench.agents.base import AgentToolCallRequest, AgentTurnResponse, BaseAgent


class OpenCodeAgent(BaseAgent):
    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(runtime_context=runtime_context)
        self.client = client
        self.provider_request_audit_id = uuid.uuid4().hex
        domain = str(getattr(runtime_context, "domain", ""))
        task_id = str(getattr(runtime_context, "task_id", ""))
        self.provider_task_key = hashlib.sha256(f"{domain}|{task_id}".encode()).hexdigest()

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        result = self.client.generate(
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
            audit_id=self.provider_request_audit_id,
            task_key=self.provider_task_key,
        )
        usage = result.usage
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        self.add_token_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            cached_input_tokens=getattr(prompt_details, "cached_tokens", None),
            reasoning_output_tokens=getattr(completion_details, "reasoning_tokens", None),
        )
        return AgentTurnResponse(
            text=result.text,
            tool_calls=[
                AgentToolCallRequest(name=call["name"], arguments=call["arguments"])
                for call in result.tool_calls
            ],
        )

    def ingest_trajectory(self, trajectory: Any) -> None:
        metadata = getattr(trajectory, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(trajectory, "metadata", metadata)
        metadata["provider_request_audit_id"] = self.provider_request_audit_id
        metadata["provider_task_key"] = self.provider_task_key
