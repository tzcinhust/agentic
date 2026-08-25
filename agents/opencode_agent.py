"""Custom STATE-Bench agent backed by an OpenAI-compatible chat client."""

from __future__ import annotations

from typing import Any

from state_bench.agents.base import AgentToolCallRequest, AgentTurnResponse, BaseAgent


class OpenCodeAgent(BaseAgent):
    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(runtime_context=runtime_context)
        self.client = client

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
        )
        usage = result.usage
        self.add_token_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )
        return AgentTurnResponse(
            text=result.text,
            tool_calls=[
                AgentToolCallRequest(name=call["name"], arguments=call["arguments"])
                for call in result.tool_calls
            ],
        )
