"""OpenAI-compatible client adapter for STATE-Bench."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from state_bench.client import BaseLLMClient


@dataclass
class AgentCompletion:
    text: str
    tool_calls: list[dict[str, Any]]
    usage: Any = None


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            converted.append(tool)
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        )
    return converted


def _chat_messages(system_prompt: str, conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_parts = [system_prompt]
    messages: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []
    call_index = 0

    for item_index, item in enumerate(conversation):
        role = item.get("role")
        content = item.get("content", "")

        if role == "system":
            system_parts.append(_as_text(content))
        elif role == "user":
            messages.append({"role": "user", "content": _as_text(content)})
        elif role == "assistant":
            calls = item.get("tool_calls") or []
            next_is_tool = (
                item_index + 1 < len(conversation)
                and conversation[item_index + 1].get("role") == "tool"
            )
            message: dict[str, Any] = {
                "role": "assistant",
                "content": content or "" if next_is_tool else "",
            }
            pending_call_ids = []
            if calls:
                converted_calls = []
                for call in calls:
                    call_id = f"statebench_call_{call_index}"
                    call_index += 1
                    pending_call_ids.append(call_id)
                    converted_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                            },
                        }
                    )
                message["tool_calls"] = converted_calls
            messages.append(message)
            if calls and not next_is_tool:
                for call_id, call in zip(pending_call_ids, calls):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": _as_text(call.get("result", "")),
                        }
                    )
                pending_call_ids = []
                if content:
                    messages.append({"role": "assistant", "content": _as_text(content)})
        elif role == "tool":
            records = content if isinstance(content, list) else []
            for index, record in enumerate(records):
                call_id = pending_call_ids[index] if index < len(pending_call_ids) else f"statebench_call_{call_index}"
                if index >= len(pending_call_ids):
                    call_index += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _as_text(record.get("result", record)),
                    }
                )
            pending_call_ids = []

    return [{"role": "system", "content": "\n\n".join(system_parts)}, *messages]


class OpenCodeLLMClient(BaseLLMClient):
    """Minimal stateless Chat Completions adapter for an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        timeout_seconds: float = 120.0,
        max_retries: int = 6,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @classmethod
    def from_env(cls) -> "OpenCodeLLMClient":
        return cls(
            base_url=os.environ["STATE_BENCH_AGENT_BASE_URL"],
            api_key=os.environ["STATE_BENCH_AGENT_API_KEY"],
            model=os.environ.get("STATE_BENCH_AGENT_MODEL", "qwen3.7-max"),
            max_tokens=int(os.environ.get("STATE_BENCH_AGENT_MAX_TOKENS", "4096")),
            timeout_seconds=float(os.environ.get("STATE_BENCH_AGENT_TIMEOUT_SECONDS", "120")),
            max_retries=int(os.environ.get("STATE_BENCH_AGENT_MAX_RETRIES", "6")),
        )

    @property
    def provider_name(self) -> str:
        return "opencode-compatible"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        audit_id: str | None = None,
        task_key: str | None = None,
    ) -> AgentCompletion:
        extra_headers = {}
        if audit_id:
            extra_headers["X-PWM-Audit-ID"] = audit_id
        if task_key:
            extra_headers["X-PWM-Task-Key"] = task_key
        response = self._client.chat.completions.create(
            model=self.model,
            messages=_chat_messages(system_prompt, conversation),
            tools=_chat_tools(tools),
            temperature=0,
            max_tokens=self.max_tokens,
            extra_headers=extra_headers or None,
        )
        message = response.choices[0].message
        tool_calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append({"name": call.function.name, "arguments": arguments})
        return AgentCompletion(text=message.content or "", tool_calls=tool_calls, usage=response.usage)
