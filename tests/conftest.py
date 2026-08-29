from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import state_bench  # noqa: F401
except ModuleNotFoundError:
    state_bench = types.ModuleType("state_bench")
    agents = types.ModuleType("state_bench.agents")
    base = types.ModuleType("state_bench.agents.base")
    client = types.ModuleType("state_bench.client")

    @dataclass
    class AgentToolCallRequest:
        name: str
        arguments: dict[str, Any]

    @dataclass
    class AgentTurnResponse:
        text: str = ""
        tool_calls: list[Any] = field(default_factory=list)

    class BaseAgent:
        def __init__(self, runtime_context=None):
            self.runtime_context = runtime_context
            self.token_usage = SimpleNamespace(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )

        def add_token_usage(self, *, input_tokens=None, output_tokens=None, **_kwargs):
            if input_tokens is None or output_tokens is None:
                return
            self.token_usage.input_tokens += int(input_tokens)
            self.token_usage.output_tokens += int(output_tokens)
            self.token_usage.total_tokens += int(input_tokens) + int(output_tokens)

        @staticmethod
        def inject_system_message(conversation, content, *, before_last_user=True):
            if not content:
                return conversation
            message = {"role": "system", "content": content}
            if before_last_user and conversation:
                return [*conversation[:-1], message, conversation[-1]]
            return [*conversation, message]

        def ingest_trajectory(self, _trajectory):
            return None

    class BaseLLMClient:
        pass

    base.AgentToolCallRequest = AgentToolCallRequest
    base.AgentTurnResponse = AgentTurnResponse
    base.BaseAgent = BaseAgent
    client.BaseLLMClient = BaseLLMClient
    sys.modules["state_bench"] = state_bench
    sys.modules["state_bench.agents"] = agents
    sys.modules["state_bench.agents.base"] = base
    sys.modules["state_bench.client"] = client
