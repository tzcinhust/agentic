"""Small compatibility shim for running the archived tests outside STATE-Bench.

The real package is used whenever installed.  The shim only supplies the tiny
interface exercised by these repository-level unit tests.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any


if importlib.util.find_spec("state_bench") is None:
    package = ModuleType("state_bench")
    package.__path__ = []
    agents_package = ModuleType("state_bench.agents")
    agents_package.__path__ = []
    base_module = ModuleType("state_bench.agents.base")
    client_module = ModuleType("state_bench.client")

    @dataclass(eq=True)
    class AgentToolCallRequest:
        name: str
        arguments: dict[str, Any]

    @dataclass(eq=True)
    class AgentTurnResponse:
        text: str
        tool_calls: list[AgentToolCallRequest]

    class BaseAgent:
        def __init__(self, runtime_context=None, **kwargs):
            self.runtime_context = runtime_context
            self.token_usage = SimpleNamespace(input_tokens=0, output_tokens=0)

        def add_token_usage(self, input_tokens=None, output_tokens=None, **kwargs):
            self.token_usage.input_tokens += int(input_tokens or 0)
            self.token_usage.output_tokens += int(output_tokens or 0)

        @staticmethod
        def inject_system_message(conversation, content, before_last_user=True):
            output = [dict(item) for item in conversation]
            message = {"role": "system", "content": content}
            if before_last_user:
                position = next(
                    (
                        index
                        for index in range(len(output) - 1, -1, -1)
                        if output[index].get("role") == "user"
                    ),
                    len(output),
                )
                output.insert(position, message)
            else:
                output.append(message)
            return output

        def ingest_trajectory(self, trajectory):
            return None

    class BaseLLMClient:
        pass

    base_module.AgentToolCallRequest = AgentToolCallRequest
    base_module.AgentTurnResponse = AgentTurnResponse
    base_module.BaseAgent = BaseAgent
    client_module.BaseLLMClient = BaseLLMClient
    sys.modules["state_bench"] = package
    sys.modules["state_bench.agents"] = agents_package
    sys.modules["state_bench.agents.base"] = base_module
    sys.modules["state_bench.client"] = client_module
