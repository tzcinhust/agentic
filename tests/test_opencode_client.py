from types import SimpleNamespace

from agents.opencode_agent import OpenCodeAgent
from clients.opencode_client import AgentCompletion, _chat_messages


def test_chat_messages_reconstructs_folded_tool_results() -> None:
    messages = _chat_messages(
        "system",
        [
            {"role": "user", "content": "Cancel order 42"},
            {
                "role": "assistant",
                "content": "The cancellation is ready for confirmation.",
                "tool_calls": [
                    {
                        "name": "get_order",
                        "arguments": {"order_id": "42"},
                        "result": {"status": "processing"},
                    }
                ],
            },
            {"role": "user", "content": "Confirm"},
        ],
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert messages[2]["tool_calls"][0]["function"]["name"] == "get_order"
    assert messages[3]["tool_call_id"] == messages[2]["tool_calls"][0]["id"]
    assert messages[4]["content"] == "The cancellation is ready for confirmation."


def test_chat_messages_keeps_live_tool_pairing() -> None:
    messages = _chat_messages(
        "system",
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "get_order", "arguments": {"order_id": "42"}}],
            },
            {"role": "tool", "content": [{"result": {"status": "processing"}}]},
        ],
    )

    assert [message["role"] for message in messages] == ["system", "assistant", "tool"]
    assert messages[2]["tool_call_id"] == messages[1]["tool_calls"][0]["id"]


def test_agent_records_cached_and_reasoning_token_buckets() -> None:
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=30,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
    )

    class Client:
        kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return AgentCompletion(text="done", tool_calls=[], usage=usage)

    client = Client()
    runtime = SimpleNamespace(domain="travel", task_id="task-1")
    agent = OpenCodeAgent(client, "system", [], {}, runtime_context=runtime)
    agent.generate_next_turn(system_prompt="system", conversation=[], tools=[])

    assert agent.token_usage.input_tokens == 120
    assert agent.token_usage.cached_input_tokens == 40
    assert agent.token_usage.output_tokens == 30
    assert agent.token_usage.reasoning_output_tokens == 12
    assert len(client.kwargs["audit_id"]) == 32
    assert len(client.kwargs["task_key"]) == 64

    trajectory = SimpleNamespace(metadata={})
    agent.ingest_trajectory(trajectory)
    assert trajectory.metadata["provider_request_audit_id"] == client.kwargs["audit_id"]
    assert trajectory.metadata["provider_task_key"] == client.kwargs["task_key"]
