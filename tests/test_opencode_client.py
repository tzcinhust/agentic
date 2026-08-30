from clients.opencode_client import _chat_messages


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
