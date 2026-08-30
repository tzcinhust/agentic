import json

from tools.eval_shim import _request_route, _usage_summary


def test_usage_summary_supports_chat_completion_token_details() -> None:
    payload = json.dumps(
        {
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 12},
            }
        }
    ).encode()

    assert _usage_summary(payload) == {
        "input_tokens": 120,
        "cached_input_tokens": 40,
        "output_tokens": 30,
        "reasoning_output_tokens": 12,
        "total_tokens": 150,
    }


def test_usage_summary_supports_responses_api_and_rejects_missing_usage() -> None:
    payload = json.dumps(
        {
            "usage": {
                "input_tokens": 200,
                "output_tokens": 50,
                "total_tokens": 250,
                "input_tokens_details": {"cached_tokens": 80},
                "output_tokens_details": {"reasoning_tokens": 20},
            }
        }
    ).encode()

    assert _usage_summary(payload)["cached_input_tokens"] == 80
    assert _usage_summary(b'{"error": {"message": "busy"}}') is None


def test_request_route_separates_agent_from_official_eval_traffic() -> None:
    assert _request_route("/v1/chat/completions") == "agent_chat_completions"
    assert _request_route("/openai/v1/responses") == "official_eval_responses"
    assert _request_route("/health") == "other"
