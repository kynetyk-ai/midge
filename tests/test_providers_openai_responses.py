"""The OpenAI Responses adapter: encoding, decoding, and error classification.

Stream events are built from the SDK's own item types wherever the adapter
calls a method on them (`model_dump`), and from `SimpleNamespace` otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)

from midge import providers
from midge.messages import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from midge.providers.openai_responses import (
    OUTPUT_KEY,
    OpenAIResponsesProvider,
    ResponseFailed,
    encode_body,
)


def _provider() -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(api_key="test")


def _body(messages: list[Any], **kw: Any) -> dict[str, Any]:
    return encode_body(
        messages=messages,
        model=kw.pop("model", "m"),
        tools=kw.pop("tools", None),
        system=kw.pop("system", None),
        **kw,
    )


def _call(call_id: str = "call_1", *, item_id: str = "fc_1") -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        id=item_id,
        call_id=call_id,
        name="read",
        arguments='{"path": "a"}',
        status="completed",
    )


def _reasoning() -> ResponseReasoningItem:
    return ResponseReasoningItem(
        type="reasoning", id="rs_1", summary=[], encrypted_content="opaque"
    )


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        type="message",
        id="msg_1",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _response(output: list[Any], *, status: str = "completed", reason: str | None = None) -> Any:
    return SimpleNamespace(
        status=status,
        output=output,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
        ),
        error=None,
    )


def _event(kind: str, **fields: Any) -> Any:
    return SimpleNamespace(type=kind, **fields)


# --- encoding -------------------------------------------------------------


def test_the_request_is_stateless_and_asks_for_encrypted_reasoning() -> None:
    body = _body([UserMessage(content="hi")], system="be brief")
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["instructions"] == "be brief"
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert body["stream"] is True


def test_tools_are_flat_and_not_strict() -> None:
    # Strict mode would reject any tool with a defaulted parameter.
    spec = {"name": "read", "description": "d", "parameters": {"type": "object"}}
    body = _body([UserMessage(content="hi")], tools=[spec])
    assert body["tools"] == [
        {
            "type": "function",
            "name": "read",
            "description": "d",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]


def test_user_blocks_become_input_parts() -> None:
    msg = UserMessage(
        content=[TextContent(text="see"), ImageContent(data="AAA", mime_type="image/png")]
    )
    assert _body([msg])["input"][0]["content"] == [
        {"type": "input_text", "text": "see"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAA"},
    ]


def test_an_assistant_turn_without_stored_output_is_rebuilt_from_content() -> None:
    msg = AssistantMessage(
        model="m",
        content=[
            TextContent(text="looking"),
            ToolCall(id="call_1", name="read", arguments={"path": "a"}),
        ],
    )
    assert _body([msg])["input"] == [
        {"role": "assistant", "content": "looking"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read",
            "arguments": '{"path": "a"}',
        },
    ]


def test_stored_output_is_replayed_verbatim_to_the_same_model() -> None:
    stored = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "x", "summary": []}]
    msg = AssistantMessage(model="m", content=[TextContent(text="hi")], extra={OUTPUT_KEY: stored})
    assert _body([msg], model="m")["input"] == stored


def test_stored_output_is_not_sent_to_another_model() -> None:
    # Encrypted reasoning is readable only by the model that produced it.
    stored = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "x", "summary": []}]
    msg = AssistantMessage(
        model="other", content=[TextContent(text="hi")], extra={OUTPUT_KEY: stored}
    )
    assert _body([msg], model="m")["input"] == [{"role": "assistant", "content": "hi"}]


def test_a_text_tool_result_is_a_string() -> None:
    msg = ToolResultMessage(tool_call_id="call_1", content=[TextContent(text="ok")])
    assert _body([msg])["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    ]


def test_a_tool_result_with_an_image_keeps_it() -> None:
    msg = ToolResultMessage(
        tool_call_id="call_1",
        content=[TextContent(text="shot"), ImageContent(data="AAA", mime_type="image/png")],
    )
    assert _body([msg])["input"][0]["output"] == [
        {"type": "input_text", "text": "shot"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAA"},
    ]


# --- decoding -------------------------------------------------------------


def test_text_deltas_decode() -> None:
    assert _provider().decode(_event("response.output_text.delta", delta="hel")).text == "hel"


def test_a_function_call_opens_on_item_added_and_fills_from_deltas() -> None:
    p = _provider()
    added = p.decode(
        _event(
            "response.output_item.added",
            output_index=2,
            item=SimpleNamespace(type="function_call", call_id="call_1", name="read", arguments=""),
        )
    )
    (start,) = added.tool_calls
    assert (start.index, start.id, start.name, start.arguments) == (2, "call_1", "read", "")

    delta = p.decode(_event("response.function_call_arguments.delta", output_index=2, delta='{"p'))
    (frag,) = delta.tool_calls
    assert (frag.index, frag.id, frag.arguments) == (2, None, '{"p')


def test_a_message_item_added_is_nothing() -> None:
    ev = _event("response.output_item.added", output_index=0, item=SimpleNamespace(type="message"))
    assert _provider().decode(ev) == providers.Delta()


def test_completion_carries_usage_and_the_output_to_replay() -> None:
    d = _provider().decode(
        _event("response.completed", response=_response([_reasoning(), _message("hi")]))
    )
    assert d.stop_reason == "stop"
    assert d.usage is not None and (d.usage.input, d.usage.output, d.usage.cached) == (10, 5, 3)
    assert d.extra is not None
    assert [i["type"] for i in d.extra[OUTPUT_KEY]] == ["reasoning", "message"]
    assert d.extra[OUTPUT_KEY][0]["encrypted_content"] == "opaque"


def test_a_completion_with_a_function_call_is_tool_use() -> None:
    d = _provider().decode(
        _event("response.completed", response=_response([_reasoning(), _call()]))
    )
    assert d.stop_reason == "tool_use"


@pytest.mark.parametrize(
    ("reason", "expected"), [("max_output_tokens", "length"), ("content_filter", "error")]
)
def test_an_incomplete_response_maps_its_reason(reason: str, expected: str) -> None:
    ev = _event("response.incomplete", response=_response([], status="incomplete", reason=reason))
    assert _provider().decode(ev).stop_reason == expected


def test_a_failed_response_raises_with_its_message() -> None:
    response = _response([], status="failed")
    response.error = SimpleNamespace(message="overloaded", code="server_error")
    with pytest.raises(ResponseFailed, match="overloaded") as info:
        _provider().decode(_event("response.failed", response=response))
    assert info.value.code == "server_error"


def test_an_error_event_raises() -> None:
    with pytest.raises(ResponseFailed, match="bad"):
        _provider().decode(_event("error", message="bad", code="invalid_prompt"))


def test_bookkeeping_events_decode_to_nothing() -> None:
    assert _provider().decode(_event("response.in_progress")) == providers.Delta()


# --- errors and registration ----------------------------------------------


def test_only_transient_stream_failures_are_retryable() -> None:
    p = _provider()
    assert p.is_retryable(ResponseFailed("x", "server_error"))
    assert p.is_retryable(ResponseFailed("x", "rate_limit_exceeded"))
    assert not p.is_retryable(ResponseFailed("x", "invalid_prompt"))
    assert not p.is_retryable(ResponseFailed("x"))


def test_openai_is_the_responses_adapter() -> None:
    p = providers.get("openai")(api_key="k", base_url=None)
    assert isinstance(p, OpenAIResponsesProvider)
    assert p.name == "openai"
    assert not isinstance(
        providers.get("openai-compatible")(api_key="k", base_url=None), OpenAIResponsesProvider
    )
