"""The OpenAI adapter: encoding, decoding, and error classification.

The only place in the suite that constructs SDK-shaped chunks. Everything else
drives a `FakeProvider` and speaks `Delta`, so this file is where a change to the
chat-completions wire format has to be caught.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from midge.messages import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from midge.providers import Capabilities
from midge.providers.openai_compat import OpenAIProvider, encode_body, encode_messages


def _provider(**kw: Any) -> OpenAIProvider:
    return OpenAIProvider(api_key="test", **kw)


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _tcd(
    *,
    index: int,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://x")
    response = httpx.Response(status_code=status, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


# --- encoding -------------------------------------------------------------


def test_a_user_message_with_blocks_becomes_content_parts() -> None:
    [out] = encode_messages(
        [
            UserMessage(
                content=[
                    TextContent(text="look"),
                    ImageContent(data="Zm8=", mime_type="image/png"),
                ]
            )
        ]
    )
    assert out["role"] == "user"
    assert out["content"][0] == {"type": "text", "text": "look"}
    assert out["content"][1]["image_url"]["url"] == "data:image/png;base64,Zm8="


def test_an_assistant_tool_call_is_serialized_as_a_function() -> None:
    [out] = encode_messages(
        [
            AssistantMessage(
                content=[ToolCall(id="c1", name="read", arguments={"path": "x"})],
                stop_reason="tool_use",
            )
        ]
    )
    assert out["tool_calls"][0]["type"] == "function"
    assert out["tool_calls"][0]["function"] == {
        "name": "read",
        "arguments": '{"path": "x"}',
    }


def test_an_image_in_a_tool_result_is_dropped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The tool role carries text only, so dropping is correct — but silently
    # losing a hook's output would leave no trace.
    with caplog.at_level("WARNING", logger="midge.providers.openai_compat"):
        [out] = encode_messages(
            [
                ToolResultMessage(
                    tool_call_id="c1",
                    tool_name="shot",
                    content=[
                        TextContent(text="ok"),
                        ImageContent(data="Zm8=", mime_type="image/png"),
                    ],
                )
            ]
        )
    assert out == {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    assert any("tool_result_image_dropped" in r.getMessage() for r in caplog.records)


def test_the_system_prompt_leads_the_message_list() -> None:
    body = encode_body(
        messages=[UserMessage(content="hi")],
        model="m",
        tools=None,
        system="be brief",
        stream_usage=False,
    )
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["stream"] is True
    assert "stream_options" not in body


def test_tools_are_wrapped_as_functions() -> None:
    body = encode_body(
        messages=[],
        model="m",
        tools=[{"name": "read", "parameters": {}}],
        system=None,
        stream_usage=False,
    )
    assert body["tools"] == [{"type": "function", "function": {"name": "read", "parameters": {}}}]


# --- stream_options, and the capability that governs it -------------------


def test_stream_usage_is_requested_when_the_capability_says_so() -> None:
    body = _provider().encode(messages=[], model="m", tools=None, system=None)
    assert body["stream_options"] == {"include_usage": True}


def test_a_provider_without_the_capability_omits_stream_options() -> None:
    p = _provider(capabilities=Capabilities(stream_usage=False))
    assert "stream_options" not in p.encode(messages=[], model="m", tools=None, system=None)


def test_the_env_override_beats_the_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """`MIDGE_INCLUDE_USAGE` exists because a server that does not support
    `stream_options` rejects the whole turn with a 400, and an operator hitting
    that needs a fix that does not require editing code."""
    monkeypatch.setenv("MIDGE_INCLUDE_USAGE", "0")
    assert "stream_options" not in _provider().encode(
        messages=[], model="m", tools=None, system=None
    )

    monkeypatch.setenv("MIDGE_INCLUDE_USAGE", "1")
    p = _provider(capabilities=Capabilities(stream_usage=False))
    assert "stream_options" in p.encode(messages=[], model="m", tools=None, system=None)


# --- decoding -------------------------------------------------------------


def test_text_and_finish_reason_decode() -> None:
    p = _provider()
    assert p.decode(_chunk(content="hi")).text == "hi"
    assert p.decode(_chunk(finish_reason="stop")).stop_reason == "stop"


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", "stop"),
        ("tool_calls", "tool_use"),
        ("length", "length"),
        ("content_filter", "error"),
        ("something_new", "stop"),
    ],
)
def test_finish_reasons_map_to_stop_reasons(finish_reason: str, expected: str) -> None:
    assert _provider().decode(_chunk(finish_reason=finish_reason)).stop_reason == expected


def test_a_tool_call_fragment_decodes() -> None:
    d = _provider().decode(
        _chunk(tool_calls=[_tcd(index=2, id="c1", name="read", arguments='{"p":')])
    )
    (frag,) = d.tool_calls
    assert (frag.index, frag.id, frag.name, frag.arguments) == (2, "c1", "read", '{"p":')


def test_a_continuation_fragment_carries_only_arguments() -> None:
    d = _provider().decode(_chunk(tool_calls=[_tcd(index=0, arguments='"x"}')]))
    (frag,) = d.tool_calls
    assert frag.id is None and frag.name is None and frag.arguments == '"x"}'


def test_usage_rides_a_chunk_with_no_choices() -> None:
    # The reason usage is read before `choices` is inspected: the final chunk has
    # an empty `choices`, and skipping it early is exactly what threw usage away.
    chunk = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=1200,
            completion_tokens=35,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1024),
        ),
    )
    d = _provider().decode(chunk)
    assert d.usage is not None
    assert (d.usage.input, d.usage.output, d.usage.cached) == (1200, 35, 1024)
    assert d.text == ""


def test_usage_tolerates_a_server_that_omits_fields() -> None:
    # Local OpenAI-compatible servers routinely omit these.
    d = _provider().decode(SimpleNamespace(choices=[], usage=SimpleNamespace()))
    assert d.usage is not None
    assert (d.usage.input, d.usage.output, d.usage.cached) == (0, 0, 0)


def test_a_chunk_with_no_usage_decodes_to_none() -> None:
    assert _provider().decode(_chunk(content="x")).usage is None


# --- error classification -------------------------------------------------


def test_retryable_errors() -> None:
    p = _provider()
    assert p.is_retryable(_status_error(429))
    assert p.is_retryable(_status_error(500))
    assert p.is_retryable(_status_error(503))
    assert p.is_retryable(openai.APIConnectionError(request=httpx.Request("POST", "http://x")))
    # APITimeoutError subclasses APIConnectionError.
    assert p.is_retryable(openai.APITimeoutError(request=httpx.Request("POST", "http://x")))


def test_non_retryable_errors() -> None:
    p = _provider()
    assert not p.is_retryable(_status_error(400))
    assert not p.is_retryable(_status_error(401))
    assert not p.is_retryable(_status_error(404))
    assert not p.is_retryable(RuntimeError("boom"))


# --- construction ---------------------------------------------------------


def test_the_sdk_does_not_retry_on_its_own() -> None:
    # The SDK's backoff sleep ignores cancellation, so the retry must be ours.
    assert _provider()._client.max_retries == 0


def test_a_compatible_server_needs_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = OpenAIProvider(
        name="openai-compatible",
        capabilities=Capabilities(requires_api_key=False),
        base_url="http://localhost:11434/v1",
    )
    assert p.capabilities.requires_api_key is False
    # The SDK insists on something; a placeholder is what local servers accept.
    assert p._client.api_key == "not-needed"


def test_both_names_are_registered() -> None:
    from midge import providers

    assert providers.names() == ["openai", "openai-compatible"]
    assert providers.get("openai")(api_key="k", base_url=None).name == "openai"


def test_resolution_prefers_explicit_then_env_then_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from midge import providers

    monkeypatch.delenv("MIDGE_PROVIDER", raising=False)
    assert providers.resolve(provider=None, base_url=None) == "openai"
    assert providers.resolve(provider=None, base_url="http://x/v1") == "openai-compatible"

    monkeypatch.setenv("MIDGE_PROVIDER", "openai")
    assert providers.resolve(provider=None, base_url="http://x/v1") == "openai"
    assert providers.resolve(provider="openai-compatible", base_url=None) == "openai-compatible"


def test_an_unknown_provider_names_the_registered_ones() -> None:
    from midge import providers

    with pytest.raises(KeyError, match="openai-compatible"):
        providers.get("anthropic")
