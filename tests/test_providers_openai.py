"""The OpenAI adapter: encoding, decoding, and error classification.

The only place in the suite that constructs SDK-shaped chunks. Everything else
drives a `FakeProvider` and speaks `Delta`, so this file is where a change to the
chat-completions wire format has to be caught.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
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
from midge.providers.openai_compat import (
    CoolOff,
    OpenAIProvider,
    encode_body,
    encode_messages,
)


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


def _status_error(
    status: int, headers: dict[str, str] | None = None, model: str | None = None
) -> openai.APIStatusError:
    # `model` rides in the request body because that is where it really is —
    # the cool-off reads it back off the rejected request rather than being
    # told, so nothing has to thread it through the retry loop.
    request = httpx.Request(
        "POST", "http://x", json={"model": model} if model else None
    )
    response = httpx.Response(status_code=status, request=request, headers=headers)
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


def test_a_registered_provider_accepts_a_capability_override() -> None:
    """How `include_usage` reaches the adapter.

    A server that does not support `stream_options` rejects the whole turn with
    a 400 rather than ignoring the field, so an operator has to be able to turn
    it off without editing code. Where that instruction comes from is
    `midge.config`'s business; the adapter only has to be overridable.
    """
    from midge import providers

    factory = providers.get("openai-compatible")
    default = factory(api_key="k", base_url=None)
    assert default.capabilities.stream_usage is True

    overridden = factory(api_key="k", base_url=None, capabilities=Capabilities(stream_usage=False))
    assert "stream_options" not in overridden.encode(
        messages=[], model="m", tools=None, system=None
    )


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


# --- the wait the server asked for ----------------------------------------


def test_retry_after_in_seconds() -> None:
    assert _provider().retry_after(_status_error(429, {"retry-after": "20"})) == 20


def test_retry_after_ms_wins_when_both_are_present() -> None:
    # `Retry-After: 1` is 200ms rounded up to a whole second; the precise one is
    # what OpenAI's own SDK prefers.
    p = _provider()
    exc = _status_error(429, {"retry-after": "1", "retry-after-ms": "200"})
    assert p.retry_after(exc) == pytest.approx(0.2)


def test_retry_after_as_an_http_date() -> None:
    when = datetime.now(UTC) + timedelta(seconds=30)
    exc = _status_error(429, {"retry-after": format_datetime(when, usegmt=True)})
    delay = _provider().retry_after(exc)
    assert delay is not None
    # The date has one-second resolution and time passes between the two calls.
    assert 28 <= delay <= 30


def test_a_date_already_past_is_not_a_hint() -> None:
    when = datetime.now(UTC) - timedelta(seconds=30)
    exc = _status_error(429, {"retry-after": format_datetime(when, usegmt=True)})
    assert _provider().retry_after(exc) is None


@pytest.mark.parametrize("value", ["", "soon", "-5", "0", "NaN "])
def test_an_unusable_retry_after_is_no_hint(value: str) -> None:
    # Falls back to the jittered backoff rather than retrying instantly.
    assert _provider().retry_after(_status_error(429, {"retry-after": value})) is None


def test_no_headers_at_all_is_no_hint() -> None:
    assert _provider().retry_after(_status_error(429)) is None


def test_a_connection_error_carries_no_hint() -> None:
    # Nothing answered, so there is no response to read a header off.
    exc = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
    assert _provider().retry_after(exc) is None


# --- the shared cool-off --------------------------------------------------


def _limited(model: str, after: str = "20") -> openai.APIStatusError:
    return _status_error(429, {"retry-after": after}, model=model)


async def test_nothing_is_held_off_to_begin_with() -> None:
    assert CoolOff().wait_for("gpt-4o") == 0


async def test_a_rate_limit_parks_the_model_it_names() -> None:
    cool = CoolOff()
    cool.penalize(_limited("gpt-4o"))
    assert cool.wait_for("gpt-4o") == pytest.approx(20, abs=0.5)


async def test_one_model_being_limited_does_not_park_another() -> None:
    # The whole reason the deadline is keyed by model: OpenAI counts per model,
    # so parking gpt-4o-mini on a gpt-4o rejection is a self-inflicted outage.
    cool = CoolOff()
    cool.penalize(_limited("gpt-4o"))
    assert cool.wait_for("gpt-4o-mini") == 0


async def test_the_longest_deadline_wins() -> None:
    # A later, shorter rejection must not release requests an earlier one parked.
    cool = CoolOff()
    cool.penalize(_limited("gpt-4o", after="60"))
    cool.penalize(_limited("gpt-4o", after="5"))
    assert cool.wait_for("gpt-4o") == pytest.approx(60, abs=0.5)


async def test_the_deadline_runs_down() -> None:
    cool = CoolOff()
    cool.penalize(_limited("gpt-4o", after="0.2"))
    first = cool.wait_for("gpt-4o")
    await asyncio.sleep(0.05)
    assert 0 < cool.wait_for("gpt-4o") < first


async def test_a_rate_limit_naming_no_wait_parks_nothing() -> None:
    # Without a `Retry-After` there is no evidence of how long, and inventing
    # one would park every other request on a guess.
    cool = CoolOff()
    cool.penalize(_status_error(429, model="gpt-4o"))
    assert cool.wait_for("gpt-4o") == 0


async def test_a_server_fault_is_not_a_rate_limit() -> None:
    cool = CoolOff()
    cool.penalize(_status_error(503, {"retry-after": "20"}, model="gpt-4o"))
    assert cool.wait_for("gpt-4o") == 0


async def test_an_unreadable_request_body_parks_nothing() -> None:
    # Nothing says which model was limited, so there is nothing safe to park.
    cool = CoolOff()
    cool.penalize(_limited(""))
    assert cool.wait_for("gpt-4o") == 0


def test_the_provider_carries_a_limiter() -> None:
    assert isinstance(_provider().limiter, CoolOff)


# --- construction ---------------------------------------------------------


def test_the_sdk_does_not_retry_on_its_own() -> None:
    # The SDK's backoff sleep ignores cancellation, so the retry must be ours.
    assert _provider()._client.max_retries == 0


def test_both_names_are_registered() -> None:
    from midge import providers

    assert providers.names() == ["openai", "openai-compatible"]
    assert providers.get("openai")(api_key="k", base_url=None).name == "openai"


def test_resolution_prefers_an_explicit_name_over_the_base_url_heuristic() -> None:
    # An explicit name arrives from `Config.provider`; this only fills the gap
    # when nobody said. The environment is not consulted here — that is
    # `midge.config`'s job, tested in test_config.py.
    from midge import providers

    assert providers.resolve(provider=None, base_url=None) == "openai"
    assert providers.resolve(provider=None, base_url="http://x/v1") == "openai-compatible"
    assert providers.resolve(provider="openai", base_url="http://x/v1") == "openai"
    assert providers.resolve(provider="openai-compatible", base_url=None) == "openai-compatible"


def test_an_unknown_provider_names_the_registered_ones() -> None:
    from midge import providers

    with pytest.raises(KeyError, match="openai-compatible"):
        providers.get("anthropic")


def test_the_env_var_supplies_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert OpenAIProvider(name="openai")._client.api_key == "sk-from-env"


def test_a_local_server_gets_a_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SDK insists on a credential; local servers accept anything.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = OpenAIProvider(name="openai-compatible", base_url="http://localhost:11434/v1")
    assert p._client.api_key == "not-needed"
