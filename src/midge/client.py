"""The streaming loop: one state machine, any provider.

`Client` owns everything that is the same whatever you are talking to — the
retry policy, assembling a partial `AssistantMessage`, tracking content indices,
buffering tool-call arguments, and emitting the `StreamEvent` taxonomy the rest
of midge consumes.

Wire formats live in `midge.providers`. The seam is `Delta`: a provider turns
its own chunks into one, and this module never sees a vendor type.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from midge import providers
from midge.logs import payload
from midge.messages import (
    AssistantMessage,
    Message,
    StopReason,
    TextContent,
    ToolCall,
    repair_history,
)

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamStart:
    partial: AssistantMessage
    type: Literal["start"] = "start"


@dataclass(slots=True)
class TextStart:
    content_index: int
    partial: AssistantMessage
    type: Literal["text_start"] = "text_start"


@dataclass(slots=True)
class TextDelta:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class TextEnd:
    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["text_end"] = "text_end"


@dataclass(slots=True)
class ToolCallStart:
    content_index: int
    partial: AssistantMessage
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass(slots=True)
class ToolCallDelta:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass(slots=True)
class ToolCallEnd:
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass(slots=True)
class Done:
    message: AssistantMessage
    type: Literal["done"] = "done"


@dataclass(slots=True)
class Error:
    message: AssistantMessage
    type: Literal["error"] = "error"


StreamEvent = (
    StreamStart
    | TextStart
    | TextDelta
    | TextEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | Done
    | Error
)


class Client:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_attempts: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 60.0,
        provider: str | providers.Provider | None = None,
        capabilities: providers.Capabilities | None = None,
        registry: providers.ModelRegistry | None = None,
    ) -> None:
        if isinstance(provider, str) or provider is None:
            name = providers.resolve(provider=provider, base_url=base_url)
            self.provider = providers.get(name)(
                api_key=api_key, base_url=base_url, capabilities=capabilities
            )
            # The base_url heuristic in `resolve` is a guess, so which provider
            # it landed on has to be visible or a misrouted request is a mystery.
            _logger.info("provider_selected name=%s", self.provider.name)
        else:
            self.provider = provider
        # An empty registry means permissive: every model goes to the single
        # provider above, which is what makes this additive — without a
        # `[models]` table nothing behaves differently than it did before.
        self.registry = registry or providers.ModelRegistry(models={}, providers={})
        self._max_attempts = max(1, max_attempts)
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay

    async def stream(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        partial = AssistantMessage(model=model)
        yield StreamStart(partial=partial)

        # The model chooses the provider, per call — `Agent.model` is mutable, a
        # hook can override it, and a sub-agent runs its own model against this
        # same Client. `self.provider` is read here rather than captured, so
        # reassigning it after construction still takes effect.
        #
        # An unroutable model is reported as an Error event rather than raised:
        # by here a consumer is already mid-stream, and every other request
        # failure arrives that way.
        try:
            provider = self.registry.provider_for(model) if self.registry else self.provider
        except providers.UnknownModel as e:
            _logger.warning("model_unroutable model=%s", model)
            partial.stop_reason = "error"
            partial.error_message = f"{type(e).__name__}: {e}"
            yield Error(message=partial)
            return

        # Repaired here, once, so a provider's encoder never has to know which
        # turns are coherent — that is a fact about midge's history.
        body = provider.encode(
            messages=repair_history(messages),
            model=model,
            tools=tools,
            system=system,
            **kwargs,
        )

        _logger.debug(
            # Every argument here is O(1) — `payload` defers the expensive part,
            # but a plain argument is evaluated whether or not DEBUG is on.
            "provider_request provider=%s model=%s tools=%d body=%s",
            provider.name,
            model,
            len(tools or ()),
            payload(body),
        )

        # A hint from the server wins, because it knows when its window resets
        # and we are guessing. The fallback is full jitter — a random point in
        # `[0, base * 2**n]` rather than the endpoint — so the N sub-agents
        # sharing this client do not wake together and collide again.
        backoff = wait_random_exponential(
            multiplier=self._retry_base_delay, max=self._retry_max_delay
        )
        # Recorded rather than recomputed in the log: the date form of
        # `Retry-After` is time-dependent, so asking twice can answer twice.
        wait_source = "backoff"

        def choose_wait(rs: RetryCallState) -> float:
            nonlocal wait_source
            exc = rs.outcome.exception() if rs.outcome else None
            hint = provider.retry_after(exc) if exc is not None else None
            if hint is None:
                wait_source = "backoff"
                return backoff(rs)
            wait_source = "header"
            return min(hint, self._retry_max_delay)

        def log_retry(rs: RetryCallState) -> None:
            _logger.warning(
                "provider_retry model=%s attempt=%d/%d delay=%.2f source=%s error=%s",
                model,
                rs.attempt_number,
                self._max_attempts,
                rs.upcoming_sleep,
                wait_source,
                type(rs.outcome.exception()).__name__ if rs.outcome else "none",
            )

        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=choose_wait,
            # Once a content event is out the consumer has already seen part of
            # this response, and a restart would duplicate it. `partial.content`
            # is read live from the enclosing scope.
            #
            # This predicate is also the only thing standing between a cancelled
            # or abandoned stream and a re-entered body: `AttemptManager.__exit__`
            # swallows whatever was raised and hands it here, and re-entering
            # after a `GeneratorExit` would yield during cleanup. Neither it nor
            # `CancelledError` is retryable, so tenacity re-raises instead.
            retry=retry_if_exception(
                lambda e: not partial.content and provider.is_retryable(e)
            ),
            reraise=True,
            before_sleep=log_retry,
        )

        try:
            async for att in retryer:
                with att:
                    # Each append is immediately followed by its start event, so
                    # a retry always resumes from an empty message. `partial`
                    # itself is never rebound: consumers hold the reference
                    # handed out by StreamStart.
                    assert not partial.content
                    text_index: int | None = None
                    tool_idx_map: dict[int, int] = {}
                    tool_arg_buffers: dict[int, str] = {}
                    stop_reason: StopReason | None = None
                    partial.usage = None

                    try:
                        stream = await provider.open(body)

                        async for chunk in stream:
                            ev = provider.decode(chunk)

                            if ev.usage is not None:
                                partial.usage = ev.usage

                            if ev.text:
                                if text_index is None:
                                    partial.content.append(TextContent(text=""))
                                    text_index = len(partial.content) - 1
                                    yield TextStart(content_index=text_index, partial=partial)
                                text_block = partial.content[text_index]
                                assert isinstance(text_block, TextContent)
                                text_block.text += ev.text
                                yield TextDelta(
                                    content_index=text_index,
                                    delta=ev.text,
                                    partial=partial,
                                )

                            for frag in ev.tool_calls:
                                idx = frag.index
                                if idx not in tool_idx_map:
                                    tool_arg_buffers[idx] = ""
                                    partial.content.append(
                                        ToolCall(
                                            id=frag.id or f"call_{idx}",
                                            name=frag.name or "",
                                            arguments={},
                                        )
                                    )
                                    content_idx = len(partial.content) - 1
                                    tool_idx_map[idx] = content_idx
                                    yield ToolCallStart(content_index=content_idx, partial=partial)

                                if frag.arguments:
                                    tool_arg_buffers[idx] += frag.arguments
                                    yield ToolCallDelta(
                                        content_index=tool_idx_map[idx],
                                        delta=frag.arguments,
                                        partial=partial,
                                    )

                            if ev.stop_reason is not None:
                                stop_reason = ev.stop_reason

                        if text_index is not None:
                            text_block = partial.content[text_index]
                            assert isinstance(text_block, TextContent)
                            yield TextEnd(
                                content_index=text_index,
                                content=text_block.text,
                                partial=partial,
                            )

                        for idx, content_idx in tool_idx_map.items():
                            tc = partial.content[content_idx]
                            assert isinstance(tc, ToolCall)
                            buf = tool_arg_buffers[idx]
                            try:
                                tc.arguments = json.loads(buf) if buf else {}
                            except json.JSONDecodeError as e:
                                tc.arguments = {}
                                tc.arguments_error = str(e)
                                _logger.warning(
                                    "tool_call_args_unparseable tool=%s id=%s error=%s buf=%.200r",
                                    tc.name,
                                    tc.id,
                                    e,
                                    buf,
                                )
                            yield ToolCallEnd(
                                content_index=content_idx,
                                tool_call=tc,
                                partial=partial,
                            )

                        # A provider that never sent one still finished; treating
                        # that as `stop` matches what every server without an
                        # explicit reason means by closing the stream.
                        partial.stop_reason = stop_reason or "stop"
                        _logger.debug(
                            "provider_response provider=%s model=%s stop=%s blocks=%d attempt=%d",
                            provider.name,
                            model,
                            partial.stop_reason,
                            len(partial.content),
                            att.retry_state.attempt_number,
                        )
                        if partial.usage is not None:
                            u = partial.usage
                            _logger.info(
                                "provider_usage model=%s input=%d output=%d cached=%d",
                                model,
                                u.input,
                                u.output,
                                u.cached,
                            )
                        if att.retry_state.attempt_number > 1:
                            _logger.info(
                                "provider_recovered model=%s attempts=%d",
                                model,
                                att.retry_state.attempt_number,
                            )
                        yield Done(message=partial)
                        return

                    except asyncio.CancelledError:
                        # Reported here, inside the attempt, because only a
                        # cancel with a request in flight has anything to report.
                        # One during the backoff sleep lands below instead, and
                        # deliberately emits nothing.
                        _logger.info(
                            "provider_stream_cancelled model=%s blocks=%d",
                            model,
                            len(partial.content),
                        )
                        partial.stop_reason = "aborted"
                        partial.error_message = "cancelled"
                        yield Error(message=partial)
                        raise

        except Exception as e:
            # Retries exhausted, not retryable, or the stream dropped after
            # content escaped — `reraise=True` brings all three here.
            #
            # `str(e)` is all the caller gets; without this the type and
            # traceback are gone and there is nowhere else to look.
            _logger.exception("provider_stream_failed model=%s", model)
            partial.stop_reason = "error"
            partial.error_message = f"{type(e).__name__}: {e}"
            yield Error(message=partial)
            return
