"""A fake provider, plus `Delta` builders, shared across the test suite.

What is faked is the **transport and the vendor chunk format** — nothing else.
`encode` calls the real OpenAI body builder and `is_retryable` the real
classifier, because those are midge's actual behaviour and a test asserting on a
request body should be asserting on the body midge would really send.

That split is the point. Before this existed, every test file stubbed
`client._client` with a `SimpleNamespace` shaped like the OpenAI SDK, so the
whole suite was coupled to a vendor's chunk format to test things like steering
and compaction. Now a turn is a list of `Delta`s.

The SDK's chunk shape is still exercised, in `test_providers_openai.py` against
`decode`, and end-to-end in `test_client.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from midge.client import Client
from midge.messages import Message, StopReason, Usage
from midge.providers import Capabilities, Delta, RateLimiter, ToolCallFragment
from midge.providers.openai_compat import OpenAIProvider, encode_body

# --- Delta builders -------------------------------------------------------
#
# One per kind of thing a chunk can carry, because a provider may split them
# however it likes and tests read better as a sequence of small facts.
#
# Named `say`/`finish`/`tokens` rather than the obvious `text`/`stop`/`usage`
# because all three of those are already parameter or local names in the suite
# — `_says(text: str)` and `_assistant(..., usage=, stop=)` would shadow the
# builder inside functions that need it.


def say(s: str) -> Delta:
    return Delta(text=s)


def finish(reason: StopReason = "stop") -> Delta:
    return Delta(stop_reason=reason)


def tcall(
    *, index: int = 0, id: str | None = None, name: str | None = None, args: str = ""
) -> Delta:
    return Delta(
        tool_calls=(ToolCallFragment(index=index, id=id, name=name, arguments=args),)
    )


def tokens(*, input: int = 0, output: int = 0, cached: int = 0) -> Delta:
    return Delta(usage=Usage(input=input, output=output, cached=cached))


def whole_call(
    name: str, args: str, *, index: int = 0, id: str | None = None
) -> list[Delta]:
    """A complete tool call as two fragments, the way a provider streams one."""
    return [
        tcall(index=index, id=id or f"call_{index}", name=name),
        tcall(index=index, args=args),
    ]


# --- the fake provider ----------------------------------------------------


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        turns: list[list[Any]] | None = None,
        *,
        capabilities: Capabilities | None = None,
    ) -> None:
        self.capabilities = capabilities or Capabilities()
        # No limits by default, so the streaming tests never wait. A rate-limit
        # test opts in by assigning the real `CoolOff`.
        self.limiter: RateLimiter | None = None
        # Every body midge produced, in order. Tests assert on these.
        self.bodies: list[dict[str, Any]] = []
        # How many times a request was opened — what a retry test counts.
        self.attempts = 0
        self._turns: list[list[Any]] = list(turns or [])
        self._turn = 0

    def encode(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[dict[str, Any]] | None,
        system: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body = encode_body(
            messages=messages,
            model=model,
            tools=tools,
            system=system,
            stream_usage=self.capabilities.stream_usage,
            **kwargs,
        )
        self.bodies.append(body)
        return body

    async def open(self, body: dict[str, Any]) -> AsyncIterator[Any]:
        if self._turn >= len(self._turns):
            raise AssertionError(
                f"provider asked for turn {self._turn + 1} but only "
                f"{len(self._turns)} were installed"
            )
        chunks = self._turns[self._turn]
        self._turn += 1
        self.attempts += 1
        # An exception anywhere in the list is raised when iteration reaches it.
        # First position means the request itself failed; later means the stream
        # dropped mid-response, which is the case a retry must *not* replay.
        if chunks and isinstance(chunks[0], BaseException):
            raise chunks[0]
        return _aiter(chunks)

    def decode(self, chunk: Any) -> Delta:
        # Chunks already *are* Deltas: this provider's wire format is midge's.
        assert isinstance(chunk, Delta), f"expected a Delta, got {type(chunk).__name__}"
        return chunk

    def is_retryable(self, exc: BaseException) -> bool:
        # The real classifier, so retry tests exercise the real policy.
        return _CLASSIFIER.is_retryable(exc)

    def retry_after(self, exc: BaseException) -> float | None:
        # Likewise the real header parsing, so a test can hand this provider an
        # error carrying `Retry-After` and see the wait it produces.
        return _CLASSIFIER.retry_after(exc)


class GatedProvider(FakeProvider):
    """Yields its chunks, then blocks until released.

    For tests that need to observe in-flight state — steering, abort, refusing a
    reload mid-turn.
    """

    def __init__(self, turns: list[list[Any]], gate: Any) -> None:
        super().__init__(turns)
        self._gate = gate

    async def open(self, body: dict[str, Any]) -> AsyncIterator[Any]:
        chunks = self._turns[min(self._turn, len(self._turns) - 1)]
        self._turn += 1
        return _gated_aiter(chunks, self._gate)


class ScriptedProvider(FakeProvider):
    """`open` runs a coroutine you supply, which returns the chunks.

    For tests about *timing* rather than content — a request that hangs until
    cancelled, one that counts how many are in flight, one that fails the first
    attempt. Those need to run code at request time, not just hand back a list.
    """

    def __init__(self, on_open: Any) -> None:
        super().__init__([])
        self._on_open = on_open

    async def open(self, body: dict[str, Any]) -> AsyncIterator[Any]:
        chunks = await self._on_open(body)
        return _aiter(list(chunks or ()))


async def _aiter(chunks: list[Any]) -> AsyncIterator[Any]:
    for c in chunks:
        if isinstance(c, BaseException):
            raise c
        yield c


async def _gated_aiter(chunks: list[Any], gate: Any) -> AsyncIterator[Any]:
    for c in chunks:
        yield c
    await gate.wait()


# Constructed once, without a base_url, purely to borrow `is_retryable`. It makes
# no network calls.
_CLASSIFIER = OpenAIProvider(api_key="test")


# --- installation ---------------------------------------------------------


def install(client: Client, turns: list[list[Any]]) -> list[dict[str, Any]]:
    """Point `client` at a fake provider; return the list its bodies land in.

    Returns the list rather than the provider so it drops straight into the
    `captured = _install_turns(...)` shape the suite already used — the migration
    changes how a turn is *spelled*, not what tests assert.
    """
    provider = FakeProvider(turns)
    client.provider = provider
    return provider.bodies


def install_provider(client: Client, turns: list[list[Any]]) -> FakeProvider:
    """Like `install`, but hands back the provider — for `.attempts`."""
    provider = FakeProvider(turns)
    client.provider = provider
    return provider


def install_gated(client: Client, chunks: list[Any], gate: Any) -> GatedProvider:
    provider = GatedProvider([chunks], gate)
    client.provider = provider
    return provider


def fake_client(turns: list[list[Any]], **kw: Any) -> tuple[Client, FakeProvider]:
    client = Client(provider=FakeProvider(turns), **kw)
    assert isinstance(client.provider, FakeProvider)
    return client, client.provider
