"""The provider contract and registry.

See `midge.providers` for what a provider is and why the split is here.

A provider knows a wire format and nothing else. It does not own the streaming
state machine — assembling partial messages, tracking content indices, buffering
tool-call arguments and deciding when a retry is still safe all live in
`client.py`, written once. That code is subtle enough that a second copy per
vendor would be the main source of bugs in this package.

So the contract is deliberately small:

    encode()        midge messages  ->  a request body
    open()          a request body  ->  an async iterator of the vendor's chunks
    decode()        one vendor chunk -> a `Delta`
    is_retryable()  an exception    ->  worth another attempt?
    retry_after()   an exception    ->  how long the server asked us to wait
    limiter         this vendor's rate limits, or None if it has none

`Delta` is the normalization point. Everything above it speaks midge's own
vocabulary; everything below is the vendor's.

Two names are registered against one adapter today, because OpenAI and an
OpenAI-compatible server (ollama, vLLM, LM Studio, llama.cpp) share a wire
format and differ only in what they tolerate. Those differences are declared as
`Capabilities` rather than discovered at runtime by catching a 400.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from midge.messages import Message, StopReason, Usage


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a provider tolerates, declared rather than probed.

    `stream_usage` says the provider accepts a request for token counts on the
    stream. A server that does not may reject the whole turn with a 400 rather
    than ignoring the field, which is why this is declared per provider instead
    of always sent.

    Deliberately one field. A `requires_api_key` flag was tried and removed: the
    behaviour it was supposed to express is already the `or "not-needed"`
    fallback, which applies to every provider here, so the flag had no reader.
    """

    stream_usage: bool = True


@dataclass(slots=True)
class ToolCallFragment:
    """Part of one tool call. Providers stream these split arbitrarily.

    `index` groups fragments belonging to the same call; `id` and `name` arrive
    once, usually on the first fragment, and `arguments` accumulates.
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(slots=True)
class Delta:
    """One chunk, in midge's vocabulary.

    Every field is optional because providers split content freely: a chunk may
    carry text, tool-call fragments, a stop reason, usage, or nothing at all.
    An all-empty `Delta` is valid and is skipped by the caller.
    """

    text: str = ""
    tool_calls: tuple[ToolCallFragment, ...] = ()
    stop_reason: StopReason | None = None
    usage: Usage | None = None


@runtime_checkable
class RateLimiter(Protocol):
    """What one vendor's rate limits look like, and when it is safe to send.

    Owned by the provider rather than by `client.py` because almost nothing
    about a rate limit is portable. OpenAI counts per model per org and resets
    on `x-ratelimit-reset-*` durations like `6m0s`; Anthropic counts per org
    across a tier and resets on an RFC 3339 timestamp; ollama has no limits at
    all. Key any of that into the core and the next provider has to fight it.

    So the core owns only the mechanics — making N in-flight requests respect
    one answer — and asks three questions:

        wait_for()  how long before it is safe to send?
        observe()   a response arrived; learn from it
        penalize()  a request was rejected; learn from that

    `wait_for` reports rather than sleeps. The wait has to pass the ceiling in
    `[retry] max_delay` and get logged, and both of those live in `client.py`;
    a limiter that blocked internally would put the duration out of reach of
    either. Same division as `retry_after`.

    One instance per provider instance, which `ModelRegistry` already caches —
    so everything routed to a provider shares its limiter, sub-agents included,
    without the core holding any of it.
    """

    def wait_for(self, model: str) -> float:
        """Seconds to hold off before sending, or 0 when clear."""
        ...

    def observe(self, response: Any) -> None:
        """A request succeeded. The argument is the vendor's own object."""
        ...

    def penalize(self, exc: BaseException) -> None:
        """A request failed. Only the vendor knows which failures mean a limit."""
        ...


@runtime_checkable
class Provider(Protocol):
    name: str
    capabilities: Capabilities
    # None means this server has no rate limits worth modelling — a local
    # llama.cpp does not — and the core skips the whole path.
    limiter: RateLimiter | None

    def encode(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[dict[str, Any]] | None,
        system: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the request body. Receives already-repaired history."""
        ...

    def open(self, body: dict[str, Any]) -> Any:
        """Start the request. Returns an awaitable yielding an async iterator."""
        ...

    def decode(self, chunk: Any) -> Delta: ...

    def is_retryable(self, exc: BaseException) -> bool: ...

    def retry_after(self, exc: BaseException) -> float | None:
        """Seconds the server asked us to wait, or None if it did not say.

        Separate from `is_retryable` rather than folded into it: whether to try
        again is a policy question with a boolean answer, and how long to wait
        is a fact the response carries. The header names and their formats are
        the vendor's, which is why the parsing is here and the ceiling that
        caps the answer is in `client.py`.
        """
        ...


# --- registry -------------------------------------------------------------
#
# Factories rather than instances: a provider is constructed with the api key
# and base url, which are per-Client rather than global.

ProviderFactory = Callable[..., Provider]

_REGISTRY: dict[str, ProviderFactory] = {}


def register(name: str, factory: ProviderFactory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Provider {name!r} already registered")
    _REGISTRY[name] = factory


def get(name: str) -> ProviderFactory:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"Unknown provider {name!r}. Registered: {known}") from None


def names() -> list[str]:
    return sorted(_REGISTRY)


def resolve(*, provider: str | None, base_url: str | None) -> str:
    """Pick a provider name: explicit, else a base_url heuristic.

    The heuristic is that a `base_url` means someone pointed midge at a server
    that is not OpenAI. It is a guess, so the caller logs what it resolved to —
    an implicit choice that is invisible is a debugging trap.

    Configuration does not reach here. An explicit name arrives as `provider`,
    which the entrypoint takes from `Config.provider` (itself resolved from
    `MIDGE_PROVIDER` or the config file); this function only fills the gap when
    nobody said.
    """
    if provider:
        return provider
    return "openai-compatible" if base_url else "openai"
