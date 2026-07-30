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

`Delta` is the normalization point. Everything above it speaks midge's own
vocabulary; everything below is the vendor's.

Two names are registered against one adapter today, because OpenAI and an
OpenAI-compatible server (ollama, vLLM, LM Studio, llama.cpp) share a wire
format and differ only in what they tolerate. Those differences are declared as
`Capabilities` rather than discovered at runtime by catching a 400.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from midge.messages import Message, StopReason, Usage


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a provider tolerates, declared rather than probed.

    `requires_api_key` is False for local servers, which accept any placeholder.
    `stream_usage` says the provider accepts a request for token counts on the
    stream; a server that does not may reject the whole turn with a 400 rather
    than ignoring the field, which is why this is declared per provider instead
    of always sent.
    """

    requires_api_key: bool = True
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
class Provider(Protocol):
    name: str
    capabilities: Capabilities

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

    def decode(self, chunk: Any) -> Delta:
        ...

    def is_retryable(self, exc: BaseException) -> bool:
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
    """Pick a provider name: explicit, then env, then a base_url heuristic.

    The heuristic is that a `base_url` means someone pointed midge at a server
    that is not OpenAI. It is a guess, so the caller logs what it resolved to —
    an implicit choice that is invisible is a debugging trap.
    """
    if provider:
        return provider
    from_env = os.getenv("MIDGE_PROVIDER")
    if from_env:
        return from_env
    return "openai-compatible" if base_url else "openai"
