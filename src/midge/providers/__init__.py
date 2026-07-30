"""Providers: translation and transport for one model API.

A provider knows a wire format and nothing else. It does not own the streaming
state machine — assembling partial messages, tracking content indices, buffering
tool-call arguments and deciding when a retry is still safe all live in
`client.py`, written once. That code is subtle enough that a second copy per
vendor would be the main source of bugs here.

So the contract is deliberately small:

    encode()        midge messages   ->  a request body
    open()          a request body   ->  an async iterator of the vendor's chunks
    decode()        one vendor chunk ->  a `Delta`
    is_retryable()  an exception     ->  worth another attempt?

`Delta` is the normalization point. Everything above it speaks midge's own
vocabulary; everything below is the vendor's.

Two names are registered against one adapter today, because OpenAI and an
OpenAI-compatible server (ollama, vLLM, LM Studio, llama.cpp) share a wire
format and differ only in what they tolerate. Those differences are declared as
`Capabilities` rather than discovered at runtime by catching a 400.

The contract lives in `base` so that an adapter can import it without importing
this module, which imports every adapter to register it.
"""

from __future__ import annotations

# Imported for the side effect of registering. The module is `openai_compat`
# rather than `openai` so that its own `import openai` reaches the SDK.
from midge.providers import openai_compat  # noqa: F401
from midge.providers.base import (
    Capabilities,
    Delta,
    Provider,
    ProviderFactory,
    ToolCallFragment,
    get,
    names,
    register,
    resolve,
)
from midge.providers.registry import ModelRegistry, UnknownModel

__all__ = [
    "Capabilities",
    "Delta",
    "ModelRegistry",
    "Provider",
    "ProviderFactory",
    "ToolCallFragment",
    "UnknownModel",
    "get",
    "names",
    "register",
    "resolve",
]
