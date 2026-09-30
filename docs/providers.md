# Providers

`Client` in [`src/midge/client.py`](../src/midge/client.py) streams one model reply as midge events. A provider in [`src/midge/providers/`](../src/midge/providers/) translates between midge and one vendor's wire format, and the model registry decides which provider serves a given model.

## The split

A provider owns what is specific to a vendor. The contract is the `Provider` protocol in [`providers/base.py`](../src/midge/providers/base.py):

| Member | Does |
|---|---|
| `encode` | Builds a request body from midge messages, model, tools and system prompt |
| `open` | Starts the request and returns the vendor's chunk stream |
| `decode` | Turns one chunk into a `Delta` |
| `is_retryable` | Says whether a failure is worth another attempt |
| `retry_after` | Reads how long the server asked the client to wait, if it said |
| `describe` | Turns an exception into a sentence a person can act on (for example, a 401 names the key variable) |
| `credential_problem` | Reports at startup that the first request will be refused, such as a missing key |
| `capabilities` | What the server tolerates, declared (`stream_usage`) |
| `limiter` | The vendor's rate-limit model, or `None` |

`Client` owns everything that is the same for every vendor: the streaming state machine, the retry loop, the wait ceilings and the sleeping. A provider answers how long and whether; it never sleeps, logs a retry, or caps a wait.

`Delta` is the boundary between the two. It carries a text fragment, tool-call fragments (grouped by `index`), a `stop_reason`, `usage`, and `extra`. `extra` is provider state the client copies onto `AssistantMessage.extra` and the provider receives back in `encode` on later requests.

## Stream events

`Client.stream` yields these, in order. Each content event carries a `content_index` into the message and the shared `partial` message.

| Event | When |
|---|---|
| `start` | Before the request; carries the partial `AssistantMessage` |
| `text_start` / `text_delta` / `text_end` | Assistant text |
| `toolcall_start` / `toolcall_delta` / `toolcall_end` | One tool call; deltas are raw argument JSON |
| `done` | Success; carries the final message |
| `error` | Failure (`stop_reason="error"`) or cancellation (`"aborted"`) |

Tool-call arguments are buffered and parsed once at the end of the stream. If the JSON does not parse, the call's `arguments_error` is set and the agent loop fails the call instead of running it with empty arguments.

History passes through `messages.repair_history` before `encode` (see [agent-loop](agent-loop.md)), so an adapter only encodes.

## Retries and rate limits

A failed request is retried when the provider calls the error retryable (rate limits, server errors and transport failures for the OpenAI adapters) and no content event has been emitted yet. Once any text or tool call has reached the consumer, a failure ends the stream with an `error` event, because a restart would duplicate what was already shown.

The wait before a retry is the server's `Retry-After` hint when there is one, otherwise a random point in an exponential window (full jitter), so concurrent sub-agents sharing a client do not retry in step. Every wait is capped. The keys are `[retry] max_attempts`, `base_delay` and `max_delay`.

Before each attempt the client asks the provider's `limiter` how long to hold off, and after a failed attempt it passes the error to `limiter.penalize`. The OpenAI adapters' limiter records a deadline per model when a 429 names a wait, so every request to that model, sub-agents included, waits out one rejection instead of each earning its own.

Cancellation emits an `error` event with `stop_reason="aborted"` and re-raises.

## Adapters

Two adapters are registered:

- **`openai`** ([`openai_responses.py`](../src/midge/providers/openai_responses.py)) speaks OpenAI's Responses API. Requests are sent with `store: false`, so each response's output items, encrypted reasoning included, are kept in `extra` and replayed on the next request to the same model. A different model receives plain text and function calls instead. Failures reported inside the stream are retried when their code is `server_error` or `rate_limit_exceeded`.
- **`openai-compatible`** ([`openai_compat.py`](../src/midge/providers/openai_compat.py)) speaks chat completions, for servers that implement only that (ollama, vLLM, LM Studio, llama.cpp). It asks for token usage on the stream when `stream_usage` is on; some servers reject that field with a 400, which `include_usage = false` fixes. Images in tool results are dropped with a warning, since the tool role carries text only.

The Responses adapter subclasses the chat-completions one and inherits its error handling, since both talk to the same vendor.

## Choosing a provider

With no model registry, every request goes to one provider set by `[provider]`:

- `name` picks the adapter. Omitted, it is `openai` without a `base_url` and `openai-compatible` with one; the choice is logged as `provider_selected`.
- `base_url` points at the server.
- `include_usage` overrides the adapter's `stream_usage`.

The key comes from `OPENAI_API_KEY`. A server that needs no key gets a placeholder. A key is never read from the config file.

## The model registry

The registry lets one process use several services, with the model deciding which one serves a request. It is two tables:

- `[providers.<name>]` says how to reach a service: `kind` (the adapter), `base_url`, `api_key_env` (the name of the environment variable holding the key) and `include_usage`.
- `[models."<id>"]` says which provider a model lives on: `provider = "<name>"`.

[`providers/registry.py`](../src/midge/providers/registry.py) resolves the model on every request, since the model can change between requests (`set_model`, a `before_provider_request` hook, a [profile](profiles.md), or a [sub-agent](subagents.md) with its own model). Provider instances are built on first use and cached, so everything routed to one provider shares its rate limiter.

An empty registry is permissive: any model goes to the `[provider]` default. Once `[models]` has an entry, only listed models are accepted:

- Startup refuses a configured model that is not listed.
- `set_model` refuses one and names the alternatives.
- A resumed session whose recorded model is no longer listed falls back to the configured model with a warning (see [sessions](sessions.md)).
- A request for an unlisted model ends in an `error` event.

midge ships no list of models and does not check that a model id exists; a typo fails at the vendor with the vendor's error. It checks only the wiring and reports problems as startup diagnostics: a provider whose `kind` is not a registered adapter, a model naming an undefined provider, providers defined with no models routed to them, and `[provider]` set alongside `[providers.*]` (the singular table is then ignored).

See [`examples/config.toml`](../examples/config.toml) for every key and [config](config.md) for how configuration is loaded.

## Adding an adapter

Implement the `Provider` protocol and register a factory with `providers.register(name, factory)`. The factory is called with `api_key`, `base_url` and `capabilities` keyword arguments. Import the module from [`providers/__init__.py`](../src/midge/providers/__init__.py) so it registers at startup; its name is then valid as `[provider] name` and as a `kind`. The core needs no changes.
