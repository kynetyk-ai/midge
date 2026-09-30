# Configuration

midge reads its settings from one TOML file, overridable by environment variables and command-line flags. [`src/midge/config.py`](../src/midge/config.py) defines every setting; [`examples/config.toml`](../examples/config.toml) lists every key with its default and its environment variable.

## Where it is read from

Two files are read, project first:

1. `./.midge/config.toml`
2. `~/.midge/config.toml`

They are merged key by key, and the project file wins for any key both set. A project file that sets `model` therefore keeps your personal `[log] level`.

```toml
model = "ibm/granite-3.2-8b"

[provider]
base_url = "http://127.0.0.1:1234/v1"

[log]
level = "INFO"
file = "~/.midge/midge.log"
```

## Precedence

For each setting, the first of these that is present wins:

1. command-line flag
2. environment variable
3. config file
4. built-in default

`Config.load` resolves the last three; the entrypoint applies flags. Many keys have an environment variable (`MIDGE_MODEL`, `MIDGE_LOG_LEVEL`, `OPENAI_BASE_URL`, and others); some, such as the `[retry]` and `[subagents]` keys, are file-only. The comments in `examples/config.toml` name each variable and flag.

## Diagnostics

A bad configuration never stops midge from starting. Each problem becomes a warning logged at startup, and the affected setting falls back to its default:

| Problem | Event |
|---|---|
| A file that cannot be read or parsed (the whole file is skipped) | `config_file_unreadable` |
| A key or section no setting reads | `config_key_unknown`, `config_section_unknown` |
| A value of the wrong type or outside its allowed set | `config_value_invalid` |
| Both `[provider]` and `[providers.*]` set (the singular form is ignored) | `config_provider_singular_ignored` |

`load` returns these as `Diagnostic` objects without logging them, because it runs before logging is configured (the log level is one of the settings it resolves). The entrypoint passes them to `emit` once logging is up.

Only entrypoints construct a `Config`. Library modules receive the values they need as parameters, and there is no global accessor. To add a setting, add the field to `Config`, read it in `Config.load`, and add a commented entry to `examples/config.toml`; `tests/test_config.py` fails if the example file stops parsing cleanly.

## Credentials

Credentials come only from the environment and are never config keys. The default provider reads `OPENAI_API_KEY`. A provider in the model registry names the variable that holds its key with `api_key_env`. An `api_key` written in the file is reported as an unknown key and has no effect. midge never logs a key, and logs a `base_url` as its hostname only, since a URL can carry credentials in its userinfo.

## The model registry

By default midge sends every request to the single provider described by `[provider]` (or inferred: no `base_url` means OpenAI, a `base_url` means an OpenAI-compatible server). The model registry lets one process use several services, with the model id deciding where each request goes. This applies to the starting model, `set_model`, a profile's model and a sub-agent's `model=`.

`[providers.*]` describes how to reach a service, under a name you choose. `[models.*]` maps a model id to one of those names.

```toml
[providers.openai]
kind = "openai"
api_key_env = "OPENAI_API_KEY"

[providers.local]
kind = "openai-compatible"
base_url = "http://localhost:11434/v1"

[models."gpt-6-luna"]
provider = "openai"

[models."ibm/granite-3.2-8b"]
provider = "local"
```

A provider entry accepts `kind` (the adapter, default `openai`), `base_url`, `api_key_env` and `include_usage`. The adapters themselves are described in [providers](providers.md).

An empty registry is permissive: any model id is accepted and goes to the single provider. Once at least one `[models]` entry exists, only listed models are usable. Startup exits if the configured model is not listed, and `set_model` refuses an unlisted one. A profile or sub-agent naming an unlisted model is dropped at startup (see [profiles](profiles.md) and [subagents](subagents.md)).

midge ships no list of models and does not check that a model id exists; ids pass to the provider unchanged, so a typo fails with the vendor's own error. Only the wiring is validated (mostly in [`src/midge/providers/registry.py`](../src/midge/providers/registry.py)), and each of these entries is dropped with a warning at startup:

| Problem | Event |
|---|---|
| A provider whose `kind` names no adapter | `provider_kind_unknown` |
| A model naming a provider that is not defined | `model_provider_undefined` |
| A model entry with no `provider` | `config_model_provider_missing` |

Providers defined with no models routed to them produce `providers_unused`.
