"""Central configuration. Only entrypoints construct a `Config`.

midge is configured by a TOML file, overridden by environment variables,
overridden by command-line flags:

    flag > env > config > default

A `Config` is built by the entrypoint and passed inward. Library modules take
parameters; they never read the environment and there is deliberately **no
module-level `get_config()`**. A global would let any module reach configuration
— which is the coupling this module exists to remove — and would be untestable
without monkeypatching. It is the same rule `logs.configure` already states for
handlers, applied to the rest of configuration.

The one value that stays in the environment is `OPENAI_API_KEY`. A credential
does not belong in a file that gets committed, so it is not a field here and
`config_key_unknown` is reported if someone puts it in the file anyway.

`load` parses and returns diagnostics rather than logging them, because it has to
run *before* `logs.configure` — the log level is one of the things it resolves.
Entrypoints call `emit` once logging is up. A malformed file is a diagnostic and
never an exception: a typo should not stop the harness from starting, but it also
should not silently run on settings nobody asked for.

Discovery, project before user, merged per key:

    ./.midge/config.toml
    ~/.midge/config.toml
"""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PAYLOAD_CHARS = 2000
DEFAULT_KEEP_RECENT = 20_000

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")

# Marks a section whose sub-table names are user data rather than a fixed set.
_OPEN = "*"


# --- diagnostics ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A parse problem, held until there is somewhere to log it."""

    event: str
    fields: dict[str, Any] = dataclass_field(default_factory=dict)


def emit(diagnostics: Iterable[Diagnostic]) -> None:
    """Log deferred parse diagnostics. Call after `logs.configure`."""
    for d in diagnostics:
        _logger.warning("%s %s", d.event, " ".join(f"{k}={v}" for k, v in d.fields.items()))


# --- the shape -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogConfig:
    level: str = "WARNING"
    openai_level: str = "WARNING"
    file: Path | None = None
    payload_chars: int = DEFAULT_PAYLOAD_CHARS


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 0.5


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """How to reach one service — an entry in `[providers.*]`.

    Separate from `[models.*]`, which says which service a model lives on. Two
    tables because they answer different questions, and a service is named once
    however many models sit on it.
    """

    kind: str = "openai"
    base_url: str | None = None
    # The *name* of the variable holding the credential, never the credential.
    # A config file gets committed; see the module docstring.
    api_key_env: str | None = None
    include_usage: bool | None = None


@dataclass(frozen=True, slots=True)
class Config:
    model: str = DEFAULT_MODEL
    # None means "infer": no base_url is OpenAI, a base_url is something
    # OpenAI-compatible. `providers.resolve` owns that heuristic.
    provider: str | None = None
    base_url: str | None = None
    # None means "whatever the provider declares"; True/False overrides it.
    include_usage: bool | None = None
    compaction_threshold: int | None = None
    compaction_keep_recent: int = DEFAULT_KEEP_RECENT
    log: LogConfig = LogConfig()
    retry: RetryConfig = RetryConfig()
    # The model registry. Empty is permissive — any model string is accepted and
    # goes to the single provider above, which is every install that predates
    # these tables. Writing a `[models]` table is what turns enforcement on.
    providers: Mapping[str, ProviderConfig] = dataclass_field(default_factory=dict)
    models: Mapping[str, str] = dataclass_field(default_factory=dict)

    @classmethod
    def load(
        cls, *, cwd: Path | None = None, home: Path | None = None
    ) -> tuple[Config, list[Diagnostic]]:
        merged, diagnostics = _read(config_paths(cwd=cwd, home=home))
        src = _Source(merged, diagnostics)
        config = cls(
            model=src.text(None, "model", "MIDGE_MODEL", default=DEFAULT_MODEL),
            provider=src.text("provider", "name", "MIDGE_PROVIDER"),
            base_url=src.text("provider", "base_url", "OPENAI_BASE_URL"),
            include_usage=src.flag("provider", "include_usage", "MIDGE_INCLUDE_USAGE"),
            compaction_threshold=src.integer("compaction", "threshold"),
            compaction_keep_recent=src.integer(
                "compaction", "keep_recent", default=DEFAULT_KEEP_RECENT
            ),
            log=LogConfig(
                level=src.text("log", "level", "MIDGE_LOG_LEVEL", default="WARNING"),
                openai_level=src.text(
                    "log", "openai_level", "MIDGE_LOG_LEVEL_OPENAI", default="WARNING"
                ),
                file=src.path("log", "file", "MIDGE_LOG_FILE"),
                payload_chars=src.integer(
                    "log",
                    "payload_chars",
                    "MIDGE_LOG_PAYLOAD_CHARS",
                    default=DEFAULT_PAYLOAD_CHARS,
                ),
            ),
            retry=RetryConfig(
                max_attempts=src.integer("retry", "max_attempts", default=3),
                base_delay=src.number("retry", "base_delay", default=0.5),
            ),
            providers=_providers(src),
            models=_models(src),
        )
        diagnostics.extend(src.unrecognized())
        if config.providers and (config.provider or config.base_url):
            # Two ways to say the same thing. `[providers.*]` is the general
            # form, so it wins and the singular is reported rather than silently
            # merged — a base_url that looks configured but is not would be
            # worse than being told.
            diagnostics.append(
                Diagnostic("config_provider_singular_ignored", {"in_favour_of": "providers"})
            )
        return config, diagnostics


def config_paths(*, cwd: Path | None = None, home: Path | None = None) -> list[Path]:
    """Project config before personal, most specific first.

    A function rather than a module constant, for the reason `default_skill_dirs`
    gives: `Path.cwd()` resolved at import time freezes whatever directory the
    interpreter started in.
    """
    return [
        (cwd or Path.cwd()) / ".midge" / "config.toml",
        (home or Path.home()) / ".midge" / "config.toml",
    ]


_PROVIDER_KEYS = ("kind", "base_url", "api_key_env", "include_usage")


def _providers(src: _Source) -> dict[str, ProviderConfig]:
    out: dict[str, ProviderConfig] = {}
    for name, table in src.tables("providers", keys=_PROVIDER_KEYS).items():
        out[name] = ProviderConfig(
            kind=_as(src, f"providers.{name}.kind", table.get("kind"), str) or "openai",
            base_url=_as(src, f"providers.{name}.base_url", table.get("base_url"), str),
            api_key_env=_as(src, f"providers.{name}.api_key_env", table.get("api_key_env"), str),
            include_usage=_as(
                src, f"providers.{name}.include_usage", table.get("include_usage"), bool
            ),
        )
    return out


def _models(src: _Source) -> dict[str, str]:
    """Model id -> provider name. A model with no provider is not a model entry."""
    out: dict[str, str] = {}
    for name, table in src.tables("models", keys=("provider",)).items():
        raw = table.get("provider")
        provider = _as(src, f"models.{name}.provider", raw, str)
        if provider:
            out[name] = provider
        elif raw is None:
            # Only when it is genuinely absent — a wrong type already reported.
            src.report("config_model_provider_missing", {"model": name})
    return out


def _as(src: _Source, key: str, raw: Any, want: type) -> Any:
    """A TOML value of exactly the expected type, or None with a diagnostic.

    Narrower than `_Source`'s accessors on purpose: these come only from the
    file, never from the environment, so there is no string to coerce — a wrong
    type is a mistake rather than a spelling. Exact type rather than `isinstance`
    because `bool` is a subclass of `int`, and `kind = true` should be reported,
    not read as `1`.
    """
    if raw is None or type(raw) is want:
        return raw
    src.report("config_value_invalid", {"key": key, "want": want.__name__, "got": repr(raw)})
    return None


# --- reading and merging ---------------------------------------------------


def _read(paths: Iterable[Path]) -> tuple[dict[str, Any], list[Diagnostic]]:
    merged: dict[str, Any] = {}
    diagnostics: list[Diagnostic] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
            diagnostics.append(
                Diagnostic(
                    "config_file_unreadable",
                    {"path": path, "error": f"{type(e).__name__}: {e}"},
                )
            )
            continue
        _fold(merged, data)
    return merged, diagnostics


def _fold(into: dict[str, Any], data: dict[str, Any]) -> None:
    """Merge per key, earliest source winning.

    Per key rather than whole-file, so a project file that sets `model` does not
    silently discard the user file's `[log] level`. Same shape as the skills
    loader, where each *name* is claimed independently.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            table = into.setdefault(key, {})
            if isinstance(table, dict):
                for inner, v in value.items():
                    table.setdefault(inner, v)
        else:
            into.setdefault(key, value)


class _Source:
    """One merged document plus the environment, resolving in precedence order.

    Precedence lives here once so it cannot be got wrong field by field, and
    every key that is asked for is recorded — which is what makes reporting the
    keys nobody asked for possible.
    """

    def __init__(self, data: dict[str, Any], diagnostics: list[Diagnostic]) -> None:
        self._data = data
        self._diagnostics = diagnostics
        self._asked: set[tuple[str | None, str]] = set()

    def text(
        self, section: str | None, key: str, env: str | None = None, *, default: Any = None
    ) -> Any:
        raw, _ = self._raw(section, key, env)
        if raw is None:
            return default
        if isinstance(raw, str):
            return raw
        return self._bad(section, key, raw, "string", default)

    def integer(
        self, section: str | None, key: str, env: str | None = None, *, default: Any = None
    ) -> Any:
        raw, _ = self._raw(section, key, env)
        if raw is None:
            return default
        # bool is an int in Python; accepting `true` as 1 would be a silent
        # misreading of an obvious mistake.
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
        try:
            return int(str(raw).strip())
        except ValueError:
            return self._bad(section, key, raw, "integer", default)

    def number(
        self, section: str | None, key: str, env: str | None = None, *, default: Any = None
    ) -> Any:
        raw, _ = self._raw(section, key, env)
        if raw is None:
            return default
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        try:
            return float(str(raw).strip())
        except ValueError:
            return self._bad(section, key, raw, "number", default)

    def flag(
        self, section: str | None, key: str, env: str | None = None, *, default: Any = None
    ) -> Any:
        raw, _ = self._raw(section, key, env)
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        lowered = str(raw).strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        return self._bad(section, key, raw, "boolean", default)

    def path(
        self, section: str | None, key: str, env: str | None = None, *, default: Any = None
    ) -> Any:
        raw = self.text(section, key, env)
        if raw is None:
            return default
        return Path(raw).expanduser()

    def tables(self, section: str, *, keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        """A section whose sub-table *names* belong to the user — `[providers.foo]`.

        Their names are data, so they cannot be checked against a list of known
        keys the way every other section is. What can be checked, and is, are the
        keys inside each one.
        """
        self._asked.add((section, _OPEN))
        raw = self._data.get(section)
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for name, table in raw.items():
            if not isinstance(table, dict):
                self.report(
                    "config_value_invalid",
                    {"key": f"{section}.{name}", "want": "table", "got": repr(table)},
                )
                continue
            for key in table:
                if key not in keys:
                    self.report("config_key_unknown", {"key": f"{section}.{name}.{key}"})
            out[name] = table
        return out

    def report(self, event: str, fields: dict[str, Any]) -> None:
        self._diagnostics.append(Diagnostic(event, fields))

    def unrecognized(self) -> list[Diagnostic]:
        """Keys and sections present in the file that no field asked for.

        A typo in a key name would otherwise be indistinguishable from a setting
        that has no effect — and `api_key` in the file has to be visibly ignored
        rather than quietly.
        """
        known = {s for s, _ in self._asked if s is not None}
        out: list[Diagnostic] = []
        for key, value in self._data.items():
            if isinstance(value, dict):
                if key not in known:
                    out.append(Diagnostic("config_section_unknown", {"section": key}))
                    continue
                if (key, _OPEN) in self._asked:
                    continue  # `tables()` already checked what it could
                out.extend(
                    Diagnostic("config_key_unknown", {"key": f"{key}.{inner}"})
                    for inner in value
                    if (key, inner) not in self._asked
                )
            elif (None, key) not in self._asked:
                out.append(Diagnostic("config_key_unknown", {"key": key}))
        return out

    def _raw(self, section: str | None, key: str, env: str | None) -> tuple[Any, str]:
        self._asked.add((section, key))
        if env is not None:
            from_env = os.getenv(env)
            if from_env is not None:
                return from_env, env
        table = self._data if section is None else self._data.get(section)
        if isinstance(table, dict) and key in table:
            return table[key], key if section is None else f"{section}.{key}"
        return None, ""

    def _bad(self, section: str | None, key: str, raw: Any, want: str, default: Any) -> Any:
        self._diagnostics.append(
            Diagnostic(
                "config_value_invalid",
                {
                    "key": key if section is None else f"{section}.{key}",
                    "want": want,
                    "got": repr(raw),
                    "using": repr(default),
                },
            )
        )
        return default


__all__ = [
    "DEFAULT_KEEP_RECENT",
    "DEFAULT_MODEL",
    "DEFAULT_PAYLOAD_CHARS",
    "Config",
    "Diagnostic",
    "LogConfig",
    "ProviderConfig",
    "RetryConfig",
    "config_paths",
    "emit",
]
