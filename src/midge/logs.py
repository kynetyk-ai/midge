"""Logging configuration. Only entrypoints call `configure`.

Library modules never configure logging — they only ever do
`_logger = logging.getLogger(__name__)`. The reason is that "log to stderr"
means three different things across midge's modes, and only the entrypoint
knows which one it is:

- RPC: stdout is the protocol, so no handler may ever target it. stderr is free.
- Headless CLI: stdout is the rendered transcript. stderr is free.
- TUI: Textual wraps its message loop in `redirect_stderr`, but
  `logging.StreamHandler` captures the stream at construction and holds that
  reference — so a handler built before `App.run()` writes past the redirect
  and corrupts the display. Pass `TextualHandler` (it resolves the target per
  record) or set a log file.

What to log is a `LogConfig`, which the entrypoint gets from `midge.config` —
this module reads no environment variables of its own. `configure` therefore has
to be called before the first log line worth keeping, and after configuration has
been parsed; `config.load` defers its own diagnostics for exactly that reason.
"""

from __future__ import annotations

import logging
import sys
from urllib.parse import urlsplit

from midge.config import DEFAULT_PAYLOAD_CHARS, LogConfig

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"

_logger = logging.getLogger(__name__)

# Set by `configure`. Module state because `payload` is rendered lazily inside a
# log record's formatting, long after any config object is in scope.
_payload_cap = DEFAULT_PAYLOAD_CHARS


def configure(handler: logging.Handler | None = None, *, log: LogConfig | None = None) -> None:
    global _payload_cap

    log = log or LogConfig()
    if handler is None:
        handler = (
            logging.FileHandler(log.file, encoding="utf-8")
            if log.file
            else logging.StreamHandler(sys.stderr)
        )
    handler.setFormatter(logging.Formatter(_FORMAT))

    _payload_cap = log.payload_chars
    level, bad_level = _level(log.level)
    openai_level, bad_openai = _level(log.openai_level)

    root = logging.getLogger("midge")
    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()
    root.addHandler(handler)
    root.setLevel(level)
    # pytest's caplog installs on the root logger and receives records by
    # propagation; turning this off empties every caplog assertion in the suite.
    root.propagate = True

    # Pinned separately from midge's own level: the SDK's DEBUG output can carry
    # the Authorization header, which is a different kind of secret from a
    # conversation and is never worth emitting.
    logging.getLogger("openai").setLevel(openai_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    for key, value in (("log.level", bad_level), ("log.openai_level", bad_openai)):
        if value is not None:
            _logger.warning("log_level_invalid key=%s value=%r using=WARNING", key, value)


class _Payload:
    """Deferred `str()` so a DEBUG payload costs nothing when DEBUG is off.

    `logging` only formats arguments once a handler will emit the record, but
    `payload(x)` as an argument is evaluated either way — so the truncation
    work has to live in `__str__`, not in the call.
    """

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __str__(self) -> str:
        text = repr(self.value)
        cap = _payload_cap
        if cap <= 0 or len(text) <= cap:
            return text
        return f"{text[:cap]}… (truncated, {len(text) - cap} chars)"


def provider_host(base_url: str | None) -> str:
    """The loggable part of a `base_url` — its hostname and nothing else.

    A base_url can carry credentials in userinfo (`https://user:pw@host/v1`) or
    a query string, so the whole string is never safe to log. Always route a
    base_url through this; never log an api_key at all, at any level.
    """
    if not base_url:
        return "default"
    host = urlsplit(base_url).hostname
    return host or "invalid"


def payload(value: object) -> _Payload:
    """Wrap a DEBUG-only payload so it is truncated and rendered lazily.

    Payloads are logged at DEBUG and nowhere else. Credentials are not payload:
    an api_key is never logged at any level, and a base_url is logged as its
    hostname only.
    """
    return _Payload(value)


def _level(name: str) -> tuple[int, str | None]:
    """Resolve a level name, reporting the raw value back when it is unusable.

    A typo should not stop the harness from starting, but it also should not
    silently run at a level the user did not ask for.
    """
    resolved = logging.getLevelNamesMapping().get(name.strip().upper())
    if not isinstance(resolved, int):
        return logging.WARNING, name
    return resolved, None


__all__ = ["configure", "payload", "provider_host"]
