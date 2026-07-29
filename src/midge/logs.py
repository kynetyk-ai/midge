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
  record) or set `MIDGE_LOG_FILE`.

Env:
    MIDGE_LOG_LEVEL          level for the `midge` logger (default WARNING)
    MIDGE_LOG_LEVEL_OPENAI   level for the `openai` logger (default WARNING)
    MIDGE_LOG_FILE           write to this path instead of stderr
    MIDGE_LOG_PAYLOAD_CHARS  truncation cap for `payload` (default 2000, 0 = off)
"""

from __future__ import annotations

import logging
import os
import sys
from urllib.parse import urlsplit

DEFAULT_PAYLOAD_CHARS = 2000
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"

_logger = logging.getLogger(__name__)


def configure(handler: logging.Handler | None = None) -> None:
    if handler is None:
        log_file = os.getenv("MIDGE_LOG_FILE")
        handler = (
            logging.FileHandler(log_file, encoding="utf-8")
            if log_file
            else logging.StreamHandler(sys.stderr)
        )
    handler.setFormatter(logging.Formatter(_FORMAT))

    level, bad_level = _level("MIDGE_LOG_LEVEL")
    openai_level, bad_openai = _level("MIDGE_LOG_LEVEL_OPENAI")

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

    for var, value in (("MIDGE_LOG_LEVEL", bad_level), ("MIDGE_LOG_LEVEL_OPENAI", bad_openai)):
        if value is not None:
            _logger.warning("log_level_invalid var=%s value=%r using=WARNING", var, value)


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
        cap = _payload_chars()
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


def _payload_chars() -> int:
    raw = os.getenv("MIDGE_LOG_PAYLOAD_CHARS")
    if raw is None:
        return DEFAULT_PAYLOAD_CHARS
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_PAYLOAD_CHARS


def _level(var: str) -> tuple[int, str | None]:
    """Resolve a level name, reporting the raw value back when it is unusable.

    A typo in an env var should not stop the harness from starting, but it also
    should not silently run at a level the user did not ask for.
    """
    raw = os.getenv(var)
    if raw is None:
        return logging.WARNING, None
    resolved = logging.getLevelNamesMapping().get(raw.strip().upper())
    if not isinstance(resolved, int):
        return logging.WARNING, raw
    return resolved, None


__all__ = ["DEFAULT_PAYLOAD_CHARS", "configure", "payload", "provider_host"]
