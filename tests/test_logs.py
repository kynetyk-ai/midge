from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from midge.logs import DEFAULT_PAYLOAD_CHARS, configure, payload, provider_host

_VARS = (
    "MIDGE_LOG_LEVEL",
    "MIDGE_LOG_LEVEL_OPENAI",
    "MIDGE_LOG_FILE",
    "MIDGE_LOG_PAYLOAD_CHARS",
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """`configure` mutates global logger state; put it back for the next test."""
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)

    midge = logging.getLogger("midge")
    saved = (list(midge.handlers), midge.level, midge.propagate)
    saved_openai = logging.getLogger("openai").level
    saved_httpx = logging.getLogger("httpx").level
    yield
    midge.handlers, midge.level, midge.propagate = saved
    logging.getLogger("openai").setLevel(saved_openai)
    logging.getLogger("httpx").setLevel(saved_httpx)


def test_defaults_to_warning_on_stderr() -> None:
    configure()
    midge = logging.getLogger("midge")

    assert midge.level == logging.WARNING
    assert len(midge.handlers) == 1
    assert isinstance(midge.handlers[0], logging.StreamHandler)


def test_level_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIDGE_LOG_LEVEL", "debug")
    configure()
    assert logging.getLogger("midge").level == logging.DEBUG
    assert logging.getLogger("midge.client").isEnabledFor(logging.DEBUG)


def test_invalid_level_falls_back_and_says_so(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIDGE_LOG_LEVEL", "LOUD")
    with caplog.at_level(logging.WARNING, logger="midge.logs"):
        configure()

    assert logging.getLogger("midge").level == logging.WARNING
    assert any("log_level_invalid" in r.getMessage() for r in caplog.records)


def test_configure_is_idempotent() -> None:
    configure()
    configure()
    configure()
    assert len(logging.getLogger("midge").handlers) == 1


def test_records_still_propagate_to_root() -> None:
    """The whole caplog suite depends on this; assert it rather than imply it."""
    configure()
    seen: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    root_handler = _Capture()
    logging.getLogger().addHandler(root_handler)
    try:
        logging.getLogger("midge.client").warning("provider_retry attempt=%d", 1)
    finally:
        logging.getLogger().removeHandler(root_handler)

    assert logging.getLogger("midge").propagate is True
    assert [r.getMessage() for r in seen] == ["provider_retry attempt=1"]


def test_log_file_receives_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "midge.log"
    monkeypatch.setenv("MIDGE_LOG_FILE", str(path))
    monkeypatch.setenv("MIDGE_LOG_LEVEL", "INFO")
    configure()

    logging.getLogger("midge.agent").info("turn_start model=%s", "gpt-4o")
    for h in logging.getLogger("midge").handlers:
        h.flush()

    text = path.read_text(encoding="utf-8")
    assert "turn_start model=gpt-4o" in text
    assert "midge.agent" in text
    assert "INFO" in text


def test_explicit_handler_wins_over_log_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIDGE_LOG_FILE", str(tmp_path / "unused.log"))
    handler = logging.NullHandler()
    configure(handler)

    assert logging.getLogger("midge").handlers == [handler]
    assert not (tmp_path / "unused.log").exists()


def test_openai_stays_quiet_when_midge_is_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SDK logs request bodies and can carry the Authorization header at
    # DEBUG; midge's own level must never drag it along.
    monkeypatch.setenv("MIDGE_LOG_LEVEL", "DEBUG")
    configure()

    assert logging.getLogger("midge").level == logging.DEBUG
    assert logging.getLogger("openai").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING


def test_openai_level_is_separately_openable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIDGE_LOG_LEVEL_OPENAI", "DEBUG")
    configure()
    assert logging.getLogger("openai").level == logging.DEBUG


def test_payload_is_untouched_when_short() -> None:
    assert str(payload("hello")) == "'hello'"


def test_payload_truncates_at_the_default() -> None:
    rendered = str(payload("x" * (DEFAULT_PAYLOAD_CHARS + 500)))

    assert len(rendered) < DEFAULT_PAYLOAD_CHARS + 100
    assert "truncated, 502 chars" in rendered  # +2 for the repr quotes


def test_payload_cap_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIDGE_LOG_PAYLOAD_CHARS", "10")
    assert str(payload("y" * 100)).startswith("'yyyyyyyyy")
    assert "truncated" in str(payload("y" * 100))


def test_payload_cap_of_zero_disables_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIDGE_LOG_PAYLOAD_CHARS", "0")
    assert "truncated" not in str(payload("z" * 10_000))


def test_payload_cap_is_read_at_render_time(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapped = payload("w" * 100)
    monkeypatch.setenv("MIDGE_LOG_PAYLOAD_CHARS", "10")
    assert "truncated" in str(wrapped)


def test_payload_bad_cap_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIDGE_LOG_PAYLOAD_CHARS", "lots")
    assert "truncated" not in str(payload("a" * 100))


def test_payload_renders_structures_single_line() -> None:
    rendered = str(payload({"messages": [{"role": "user", "content": "a\nb"}]}))
    assert "\n" not in rendered
    assert "role" in rendered


def test_payload_is_not_rendered_when_the_level_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`payload(x)` is evaluated as an argument either way; the cost must be in __str__."""
    rendered = False

    class _Tracking:
        def __repr__(self) -> str:
            nonlocal rendered
            rendered = True
            return "expensive"

    class _Formatting(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            self.format(record)

    configure(_Formatting())
    logging.getLogger("midge.client").debug("request body=%s", payload(_Tracking()))
    assert rendered is False

    monkeypatch.setenv("MIDGE_LOG_LEVEL", "DEBUG")
    configure(_Formatting())
    logging.getLogger("midge.client").debug("request body=%s", payload(_Tracking()))
    assert rendered is True


def test_provider_host_defaults_when_unset() -> None:
    assert provider_host(None) == "default"
    assert provider_host("") == "default"


def test_provider_host_strips_everything_but_the_hostname() -> None:
    assert provider_host("https://api.openai.com/v1") == "api.openai.com"
    assert provider_host("http://127.0.0.1:1234/v1") == "127.0.0.1"


def test_provider_host_drops_credentials_in_userinfo() -> None:
    # A base_url may carry a secret; only the hostname is ever loggable.
    rendered = provider_host("https://user:hunter2@proxy.internal:8443/v1?token=abc")
    assert rendered == "proxy.internal"
    assert "hunter2" not in rendered
    assert "abc" not in rendered


def test_provider_host_marks_unparseable_input() -> None:
    assert provider_host("::::") == "invalid"


def test_tui_handler_is_never_an_eagerly_bound_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """A StreamHandler built before App.run() writes past Textual's
    redirect_stderr and corrupts the display. TextualHandler resolves the
    target per record instead."""
    from textual.logging import TextualHandler

    from midge.tui import tui_log_handler

    monkeypatch.delenv("MIDGE_LOG_FILE", raising=False)
    handler = tui_log_handler()

    assert isinstance(handler, TextualHandler)
    assert not isinstance(handler, logging.StreamHandler)


def test_tui_handler_defers_to_log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from midge.tui import tui_log_handler

    monkeypatch.setenv("MIDGE_LOG_FILE", str(tmp_path / "midge.log"))
    # None hands the case back to configure(), which opens the file itself.
    assert tui_log_handler() is None
