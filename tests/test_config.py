"""Configuration: discovery, per-key merge, precedence, and bad input.

`Config.load` is a pure transform over two file paths and the environment, so
every test here writes real files into `tmp_path` and passes them as `cwd`/`home`
rather than monkeypatching a loader.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from midge.config import (
    DEFAULT_KEEP_RECENT,
    DEFAULT_MODEL,
    DEFAULT_PAYLOAD_CHARS,
    Config,
    LogConfig,
    ProviderConfig,
    RetryConfig,
    config_paths,
    emit,
)

_VARS = (
    "MIDGE_MODEL",
    "MIDGE_PROVIDER",
    "MIDGE_INCLUDE_USAGE",
    "MIDGE_LOG_LEVEL",
    "MIDGE_LOG_LEVEL_OPENAI",
    "MIDGE_LOG_FILE",
    "MIDGE_LOG_PAYLOAD_CHARS",
    "MIDGE_PROFILE",
    "OPENAI_BASE_URL",
)


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A developer's own environment must not decide what these tests see."""
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
    yield


def _write(root: Path, body: str) -> Path:
    """Write a config file under `root` the way discovery expects to find it."""
    path = root / ".midge" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _load(tmp_path: Path, *, project: str | None = None, user: str | None = None) -> Config:
    cwd, home = tmp_path / "project", tmp_path / "home"
    cwd.mkdir(exist_ok=True)
    home.mkdir(exist_ok=True)
    if project is not None:
        _write(cwd, project)
    if user is not None:
        _write(home, user)
    config, diagnostics = Config.load(cwd=cwd, home=home)
    assert diagnostics == [], f"unexpected diagnostics: {diagnostics}"
    return config


def _project(tmp_path: Path, body: str) -> Path:
    cwd = tmp_path / "project"
    cwd.mkdir(exist_ok=True)
    _write(cwd, body)
    return cwd


def _diagnose(tmp_path: Path, body: str) -> tuple[Config, list[str]]:
    config, diagnostics = Config.load(cwd=_project(tmp_path, body), home=tmp_path / "nowhere")
    return config, [d.event for d in diagnostics]


# --- absent config --------------------------------------------------------


def test_no_file_anywhere_is_the_built_in_defaults(tmp_path: Path) -> None:
    # Everything downstream assumes this: introducing a config file must not
    # change what midge does for someone who does not have one.
    assert _load(tmp_path) == Config()
    assert Config() == Config(
        model=DEFAULT_MODEL,
        provider=None,
        base_url=None,
        include_usage=None,
        default_profile=None,
        compaction_threshold=None,
        compaction_keep_recent=DEFAULT_KEEP_RECENT,
        log=LogConfig(
            level="WARNING",
            openai_level="WARNING",
            file=None,
            payload_chars=DEFAULT_PAYLOAD_CHARS,
        ),
        retry=RetryConfig(max_attempts=3, base_delay=0.5),
    )


def test_an_empty_file_is_also_the_defaults(tmp_path: Path) -> None:
    assert _load(tmp_path, project="") == Config()


# --- discovery and merge --------------------------------------------------


def test_discovery_looks_at_the_project_then_the_user(tmp_path: Path) -> None:
    assert config_paths(cwd=tmp_path / "p", home=tmp_path / "h") == [
        tmp_path / "p" / ".midge" / "config.toml",
        tmp_path / "h" / ".midge" / "config.toml",
    ]


def test_the_project_file_wins_the_keys_it_sets(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        project='model = "project-model"',
        user='model = "user-model"',
    )
    assert config.model == "project-model"


def test_merging_is_per_key_not_per_file(tmp_path: Path) -> None:
    """The point of merging rather than first-file-wins.

    A project file naming a model must not silently discard the user's log
    level — that is a setting they configured once and expect everywhere.
    """
    config = _load(
        tmp_path,
        project='model = "project-model"\n[retry]\nmax_attempts = 9\n',
        user='[log]\nlevel = "DEBUG"\npayload_chars = 50\n[retry]\nbase_delay = 2.5\n',
    )
    assert config.model == "project-model"
    assert config.log.level == "DEBUG"
    assert config.log.payload_chars == 50
    # Within one section, too: the project claims max_attempts, the user's
    # base_delay survives.
    assert config.retry == RetryConfig(max_attempts=9, base_delay=2.5)


def test_the_user_file_alone_is_used(tmp_path: Path) -> None:
    assert _load(tmp_path, user='model = "only-user"').model == "only-user"


# --- precedence -----------------------------------------------------------


def test_env_beats_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIDGE_MODEL", "from-env")
    assert _load(tmp_path, project='model = "from-file"').model == "from-env"


def test_all_four_levels_of_precedence_for_one_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The flag layer is the entrypoint's, not this module's — it is pinned in
    # test_cli_config.py. Here: default, then file, then env.
    assert _load(tmp_path).log.level == "WARNING"
    assert _load(tmp_path, project='[log]\nlevel = "INFO"').log.level == "INFO"
    monkeypatch.setenv("MIDGE_LOG_LEVEL", "ERROR")
    assert _load(tmp_path, project='[log]\nlevel = "INFO"').log.level == "ERROR"


def test_an_empty_env_var_still_counts_as_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `MIDGE_MODEL=` in a shell profile is a deliberate blanking, not an absence;
    # silently falling through to the file would be surprising.
    monkeypatch.setenv("MIDGE_MODEL", "")
    assert _load(tmp_path, project='model = "from-file"').model == ""


# --- the whole surface ----------------------------------------------------


def test_every_field_is_reachable_from_the_file(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        project="""
        model = "granite"

        [provider]
        name = "openai-compatible"
        base_url = "http://localhost:11434/v1"
        include_usage = false

        [retry]
        max_attempts = 5
        base_delay = 1.5

        [compaction]
        threshold = 120000
        keep_recent = 8000

        [log]
        level = "DEBUG"
        openai_level = "INFO"
        payload_chars = 0
        """,
    )
    assert config == Config(
        model="granite",
        provider="openai-compatible",
        base_url="http://localhost:11434/v1",
        include_usage=False,
        compaction_threshold=120_000,
        compaction_keep_recent=8_000,
        log=LogConfig(level="DEBUG", openai_level="INFO", payload_chars=0),
        retry=RetryConfig(max_attempts=5, base_delay=1.5),
    )


def test_the_shipped_example_parses_with_no_diagnostics(tmp_path: Path) -> None:
    """`examples/config.toml` documents the surface, so it has to match it.

    A key renamed in `Config` and not in the example would otherwise leave the
    documentation quietly wrong.
    """
    example = Path(__file__).parent.parent / "examples" / "config.toml"
    config = _load(tmp_path, project=example.read_text(encoding="utf-8"))
    # Everything uncommented in the example is a default, so this round-trips.
    assert config == Config()


def test_the_default_profile_is_named_in_the_file(tmp_path: Path) -> None:
    config = _load(tmp_path, project='[profiles]\ndefault = "adversarial-reviewer"')
    assert config.default_profile == "adversarial-reviewer"


def test_the_environment_beats_the_file_for_the_default_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIDGE_PROFILE", "from-env")
    config = _load(tmp_path, project='[profiles]\ndefault = "from-file"')
    assert config.default_profile == "from-env"


def test_the_session_directory_is_named_in_the_file(tmp_path: Path) -> None:
    config = _load(tmp_path, project='[session]\ndir = "~/transcripts"')
    assert config.session.dir == Path.home() / "transcripts"
    assert config.session.enabled is True


def test_transcripts_can_be_turned_off_in_the_file(tmp_path: Path) -> None:
    config = _load(tmp_path, project="[session]\nenabled = false")
    assert config.session.enabled is False


def test_the_environment_beats_the_file_for_the_session_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIDGE_SESSION_DIR", "/tmp/from-env")
    config = _load(tmp_path, project='[session]\ndir = "/tmp/from-file"')
    assert config.session.dir == Path("/tmp/from-env")


def test_a_log_file_path_expands_a_tilde(tmp_path: Path) -> None:
    config = _load(tmp_path, project='[log]\nfile = "~/logs/midge.log"')
    assert config.log.file == Path.home() / "logs" / "midge.log"


def test_booleans_accept_the_usual_env_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for raw, expected in (
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("OFF", False),
    ):
        monkeypatch.setenv("MIDGE_INCLUDE_USAGE", raw)
        assert _load(tmp_path).include_usage is expected, raw


def test_include_usage_is_none_when_nobody_says(tmp_path: Path) -> None:
    # None is not False: it means "whatever the provider declares", which is how
    # the capability stays the default rather than being overridden by absence.
    assert _load(tmp_path).include_usage is None


# --- the registry tables --------------------------------------------------


def test_providers_and_models_parse_into_tables(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        project="""
        [providers.openai]
        api_key_env = "OPENAI_API_KEY"

        [providers.local]
        kind = "openai-compatible"
        base_url = "http://localhost:11434/v1"
        include_usage = false

        [models."gpt-4o-mini"]
        provider = "openai"

        [models."ibm/granite-3.2-8b"]
        provider = "local"
        """,
    )
    assert config.providers == {
        "openai": ProviderConfig(kind="openai", api_key_env="OPENAI_API_KEY"),
        "local": ProviderConfig(
            kind="openai-compatible",
            base_url="http://localhost:11434/v1",
            include_usage=False,
        ),
    }
    assert config.models == {"gpt-4o-mini": "openai", "ibm/granite-3.2-8b": "local"}


def test_both_tables_default_to_empty(tmp_path: Path) -> None:
    # Empty is permissive, and every install predating these tables is empty.
    config = _load(tmp_path, project='model = "x"')
    assert config.providers == {} and config.models == {}


def test_a_provider_name_is_user_data_not_a_known_key(tmp_path: Path) -> None:
    # `[providers.whatever]` cannot be checked against a list of known names —
    # only the keys inside it can be.
    config = _load(tmp_path, project="[providers.anything_at_all]\nkind = 'openai'")
    assert "anything_at_all" in config.providers


def test_an_unknown_key_inside_a_provider_is_reported(tmp_path: Path) -> None:
    config, events = _diagnose(tmp_path, "[providers.p]\nkind = 'openai'\napi_key = 'sk-nope'")
    assert events == ["config_key_unknown"]
    # The credential rule does not bend for a second provider: `api_key_env`
    # names a variable, and there is nowhere to put a key itself.
    assert config.providers["p"].api_key_env is None


def test_a_model_without_a_provider_is_dropped(tmp_path: Path) -> None:
    config, events = _diagnose(
        tmp_path,
        '[models."a"]\nprovider = "p"\n[models."b"]\n',
    )
    assert events == ["config_model_provider_missing"]
    assert config.models == {"a": "p"}


def test_a_wrongly_typed_provider_name_reports_once(tmp_path: Path) -> None:
    # One diagnostic, not two — a wrong type must not also read as "missing".
    config, events = _diagnose(tmp_path, '[models."a"]\nprovider = 3')
    assert events == ["config_value_invalid"]
    assert config.models == {}


def test_kind_defaults_to_openai(tmp_path: Path) -> None:
    assert _load(tmp_path, project="[providers.p]\nbase_url = 'http://x/v1'").providers[
        "p"
    ].kind == "openai"


def test_the_singular_provider_is_reported_when_both_are_set(tmp_path: Path) -> None:
    """`[providers.*]` is the general form and wins.

    Silently merging would leave a `base_url` that looks configured but is not,
    which is worse than being told it was ignored.
    """
    config, events = _diagnose(
        tmp_path,
        "[provider]\nbase_url = 'http://old/v1'\n[providers.p]\nkind = 'openai'\n",
    )
    assert events == ["config_provider_singular_ignored"]
    assert set(config.providers) == {"p"}


def test_the_singular_provider_alone_is_silent(tmp_path: Path) -> None:
    # It shipped in #70 three commits ago; using it must not warn.
    config, events = _diagnose(tmp_path, "[provider]\nbase_url = 'http://x/v1'")
    assert events == []
    assert config.base_url == "http://x/v1"


# --- bad input is a diagnostic, never an exception ------------------------


def test_malformed_toml_falls_back_to_defaults_and_says_so(tmp_path: Path) -> None:
    config, events = _diagnose(tmp_path, 'model = "unterminated\n[log\n')
    assert config == Config()
    assert events == ["config_file_unreadable"]


def test_an_unknown_key_is_reported(tmp_path: Path) -> None:
    _, events = _diagnose(tmp_path, 'modle = "typo"')
    assert events == ["config_key_unknown"]


def test_an_unknown_section_is_reported_once(tmp_path: Path) -> None:
    # Once for the section, not once per key inside it — a whole mistyped table
    # is one mistake.
    _, events = _diagnose(tmp_path, '[compactions]\nthreshold = 1\nkeep_recent = 2\n')
    assert events == ["config_section_unknown"]


def test_the_singular_plural_near_miss_says_what_is_wrong(tmp_path: Path) -> None:
    """`[providers]` where `[provider]` was meant.

    Both spellings are real sections now, so this cannot be caught as an unknown
    section. What it can do is name the key and say a table was expected, which
    is enough to see the mistake.
    """
    config, diagnostics = Config.load(cwd=_project(tmp_path, '[providers]\nname = "x"\n'), home=tmp_path / "nowhere")
    assert config.providers == {}
    [d] = diagnostics
    assert d.event == "config_value_invalid"
    assert d.fields["key"] == "providers.name" and d.fields["want"] == "table"


def test_an_api_key_in_the_file_is_visibly_ignored(tmp_path: Path) -> None:
    """A credential does not belong in a committed file.

    Being unknown rather than merely unused is the point: someone who puts it
    there is told it does nothing, instead of assuming it worked.
    """
    config, events = _diagnose(tmp_path, 'api_key = "sk-secret"')
    assert events == ["config_key_unknown"]
    assert not hasattr(config, "api_key")


def test_a_wrongly_typed_value_falls_back_to_the_default(tmp_path: Path) -> None:
    config, events = _diagnose(tmp_path, "[retry]\nmax_attempts = 'lots'")
    assert events == ["config_value_invalid"]
    assert config.retry.max_attempts == 3


def test_a_boolean_is_not_accepted_as_an_integer(tmp_path: Path) -> None:
    # `bool` is a subclass of `int` in Python; reading `true` as 1 would be a
    # silent misreading of an obvious mistake.
    config, events = _diagnose(tmp_path, "[compaction]\nkeep_recent = true")
    assert events == ["config_value_invalid"]
    assert config.compaction_keep_recent == DEFAULT_KEEP_RECENT


def test_an_unparseable_boolean_is_reported(tmp_path: Path) -> None:
    config, events = _diagnose(tmp_path, '[provider]\ninclude_usage = "maybe"')
    assert events == ["config_value_invalid"]
    assert config.include_usage is None


def test_a_numeric_string_is_still_accepted(tmp_path: Path) -> None:
    # Env values are always strings, so the coercion has to work for the file
    # too rather than being special-cased per source.
    assert _load(tmp_path, project='[retry]\nbase_delay = "2"').retry.base_delay == 2.0


def test_diagnostics_accumulate_rather_than_stopping_at_the_first(tmp_path: Path) -> None:
    _, events = _diagnose(tmp_path, "wrong = 1\n[retry]\nmax_attempts = 'x'\n")
    assert sorted(events) == ["config_key_unknown", "config_value_invalid"]


def test_an_unreadable_project_file_does_not_stop_the_user_file(tmp_path: Path) -> None:
    cwd, home = tmp_path / "project", tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    _write(cwd, "[log\n")
    _write(home, 'model = "survivor"')

    config, diagnostics = Config.load(cwd=cwd, home=home)
    assert config.model == "survivor"
    assert [d.event for d in diagnostics] == ["config_file_unreadable"]


# --- emitting -------------------------------------------------------------


def test_emit_logs_one_greppable_line_per_diagnostic(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Diagnostics are deferred, so they are useless unless `emit` renders them.

    The event identity has to lead the line: these are counted with `grep -c`,
    not matched with a regex over English.
    """
    _, diagnostics = Config.load(cwd=tmp_path / "p", home=tmp_path / "h")
    cwd = tmp_path / "project"
    cwd.mkdir()
    _write(cwd, "nope = 1\n")
    _, diagnostics = Config.load(cwd=cwd, home=tmp_path / "h")

    with caplog.at_level(logging.WARNING, logger="midge.config"):
        emit(diagnostics)

    [record] = caplog.records
    assert record.getMessage().startswith("config_key_unknown ")
    assert "key=nope" in record.getMessage()


def test_emit_of_nothing_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="midge.config"):
        emit([])
    assert caplog.records == []
