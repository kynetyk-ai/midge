"""Profiles: discovery through `load_extensions`, and validation.

Discovery is exercised through the real loader against files on disk rather than
by constructing a `ProfileSet` directly — the thing worth pinning is that a
profile is found the same way a tool is, from one import of one file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from midge.config import ProviderConfig
from midge.extensions import load_extensions
from midge.hooks import Hooks
from midge.profiles import Profile, ProfileSet, validate
from midge.providers import ModelRegistry
from midge.tools import ToolRegistry, tool

_PROFILE = """
from midge.profiles import Profile

P = Profile(
    name={name!r},
    description="A profile.",
    prompt="Be adversarial.",
    tools={tools!r},
    hooks={hooks!r},
    model={model!r},
)
"""


def _write_profile(
    path: Path,
    *,
    name: str = "reviewer",
    tools: tuple[str, ...] = (),
    hooks: dict[str, bool] | None = None,
    model: str = "",
) -> None:
    path.write_text(
        _PROFILE.format(name=name, tools=tools, hooks=hooks or {}, model=model)
    )


def _discover(*sources: Path) -> ProfileSet:
    profiles = ProfileSet()
    load_extensions(list(sources), profiles=profiles)
    return profiles


async def _noop() -> str:
    """A tool."""
    return "ok"


def _tools(*names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for n in names:
        registry.add(tool(name=n)(_noop))
    return registry


# --- discovery ------------------------------------------------------------


def test_a_profile_is_discovered_from_an_extension_dir(tmp_path: Path) -> None:
    _write_profile(tmp_path / "reviewer.py", tools=("read",))
    profiles = _discover(tmp_path)

    assert profiles.names() == ["reviewer"]
    found = profiles.get("reviewer")
    assert found is not None
    assert found.tools == ("read",)
    assert profiles.path_of("reviewer") == tmp_path / "reviewer.py"


def test_one_file_may_declare_a_tool_and_a_profile(tmp_path: Path) -> None:
    """The argument for reusing `load_extensions` rather than adding a loader:
    the file is imported once, so its side effects happen once."""
    (tmp_path / "both.py").write_text(
        "from midge.tools import tool\n"
        "from midge.profiles import Profile\n\n"
        "@tool\n"
        "async def greet() -> str:\n"
        '    """Greet."""\n'
        "    return 'hi'\n\n"
        "P = Profile(name='p', description='d', prompt='go', tools=('greet',))\n"
    )
    profiles = ProfileSet()
    registry, _ = load_extensions([tmp_path], profiles=profiles)

    assert "greet" in registry
    assert "p" in profiles


def test_no_sink_means_no_profiles_and_no_error(tmp_path: Path) -> None:
    """Every entrypoint that predates profiles keeps working unchanged."""
    _write_profile(tmp_path / "reviewer.py")
    registry, prompt = load_extensions([tmp_path])
    assert len(registry) == 0
    assert prompt == ""


def test_a_duplicate_name_is_first_wins_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    _write_profile(first / "one.py", name="dup", tools=("read",))
    _write_profile(second / "two.py", name="dup", tools=("bash",))

    with caplog.at_level(logging.WARNING, logger="midge.extensions"):
        profiles = _discover(first, second)

    winner = profiles.get("dup")
    assert winner is not None and winner.tools == ("read",)
    assert "profile_name_shadowed" in caplog.text


# --- validation -----------------------------------------------------------


def test_a_profile_naming_a_tool_from_another_file_loads(tmp_path: Path) -> None:
    """Why validation is deferred until every source has been read: there is no
    file ordering that makes a per-file check correct."""
    (tmp_path / "a_profile.py").write_text(
        "from midge.profiles import Profile\n"
        "P = Profile(name='p', description='d', prompt='go', tools=('later',))\n"
    )
    (tmp_path / "z_tool.py").write_text(
        "from midge.tools import tool\n"
        "@tool(name='later')\n"
        "async def later() -> str:\n"
        '    """Later."""\n'
        "    return 'ok'\n"
    )
    profiles = ProfileSet()
    registry, _ = load_extensions([tmp_path], profiles=profiles)

    assert validate(profiles, tools=registry, hook_names=set()) == []
    assert "p" in profiles


def test_an_unknown_tool_drops_the_profile(tmp_path: Path) -> None:
    _write_profile(tmp_path / "p.py", tools=("read", "nonexistent"))
    profiles = _discover(tmp_path)

    diagnostics = validate(profiles, tools=_tools("read"), hook_names=set())

    assert [d.event for d in diagnostics] == ["profile_tool_unknown"]
    assert diagnostics[0].fields["tool"] == "nonexistent"
    assert "reviewer" not in profiles


def test_an_unknown_hook_drops_the_profile(tmp_path: Path) -> None:
    _write_profile(tmp_path / "p.py", hooks={"approve": True})
    profiles = _discover(tmp_path)

    diagnostics = validate(profiles, tools=_tools(), hook_names={"audit"})

    # Both halves of the rule fire: `approve` does not exist, and the `audit`
    # that does exist was never decided.
    assert [d.event for d in diagnostics] == [
        "profile_hook_unknown",
        "profile_hook_undecided",
    ]
    assert len(profiles) == 0


def test_a_known_hook_is_the_extension_file_stem(tmp_path: Path) -> None:
    """What a profile writes is `approve`, not the absolute path the diagnostic
    carries. Pinned end-to-end because the two names are stamped together."""
    (tmp_path / "approve.py").write_text(
        "def register_hooks(hooks):\n"
        "    hooks.on('tool_call', lambda event, ctx: None)\n"
    )
    _write_profile(tmp_path / "p.py", hooks={"approve": True})
    hooks = Hooks()
    profiles = ProfileSet()
    registry, _ = load_extensions([tmp_path], hooks=hooks, profiles=profiles)

    assert hooks.source_names() == {"approve"}
    assert validate(profiles, tools=registry, hook_names=hooks.source_names()) == []
    assert "reviewer" in profiles


def test_a_blank_prompt_is_invalid(tmp_path: Path) -> None:
    (tmp_path / "p.py").write_text(
        "from midge.profiles import Profile\n"
        "P = Profile(name='p', description='d', prompt='   ')\n"
    )
    profiles = _discover(tmp_path)

    diagnostics = validate(profiles, tools=_tools(), hook_names=set())

    assert [d.event for d in diagnostics] == ["profile_invalid"]
    assert len(profiles) == 0


def test_every_problem_is_reported_not_just_the_first(tmp_path: Path) -> None:
    _write_profile(tmp_path / "p.py", tools=("gone", "also-gone"), hooks={"missing": True})
    profiles = _discover(tmp_path)

    events = [d.event for d in validate(profiles, tools=_tools(), hook_names=set())]

    assert events == ["profile_tool_unknown", "profile_tool_unknown", "profile_hook_unknown"]


def test_the_diagnostic_names_the_file(tmp_path: Path) -> None:
    _write_profile(tmp_path / "p.py", tools=("gone",))
    profiles = _discover(tmp_path)

    diagnostics = validate(profiles, tools=_tools(), hook_names=set())

    assert diagnostics[0].fields["path"] == tmp_path / "p.py"


# --- the model, against the registry --------------------------------------


def _registry(*models: str) -> ModelRegistry:
    return ModelRegistry(
        models=dict.fromkeys(models, "svc"),
        providers={"svc": ProviderConfig(kind="openai")},
    )


def test_an_empty_model_registry_accepts_any_model(tmp_path: Path) -> None:
    """The rule everywhere the registry is consulted. A profile naming a model
    is not an error in an install that has declared nothing."""
    _write_profile(tmp_path / "p.py", model="whatever-they-called-it")
    profiles = _discover(tmp_path)

    assert validate(profiles, tools=_tools(), hook_names=set(), models=_registry()) == []
    assert "reviewer" in profiles


def test_an_unregistered_model_drops_the_profile(tmp_path: Path) -> None:
    _write_profile(tmp_path / "p.py", model="gpt-4o")
    profiles = _discover(tmp_path)

    diagnostics = validate(
        profiles, tools=_tools(), hook_names=set(), models=_registry("gpt-4o-mini")
    )

    assert [d.event for d in diagnostics] == ["profile_model_unregistered"]
    assert diagnostics[0].fields["registered"] == "gpt-4o-mini"
    assert len(profiles) == 0


def test_a_profile_with_no_model_is_never_checked(tmp_path: Path) -> None:
    """An unset model means "keep the current one", which no registry can
    refuse."""
    _write_profile(tmp_path / "p.py")
    profiles = _discover(tmp_path)

    assert (
        validate(profiles, tools=_tools(), hook_names=set(), models=_registry("gpt-4o-mini")) == []
    )
    assert "reviewer" in profiles


# --- the shipped example --------------------------------------------------


def test_the_shipped_example_profile_loads_and_validates() -> None:
    """`examples/profile_extension/` is documentation, so it has to work."""
    root = Path(__file__).parent.parent / "examples"
    hooks = Hooks()
    profiles = ProfileSet()
    registry, _ = load_extensions(
        [root / "profile_extension", root / "approval_extension", *_builtin()],
        hooks=hooks,
        profiles=profiles,
    )

    assert validate(profiles, tools=registry, hook_names=hooks.source_names()) == []
    assert "adversarial-reviewer" in profiles


def _builtin() -> list[Path]:
    from midge.extensions import BUILTIN_TOOL_DIRS

    return list(BUILTIN_TOOL_DIRS)


# --- the collection -------------------------------------------------------


def test_adding_a_duplicate_raises() -> None:
    """`ProfileSet` refuses rather than warns; the loader owns the warning, so
    it can name both files."""
    profiles = ProfileSet()
    profiles.add(Profile(name="p", description="d", prompt="go"))
    with pytest.raises(ValueError, match="already registered"):
        profiles.add(Profile(name="p", description="other", prompt="go"))


# --- hooks must be decided exhaustively -----------------------------------
#
# The asymmetry with `tools` is the whole point. Omitting a tool yields a less
# capable agent; omitting a hook would yield an unguarded one. So `tools` is an
# allowlist and `hooks` is a decision per discovered source.


def test_an_undecided_hook_drops_the_profile(tmp_path: Path) -> None:
    """The mistake this catches: writing only the hooks you mean to change,
    and silently switching off the approval gate you never mentioned."""
    _write_profile(tmp_path / "p.py", hooks={})
    profiles = _discover(tmp_path)

    diagnostics = validate(profiles, tools=_tools(), hook_names={"approve"})

    assert [d.event for d in diagnostics] == ["profile_hook_undecided"]
    assert diagnostics[0].fields["hook"] == "approve"
    assert "reviewer" not in profiles


def test_deciding_a_hook_false_is_valid(tmp_path: Path) -> None:
    """Turning a gate off is allowed — it just has to be written down."""
    _write_profile(tmp_path / "p.py", hooks={"approve": False})
    profiles = _discover(tmp_path)

    assert validate(profiles, tools=_tools(), hook_names={"approve"}) == []
    decided = profiles.get("reviewer")
    assert decided is not None and decided.hooks == {"approve": False}


def test_every_undecided_source_is_named(tmp_path: Path) -> None:
    """One diagnostic per source, so the fix is mechanical rather than a hunt."""
    _write_profile(tmp_path / "p.py", hooks={"approve": True})
    profiles = _discover(tmp_path)

    diagnostics = validate(
        profiles, tools=_tools(), hook_names={"approve", "audit", "telemetry"}
    )

    assert [d.fields["hook"] for d in diagnostics] == ["audit", "telemetry"]


def test_no_hook_sources_means_an_empty_decision_is_complete(tmp_path: Path) -> None:
    """Exhaustiveness must not tax the common case: an install with no
    hook-bearing extensions needs no `hooks` at all."""
    _write_profile(tmp_path / "p.py", tools=("read",))
    profiles = _discover(tmp_path)

    assert validate(profiles, tools=_tools("read"), hook_names=set()) == []
    assert "reviewer" in profiles


def test_tools_stay_an_allowlist_not_a_decision(tmp_path: Path) -> None:
    """The asymmetry, pinned. A tool left out is simply not granted; there is
    no `profile_tool_undecided`, because omission there is fail-safe."""
    _write_profile(tmp_path / "p.py", tools=("read",))
    profiles = _discover(tmp_path)

    assert validate(profiles, tools=_tools("read", "bash", "write"), hook_names=set()) == []
    assert "reviewer" in profiles
