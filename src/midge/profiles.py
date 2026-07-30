"""Profiles: named configurations the agent can be retargeted to.

A profile is *what the agent is* — a system prompt, a model, a subset of the
discovered tools, and a set of active hooks, bundled under a name so that
"the adversarial reviewer" is one thing rather than three facts a reader has to
correlate. See `docs/adr/0001-session-profiles.md`.

This module is the declaration half only: the dataclass, the collection, and
validation. **Nothing here applies a profile** — that is `use_profile` (#67),
which needs source-scoped hook activation (#60) first.

The format is a `.py` file holding a `Profile` instance, collected from the
module namespace exactly as `load_extensions` collects `Tool` instances:

    from midge.profiles import Profile

    ADVERSARIAL = Profile(
        name="adversarial-reviewer",
        description="Reviews recent work looking for what is wrong with it.",
        model="gpt-4o",
        tools=("read", "bash"),
        hooks={"approve": True, "audit": False},
        prompt="Assume the work is wrong and find out how.",
    )

**`hooks` must decide every discovered hook source, and a profile that leaves
one out does not load.** That is deliberately stricter than `tools`, and the
asymmetry is the point: omitting a tool yields a *less* capable agent, while
omitting a hook would yield an *unguarded* one. A profile reads like a
convenience layer, so the natural mistake is to write only what you mean to
change — and under "the list is what is active" that instinct silently switches
off an approval gate, while under "the list is what changes" a profile can never
turn anything off at all. Requiring a decision per source makes the question
unaskable rather than picking a safer answer to it, and turns the mistake into a
startup diagnostic naming the source you forgot.

The cost, stated plainly: installing a new hook-bearing extension invalidates
every existing profile until each names it. That is loud and one line to fix,
which is the trade — a noisy failure over a silent capability grant.

Chosen over a Markdown file with frontmatter because a profile intersects code:
`tools` and `hooks` name symbols that must exist, so a dataclass gets structural
validation at import and is checkable by pyright, where frontmatter defers every
error to runtime. One `.py` file may declare a tool, a sub-agent and a profile
together, so there is no new loader and no new flag.

`Profile` deliberately does **not** converge with `SubagentSpec`. Their fields
nearly coincide — `name`, `prompt`, `tools`, `model` — but a sub-agent is a tool
the agent uses and a profile is what the agent is. Unifying them would put a
`timeout` on a profile and imply either can stand in for the other.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from midge.config import Diagnostic
from midge.providers import ModelRegistry
from midge.tools import ToolRegistry


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    description: str
    prompt: str
    # "" rather than None: an unset model means "whatever the agent is already
    # running", which is a real and common choice — a profile is often a prompt
    # and a toolset with no opinion about the provider.
    model: str = ""
    tools: tuple[str, ...] = ()
    # A decision for *every* discovered hook source, keyed by the extension
    # file's stem — which is what `load_extensions` stamps on every registration
    # it makes. Exhaustive on purpose; see the module docstring.
    hooks: Mapping[str, bool] = dataclass_field(default_factory=dict)


class ProfileSet:
    """Discovered profiles by name, remembering which file each came from.

    The path is held here rather than on `Profile` so a hand-written profile —
    in a test, or built by an embedder — needs no file to exist. It is also not
    something a profile's author writes.

    Like `ToolRegistry`, this does not warn on collision; the loader does, so
    the warning can name both files.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, Profile] = {}
        self._paths: dict[str, Path] = {}

    def add(self, profile: Profile, *, path: Path | None = None) -> None:
        if profile.name in self._profiles:
            raise ValueError(f"Profile {profile.name!r} already registered")
        self._profiles[profile.name] = profile
        if path is not None:
            self._paths[profile.name] = path

    def get(self, name: str) -> Profile | None:
        return self._profiles.get(name)

    def path_of(self, name: str) -> Path | None:
        return self._paths.get(name)

    def remove(self, name: str) -> None:
        self._profiles.pop(name, None)
        self._paths.pop(name, None)

    def names(self) -> list[str]:
        return list(self._profiles)

    def __contains__(self, name: str) -> bool:
        return name in self._profiles

    def __iter__(self) -> Iterator[Profile]:
        return iter(self._profiles.values())

    def __len__(self) -> int:
        return len(self._profiles)


def validate(
    profiles: ProfileSet,
    *,
    tools: ToolRegistry,
    hook_names: set[str],
    models: ModelRegistry | None = None,
) -> list[Diagnostic]:
    """Drop every profile that names something which does not exist.

    Run once *after all sources are loaded*, never per file: a profile may name
    a tool declared in another extension, and there is no ordering that makes
    per-file validation correct.

    A failing profile is removed rather than degraded, so a later switch to it
    fails with a clear error instead of silently granting fewer tools than it
    claims (ADR Decision 9). Diagnostics are returned rather than logged, the
    shape `ModelRegistry` already uses, so an entrypoint emits profile, config
    and registry problems through one call.
    """
    diagnostics: list[Diagnostic] = []
    for profile in list(profiles):
        problems = _problems(profile, tools=tools, hook_names=hook_names, models=models)
        if not problems:
            continue
        where = profiles.path_of(profile.name)
        for d in problems:
            diagnostics.append(Diagnostic(d.event, {**d.fields, "path": where or "-"}))
        profiles.remove(profile.name)
    return diagnostics


def _problems(
    profile: Profile,
    *,
    tools: ToolRegistry,
    hook_names: set[str],
    models: ModelRegistry | None,
) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    if not profile.name.strip() or not profile.prompt.strip():
        out.append(Diagnostic("profile_invalid", {"name": profile.name or "-"}))
        # Nothing below can say anything useful about a profile this broken.
        return out
    out.extend(
        Diagnostic("profile_tool_unknown", {"profile": profile.name, "tool": name})
        for name in profile.tools
        if name not in tools
    )
    out.extend(
        Diagnostic("profile_hook_unknown", {"profile": profile.name, "hook": name})
        for name in profile.hooks
        if name not in hook_names
    )
    # The dual of the check above, and the reason `hooks` is a mapping rather
    # than a list of active names: a source nobody decided about would have to
    # default to on or off, and both defaults are a silent answer to the
    # question the author did not know they were being asked.
    out.extend(
        Diagnostic("profile_hook_undecided", {"profile": profile.name, "hook": name})
        for name in sorted(hook_names - set(profile.hooks))
    )
    # Empty means permissive, the rule everywhere the registry is consulted: a
    # profile naming a model is not an error in an unconfigured install, and
    # becomes one only once the user has declared what they want available.
    if models and profile.model and profile.model not in models:
        out.append(
            Diagnostic(
                "profile_model_unregistered",
                {
                    "profile": profile.name,
                    "model": profile.model,
                    "registered": ",".join(models.names()),
                },
            )
        )
    return out


__all__ = ["Profile", "ProfileSet", "validate"]
