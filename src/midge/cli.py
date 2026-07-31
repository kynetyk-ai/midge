"""`midge` CLI: launches the interactive TUI.

Usage:
    midge [--extension-dir DIR] [--skill-dir DIR] [--session PATH] \\
       [--profile NAME] [--compaction-threshold N] [--compaction-keep-recent N]

Configuration is `.midge/config.toml`, overridden by environment variables,
overridden by these flags — see `midge.config`. `OPENAI_API_KEY` is read from the
environment only, and never from the file.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from midge.agent import Agent
from midge.client import Client
from midge.config import Config
from midge.config import emit as emit_config_diagnostics
from midge.extensions import BUILTIN_TOOL_DIRS, load_extensions
from midge.hooks import Hooks, SessionEnd, SessionStart
from midge.logs import configure as configure_logging
from midge.logs import provider_host
from midge.persistence import Session, resolve_session_path
from midge.profiles import ProfileSet
from midge.profiles import validate as validate_profiles
from midge.providers import Capabilities, ModelRegistry
from midge.rpc import RpcServer, serve_stdio
from midge.skills import default_skill_dirs, load_skills, skills_prompt
from midge.subagents import bind_subagents
from midge.subagents import validate as validate_subagents
from midge.tools import ToolRegistry
from midge.tui import run_tui, tui_log_handler

_logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = (
    "You are a coding assistant working in a local repository. "
    "Use the available tools to inspect and modify files. "
    "Keep responses concise and prefer doing work over describing it."
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="midge")
    parser.add_argument(
        "--extension-dir",
        action="append",
        type=Path,
        default=[],
        metavar="DIR",
        help="Directory of extension .py files to load (repeatable).",
    )
    parser.add_argument(
        "--skill-dir",
        action="append",
        type=Path,
        default=[],
        metavar="DIR",
        help=(
            "Directory of SKILL.md skills to load (repeatable). Searched before "
            "the project and user defaults."
        ),
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help=(
            "Start under a discovered profile, by name. Outranks [profiles] "
            "default. Profiles are declared in extension .py files."
        ),
    )
    parser.add_argument(
        "--rpc",
        action="store_true",
        help=(
            "Serve the JSON-over-stdio RPC protocol instead of the TUI. Stdout "
            "carries the protocol; diagnostics go to stderr."
        ),
    )
    parser.add_argument(
        "--session",
        type=Path,
        metavar="PATH",
        help=(
            "Append turns to a JSONL session file. Resumes if the file exists. "
            "A relative path is taken under the session directory. Without this, "
            "a timestamped file is created there."
        ),
    )
    # `store_true` despite the argparse-default trap `tests/test_cli_config.py`
    # pins: absent is False, which means "no opinion" here rather than a value
    # that would shadow `[session] enabled`.
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="Do not write a transcript for this run.",
    )
    parser.add_argument(
        "--compaction-threshold",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run compaction after a turn if estimated history tokens exceed N. Disabled if not set."
        ),
    )
    parser.add_argument(
        "--compaction-keep-recent",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Token budget for the suffix kept verbatim after compaction "
            "(default 20000, or [compaction] keep_recent)."
        ),
    )
    return parser.parse_args(argv)


def resume_identity(
    session: Session,
    *,
    configured: str,
    configured_explicitly: bool = False,
    registry: ModelRegistry | None = None,
) -> tuple[str, str]:
    """The model and base prompt to resume a session with.

    Not `header.model` / `header.system_prompt`: the header is what the session
    *started* as and is never rewritten. `set_model` and `set_system_prompt`
    append records that supersede it, which `Session.load` has already folded
    onto these attributes. Reading the header instead is the defect in #57 —
    both commands reported success and then silently reverted.

    **The two halves are not the same kind of thing**, and are treated
    differently on purpose.

    The **base prompt** is part of what the conversation *is*. Resuming a
    reviewer's transcript under a coding assistant's instructions would make its
    own history misleading, so it is always restored. Once a profile is recorded
    (#67) that is the better thing to restore, and this becomes its fallback.

    The **model** is infrastructure — interchangeable, machine-specific, and it
    has its own config key. So it is a *stored prior choice* that takes part in
    precedence rather than sitting above it: it beats a default, and loses to a
    model the operator asked for this run. That fixes two things. An operator
    who sets `MIDGE_MODEL` and resumes no longer has it silently discarded; and
    a session recorded against a model since retired no longer refuses to
    start, because a value nobody chose this run should degrade rather than
    block. Both warn, so a disagreement is visible and the config can be
    reconsidered.
    """
    durable = session.system_prompt or BASE_SYSTEM_PROMPT
    recorded = session.model
    if configured_explicitly:
        if recorded != configured:
            _logger.warning(
                "resume_model_overridden recorded=%s using=%s", recorded, configured
            )
        return configured, durable
    if registry and recorded not in registry:
        # A warning rather than the refusal an operator-named model gets: they
        # did not choose this one this run, so blocking startup would strand
        # the session on a model id that has simply been retired.
        _logger.warning(
            "resume_model_unregistered recorded=%s using=%s registered=%s",
            recorded,
            configured,
            ",".join(registry.names()),
        )
        return configured, durable
    return recorded, durable


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    # Config is parsed before logging is configured, because the log level is one
    # of the things it resolves. `load` therefore logs nothing and hands back
    # diagnostics, which are emitted as soon as there is somewhere to put them.
    config, diagnostics = Config.load()
    # Before `load_extensions`/`load_skills`, which are the two loudest
    # loaders — a handler installed after them loses every startup diagnostic.
    # The TUI needs a handler that resolves per record; RPC needs stderr,
    # because a handler bound to stdout would corrupt the protocol.
    configure_logging(
        None if args.rpc else tui_log_handler(config.log.file),
        log=config.log,
    )
    emit_config_diagnostics(diagnostics)

    # A flag outranks env, which outranks the file. Argparse defaults are None so
    # that an unset flag does not beat a configured value.
    keep_recent: int = (
        config.compaction_keep_recent
        if args.compaction_keep_recent is None
        else args.compaction_keep_recent
    )
    threshold: int | None = (
        config.compaction_threshold
        if args.compaction_threshold is None
        else args.compaction_threshold
    )

    hooks = Hooks()
    extension_sources = [*BUILTIN_TOOL_DIRS, *args.extension_dir]
    # Explicit paths outrank the defaults: naming a directory on the command
    # line is a deliberate override. Note this is the opposite nesting from the
    # extension sources above, where the built-ins must not be shadowed.
    skill_sources = [*args.skill_dir, *default_skill_dirs()]
    profiles = ProfileSet()
    registry, prompt_addition = load_extensions(extension_sources, hooks=hooks, profiles=profiles)
    skills = load_skills(skill_sources)
    catalogue = skills_prompt(skills) if "read" in registry else ""

    session: Session | None = None
    model = config.model
    durable = BASE_SYSTEM_PROMPT

    # Before the session opens, because resuming one consults it: a recorded
    # model that is no longer registered degrades to the configured one rather
    # than refusing to start. The registry validates its own wiring — a model
    # naming a provider that was never defined, or a provider naming an adapter
    # that does not exist — and reports it the way config parsing does.
    model_registry = ModelRegistry(models=config.models, providers=config.providers)
    emit_config_diagnostics(model_registry.diagnostics)

    if args.no_session and args.session is not None:
        # Contradictory, so say which won rather than silently picking. Same
        # treatment `Config.load` gives `[providers.*]` colliding with the
        # singular `provider`.
        _logger.warning("session_flags_conflict ignoring=--session in_favour_of=--no-session")
    requested = None if args.no_session else args.session
    session_file = resolve_session_path(
        requested,
        directory=config.session.dir,
        enabled=config.session.enabled and not args.no_session,
    )
    if session_file is not None:
        # `open` rather than the load/new pair: a generated path never exists
        # and a named one may or may not, which is exactly what it decides.
        session = Session.open(session_file, model=model, system_prompt=durable)
        # Unconditional: on a session just created these read back exactly what
        # was passed in, so there is nothing to branch on.
        model, durable = resume_identity(
            session,
            configured=config.model,
            configured_explicitly=bool(config.model_source),
            registry=model_registry,
        )

    # After every source is loaded, because a profile may name a tool declared
    # in another file — and after the model registry, which is the third thing
    # a profile can name. A profile that fails is dropped, so the selection
    # below sees only profiles that would really work.
    # Before profiles, because a profile's `tools` may name a `spawn_*` tool and
    # a sub-agent that fails here is gone — so the profile is validated against
    # the registry that will actually exist.
    emit_config_diagnostics(validate_subagents(registry, models=model_registry))

    discovered = set(profiles.names())
    emit_config_diagnostics(
        validate_profiles(
            profiles,
            tools=registry,
            hook_names=hooks.source_names(),
            models=model_registry,
        )
    )
    # Flag, then what the session was running under, then the configured
    # default. The middle term is the same rule the model follows: a session's
    # own record beats a global default and loses to something asked for this
    # run. Unlike the model it is restored rather than degraded — a profile is a
    # deliberate named identity, not an incidental setting.
    recorded = session.profile if session is not None else None
    if recorded is not None and recorded not in profiles:
        _logger.warning(
            "resume_profile_unavailable profile=%s using=%s",
            recorded,
            "the recorded prompt and model",
        )
        recorded = None
    selected = args.profile or recorded or config.default_profile
    if selected is not None and selected not in profiles:
        available = ", ".join(profiles.names()) or "none"
        # Dropped and never-there are different mistakes with different fixes —
        # a broken profile file versus a wrong name — so they do not share a
        # message. Saying "not discovered" about a profile that was discovered
        # and then rejected sends the reader to the wrong file.
        if selected in discovered:
            _logger.error("startup_profile_invalid profile=%s available=%s", selected, available)
            raise SystemExit(
                f"profile {selected!r} was found but failed validation — see the "
                f"profile_* warnings above; usable profiles: {available}"
            )
        _logger.error("startup_profile_unknown profile=%s available=%s", selected, available)
        raise SystemExit(f"profile {selected!r} was not discovered; available: {available}")

    active = profiles.get(selected) if selected is not None else None
    if active is not None:
        durable = active.prompt
        model = active.model or model
        registry = ToolRegistry([t for t in registry if t.name in active.tools])
        hooks.set_active_sources({n for n, on in active.hooks.items() if on})
        if session is not None:
            session.set_profile(name=active.name, model=model, system_prompt=durable)
    _logger.info("profiles_loaded count=%d selected=%s", len(profiles), selected or "-")

    # The header records the agent's identity. Which tools and skills exist is a
    # fact about this machine right now, so it is recomposed on every start
    # rather than restored — otherwise a skill added after the session began is
    # invisible, and its absolute paths could point at another machine entirely.
    full_prompt = "\n\n".join(p for p in (durable, prompt_addition, catalogue) if p)

    _logger.info(
        "startup mode=%s model=%s provider=%s tools=%d skills=%d session=%s",
        "rpc" if args.rpc else "tui",
        model,
        provider_host(config.base_url),
        len(registry),
        len(skills),
        session_file or "-",
    )
    if model_registry and model not in model_registry:
        # Only reachable for a model the *operator* named — `resume_identity`
        # has already degraded a recorded one to the configured model with a
        # warning. A selection made this run is subject to the same rule as
        # `set_model`, and refusing beats a first turn that fails with the
        # vendor's 404.
        _logger.error(
            "startup_model_unregistered model=%s registered=%s",
            model,
            ",".join(model_registry.names()),
        )
        raise SystemExit(
            f"model {model!r} is not in the model registry; registered: "
            f"{', '.join(model_registry.names())}"
        )


    # No api_key: the fallback provider reads `OPENAI_API_KEY` itself and a
    # registered one names its own variable, so a credential never passes
    # through configuration.
    client = Client(
        base_url=config.base_url,
        provider=config.provider,
        capabilities=(
            None
            if config.include_usage is None
            else Capabilities(stream_usage=config.include_usage)
        ),
        max_attempts=config.retry.max_attempts,
        retry_base_delay=config.retry.base_delay,
        retry_max_delay=config.retry.max_delay,
        registry=model_registry,
    )
    # Tools cannot reach the calling agent, so any sub-agent tool an extension
    # registered gets what it needs to run a child here. No-op without them.
    bind_subagents(
        registry,
        client=client,
        model=model,
        hooks=hooks,
        session=session,
        subagents=config.subagents,
    )
    agent = Agent(
        client=client,
        model=model,
        tools=registry,
        system_prompt=full_prompt,
        hooks=hooks,
    )
    if session is not None:
        agent.history = list(session.messages)

    session_path = str(session_file) if session_file is not None else None

    if args.rpc:
        # RPC owns its loop, so the session bookends run inside it rather than
        # in their own `asyncio.run` the way the TUI's do.
        server = RpcServer(
            agent,
            session=session,
            compaction_keep_recent=keep_recent,
            base_prompt=durable,
            extension_prompt=prompt_addition,
            skills=skills,
            profiles=profiles,
            # The server re-binds sub-agents on reload, `new_session` and a
            # profile switch, so it needs the limits rather than reaching for
            # the library defaults each time.
            subagents=config.subagents,
            resume_fallback="continue" if config.resume_fallback == "continue" else "fork",
            # Where `list_sessions` looks. The same directory `resolve_session_path`
            # writes into, so a listing shows what this process has been creating.
            session_dir=config.session.dir,
            # The same lists the loaders above were given, so `reload` repeats
            # that call rather than rebuilding it.
            extension_sources=extension_sources,
            skill_sources=skill_sources,
        )

        async def _serve() -> None:
            await hooks.emit(SessionStart(path=session_path))
            try:
                await serve_stdio(server)
            finally:
                if server.session is not None:
                    server.session.close()
                await hooks.emit(SessionEnd(path=session_path))

        asyncio.run(_serve())
        return

    # These bookend the TUI's own event loop, so they run in their own.
    # A handler that needs the running app's loop should use a turn-scoped
    # event instead.
    asyncio.run(hooks.emit(SessionStart(path=session_path)))
    try:
        run_tui(
            agent,
            session=session,
            compaction_threshold=threshold,
            compaction_keep_recent=keep_recent,
        )
    finally:
        if session is not None:
            session.close()
        asyncio.run(hooks.emit(SessionEnd(path=session_path)))
