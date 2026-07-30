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
from midge.persistence import Session
from midge.profiles import ProfileSet
from midge.profiles import validate as validate_profiles
from midge.providers import Capabilities, ModelRegistry
from midge.rpc import RpcServer, serve_stdio
from midge.skills import default_skill_dirs, load_skills, skills_prompt
from midge.subagents import bind_subagents
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
        help="Append turns to a JSONL session file. Resumes if the file exists.",
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


def resume_identity(session: Session) -> tuple[str, str]:
    """The model and base prompt to resume a session with.

    Not `header.model` / `header.system_prompt`: the header is what the session
    *started* as and is never rewritten. `set_model` and `set_system_prompt`
    append records that supersede it, which `Session.load` has already folded
    onto these attributes. Reading the header instead is the defect in #57 —
    both commands reported success and then silently reverted.
    """
    return session.model, session.system_prompt or BASE_SYSTEM_PROMPT


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

    if args.session is not None:
        if args.session.exists():
            session = Session.load(args.session)
            model, durable = resume_identity(session)
        else:
            session = Session.new(args.session, model=model, system_prompt=durable)

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
        args.session or "-",
    )
    # The registry validates its own wiring — a model naming a provider that was
    # never defined, or a provider naming an adapter that does not exist — and
    # reports it the same way config parsing does.
    model_registry = ModelRegistry(models=config.models, providers=config.providers)
    emit_config_diagnostics(model_registry.diagnostics)
    if model_registry and model not in model_registry:
        # The startup model is a selection like any other, so it is subject to
        # the same rule as `set_model`. Refusing here beats a first turn that
        # fails with the vendor's 404.
        _logger.error(
            "startup_model_unregistered model=%s registered=%s",
            model,
            ",".join(model_registry.names()),
        )
        raise SystemExit(
            f"model {model!r} is not in the model registry; registered: "
            f"{', '.join(model_registry.names())}"
        )

    # After every source is loaded, because a profile may name a tool declared
    # in another file — and after the model registry, which is the third thing
    # a profile can name. A profile that fails is dropped, so the selection
    # check below sees only profiles that would really work.
    discovered = set(profiles.names())
    emit_config_diagnostics(
        validate_profiles(
            profiles,
            tools=registry,
            hook_names=hooks.source_names(),
            models=model_registry,
        )
    )
    selected = args.profile or config.default_profile
    if selected is not None and selected not in profiles:
        # Nothing applies a profile yet (#67), but a name that resolves to
        # nothing is a mistake either way, and saying so now beats saying it
        # once switching exists.
        #
        # Dropped and never-there are different mistakes with different fixes —
        # a broken profile file versus a wrong name — so they are not allowed to
        # share a message. Saying "not discovered" about a profile that was
        # discovered and then rejected sends the reader to the wrong file.
        available = ", ".join(profiles.names()) or "none"
        if selected in discovered:
            _logger.error("startup_profile_invalid profile=%s available=%s", selected, available)
            raise SystemExit(
                f"profile {selected!r} was found but failed validation — see the "
                f"profile_* warnings above; usable profiles: {available}"
            )
        _logger.error("startup_profile_unknown profile=%s available=%s", selected, available)
        raise SystemExit(f"profile {selected!r} was not discovered; available: {available}")
    _logger.info("profiles_loaded count=%d selected=%s", len(profiles), selected or "-")

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
        registry=model_registry,
    )
    # Tools cannot reach the calling agent, so any sub-agent tool an extension
    # registered gets what it needs to run a child here. No-op without them.
    bind_subagents(
        registry,
        client=client,
        model=model,
        hooks=hooks,
        session_path=args.session,
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

    session_path = str(args.session) if args.session is not None else None

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
