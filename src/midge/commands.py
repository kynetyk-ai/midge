"""What an operator can do to a running agent, and the one place it is done.

Every control — compact, clear, switch model, switch profile, open another
transcript, reload from disk — used to live inside `RpcServer`'s handlers, each
one parsing JSON, checking policy, mutating the agent and formatting a response
in a single function body. That was an accident of build order rather than a
design: RPC was simply the first consumer. It left the TUI with nothing to call,
so the TUI grew its own copy of compaction and could not offer anything else.

So the split here is **front-end versus operation**:

    rpc/server.py   JSON in  -> Controls -> JSON out
    tui/            a keypress -> Controls -> a rendered result

`Controls` takes typed arguments and returns plain data. It never parses a
request, never formats a response, and never decides how a refusal is shown. A
front-end owns its own vocabulary; what an operation *means* is here, once.

`BUILTIN_COMMANDS` lives here for the same reason. It is the list of things a
human may invoke, and both surfaces enumerate it — which is what stops them
drifting apart again, since a command added to the table appears in both without
either front-end being edited.

Two things deliberately stay outside:

**Argument validation.** RPC validates JSON against a pydantic model; a TUI
parses `/model gpt-4o`. Both arrive here with a `str`. Only `Refused` — a policy
answer, not a parse error — comes back out.

**Owning the run.** Whether a turn is in flight is an `asyncio.Task` to RPC and
a Textual `Worker` to the TUI, so `Runner` is the seam. It is what lets the
clear-then-cancel ordering in `abort`, and the mid-run refusals, be stated once
rather than rediscovered per front-end.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from midge.agent import Agent
from midge.compaction import compact
from midge.config import SubagentConfig
from midge.config import emit as emit_diagnostics
from midge.extensions import load_extensions
from midge.messages import UserMessage
from midge.persistence import (
    ProfileRecord,
    Session,
    list_sessions,
    read_transcript,
    session_chain,
)
from midge.profiles import Profile, ProfileSet
from midge.profiles import validate as validate_profiles
from midge.skills import Skill, load_skills, skill_message, skills_prompt
from midge.subagents import bind_subagents
from midge.tools import ToolRegistry

_logger = logging.getLogger(__name__)

# A profile name reaches the filesystem when a fork names a transcript after it.
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")

SKILL_COMMAND_PREFIX = "/skill:"


class Refused(Exception):
    """An operation the current state does not allow.

    Distinct from a bad argument, which a front-end catches before it gets here.
    A refusal is the answer to a well-formed request — no session to name, a run
    in flight, a profile that does not exist — so it carries a message meant to
    be shown to whoever asked.
    """


class Runner(Protocol):
    """Whoever owns the turn in flight.

    RPC holds an `asyncio.Task`; the TUI holds a Textual `Worker`. Neither
    abstraction belongs here, but the *questions* do — which commands refuse
    mid-run, and that `abort` clears the queue before cancelling.
    """

    def busy(self) -> bool: ...

    def cancel(self) -> None: ...


# --- command schemas ------------------------------------------------------


class _CommandParams(BaseModel):
    """Base for built-in command schemas.

    `extra="forbid"` so the generated schema carries `additionalProperties:
    false`, matching what `Tool.schema()` produces — a consumer that can render
    a tool call can render a command with no second convention to learn.
    """

    model_config = ConfigDict(extra="forbid")


class SetModelParams(_CommandParams):
    model: str = Field(description="Provider model id, e.g. gpt-4o")


class SetSystemPromptParams(_CommandParams):
    prompt: str = Field(
        description="Replaces the durable base prompt; the generated half is re-appended"
    )


class NewSessionParams(_CommandParams):
    path: str = Field(description="Path for the new session log")


class OpenSessionParams(_CommandParams):
    path: str = Field(description="Session log to attach to; created if it does not exist")


class SessionNameParams(_CommandParams):
    name: str = Field(description="Display name for the current session")


TRANSCRIPT_OPTIONS = ("continue", "fork", "resume_last")


class UseProfileParams(_CommandParams):
    name: str = Field(description="Name of a discovered profile")
    transcript: Literal["continue", "fork", "resume_last"] = Field(
        default="continue",
        description=(
            "continue: stay on this transcript. fork: open a new one, linked to "
            "it. resume_last: reopen this session's most recent transcript under "
            "the profile, falling back per `[profiles] resume_fallback`."
        ),
    )


RELOAD_TARGETS = ("skills", "extensions")


class ReloadParams(_CommandParams):
    targets: list[Literal["skills", "extensions"]] | None = Field(
        default=None, description="Which sources to re-scan; omit for all of them"
    )


@dataclass(frozen=True, slots=True)
class BuiltinCommand:
    name: str
    description: str
    params: type[BaseModel] | None = None


# midge does get an opinion about what is a user-facing action rather than
# protocol plumbing. Out: `prompt`, `steer` and `follow_up`, which *are* the
# interaction — a UI picks between them by policy when the user hits enter — and
# the `get_*` family, which a client reads to render itself rather than offering
# to a user. `abort` is in: leaving it out assumed every consumer has an escape
# key, and a chat bot does not.
BUILTIN_COMMANDS: tuple[BuiltinCommand, ...] = (
    BuiltinCommand("abort", "Stop the run in flight and discard anything queued"),
    BuiltinCommand("compact", "Summarize older turns to reclaim context"),
    BuiltinCommand("clear_context", "Forget the conversation; keep recording to the same log"),
    BuiltinCommand("new_session", "Close the current log and start a fresh one", NewSessionParams),
    BuiltinCommand(
        "open_session",
        "Attach to an existing session log, restoring its conversation",
        OpenSessionParams,
    ),
    BuiltinCommand("set_model", "Switch the model used for subsequent turns", SetModelParams),
    BuiltinCommand(
        "set_system_prompt", "Replace the agent's base system prompt", SetSystemPromptParams
    ),
    BuiltinCommand("reload", "Re-scan skills and extensions from disk", ReloadParams),
    BuiltinCommand(
        "set_session_name", "Give the current session a display name", SessionNameParams
    ),
    BuiltinCommand(
        "use_profile",
        "Retarget the agent to a named profile — prompt, model, tools and hooks at once",
        UseProfileParams,
    ),
)

_NO_PARAMS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


class Controls:
    """The agent, plus everything an operator can change about it."""

    def __init__(
        self,
        agent: Agent,
        *,
        session: Session | None = None,
        compaction_keep_recent: int = 20_000,
        base_prompt: str | None = None,
        extension_prompt: str = "",
        skills: Sequence[Skill] | None = None,
        profiles: ProfileSet | None = None,
        subagents: SubagentConfig | None = None,
        resume_fallback: Literal["fork", "continue"] = "fork",
        extension_sources: Sequence[Path] | None = None,
        skill_sources: Sequence[Path] | None = None,
        session_dir: Path | None = None,
        runner: Runner | None = None,
        on_subagent_event: Any = None,
    ) -> None:
        self.agent = agent
        # `client` and `model` come off the agent; `session` is what the
        # persistence commands need and cannot reach otherwise.
        self.session = session
        self.compaction_keep_recent = compaction_keep_recent
        # The agent's prompt is composed: a durable base the operator owns, then
        # what midge generates — extension contributions and the skills
        # catalogue. Keeping the halves apart is what lets `set_system_prompt`
        # change the base without silently deleting the catalogue, and what lets
        # `reload` replace one generated half without disturbing the other.
        self.base_prompt = base_prompt if base_prompt is not None else (agent.system_prompt or "")
        self.extension_prompt = extension_prompt
        self.skills: Sequence[Skill] = skills or ()
        # Held as the set rather than a list because the source path is part of
        # what a client is shown, and only the set knows it.
        self.profiles = profiles if profiles is not None else ProfileSet()
        # Everything discovered, kept apart from `agent.tools` because a profile
        # *projects* a subset onto the latter. Switching from a two-tool profile
        # to a five-tool one has to project from the whole set; projecting from
        # the already-narrowed one would make each switch a ratchet.
        self.discovered_tools = agent.tools
        self.profile: str | None = session.profile if session is not None else None
        self.resume_fallback: Literal["fork", "continue"] = resume_fallback
        # Held rather than taken once, because a reload, a `new_session` and a
        # profile switch all re-bind.
        self.subagents = subagents
        self.extension_sources = extension_sources
        self.skill_sources = skill_sources
        # Where to look when asked what sessions exist. `None` means the
        # default, resolved at call time rather than here — `Path.cwd()` frozen
        # at construction is the trap `default_session_dir` exists to avoid.
        self.session_dir = session_dir
        self.runner = runner
        self.on_subagent_event = on_subagent_event

    # --- what the front-ends read to render themselves --------------------

    def busy(self) -> bool:
        return self.runner is not None and self.runner.busy()

    def _refuse_if_busy(self, what: str) -> None:
        if self.busy():
            raise Refused(f"cannot {what} while a run is in flight")

    def generated_prompt(self) -> str:
        """Extension contributions plus the skills catalogue, gated as at startup.

        Derived rather than stored so the `read` gate cannot fall out of date:
        the catalogue tells the model to open a `SKILL.md`, so without a tool
        that can open one it is an instruction to do the impossible. An
        extensions reload can add or remove `read`, which makes this the one
        point where reloading extensions changes the skills half of the prompt.
        """
        catalogue = skills_prompt(self.skills) if "read" in self.agent.tools else ""
        return "\n\n".join(p for p in (self.extension_prompt, catalogue) if p)

    def compose_prompt(self) -> str:
        return "\n\n".join(p for p in (self.base_prompt, self.generated_prompt()) if p)

    def state(self) -> dict[str, Any]:
        return {
            "model": self.agent.model,
            "streaming": self.busy(),
            "session": str(self.session.path) if self.session is not None else None,
            "session_name": self.session.name if self.session is not None else None,
            "messages": len(self.agent.history),
        }

    def builtin_schema(self, command: BuiltinCommand) -> dict[str, Any]:
        """The command's arguments, narrowed to what this process can accept.

        The enums are why a palette needs no parser: `set_model` and
        `use_profile` are the two commands whose valid values are knowable up
        front, so a front-end can offer them as choices rather than asking for
        free text and refusing it afterwards.
        """
        schema = command.params.model_json_schema() if command.params else dict(_NO_PARAMS)
        registry = self.agent.client.registry
        if command.name == "set_model" and registry:
            schema["properties"]["model"]["enum"] = registry.names()
        if command.name == "use_profile":
            schema["properties"]["name"]["enum"] = self.profiles.names()
        return schema

    def commands(self) -> list[dict[str, Any]]:
        """Everything a user can invoke, and how to invoke it.

        Read-only; executes nothing. A projection of what already exists —
        built-ins from the table above, skills from disk — rather than a new
        concept, which is what makes one description serve a socket client and a
        command palette alike.

        `invoke` says how to transmit: `command` means send `{"type": name, …}`,
        `prompt` means put the text in a prompt/steer/follow_up message.
        `parameters` is JSON Schema in the same shape `Tool.schema()` produces,
        so an empty `properties` is the "select and fire" signal. Note it means
        slightly different things per `invoke`: for a command the properties are
        keys in the request object; for a prompt the single property is free
        text appended after the name. A prompt-invoked command takes at most one
        argument, which is what keeps that unambiguous.

        Deliberately absent: any notion of whether an entry is dangerous enough
        to confirm. That is a front-end policy — a misclick in a terminal and one
        in a shared channel are not the same risk.
        """
        out: list[dict[str, Any]] = [
            {
                "name": c.name,
                "source": "builtin",
                "invoke": "command",
                "description": c.description,
                "parameters": self.builtin_schema(c),
            }
            for c in BUILTIN_COMMANDS
        ]
        # Listed regardless of `model_invocable`: hiding a skill from the model's
        # catalogue is exactly the case where an explicit command is the only
        # way to reach it.
        out.extend(
            {
                "name": f"skill:{s.name}",
                "source": "skill",
                "invoke": "prompt",
                "description": s.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instructions": {
                            "type": "string",
                            "description": "Extra guidance appended to the skill",
                        }
                    },
                    "additionalProperties": False,
                },
                "source_info": {"path": str(s.path)},
            }
            for s in self.skills
        )
        return out

    def profile_list(self) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "description": p.description,
                # An empty declaration means "whatever the agent is running",
                # which is not the same as a model named as the empty string.
                "model": p.model or None,
                "tools": sorted(p.tools),
                "hooks": list(p.hooks),
                "prompt": p.prompt,
                "source": str(path) if (path := self.profiles.path_of(p.name)) else None,
            }
            for p in self.profiles
        ]

    def session_list(self, *, roots_only: bool = True) -> list[dict[str, Any]]:
        """What conversations exist, for something that offers a choice of them.

        A read, not a command — the same relationship `get_profiles` has to
        `use_profile`. This renders the picker; `open_session` is the act.

        `current` is here because the one thing a picker must not do is invite
        you to reopen what you are already in. Comparing paths at the call site
        would mean every front-end getting the same comparison right.
        """
        current = self.session.path if self.session is not None else None
        return [
            {
                "path": str(s.path),
                "name": s.name,
                "created_at": s.created_at,
                "model": s.model,
                "messages": s.messages,
                "modified": s.modified,
                "current": s.path == current,
            }
            for s in list_sessions(self.session_dir, roots_only=roots_only)
        ]

    def expand(self, text: str) -> str | UserMessage:
        """Resolve a `/skill:` invocation at enqueue time.

        Whatever is queued has to already be a plain message: a skill resolved
        when it *runs* rather than when it was asked for could pick up a
        different file, which is not what the caller chose.
        """
        if not text.startswith(SKILL_COMMAND_PREFIX):
            return text
        rest = text[len(SKILL_COMMAND_PREFIX) :]
        name, _, args = rest.partition(" ")
        return skill_message(self.skills, name, args.strip() or None)

    # --- the operations ---------------------------------------------------

    def bind_subagents(self, registry: ToolRegistry, *, model: str | None = None) -> None:
        """Every re-bind goes through here so none can forget a setting.

        Reload, `new_session`, `open_session` and a profile switch all re-bind;
        each rebuilding the argument list by hand is how the operator's limits
        used to get reset to the defaults.
        """
        bind_subagents(
            registry,
            client=self.agent.client,
            model=model if model is not None else self.agent.model,
            hooks=self.agent.hooks,
            session=self.session,
            subagents=self.subagents,
            on_event=self.on_subagent_event,
        )

    async def compact(self) -> dict[str, Any]:
        """Summarize older turns to reclaim context.

        Not refused mid-run, and that is a known hazard rather than a decision:
        summarizing is itself a provider call, so `agent.history` is reassigned
        *after* an await, and a turn running across that window appends to the
        list being replaced. Those messages are dropped from context with
        nothing saying so. Unreachable today — nothing invokes this during a run
        — and guarded when the TUI puts it on a keystroke.
        """
        result = await compact(
            self.agent.history,
            client=self.agent.client,
            model=self.agent.model,
            keep_recent_tokens=self.compaction_keep_recent,
            hooks=self.agent.hooks,
        )
        if result is None:
            return {
                "summary": None,
                "cut_index": None,
                "message_count": len(self.agent.history),
            }
        new_history, summary, cut_index = result
        self.agent.history = new_history
        if self.session is not None:
            self.session.append_compaction(summary=summary, cut_index=cut_index)
        return {"summary": summary, "cut_index": cut_index, "message_count": len(new_history)}

    def clear_context(self) -> dict[str, Any]:
        """Forget the conversation; keep recording to the same log.

        Durable. The messages stay in the file — it is the record of what
        happened, and anything reading the JSONL still sees them — but a `clear`
        marker is appended so a resume does not replay them. Clearing because a
        conversation went sideways and finding it restored on the next resume
        was the surprising behaviour. Use `new_session` for a fresh log.

        Carries the same unguarded-mid-run hazard as `compact`.
        """
        cleared = len(self.agent.history)
        self.agent.history = []
        if self.session is not None:
            self.session.append_clear(cut_index=cleared)
        _logger.info("context_cleared messages=%d", cleared)
        return {
            "cleared": cleared,
            "session": str(self.session.path) if self.session is not None else None,
        }

    def set_model(self, model: str) -> dict[str, Any]:
        # An empty registry is permissive, so this only refuses once a user has
        # written a `[models]` table and thereby said what they want available.
        # Reporting success and then misrouting the next turn is the defect this
        # command had; a refusal that names the alternatives is the fix.
        registry = self.agent.client.registry
        if registry and model not in registry:
            _logger.warning("model_unknown model=%s", model)
            raise Refused(
                f"unknown model {model!r}; registered: {', '.join(registry.names())}"
            )
        self.agent.model = model
        if self.session is not None:
            self.session.set_model(model)
        _logger.info("model_set model=%s durable=%s", model, self.session is not None)
        return {"durable": self.session is not None}

    def set_system_prompt(self, prompt: str) -> dict[str, Any]:
        # Sets the base only; the generated half is re-appended. Replacing the
        # whole composed prompt would delete the skills catalogue and every
        # extension's guidance, and a caller could not put them back — the
        # composed string is undelimited and the catalogue carries absolute
        # paths, so it is not reconstructable off-machine.
        #
        # `_stream` snapshots the prompt once outside its turn loop, so this
        # lands on the next turn rather than corrupting the one in flight.
        self.base_prompt = prompt
        self.agent.system_prompt = self.compose_prompt()
        # Appended, not written to the header: the header is never rewritten,
        # so an identity that only lived in this process would silently revert
        # on the next resume, with nothing on the wire saying so.
        if self.session is not None:
            self.session.set_system_prompt(prompt)
        _logger.info(
            "system_prompt_set base_chars=%d composed_chars=%d durable=%s",
            len(prompt),
            len(self.agent.system_prompt or ""),
            self.session is not None,
        )
        return {"durable": self.session is not None}

    def set_session_name(self, name: str) -> dict[str, Any]:
        if self.session is None:
            raise Refused("no session; a name needs a transcript on disk to live in")
        # Newlines would split one record across two JSONL lines and corrupt the
        # file, which is worth stripping rather than rejecting.
        cleaned = " ".join(name.split())
        try:
            self.session.set_name(cleaned)
        except OSError as e:
            raise Refused(str(e)) from e
        _logger.info("session_named name=%s", cleaned)
        return {"name": cleaned}

    def new_session(self, path: Path) -> dict[str, Any]:
        """Close the current log and start a fresh one."""
        # Open the new log *before* touching anything, so a failure here leaves
        # the agent exactly as it was rather than reporting an error from a
        # state it has already half-destroyed.
        try:
            opened = Session.new(
                path,
                model=self.agent.model,
                # The base, not the composed prompt: the header is what a resume
                # reads back as `durable`, and cli.py re-appends the generated
                # half itself. Storing the composed string would duplicate the
                # skills catalogue on every resume.
                system_prompt=self.base_prompt or None,
            )
        except (OSError, FileExistsError) as e:
            raise Refused(str(e)) from e

        self.agent.history = []
        if self.session is not None:
            self.session.close()
        self.session = opened
        # Sub-agents hold the parent session to write their transcripts beside
        # it and to record the forward link in it. Without this they would keep
        # the one just closed, and the next spawn would write to a closed file.
        self.bind_subagents(self.agent.tools)
        _logger.info("new_session path=%s", path)
        return {"session": str(opened.path)}

    def open_session(self, path: Path) -> dict[str, Any]:
        """Attach to an existing transcript, restoring its conversation.

        The gap this closes: `--session PATH` bound a transcript at startup and
        held it until exit, so a running process could not move to another one.
        `new_session` only creates. Reopening is what makes an excursion a round
        trip rather than a one-way door — `use_profile(..., fork)` leaves the
        thread you were on, and something has to get you back to it (#67).

        **History and the base prompt come back; the model does not.** The
        prompt is part of what the conversation *is*, so restoring a reviewer's
        transcript without it would leave its own history misleading. The model
        is infrastructure, and mid-run whatever the agent is on is a live choice
        — possibly a `set_model` a minute ago — which a recorded value should
        not silently override. A disagreement is reported instead, so a caller
        can offer to apply it. Same rule `resume_identity` follows at startup,
        where "asked for this run" is answered by config provenance rather than
        by there being a running agent.
        """
        # Swapping the session under a running turn puts two writers on one
        # history.
        self._refuse_if_busy("open a session")

        if self.session is not None and self.session.path == path:
            # Already attached. Closing and reloading the file we are writing to
            # would be a no-op at best, so say so rather than doing it.
            return {
                "session": str(path),
                "reopened": False,
                "messages": len(self.agent.history),
            }

        # Opened before anything is discarded, so a failure leaves the agent
        # exactly as it was — the sequencing `new_session` already uses.
        try:
            opened = Session.open(
                path, model=self.agent.model, system_prompt=self.base_prompt or None
            )
        except (OSError, ValueError) as e:
            raise Refused(str(e)) from e

        recorded = opened.model
        if recorded != self.agent.model:
            _logger.warning(
                "open_session_model_differs recorded=%s running=%s path=%s",
                recorded,
                self.agent.model,
                path,
            )
        self.agent.history = list(opened.messages)
        if opened.system_prompt:
            # Not re-recorded: it came from this transcript, so appending an
            # `identity` record would only restate what the file already says.
            self.base_prompt = opened.system_prompt
            self.agent.system_prompt = self.compose_prompt()
        if self.session is not None:
            # No forward record: a return is not a branch. `continued` means
            # another transcript *started* here, and writing one would make the
            # chain cyclic — which is what #67 walks to resolve `resume_last`.
            self.session.close()
        self.session = opened
        self.bind_subagents(self.agent.tools)
        _logger.info(
            "open_session path=%s messages=%d name=%s",
            path,
            len(self.agent.history),
            opened.name or "-",
        )
        return {
            "session": str(opened.path),
            "reopened": True,
            "name": opened.name,
            "messages": len(self.agent.history),
            "model": self.agent.model,
            "recorded_model": recorded,
            # So a caller can say "this ran under X, you are on Y" and offer to
            # apply it, rather than the switch doing it silently.
            "model_differs": recorded != self.agent.model,
        }

    def use_profile(
        self, name: str, transcript: Literal["continue", "fork", "resume_last"] = "continue"
    ) -> dict[str, Any]:
        """Retarget the agent to a named profile — every dimension or none.

        One operation rather than documented guidance, because no caller
        discipline can substitute: hand-orchestrating a switch gets a success
        from `set_system_prompt` while the whole previous toolset and every hook
        stay active. A partial switch reporting success is worse than no switch.

        Refused mid-turn, as `reload` is. **There is no revert** — going back is
        naming the other profile, so the operation stays stateless and a caller
        never reasons about how deep it is in a sequence of excursions.

        **History is untouched**, per ADR Decision 4. `fork` changes which file
        the turns are written to, not what the agent still has in context; a
        caller wanting a clean slate composes `clear_context` after.
        """
        self._refuse_if_busy("switch profile")

        profile = self.profiles.get(name)
        if profile is None:
            available = ", ".join(self.profiles.names()) or "none"
            raise Refused(f"unknown profile {name!r}; available: {available}")

        # Everything that can fail happens before anything is applied.
        requested = transcript
        resolved: str = transcript
        target: Path | None = None
        if transcript == "resume_last":
            target = self.resume_target(name)
            if target is None:
                # An ordinary first use of a profile, not an error: a caller
                # cannot know whether one has been used before without asking.
                resolved = self.resume_fallback
                transcript = self.resume_fallback
        if transcript == "fork" and self.session is None:
            # Nothing to fork from. Degrading beats refusing — the caller asked
            # to be retargeted, and which file the turns land in is the part
            # that cannot be honoured.
            _logger.warning("use_profile_fork_without_session profile=%s", name)
            resolved = "continue"
            transcript = "continue"

        opened: Session | None = None
        try:
            if transcript == "fork":
                assert self.session is not None
                path = self.fork_path(self.session.path, name)
                opened = Session.new(
                    path,
                    model=profile.model or self.agent.model,
                    system_prompt=profile.prompt,
                    origin="profile",
                    parent_session=str(self.session.path),
                )
                self.session.append_continued(path=path.name, reason="profile")
            elif transcript == "resume_last" and target is not None:
                opened = Session.open(
                    target,
                    model=profile.model or self.agent.model,
                    system_prompt=profile.prompt,
                )
        except (OSError, ValueError) as e:
            raise Refused(str(e)) from e

        if opened is not None:
            if self.session is not None:
                self.session.close()
            self.session = opened
            if transcript == "resume_last":
                # Only a resume brings a conversation with it; a fork is a new
                # file for the turns this agent is already mid-way through.
                self.agent.history = list(opened.messages)

        model = self.apply_profile(profile)
        if self.session is not None:
            self.session.set_profile(name=profile.name, model=model, system_prompt=profile.prompt)
        _logger.info(
            "use_profile profile=%s transcript=%s model=%s tools=%d session=%s",
            profile.name,
            resolved,
            model,
            len(self.agent.tools),
            self.session.path if self.session is not None else "-",
        )
        return {
            "profile": profile.name,
            # What was asked for and what happened can differ — a `resume_last`
            # with nothing to resume, or a `fork` with no transcript — so a
            # caller can render which it got.
            "requested": requested,
            "transcript": resolved,
            "model": model,
            "tools": sorted(t.name for t in self.agent.tools),
            "session": str(self.session.path) if self.session is not None else None,
            "messages": len(self.agent.history),
            "durable": self.session is not None,
        }

    def apply_profile(self, profile: Profile) -> str:
        """Set every dimension a profile owns. Returns the model applied.

        Called only once nothing fallible remains, which is what makes the
        switch atomic: a profile that does not exist, or a transcript that will
        not open, has already failed before any of this runs.

        An empty `model` means "whatever the agent is already running" — a
        profile is often a prompt and a toolset with no opinion about the
        provider — so the *applied* model is returned for recording, never the
        declared one.
        """
        model = profile.model or self.agent.model
        self.base_prompt = profile.prompt
        self.agent.system_prompt = self.compose_prompt()
        self.agent.model = model
        # Projection, not re-discovery: which sources are loaded never changes,
        # so an embedder's deliberately restricted registry cannot be widened by
        # a switch. Same shape `_child_registry` uses for a sub-agent.
        projected = ToolRegistry([t for t in self.discovered_tools if t.name in profile.tools])
        self.agent.tools = projected
        # Re-bound to the *projected* registry, so a sub-agent's own allowlist
        # is intersected with what this profile can see. Otherwise a reviewer
        # with no `write` could delegate to a child that has one.
        self.bind_subagents(projected, model=model)
        hooks = self.agent.hooks
        if hooks is not None:
            hooks.set_active_sources({n for n, on in profile.hooks.items() if on})
        self.profile = profile.name
        return model

    def fork_path(self, parent: Path, profile: str) -> Path:
        stem = _UNSAFE.sub("-", profile).strip("-")[:64]
        for i in range(100):
            candidate = parent.with_name(f"{parent.stem}.{stem}-{i}{parent.suffix}")
            if not candidate.exists():
                return candidate
        raise OSError(f"no free transcript name beside {parent}")

    def resume_target(self, name: str) -> Path | None:
        """This session's most recent transcript running under `name`.

        The chain, never a directory scan — which is what keeps `resume_last`
        independent of session discovery. Sub-agent runs are excluded by their
        `origin`: they are delegations, not profile excursions, and resuming one
        as though it were a thread would be wrong rather than untidy.

        Recency is the timestamp of the *last profile record*, not the file's
        creation: a transcript opened first but switched back into most recently
        is the one "where I left off" means. That also makes a thread which has
        since switched away no candidate at all, which is ADR Decision 5 read
        backwards.
        """
        if self.session is None:
            return None
        best: tuple[int, Path] | None = None
        for path in session_chain(self.session.path):
            try:
                header, entries = read_transcript(path)
            except (OSError, ValueError):
                continue
            if header.origin == "subagent":
                continue
            last = None
            for entry in reversed(entries):
                if isinstance(entry, ProfileRecord):
                    last = entry
                    break
            if last is None or last.name != name:
                continue
            if best is None or last.timestamp > best[0]:
                best = (last.timestamp, path)
        return best[1] if best is not None else None

    async def reload(self, targets: Sequence[str] | None = None) -> dict[str, Any]:
        """Re-scan skills and extensions from disk.

        Both targets are a discard and re-run rather than anything incremental.
        That works because the two loaders already have the right shapes:
        `load_extensions` returns a *fresh* registry, so tools are a swap with
        no residue, and `Hooks.clear()` runs every cleanup before wiping, which
        is exactly what unloading an extension should do. Hooks would otherwise
        double-register, because they are mutated into a shared `Hooks` rather
        than returned.

        Per-file failures are not reported. Both loaders already skip a bad
        file, log it, and carry on; surfacing them here would mean changing both
        return types to serve a caller that does not exist yet.
        """
        # Refused rather than queued: swapping the tool registry under a running
        # turn breaks the tool-call/result pairing the loop depends on. It also
        # disposes of the one genuinely hard case — a sub-agent holding a child
        # registry bound to the old runtime can only exist inside a turn.
        self._refuse_if_busy("reload")

        configured = {
            "extensions": self.extension_sources is not None,
            "skills": self.skill_sources is not None,
        }
        if targets is not None:
            chosen = tuple(targets)
            missing = [t for t in chosen if not configured[t]]
            if missing:
                raise Refused(f"no sources configured for: {', '.join(missing)}")
        else:
            # The bare form reloads what is wired up. Failing it because an
            # embedder wired only skills would make the convenient spelling the
            # one that does not work.
            chosen = tuple(t for t in RELOAD_TARGETS if configured[t])

        if "extensions" in chosen:
            await self._reload_extensions()
        if "skills" in chosen:
            self._reload_skills()
        # After both, so an extensions reload that added or removed `read` is
        # reflected even when only extensions were asked for.
        self.agent.system_prompt = self.compose_prompt()
        _logger.info(
            "reloaded targets=%s tools=%d skills=%d profiles=%d",
            ",".join(chosen),
            len(self.agent.tools),
            len(self.skills),
            len(self.profiles),
        )
        return {
            "targets": list(chosen),
            "tools": len(self.agent.tools),
            "skills": len(self.skills),
            "profiles": len(self.profiles),
        }

    async def _reload_extensions(self) -> None:
        assert self.extension_sources is not None
        hooks = self.agent.hooks
        if hooks is not None:
            # A whole-registry wipe, which is right because `load_extensions` is
            # the only thing that registers into this `Hooks` — and it runs every
            # `add_cleanup` handler first, which is what unloading should do.
            # `_Registration.source` is already stamped if this ever needs to
            # become a scoped removal.
            await hooks.clear()
        # A fresh set, for the same reason the registry is: a re-import produces
        # new instances, and a profile deleted from disk has to disappear.
        profiles = ProfileSet()
        registry, self.extension_prompt = load_extensions(
            self.extension_sources, hooks=hooks, profiles=profiles
        )
        # After the load, so it validates against the tools and hooks that now
        # exist rather than the ones that just went away.
        emit_diagnostics(
            validate_profiles(
                profiles,
                tools=registry,
                hook_names=hooks.source_names() if hooks is not None else set(),
                models=self.agent.client.registry,
            )
        )
        self.profiles = profiles
        # Re-import produces new `SubagentTool` instances, so this binds the
        # fresh ones. The model comes off the agent so a `set_model` since
        # startup is respected rather than reverted.
        self.bind_subagents(registry)
        self.discovered_tools = registry
        self.agent.tools = registry
        # A reload under a profile re-projects rather than silently widening the
        # agent back to everything on disk. `set_active_sources` likewise: the
        # wipe above cleared it, and the profile's decisions have to be reapplied
        # against the freshly registered sources.
        active = self.profiles.get(self.profile) if self.profile else None
        if active is not None:
            self.apply_profile(active)
        elif hooks is not None:
            hooks.set_active_sources(None)

    def _reload_skills(self) -> None:
        assert self.skill_sources is not None
        self.skills = load_skills(self.skill_sources)

    def abort(self) -> list[Any]:
        """Stop the run in flight and discard anything queued.

        Returns what was dropped. Clearing before cancelling is the whole point:
        leaving the queues alone means the run abort just stopped is immediately
        replaced by one built from whatever was pending, which is not what
        "stop" means.
        """
        if self.runner is None or not self.runner.busy():
            raise Refused("no prompt in flight")
        dropped = self.agent.steering.clear() if self.agent.steering is not None else []
        self.runner.cancel()
        return dropped
