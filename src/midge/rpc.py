"""JSON-over-stdio RPC server for embedding the agent in external tools.

Wire format: newline-delimited JSON, LF-only. Each emitted record is
`json.dumps(obj, ensure_ascii=False) + "\n"`.

Two outbound shapes:
    - Responses correlated to inbound commands by an optional `id`:
      {"id": "...", "type": "response", "command": "...", "success": bool, ...}
    - Async events streamed during a prompt run, uncorrelated:
      {"type": "assistant_text_delta" | "tool_call_start" | ... }

Stdout is the protocol; stderr is for diagnostics. Call `claim_stdout()` before
anything else can write, so a stray `print()` anywhere in the process lands on
stderr instead of corrupting the stream.

Inbound: prompt, steer, follow_up, abort, and a set of state and control
commands `get_commands` enumerates for clients that would rather discover the
surface than hardcode it — including `reload`, which re-scans skills and
extensions from disk so a long-lived process picks up edits.
Commands are dispatched serially; a `prompt` returns its response immediately
after preflight and runs the agent in a background task while the dispatch
loop continues reading stdin (so `abort` can interrupt).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import signal
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, Field

from midge.agent import (
    Agent,
    AgentEnd,
    Steered,
    SteeringQueue,
    ToolExecutionEnd,
    ToolExecutionStart,
)
from midge.client import (
    Done,
    Error,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from midge.compaction import compact
from midge.extensions import load_extensions
from midge.messages import AssistantMessage, TextContent, ToolCall, UserMessage
from midge.persistence import Session, read_transcript
from midge.session import export_html
from midge.skills import Skill, load_skills, skill_message, skills_prompt
from midge.subagents import bind_subagents

_logger = logging.getLogger(__name__)

READ_LIMIT = 16 * 1024 * 1024
# Roughly ten ordinary answers' worth of frames — deep enough that a client
# pausing to render or collect garbage never stalls anything, shallow enough
# that a client which has died applies backpressure instead of exhausting memory.
OUTBOX_FRAMES = 4096
FLUSH_TIMEOUT = 5.0

WriteFn = Callable[[bytes], Awaitable[None]]
ReadLineFn = Callable[[], Awaitable[bytes]]

_claimed_stdout: BinaryIO | None = None
# The wrapper we displace is kept alive deliberately: dropping the last
# reference to a TextIOWrapper closes the buffer underneath it, which here is
# the fd carrying the protocol. In a real process `sys.__stdout__` happens to
# hold one too, but relying on that is a footgun.
_displaced_stdout: Any = None


def claim_stdout() -> BinaryIO:
    """Take fd 1 for the protocol and point `sys.stdout` at stderr.

    Stdout is the wire here, so a single stray `print()` — from a tool, a hook,
    a user extension, or a dependency — corrupts it. The corruption is quiet
    rather than loud: the protocol writes through the buffered binary layer
    while `print` goes through the text wrapper above it, so under a pipe the
    stray text is block-buffered and surfaces at some arbitrary later point.
    Individual frames stay intact; their ordering does not.

    Returns the real stdout for the protocol writer to hold. Idempotent.
    """
    global _claimed_stdout, _displaced_stdout
    if _claimed_stdout is not None:
        return _claimed_stdout

    _displaced_stdout = sys.stdout
    real = sys.stdout.buffer
    sys.stdout = io.TextIOWrapper(
        sys.stderr.buffer, encoding=sys.stderr.encoding, errors="replace", line_buffering=True
    )
    _claimed_stdout = real
    return real


def event_to_wire(ev: Any) -> dict[str, Any] | None:
    if isinstance(ev, StreamStart | TextStart | TextEnd):
        return None
    if isinstance(ev, TextDelta):
        return {"type": "assistant_text_delta", "delta": ev.delta}
    if isinstance(ev, ToolCallStart):
        tc = ev.partial.content[ev.content_index]
        assert isinstance(tc, ToolCall)
        return {"type": "tool_call_start", "id": tc.id, "name": tc.name}
    if isinstance(ev, ToolCallDelta):
        tc = ev.partial.content[ev.content_index]
        assert isinstance(tc, ToolCall)
        return {"type": "tool_call_delta", "id": tc.id, "delta": ev.delta}
    if isinstance(ev, ToolCallEnd):
        return {
            "type": "tool_call_end",
            "id": ev.tool_call.id,
            "name": ev.tool_call.name,
            "arguments": ev.tool_call.arguments,
        }
    if isinstance(ev, Done):
        return {
            "type": "assistant_message_end",
            "stop_reason": ev.message.stop_reason,
            "model": ev.message.model,
        }
    if isinstance(ev, Error):
        return {
            "type": "error",
            "message": ev.message.error_message or "",
            "stop_reason": ev.message.stop_reason,
        }
    if isinstance(ev, ToolExecutionStart):
        return {
            "type": "tool_execution_start",
            "id": ev.tool_call.id,
            "name": ev.tool_call.name,
        }
    if isinstance(ev, ToolExecutionEnd):
        text = ""
        if ev.result.content and isinstance(ev.result.content[0], TextContent):
            text = ev.result.content[0].text
        return {
            "type": "tool_result",
            "tool_call_id": ev.result.tool_call_id,
            "content": text,
            "is_error": ev.result.is_error,
        }
    if isinstance(ev, Steered):
        return {
            "type": "user_message",
            "content": str(ev.message.content),
            "source": "steer",
            "queue_id": ev.queue_id,
        }
    if isinstance(ev, AgentEnd):
        return {"type": "agent_end"}
    return None


class _CommandParams(BaseModel):
    """Base for built-in command schemas.

    `extra="forbid"` so the generated schema carries `additionalProperties:
    false`, matching what `Tool.schema()` produces — a consumer that can render
    a tool call can render a command with no second convention to learn.
    """

    model_config = ConfigDict(extra="forbid")


class _SetModelParams(_CommandParams):
    model: str = Field(description="Provider model id, e.g. gpt-4o")


class _SetSystemPromptParams(_CommandParams):
    prompt: str = Field(
        description="Replaces the durable base prompt; the generated half is re-appended"
    )


class _NewSessionParams(_CommandParams):
    path: str = Field(description="Path for the new session log")


class _SessionNameParams(_CommandParams):
    name: str = Field(description="Display name for the current session")


class _ExportHtmlParams(_CommandParams):
    output_path: str | None = Field(
        default=None, description="Where to write; defaults to the session path with .html"
    )


RELOAD_TARGETS = ("skills", "extensions")


class _ReloadParams(_CommandParams):
    targets: list[Literal["skills", "extensions"]] | None = Field(
        default=None, description="Which sources to re-scan; omit for all of them"
    )


@dataclass(frozen=True, slots=True)
class BuiltinCommand:
    name: str
    description: str
    params: type[BaseModel] | None = None


# The server does get an opinion about what is a user-facing action rather than
# protocol plumbing. Out: `prompt`, `steer` and `follow_up`, which *are* the
# interaction — a UI picks between them by policy when the user hits enter — and
# the `get_*` family, which a client reads to render itself rather than offering
# to a user. `abort` is in: leaving it out assumed every consumer has an escape
# key, and a chat bot does not.
BUILTIN_COMMANDS: tuple[BuiltinCommand, ...] = (
    BuiltinCommand("abort", "Stop the run in flight and discard anything queued"),
    BuiltinCommand("compact", "Summarize older turns to reclaim context"),
    BuiltinCommand("clear_context", "Forget the conversation; keep recording to the same log"),
    BuiltinCommand("new_session", "Close the current log and start a fresh one", _NewSessionParams),
    BuiltinCommand("export_html", "Write an HTML transcript of the session", _ExportHtmlParams),
    BuiltinCommand("set_model", "Switch the model used for subsequent turns", _SetModelParams),
    BuiltinCommand(
        "set_system_prompt", "Replace the agent's base system prompt", _SetSystemPromptParams
    ),
    BuiltinCommand(
        "reload", "Re-scan skills and extensions from disk", _ReloadParams
    ),
    BuiltinCommand(
        "set_session_name", "Give the current session a display name", _SessionNameParams
    ),
)

_NO_PARAMS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

SKILL_COMMAND_PREFIX = "/skill:"


class RpcServer:
    def __init__(
        self,
        agent: Agent,
        *,
        session: Session | None = None,
        compaction_keep_recent: int = 20_000,
        base_prompt: str | None = None,
        extension_prompt: str = "",
        skills: Sequence[Skill] | None = None,
        extension_sources: Sequence[Path] | None = None,
        skill_sources: Sequence[Path] | None = None,
    ) -> None:
        self.agent = agent
        # `client` and `model` come off the agent; `session` is what the export
        # and persistence commands need and cannot reach otherwise.
        self.session = session
        self.compaction_keep_recent = compaction_keep_recent
        # The agent's prompt is composed: a durable base the operator owns, then
        # what midge generates — extension contributions and the skills
        # catalogue. Keeping the halves apart is what lets `set_system_prompt`
        # change the base without silently deleting the catalogue, and what lets
        # `reload` replace one generated half without disturbing the other.
        self._base_prompt = base_prompt if base_prompt is not None else (agent.system_prompt or "")
        self._extension_prompt = extension_prompt
        self._skills: Sequence[Skill] = skills or ()
        # The exact source lists the entrypoint loaded from, so `reload` re-runs
        # the same call rather than reconstructing one. Reconstructing would mean
        # knowing which sources are built-in, and an embedder that handed the
        # agent a deliberately restricted registry would find reload silently
        # widening it. `None` means the entrypoint did not wire that target up,
        # which is not the same as an empty list.
        self._extension_sources = list(extension_sources) if extension_sources is not None else None
        self._skill_sources = list(skill_sources) if skill_sources is not None else None
        # The queue is shared with the agent: it drains steering at its own
        # boundaries, the server drains follow-ups once a run is done.
        self.steering = agent.steering if agent.steering is not None else SteeringQueue()
        agent.steering = self.steering
        self._current_run: asyncio.Task[None] | None = None
        # Frames are queued, not written inline. The dispatch loop must keep
        # reading while a slow client is being written to, or `abort` — the one
        # command that can stop a runaway — cannot be delivered.
        self._outbox: asyncio.Queue[bytes] = asyncio.Queue(maxsize=OUTBOX_FRAMES)
        self._write: WriteFn | None = None

    async def serve(self, *, read_line: ReadLineFn, write: WriteFn) -> None:
        self._write = write
        pump = asyncio.ensure_future(self._pump())
        try:
            while True:
                line = await read_line()
                if not line:
                    break
                # `strip`, not `rstrip("\r\n")`: a whitespace-only line is a
                # blank line, not a malformed command, and answering it with a
                # parse error is noise.
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    cmd = json.loads(stripped)
                except json.JSONDecodeError as e:
                    await self._respond(None, "parse", success=False, error=str(e))
                    continue
                if not isinstance(cmd, dict):
                    await self._respond(
                        None, "parse", success=False, error="command must be a JSON object"
                    )
                    continue
                await self._dispatch(cmd)
        finally:
            run = self._current_run
            if run is not None and not run.done():
                run.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run
            # Give whatever is queued a chance to land before the pipe closes.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._outbox.join(), timeout=FLUSH_TIMEOUT)
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    async def _dispatch(self, cmd: dict[str, Any]) -> None:
        cmd_id_raw = cmd.get("id")
        cmd_id = cmd_id_raw if isinstance(cmd_id_raw, str) else None
        cmd_type = cmd.get("type")
        _logger.info("rpc_command type=%s id=%s", cmd_type, cmd_id or "-")
        match cmd_type:
            case "prompt":
                await self._handle_prompt(cmd_id, cmd)
            case "abort":
                await self._handle_abort(cmd_id)
            case "steer":
                await self._handle_queue(cmd_id, cmd, "steer")
            case "follow_up":
                await self._handle_queue(cmd_id, cmd, "follow_up")
            case "get_commands":
                await self._handle_get_commands(cmd_id)
            case "get_messages":
                await self._handle_get_messages(cmd_id)
            case "get_state":
                await self._handle_get_state(cmd_id)
            case "get_last_assistant_text":
                await self._handle_get_last_assistant_text(cmd_id)
            case "get_system_prompt":
                await self._handle_get_system_prompt(cmd_id)
            case "set_system_prompt":
                await self._handle_set_system_prompt(cmd_id, cmd)
            case "set_model":
                await self._handle_set_model(cmd_id, cmd)
            case "export_html":
                await self._handle_export_html(cmd_id, cmd)
            case "compact":
                await self._handle_compact(cmd_id)
            case "clear_context":
                await self._handle_clear_context(cmd_id)
            case "new_session":
                await self._handle_new_session(cmd_id, cmd)
            case "reload":
                await self._handle_reload(cmd_id, cmd)
            case "set_session_name":
                await self._handle_set_session_name(cmd_id, cmd)
            case _:
                _logger.warning("rpc_command_unknown type=%r", cmd_type)
                await self._respond(
                    cmd_id,
                    cmd_type if isinstance(cmd_type, str) else "unknown",
                    success=False,
                    error=f"unknown command: {cmd_type!r}",
                )

    async def _handle_prompt(self, cmd_id: str | None, cmd: dict[str, Any]) -> None:
        message = cmd.get("message")
        if not isinstance(message, str):
            await self._respond(
                cmd_id,
                "prompt",
                success=False,
                error="`message` is required and must be a string",
            )
            return
        try:
            resolved = self._expand_command(message)
        except (KeyError, OSError) as e:
            await self._respond(cmd_id, "prompt", success=False, error=str(e))
            return
        # A prompt arriving mid-run is queued rather than refused, but the
        # response still says which happened: the client should not have to
        # infer it from whether events follow.
        if self._current_run is not None and not self._current_run.done():
            self.steering.follow_up(resolved)
            await self._emit_queue_update()
            await self._respond(
                cmd_id, "prompt", success=True, data={"accepted": "queued"}
            )
            return
        await self._respond(cmd_id, "prompt", success=True, data={"accepted": "started"})
        self._current_run = asyncio.create_task(self._run_until_settled(resolved))

    async def _emit_queue_update(self) -> None:
        await self._emit({"type": "queue_update", **self.steering.snapshot()})

    async def _handle_queue(
        self, cmd_id: str | None, cmd: dict[str, Any], kind: str
    ) -> None:
        message = cmd.get("message")
        if not isinstance(message, str) or not message:
            await self._respond(
                cmd_id, kind, success=False,
                error="`message` is required and must be a non-empty string",
            )
            return
        try:
            resolved = self._expand_command(message)
        except (KeyError, OSError) as e:
            await self._respond(cmd_id, kind, success=False, error=str(e))
            return
        if kind == "steer":
            queue_id = self.steering.steer(resolved)
        else:
            queue_id = self.steering.follow_up(resolved)
        await self._emit_queue_update()
        _logger.info("rpc_queued kind=%s id=%s", kind, queue_id)
        await self._respond(cmd_id, kind, success=True, data={"queue_id": queue_id})

    async def _run_until_settled(self, message: str | UserMessage) -> None:
        """Run, then keep running while follow-ups are waiting.

        `agent_end` means one run finished, and a follow-up starts another, so
        it can fire several times for one client prompt. `agent_settled` is the
        terminal a client should wait on — emitted from a `finally` so it also
        fires when the run errors or is cancelled, which are exactly the paths
        that could previously emit no terminal at all.
        """
        try:
            nxt: str | UserMessage | None = message
            while nxt is not None:
                await self._run_prompt(nxt)
                queued = self.steering.take_follow_up()
                if queued is None:
                    nxt = None
                else:
                    await self._emit_queue_update()
                    nxt = queued.message
        finally:
            await self._emit({"type": "agent_settled"})

    async def _run_prompt(self, message: str | UserMessage) -> None:
        await self._emit({"type": "user_message", "content": str(
            message.content if isinstance(message, UserMessage) else message
        )})
        saw_error_event = False
        try:
            async for ev in self.agent.stream(message):
                if isinstance(ev, Error):
                    saw_error_event = True
                wire = event_to_wire(ev)
                if wire is not None:
                    await self._emit(wire)
        except asyncio.CancelledError:
            if not saw_error_event:
                await self._emit(
                    {"type": "error", "message": "cancelled", "stop_reason": "aborted"}
                )
            raise
        except Exception as e:
            _logger.exception("rpc_prompt_failed")
            if not saw_error_event:
                await self._emit(
                    {"type": "error", "message": str(e), "stop_reason": "error"}
                )

    def _builtin_schema(self, command: BuiltinCommand) -> dict[str, Any]:
        """A built-in's declared schema, narrowed by what this process knows.

        `set_model` is the only one so far: with a non-empty registry the set of
        models is a fact the server holds, so it becomes an `enum` and a client
        renders a picker with nothing hardcoded — the same projection `reload`
        does for its targets. Absent a registry the field stays a free string,
        which is what an empty registry means everywhere else too.
        """
        schema = command.params.model_json_schema() if command.params else dict(_NO_PARAMS)
        registry = self.agent.client.registry
        if command.name == "set_model" and registry:
            schema["properties"]["model"]["enum"] = registry.names()
        return schema

    async def _handle_get_commands(self, cmd_id: str | None) -> None:
        """Everything a user can invoke, and how to invoke it.

        Read-only; executes nothing. A projection of what already exists —
        built-ins from the dispatch table, skills from disk — rather than a new
        concept, which is what makes it safe to ship before any consumer does.

        `invoke` says how to transmit: `command` means send `{"type": name, …}`,
        `prompt` means put the text in a prompt/steer/follow_up message.
        `parameters` is JSON Schema in the same shape `Tool.schema()` produces,
        so an empty `properties` is the "select and fire" signal. Note it means
        slightly different things per `invoke`: for a command the properties are
        keys in the request object; for a prompt the single property is free
        text appended after the name. A prompt-invoked command takes at most one
        argument, which is what keeps that unambiguous.

        Deliberately absent: any notion of whether an entry is dangerous enough
        to confirm. That is a consumer policy — a misclick in a terminal and one
        in a shared channel are not the same risk — and the server cannot know
        which it is talking to.
        """
        commands: list[dict[str, Any]] = [
            {
                "name": c.name,
                "source": "builtin",
                "invoke": "command",
                "description": c.description,
                "parameters": self._builtin_schema(c),
            }
            for c in BUILTIN_COMMANDS
        ]
        # Listed regardless of `model_invocable`: hiding a skill from the model's
        # catalogue is exactly the case where an explicit command is the only
        # way to reach it.
        commands.extend(
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
            for s in self._skills
        )
        await self._respond(cmd_id, "get_commands", success=True, data={"commands": commands})

    def _expand_command(self, text: str) -> str | UserMessage:
        """Resolve `/skill:name [args]` to a real message, or pass text through.

        Done at enqueue rather than delivery, which is the rule `SteeringQueue`
        already states: whatever is queued must already be a plain message, so a
        bad name fails in the response to whoever queued it instead of surfacing
        mid-run with nothing to attribute it to. It also freezes what gets sent
        at the moment the user asked for it, so editing the SKILL.md in between
        changes nothing.
        """
        if not text.startswith(SKILL_COMMAND_PREFIX):
            return text
        rest = text[len(SKILL_COMMAND_PREFIX) :]
        name, _, args = rest.partition(" ")
        # Raises KeyError for an unknown name rather than passing the text
        # through as pi does: midge has no text-expansion path where a literal
        # `/skill:typo` would make sense, so it is a caller bug.
        return skill_message(self._skills, name, args.strip() or None)

    async def _handle_get_state(self, cmd_id: str | None) -> None:
        # Deliberately excludes the system prompt: composed from the base, every
        # extension contribution and the skills catalogue, it runs to kilobytes,
        # which is not what a state summary is for. `get_system_prompt` has it.
        await self._respond(
            cmd_id,
            "get_state",
            success=True,
            data={
                "model": self.agent.model,
                "streaming": self._current_run is not None and not self._current_run.done(),
                "session": str(self.session.path) if self.session is not None else None,
                "session_name": self.session.name if self.session is not None else None,
                "messages": len(self.agent.history),
            },
        )

    async def _handle_get_last_assistant_text(self, cmd_id: str | None) -> None:
        text: str | None = None
        for m in reversed(self.agent.history):
            if isinstance(m, AssistantMessage):
                joined = "".join(c.text for c in m.content if isinstance(c, TextContent))
                text = joined or None
                break
        await self._respond(
            cmd_id, "get_last_assistant_text", success=True, data={"text": text}
        )

    def _generated_prompt(self) -> str:
        """Extension contributions plus the skills catalogue, gated as at startup.

        Derived rather than stored so the `read` gate cannot fall out of date:
        the catalogue tells the model to open a `SKILL.md`, so without a tool
        that can open one it is an instruction to do the impossible. An
        extensions reload can add or remove `read`, which makes this the one
        point where reloading extensions changes the skills half of the prompt.
        """
        catalogue = skills_prompt(self._skills) if "read" in self.agent.tools else ""
        return "\n\n".join(p for p in (self._extension_prompt, catalogue) if p)

    def _compose_prompt(self) -> str:
        return "\n\n".join(p for p in (self._base_prompt, self._generated_prompt()) if p)

    async def _handle_get_system_prompt(self, cmd_id: str | None) -> None:
        await self._respond(
            cmd_id,
            "get_system_prompt",
            success=True,
            data={
                "prompt": self.agent.system_prompt,
                "base": self._base_prompt,
                "appended": self._generated_prompt(),
            },
        )

    async def _handle_set_system_prompt(self, cmd_id: str | None, cmd: dict[str, Any]) -> None:
        prompt = cmd.get("prompt")
        if not isinstance(prompt, str):
            await self._respond(
                cmd_id,
                "set_system_prompt",
                success=False,
                error="`prompt` is required and must be a string",
            )
            return
        # Sets the base only; the generated half is re-appended. Replacing the
        # whole composed prompt would delete the skills catalogue and every
        # extension's guidance, and a client could not put them back — the
        # composed string is undelimited and the catalogue carries absolute
        # paths, so it is not reconstructable off-machine.
        #
        # `_stream` snapshots the prompt once outside its turn loop, so this
        # lands on the next turn rather than corrupting the one in flight.
        self._base_prompt = prompt
        self.agent.system_prompt = self._compose_prompt()
        _logger.info(
            "rpc_system_prompt_set base_chars=%d composed_chars=%d",
            len(prompt),
            len(self.agent.system_prompt or ""),
        )
        await self._respond(cmd_id, "set_system_prompt", success=True)

    async def _handle_set_model(self, cmd_id: str | None, cmd: dict[str, Any]) -> None:
        model = cmd.get("model")
        if not isinstance(model, str) or not model:
            await self._respond(
                cmd_id,
                "set_model",
                success=False,
                error="`model` is required and must be a non-empty string",
            )
            return
        # An empty registry is permissive, so this only refuses once a user has
        # written a `[models]` table and thereby said what they want available.
        # Reporting success and then misrouting the next turn is the defect this
        # command had; a refusal that names the alternatives is the fix.
        registry = self.agent.client.registry
        if registry and model not in registry:
            _logger.warning("rpc_model_unknown model=%s", model)
            await self._respond(
                cmd_id,
                "set_model",
                success=False,
                error=f"unknown model {model!r}; registered: {', '.join(registry.names())}",
            )
            return
        self.agent.model = model
        _logger.info("rpc_model_set model=%s", model)
        await self._respond(cmd_id, "set_model", success=True)

    async def _handle_export_html(self, cmd_id: str | None, cmd: dict[str, Any]) -> None:
        raw_path = cmd.get("output_path")
        if raw_path is not None and not isinstance(raw_path, str):
            await self._respond(
                cmd_id, "export_html", success=False,
                error="`output_path` must be a string",
            )
            return
        if self.session is None:
            await self._respond(
                cmd_id, "export_html", success=False,
                error="no session; export needs a transcript on disk",
            )
            return

        out = Path(raw_path) if raw_path else self.session.path.with_suffix(".html")
        # From the file, not `agent.history`: history is post-compaction and
        # loses every message a compaction folded away.
        _, entries = read_transcript(self.session.path)
        try:
            out.write_text(
                export_html(
                    entries,
                    model=self.agent.model,
                    **({"title": self.session.name} if self.session.name else {}),
                ),
                encoding="utf-8",
            )
        except OSError as e:
            await self._respond(cmd_id, "export_html", success=False, error=str(e))
            return
        _logger.info("rpc_exported path=%s entries=%d", out, len(entries))
        await self._respond(cmd_id, "export_html", success=True, data={"path": str(out)})

    async def _handle_compact(self, cmd_id: str | None) -> None:
        result = await compact(
            self.agent.history,
            client=self.agent.client,
            model=self.agent.model,
            keep_recent_tokens=self.compaction_keep_recent,
            hooks=self.agent.hooks,
        )
        if result is None:
            await self._respond(
                cmd_id, "compact", success=True,
                data={"summary": None, "cut_index": None, "message_count": len(self.agent.history)},
            )
            return
        new_history, summary, cut_index = result
        self.agent.history = new_history
        if self.session is not None:
            self.session.append_compaction(summary=summary, cut_index=cut_index)
        await self._respond(
            cmd_id, "compact", success=True,
            data={
                "summary": summary,
                "cut_index": cut_index,
                "message_count": len(new_history),
            },
        )

    async def _handle_clear_context(self, cmd_id: str | None) -> None:
        """Forget the conversation; keep recording to the same log.

        Durable. The messages stay in the file — it is the record of what
        happened, and `export_html` still renders them — but a `clear` marker is
        appended so a resume does not replay them. Clearing because a
        conversation went sideways and finding it restored on the next resume
        was the surprising behaviour. Use `new_session` for a fresh log.
        """
        cleared = len(self.agent.history)
        self.agent.history = []
        if self.session is not None:
            self.session.append_clear(cut_index=cleared)
        _logger.info("rpc_context_cleared messages=%d", cleared)
        await self._respond(
            cmd_id,
            "clear_context",
            success=True,
            data={
                "cleared": cleared,
                "session": str(self.session.path) if self.session is not None else None,
            },
        )

    async def _handle_new_session(self, cmd_id: str | None, cmd: dict[str, Any]) -> None:
        """Close the current log and start a fresh one. `path` is required —
        without it there would be no new session, only a silent end to
        persistence, which is what `clear_context` is for."""
        raw_path = cmd.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            await self._respond(
                cmd_id,
                "new_session",
                success=False,
                error="`path` is required and must be a non-empty string",
            )
            return

        # Open the new log *before* touching anything, so a failure here leaves
        # the server exactly as it was rather than reporting an error from a
        # state it has already half-destroyed.
        try:
            opened = Session.new(
                Path(raw_path),
                model=self.agent.model,
                # The base, not the composed prompt: the header is what a resume
                # reads back as `durable`, and cli.py re-appends the generated
                # half itself. Storing the composed string would duplicate the
                # skills catalogue on every resume.
                system_prompt=self._base_prompt or None,
            )
        except (OSError, FileExistsError) as e:
            await self._respond(cmd_id, "new_session", success=False, error=str(e))
            return

        self.agent.history = []
        if self.session is not None:
            self.session.close()
        self.session = opened
        _logger.info("rpc_new_session path=%s", raw_path)
        await self._respond(
            cmd_id, "new_session", success=True, data={"session": str(opened.path)}
        )

    async def _handle_reload(self, cmd_id: str | None, cmd: dict[str, Any]) -> None:
        """Re-scan skills and extensions from disk.

        Both targets are a discard and re-run rather than anything incremental.
        That works because the two loaders already have the right shapes:
        `load_extensions` returns a *fresh* registry, so tools are a swap with
        no residue, and `Hooks.clear()` runs every cleanup before wiping, which
        is exactly what unloading an extension should do. Hooks would otherwise
        double-register, because they are mutated into a shared `Hooks` rather
        than returned.

        Per-file failures are not on the wire. Both loaders already skip a bad
        file, log it, and carry on; reporting them here would mean changing
        both return types to serve a client that does not exist yet.
        """
        try:
            params = _ReloadParams.model_validate(
                {k: v for k, v in cmd.items() if k not in ("id", "type")}
            )
        except ValueError as e:
            await self._respond(cmd_id, "reload", success=False, error=str(e))
            return

        # Refused rather than queued: swapping the tool registry under a running
        # turn breaks the tool-call/result pairing the loop depends on. It also
        # disposes of the one genuinely hard case — a sub-agent holding a child
        # registry bound to the old runtime can only exist inside a turn.
        if self._current_run is not None and not self._current_run.done():
            await self._respond(
                cmd_id, "reload", success=False, error="cannot reload while a run is in flight"
            )
            return

        configured = {
            "extensions": self._extension_sources is not None,
            "skills": self._skill_sources is not None,
        }
        if params.targets is not None:
            targets = tuple(params.targets)
            missing = [t for t in targets if not configured[t]]
            if missing:
                await self._respond(
                    cmd_id,
                    "reload",
                    success=False,
                    error=f"no sources configured for: {', '.join(missing)}",
                )
                return
        else:
            # The bare form reloads what is wired up. Failing it because an
            # embedder wired only skills would make the convenient spelling the
            # one that does not work.
            targets = tuple(t for t in RELOAD_TARGETS if configured[t])

        if "extensions" in targets:
            await self._reload_extensions()
        if "skills" in targets:
            self._reload_skills()
        # After both, so an extensions reload that added or removed `read` is
        # reflected even when only extensions were asked for.
        self.agent.system_prompt = self._compose_prompt()
        _logger.info(
            "rpc_reloaded targets=%s tools=%d skills=%d",
            ",".join(targets),
            len(self.agent.tools),
            len(self._skills),
        )
        await self._respond(
            cmd_id,
            "reload",
            success=True,
            data={
                "targets": list(targets),
                "tools": len(self.agent.tools),
                "skills": len(self._skills),
            },
        )

    async def _reload_extensions(self) -> None:
        assert self._extension_sources is not None
        hooks = self.agent.hooks
        if hooks is not None:
            # A whole-registry wipe, which is right because `load_extensions` is
            # the only thing that registers into this `Hooks` — and it runs every
            # `add_cleanup` handler first, which is what unloading should do.
            # `_Registration.source` is already stamped if this ever needs to
            # become a scoped removal.
            await hooks.clear()
        registry, self._extension_prompt = load_extensions(
            self._extension_sources, hooks=hooks
        )
        # Re-import produces new `SubagentTool` instances, so this binds the
        # fresh ones. The model comes off the agent so a `set_model` since
        # startup is respected rather than reverted.
        bind_subagents(
            registry,
            client=self.agent.client,
            model=self.agent.model,
            hooks=hooks,
            session_path=self.session.path if self.session is not None else None,
        )
        self.agent.tools = registry

    def _reload_skills(self) -> None:
        assert self._skill_sources is not None
        self._skills = load_skills(self._skill_sources)

    async def _handle_set_session_name(self, cmd_id: str | None, cmd: dict[str, Any]) -> None:
        name = cmd.get("name")
        if not isinstance(name, str) or not name.strip():
            await self._respond(
                cmd_id,
                "set_session_name",
                success=False,
                error="`name` is required and must be a non-empty string",
            )
            return
        if self.session is None:
            await self._respond(
                cmd_id,
                "set_session_name",
                success=False,
                error="no session; a name needs a transcript on disk to live in",
            )
            return
        # Newlines would split one record across two JSONL lines and corrupt the
        # file, which is worth stripping rather than rejecting.
        cleaned = " ".join(name.split())
        try:
            self.session.set_name(cleaned)
        except OSError as e:
            await self._respond(cmd_id, "set_session_name", success=False, error=str(e))
            return
        _logger.info("rpc_session_named name=%s", cleaned)
        await self._respond(
            cmd_id, "set_session_name", success=True, data={"name": cleaned}
        )

    async def _handle_abort(self, cmd_id: str | None) -> None:
        if self._current_run is not None and not self._current_run.done():
            # Clear before cancelling: leaving the queues alone means the run
            # that abort just stopped is immediately replaced by one built from
            # whatever was pending, which is not what "stop" means.
            dropped = self.steering.clear()
            self._current_run.cancel()
            if dropped:
                await self._emit_queue_update()
            await self._respond(
                cmd_id, "abort", success=True,
                data={"dropped": [{"id": d.id, "content": str(d.message.content)} for d in dropped]},
            )
        else:
            await self._respond(
                cmd_id, "abort", success=False, error="no prompt in flight"
            )

    async def _handle_get_messages(self, cmd_id: str | None) -> None:
        data = [m.model_dump(mode="json") for m in self.agent.history]
        await self._respond(cmd_id, "get_messages", success=True, data=data)

    async def _respond(
        self,
        cmd_id: str | None,
        command: str,
        *,
        success: bool,
        error: str | None = None,
        data: Any = None,
    ) -> None:
        out: dict[str, Any] = {"type": "response", "command": command, "success": success}
        if cmd_id is not None:
            out["id"] = cmd_id
        if error is not None:
            out["error"] = error
        if data is not None:
            out["data"] = data
        await self._emit(out)

    async def _pump(self) -> None:
        """Single writer. Being the only one is what keeps frames whole."""
        assert self._write is not None, "_pump started outside serve()"
        while True:
            line = await self._outbox.get()
            try:
                await self._write(line)
            except Exception:
                _logger.exception("rpc_write_failed")
            finally:
                self._outbox.task_done()

    async def _emit(self, obj: dict[str, Any]) -> None:
        assert self._write is not None, "_emit called outside serve()"
        line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        if self._outbox.full():
            # The queue is deep enough that reaching this means the client has
            # genuinely stopped, not paused. Stalling the producer is then the
            # right answer — better than dropping frames or growing without
            # bound — and it is the agent, not the dispatch loop, that produces
            # nearly all of them.
            _logger.warning("rpc_outbox_full frames=%d", self._outbox.qsize())
        await self._outbox.put(line)


async def serve_stdio(server: RpcServer) -> None:
    """Run `server` over this process's stdin/stdout.

    Claims stdout first, installs SIGTERM/SIGHUP handlers so a supervisor can
    stop the process cleanly, and shuts down on stdin EOF.
    """
    stdout = claim_stdout()
    loop = asyncio.get_running_loop()

    # The default 64 KiB limit turns a large pasted prompt into a ValueError
    # that escapes `serve` and kills the process.
    reader = asyncio.StreamReader(limit=READ_LIMIT)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    async def read_line() -> bytes:
        return await reader.readline()

    write, close_writer = await _stdout_writer(loop, stdout)

    serving = asyncio.ensure_future(server.serve(read_line=read_line, write=write))

    def _stop(signame: str) -> None:
        _logger.info("rpc_signal signal=%s", signame)
        serving.cancel()

    installed: list[signal.Signals] = []
    for sig in (signal.SIGTERM, signal.SIGHUP):
        with contextlib.suppress(NotImplementedError, AttributeError):
            loop.add_signal_handler(sig, _stop, sig.name)
            installed.append(sig)
    try:
        # Cancelling is a clean stop here, not a failure: `serve`'s own finally
        # cancels the in-flight run on the way out.
        with contextlib.suppress(asyncio.CancelledError):
            await serving
    finally:
        for sig in installed:
            with contextlib.suppress(NotImplementedError):
                loop.remove_signal_handler(sig)
        close_writer()


async def _stdout_writer(
    loop: asyncio.AbstractEventLoop, stdout: BinaryIO
) -> tuple[WriteFn, Callable[[], None]]:
    """A writer that suspends rather than blocks when the client stops reading.

    A pipe holds ~64 KiB, and one ordinary assistant answer is ~20 KiB of frames
    because every token is its own record — so three answers fill it. Writing
    with a plain `file.write` then blocks the *event loop*, which stops the
    agent, every tool, and the stdin reader together. `abort` cannot get through
    because it arrives on the blocked reader, and the SIGTERM handler cannot run
    because it is queued on the blocked loop. Only SIGKILL is left.

    `drain()` suspends the calling coroutine instead, so the loop keeps
    scheduling: commands are still dispatched and an abort still lands. The
    agent throttles to the speed of the client, which is the correct answer —
    pausing beats both blocking and buffering without bound.
    """
    try:
        transport, protocol = await loop.connect_write_pipe(
            lambda: asyncio.streams.FlowControlMixin(loop), stdout
        )
    except ValueError:
        # Not a pipe, socket or tty — `midge --rpc > out.jsonl`. A regular file
        # has no reader to stall behind, so blocking writes are fine here and
        # asyncio refuses to wrap it anyway.
        _logger.debug("rpc_writer mode=blocking reason=not_a_pipe")

        async def write_blocking(data: bytes) -> None:
            stdout.write(data)
            stdout.flush()

        return write_blocking, lambda: None

    writer = asyncio.StreamWriter(transport, protocol, None, loop)

    async def write(data: bytes) -> None:
        writer.write(data)
        await writer.drain()

    return write, transport.close
