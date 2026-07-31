"""The dispatch loop and the command handlers.

`serve(read_line=, write=)` takes callables rather than a stream, so the loop is
transport-agnostic — which is what lets the whole protocol be tested in-process
without pipes. `midge.rpc.transport` is one binding of that seam.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from midge.agent import Agent, SteeringQueue
from midge.client import Error
from midge.commands import (
    BUILTIN_COMMANDS,
    RELOAD_TARGETS,
    SKILL_COMMAND_PREFIX,
    TRANSCRIPT_OPTIONS,
    BuiltinCommand,
    Controls,
    Refused,
    ReloadParams,
    UseProfileParams,
)
from midge.config import SubagentConfig
from midge.messages import AssistantMessage, TextContent, UserMessage
from midge.persistence import Session
from midge.profiles import ProfileSet
from midge.rpc.transport import FLUSH_TIMEOUT, OUTBOX_FRAMES, ReadLineFn, WriteFn
from midge.rpc.wire import event_to_wire
from midge.skills import Skill

# Re-exported: `get_commands` is the protocol's view of the same table, and a
# client that imported these from here should not have to learn where they moved.
__all__ = [
    "BUILTIN_COMMANDS",
    "RELOAD_TARGETS",
    "SKILL_COMMAND_PREFIX",
    "TRANSCRIPT_OPTIONS",
    "BuiltinCommand",
    "RpcServer",
]

_logger = logging.getLogger(__name__)


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
        profiles: ProfileSet | None = None,
        subagents: SubagentConfig | None = None,
        resume_fallback: Literal["fork", "continue"] = "fork",
        extension_sources: Sequence[Path] | None = None,
        skill_sources: Sequence[Path] | None = None,
        controls: Controls | None = None,
    ) -> None:
        # An entrypoint that already built one passes it in, so the TUI and the
        # RPC server can be two front-ends onto the *same* agent rather than two
        # arrangements of the same arguments.
        self.controls = controls if controls is not None else Controls(
            agent,
            session=session,
            compaction_keep_recent=compaction_keep_recent,
            base_prompt=base_prompt,
            extension_prompt=extension_prompt,
            skills=skills,
            profiles=profiles,
            subagents=subagents if subagents is not None else SubagentConfig(),
            resume_fallback=resume_fallback,
            # The exact source lists the entrypoint loaded from, so `reload`
            # re-runs the same call rather than reconstructing one.
            # Reconstructing would mean knowing which sources are built-in, and
            # an embedder that handed the agent a deliberately restricted
            # registry would find reload silently widening it. `None` means the
            # entrypoint did not wire that target up, which is not the same as
            # an empty list.
            extension_sources=list(extension_sources) if extension_sources is not None else None,
            skill_sources=list(skill_sources) if skill_sources is not None else None,
        )
        self.controls.runner = self
        self.controls.on_subagent_event = self.subagent_event
        self.controls.bind_subagents(agent.tools)
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

    # `Controls.Runner`: what "a run is in flight" means here is a task.
    def busy(self) -> bool:
        return self._current_run is not None and not self._current_run.done()

    def cancel(self) -> None:
        assert self._current_run is not None
        self._current_run.cancel()

    @property
    def agent(self) -> Agent:
        return self.controls.agent

    @property
    def session(self) -> Session | None:
        return self.controls.session

    @session.setter
    def session(self, value: Session | None) -> None:
        self.controls.session = value

    async def _call(
        self, cmd_id: str | None, command: str, op: Callable[[], dict[str, Any]]
    ) -> None:
        """Run one operation and answer with it.

        Every control handler is this shape once argument parsing is done, which
        is the point of the split: what is left here is the protocol, and a
        `Refused` is the only thing an operation says back that is not data.
        """
        try:
            data = op()
        except Refused as e:
            await self._respond(cmd_id, command, success=False, error=str(e))
            return
        await self._respond(cmd_id, command, success=True, data=data)

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
            case "get_profiles":
                await self._handle_get_profiles(cmd_id)
            case "set_system_prompt":
                await self._handle_set_system_prompt(cmd_id, cmd)
            case "set_model":
                await self._handle_set_model(cmd_id, cmd)
            case "compact":
                await self._handle_compact(cmd_id)
            case "clear_context":
                await self._handle_clear_context(cmd_id)
            case "new_session":
                await self._handle_new_session(cmd_id, cmd)
            case "open_session":
                await self._handle_open_session(cmd_id, cmd)
            case "use_profile":
                await self._handle_use_profile(cmd_id, cmd)
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
            resolved = self.controls.expand(message)
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
            resolved = self.controls.expand(message)
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

    async def _handle_get_commands(self, cmd_id: str | None) -> None:
        await self._respond(
            cmd_id, "get_commands", success=True, data={"commands": self.controls.commands()}
        )

    async def _handle_get_state(self, cmd_id: str | None) -> None:
        await self._respond(cmd_id, "get_state", success=True, data=self.controls.state())

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

    async def _handle_get_profiles(self, cmd_id: str | None) -> None:
        await self._respond(
            cmd_id,
            "get_profiles",
            success=True,
            data={"active": self.controls.profile, "profiles": self.controls.profile_list()},
        )

    async def _handle_get_system_prompt(self, cmd_id: str | None) -> None:
        await self._respond(
            cmd_id,
            "get_system_prompt",
            success=True,
            data={
                "prompt": self.agent.system_prompt,
                "base": self.controls.base_prompt,
                "appended": self.controls.generated_prompt(),
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
        # `durable` because the caveat #57 fixed was invisible: the command
        # reported success either way, and a client had no way to learn whether
        # the change would outlive the process.
        await self._call(cmd_id, "set_system_prompt", lambda: self.controls.set_system_prompt(prompt))

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
        await self._call(cmd_id, "set_model", lambda: self.controls.set_model(model))

    async def _handle_compact(self, cmd_id: str | None) -> None:
        await self._respond(cmd_id, "compact", success=True, data=await self.controls.compact())

    async def _handle_clear_context(self, cmd_id: str | None) -> None:
        await self._call(cmd_id, "clear_context", self.controls.clear_context)

    async def _handle_new_session(self, cmd_id: str | None, cmd: dict[str, Any]) -> None:
        raw_path = cmd.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            await self._respond(
                cmd_id,
                "new_session",
                success=False,
                error="`path` is required and must be a non-empty string",
            )
            return
        await self._call(cmd_id, "new_session", lambda: self.controls.new_session(Path(raw_path)))

    async def subagent_event(self, event: Any, envelope: dict[str, Any]) -> None:
        """Relay a nested agent's event, correlated.

        Hand this to `bind_subagents(on_event=...)`. Child events go through the
        same `event_to_wire` the parent's do, so nothing internal reaches the
        protocol. The envelope is a sibling key rather than a merge, so a client
        that ignores it sees the frames it always saw.
        """
        wire = event_to_wire(event)
        if wire is None:
            return
        await self._emit({**wire, "agent": envelope})

    async def _handle_use_profile(self, cmd_id: str | None, cmd: dict[str, Any]) -> None:
        try:
            params = UseProfileParams.model_validate(
                {k: v for k, v in cmd.items() if k not in ("id", "type")}
            )
        except ValueError as e:
            await self._respond(cmd_id, "use_profile", success=False, error=str(e))
            return
        await self._call(
            cmd_id,
            "use_profile",
            lambda: self.controls.use_profile(params.name, params.transcript),
        )

    async def _handle_open_session(self, cmd_id: str | None, cmd: dict[str, Any]) -> None:
        raw_path = cmd.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            await self._respond(
                cmd_id,
                "open_session",
                success=False,
                error="`path` is required and must be a non-empty string",
            )
            return
        await self._call(cmd_id, "open_session", lambda: self.controls.open_session(Path(raw_path)))

    async def _handle_reload(self, cmd_id: str | None, cmd: dict[str, Any]) -> None:
        try:
            params = ReloadParams.model_validate(
                {k: v for k, v in cmd.items() if k not in ("id", "type")}
            )
        except ValueError as e:
            await self._respond(cmd_id, "reload", success=False, error=str(e))
            return
        try:
            data = await self.controls.reload(params.targets)
        except Refused as e:
            await self._respond(cmd_id, "reload", success=False, error=str(e))
            return
        await self._respond(cmd_id, "reload", success=True, data=data)

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
        await self._call(cmd_id, "set_session_name", lambda: self.controls.set_session_name(name))

    async def _handle_abort(self, cmd_id: str | None) -> None:
        try:
            dropped = self.controls.abort()
        except Refused as e:
            await self._respond(cmd_id, "abort", success=False, error=str(e))
            return
        if dropped:
            await self._emit_queue_update()
        await self._respond(
            cmd_id,
            "abort",
            success=True,
            data={"dropped": [{"id": d.id, "content": str(d.message.content)} for d in dropped]},
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
