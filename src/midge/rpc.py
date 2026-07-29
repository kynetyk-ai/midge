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

Inbound subset (Phase 2): prompt, abort, get_messages.
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
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, BinaryIO

from midge.agent import (
    Agent,
    AgentEnd,
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
from midge.messages import AssistantMessage, TextContent, ToolCall
from midge.persistence import Session, read_transcript
from midge.session import export_html

_logger = logging.getLogger(__name__)

READ_LIMIT = 16 * 1024 * 1024

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
    if isinstance(ev, AgentEnd):
        return {"type": "agent_end"}
    return None


class RpcServer:
    def __init__(
        self,
        agent: Agent,
        *,
        session: Session | None = None,
        compaction_keep_recent: int = 20_000,
        base_prompt: str | None = None,
        prompt_suffix: str = "",
    ) -> None:
        self.agent = agent
        # `client` and `model` come off the agent; `session` is what the export
        # and persistence commands need and cannot reach otherwise.
        self.session = session
        self.compaction_keep_recent = compaction_keep_recent
        # The agent's prompt is composed: a durable base the operator owns, then
        # what midge generates — extension contributions and the skills
        # catalogue. Keeping the halves apart is what lets `set_system_prompt`
        # change the base without silently deleting the catalogue.
        self._base_prompt = base_prompt if base_prompt is not None else (agent.system_prompt or "")
        self._prompt_suffix = prompt_suffix
        self._current_run: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._write: WriteFn | None = None

    async def serve(self, *, read_line: ReadLineFn, write: WriteFn) -> None:
        self._write = write
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
        if self._current_run is not None and not self._current_run.done():
            await self._respond(
                cmd_id,
                "prompt",
                success=False,
                error="a prompt is already in flight",
            )
            return
        await self._respond(cmd_id, "prompt", success=True)
        self._current_run = asyncio.create_task(self._run_prompt(message))

    async def _run_prompt(self, message: str) -> None:
        await self._emit({"type": "user_message", "content": message})
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

    def _compose_prompt(self) -> str:
        return "\n\n".join(p for p in (self._base_prompt, self._prompt_suffix) if p)

    async def _handle_get_system_prompt(self, cmd_id: str | None) -> None:
        await self._respond(
            cmd_id,
            "get_system_prompt",
            success=True,
            data={
                "prompt": self.agent.system_prompt,
                "base": self._base_prompt,
                "appended": self._prompt_suffix,
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
                export_html(entries, model=self.agent.model), encoding="utf-8"
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

        Runtime-only, deliberately. The session file is an append-only record of
        what happened, so it keeps every message and a resume of that file
        restores them — clearing changes what the model sees now, not what was
        written. Use `new_session` to start a fresh log.
        """
        cleared = len(self.agent.history)
        self.agent.history = []
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

    async def _handle_abort(self, cmd_id: str | None) -> None:
        if self._current_run is not None and not self._current_run.done():
            self._current_run.cancel()
            await self._respond(cmd_id, "abort", success=True)
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

    async def _emit(self, obj: dict[str, Any]) -> None:
        assert self._write is not None, "_emit called outside serve()"
        line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            await self._write(line)


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

    async def write(data: bytes) -> None:
        stdout.write(data)
        stdout.flush()

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
