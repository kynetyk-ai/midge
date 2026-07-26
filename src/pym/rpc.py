"""JSON-over-stdio RPC server for embedding the agent in external tools.

Wire format: newline-delimited JSON, LF-only. Each emitted record is
`json.dumps(obj, ensure_ascii=False) + "\n"`.

Two outbound shapes:
    - Responses correlated to inbound commands by an optional `id`:
      {"id": "...", "type": "response", "command": "...", "success": bool, ...}
    - Async events streamed during a prompt run, uncorrelated:
      {"type": "assistant_text_delta" | "tool_call_start" | ... }

Inbound subset (Phase 2): prompt, abort, get_messages.
Commands are dispatched serially; a `prompt` returns its response immediately
after preflight and runs the agent in a background task while the dispatch
loop continues reading stdin (so `abort` can interrupt).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pym.agent import (
    Agent,
    AgentEnd,
    ToolExecutionEnd,
    ToolExecutionStart,
)
from pym.client import (
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
from pym.messages import TextContent, ToolCall

_logger = logging.getLogger(__name__)

WriteFn = Callable[[bytes], Awaitable[None]]
ReadLineFn = Callable[[], Awaitable[bytes]]


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
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
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
                stripped = line.rstrip(b"\r\n")
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
        match cmd_type:
            case "prompt":
                await self._handle_prompt(cmd_id, cmd)
            case "abort":
                await self._handle_abort(cmd_id)
            case "get_messages":
                await self._handle_get_messages(cmd_id)
            case _:
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
            _logger.exception("error during prompt run")
            if not saw_error_event:
                await self._emit(
                    {"type": "error", "message": str(e), "stop_reason": "error"}
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
