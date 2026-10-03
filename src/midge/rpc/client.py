"""Client for `midge --rpc`.

Standard library only: a program that embeds midge speaks to it over a pipe
and should not need midge's dependencies. The wire protocol is `docs/rpc.md`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from queue import Empty, Queue
from typing import IO, Any

SUPPORTED_PROTOCOLS: frozenset[int] = frozenset({1})


class ProtocolError(Exception):
    """The server did not send a valid ready frame, or uses an unsupported protocol."""


class CommandError(Exception):
    """The server responded with success: false."""

    def __init__(self, command: str, error: str) -> None:
        self.command = command
        self.error = error
        super().__init__(f"{command}: {error}")


class MidgeClient:
    """Talk to a `midge --rpc` subprocess over stdio pipes.

    Use as a context manager, or call `close()` when done.
    """

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        if command is None:
            exe = shutil.which("midge")
            command = [exe, "--rpc"] if exe else [sys.executable, "-m", "midge", "--rpc"]

        self._proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=str(cwd) if isinstance(cwd, Path) else cwd,
            env=dict(env) if env else None,
        )
        self._timeout = timeout
        self._id_counter = 0
        self._waiters: dict[str, Queue[dict[str, Any]]] = {}
        self._waiters_lock = threading.Lock()
        self._events: Queue[dict[str, Any] | None] = Queue()
        self._closed = False

        # Read the ready frame synchronously
        assert self._proc.stdout is not None
        try:
            ready_frame = _read_ready(self._proc.stdout, timeout)
        except ProtocolError:
            self._proc.kill()
            self._proc.wait()
            raise

        proto = ready_frame.get("protocol")
        if proto not in SUPPORTED_PROTOCOLS:
            self._proc.terminate()
            self._proc.wait()
            raise ProtocolError(
                f"midge speaks protocol {proto}; this client knows {sorted(SUPPORTED_PROTOCOLS)}"
            )

        self.ready: dict[str, Any] = ready_frame
        self.protocol: int = proto
        self.version: str = ready_frame.get("midge", "")

        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._dispatch(frame)

        self._events.put(None)

    def _dispatch(self, frame: dict[str, Any]) -> None:
        if frame.get("type") == "response" and "id" in frame:
            waited = False
            with self._waiters_lock:
                q = self._waiters.pop(frame["id"], None)
                waited = q is not None
            if waited:
                assert q is not None
                q.put(frame)
        else:
            self._events.put(frame)

    def _next_id(self) -> str:
        self._id_counter += 1
        return str(self._id_counter)

    def request(self, type: str, **params: Any) -> Any:
        """Send a command and wait for its response.

        Returns the `data` field, or `None` when the response has no data.
        Raises `CommandError` on failure, `TimeoutError` on timeout.
        """
        assert self._proc.stdin is not None
        rid = self._next_id()
        q: Queue = Queue()
        with self._waiters_lock:
            self._waiters[rid] = q
        payload = json.dumps({"id": rid, "type": type, **params})
        self._proc.stdin.write(payload + "\n")
        self._proc.stdin.flush()
        try:
            frame = q.get(timeout=self._timeout)
        except Empty:
            with self._waiters_lock:
                self._waiters.pop(rid, None)
            raise TimeoutError(f"no response for {type!r} within {self._timeout}s") from None
        if not frame.get("success", True):
            raise CommandError(frame.get("command", type), frame.get("error", "unknown error"))
        return frame.get("data")

    def steer(self, message: str) -> str:
        """Inject a steering message into the running turn. Returns queue_id."""
        data = self.request("steer", message=message)
        return data["queue_id"]

    def follow_up(self, message: str) -> str:
        """Queue a follow-up message after the current turn. Returns queue_id."""
        data = self.request("follow_up", message=message)
        return data["queue_id"]

    def prompt(self, message: str) -> Iterator[dict[str, Any]]:
        """Submit a prompt and yield event frames until the agent settles."""
        self.request("prompt", message=message)
        while True:
            frame = self._events.get()
            if frame is None:
                raise ConnectionError("midge exited before agent_settled")
            yield frame
            if frame.get("type") == "agent_settled":
                return

    def abort(self) -> list[dict[str, Any]]:
        """Abort the running turn. Returns dropped queued messages."""
        data = self.request("abort")
        return data["dropped"]

    def get_state(self) -> dict[str, Any]:
        """Return the current agent state."""
        return self.request("get_state")

    def get_messages(self) -> list[dict[str, Any]]:
        """Return the transcript messages."""
        return self.request("get_messages")

    def set_model(self, model: str) -> dict[str, Any]:
        """Switch the active model."""
        return self.request("set_model", model=model)

    def use_profile(
        self, name: str, transcript: str = "continue",
    ) -> dict[str, Any]:
        """Switch to a named profile."""
        return self.request("use_profile", name=name, transcript=transcript)

    def clear_context(self) -> dict[str, Any]:
        """Clear the conversation context."""
        return self.request("clear_context")

    def new_session(self, path: str) -> dict[str, Any]:
        """Start a new session at the given path."""
        return self.request("new_session", path=path)

    def open_session(self, path: str) -> dict[str, Any]:
        """Open an existing session at the given path."""
        return self.request("open_session", path=path)

    def list_sessions(self, roots_only: bool = True) -> list[dict[str, Any]]:
        """List available sessions."""
        data = self.request("list_sessions", roots_only=roots_only)
        return data["sessions"]

    def close(self) -> int:
        """Close stdin, wait for the process, and return its exit code."""
        if self._closed:
            return self._proc.poll() or 0
        self._closed = True
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        return self._proc.wait(timeout=self._timeout)

    def __enter__(self) -> MidgeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _read_ready(out: IO[str], timeout: float) -> dict[str, Any]:
    """Read the first ready line from the process stdout.

    Uses a separate thread so we can block-read from a text-mode file
    and enforce the timeout. Raises ``ProtocolError`` on timeout or a
    non-ready frame.
    """

    result: list[dict[str, Any]] = []
    exc: list[Exception] = []

    def _do() -> None:
        try:
            line = out.readline()
            if line:
                try:
                    result.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    exc.append(ProtocolError(f"first frame is not JSON: {e}"))
        except Exception as e:
            exc.append(e)

    t = threading.Thread(target=_do)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise ProtocolError(f"server did not respond within {timeout}s")
    if exc:
        raise exc[0]
    if not result:
        raise ProtocolError("server produced no output")
    frame = result[0]
    if frame.get("type") != "ready":
        raise ProtocolError(f"expected ready frame, got {frame.get('type')!r}")
    return frame
