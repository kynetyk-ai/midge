from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import tempfile

from midge.tools import tool

_MAX_LINES = 2000
_MAX_BYTES = 50_000
_DEFAULT_TIMEOUT = 60


@tool
async def bash(command: str, timeout: int | None = None) -> str:
    """Run a shell command via /bin/bash -c. Stdout and stderr are interleaved.
    Output is tail-truncated at 2000 lines / 50KB; full output spills to a temp file.

    Args:
        command: shell command string
        timeout: max seconds to wait (default 60)
    """
    timeout_s = timeout if timeout is not None else _DEFAULT_TIMEOUT
    proc = await asyncio.create_subprocess_exec(
        "/bin/bash",
        "-c",
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )

    try:
        stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        _kill_group(proc.pid)
        await proc.wait()
        raise TimeoutError(f"Command timed out after {timeout_s}s") from None
    except asyncio.CancelledError:
        _kill_group(proc.pid)
        raise

    output = stdout_b.decode("utf-8", errors="replace")
    return _format_output(output, proc.returncode or 0)


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)


def _format_output(output: str, returncode: int) -> str:
    spill_path: str | None = None
    if len(output.encode("utf-8")) > _MAX_BYTES:
        fd, spill_path = tempfile.mkstemp(prefix="pi_bash_", suffix=".log")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(output)

    lines = output.splitlines()
    truncated = False
    if len(lines) > _MAX_LINES:
        lines = lines[-_MAX_LINES:]
        truncated = True

    body = "\n".join(lines)
    while len(body.encode("utf-8")) > _MAX_BYTES and lines:
        lines.pop(0)
        truncated = True
        body = "\n".join(lines)

    parts: list[str] = []
    if spill_path is not None:
        parts.append(f"[full output spilled to {spill_path}]")
    if truncated:
        parts.append(f"[output truncated; showing last {len(lines)} lines]")
    if parts:
        body = "\n".join(parts) + "\n" + body
    if returncode != 0:
        body += f"\n[exit code: {returncode}]"
    return body
