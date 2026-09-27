"""The wire contract, pinned.

`tests/golden/rpc_frames.json` records the *shape* of every frame a scripted
session produces — its type, its keys, and for a response the keys of its
`data`. A change to any of them fails here, which is the point: it is a change
to what every client parses. Update the golden file, `docs/rpc.md`, and — if a
client relying on the old shape would now break — `PROTOCOL_VERSION`.

    MIDGE_UPDATE_GOLDEN=1 poetry run pytest tests/test_rpc_contract.py

rewrites the golden file from the current code, for when the change is meant.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from midge.agent import Agent
from midge.client import Client
from midge.messages import AssistantMessage, Message, TextContent, UserMessage
from midge.persistence import Session
from midge.rpc import RpcServer
from midge.tools import ToolRegistry, tool
from tests.fakes import finish, install, install_gated, say, whole_call

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).parent / "golden" / "rpc_frames.json"
DOC = ROOT / "docs" / "rpc.md"


def shape(frame: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"type": frame["type"], "keys": sorted(frame)}
    if frame["type"] == "response":
        out["command"] = frame["command"]
        if isinstance(frame.get("data"), dict):
            out["data"] = sorted(frame["data"])
    return out


def _long_history() -> list[Message]:
    history: list[Message] = []
    for i in range(3):
        history.append(UserMessage(content=f"q{i}: " + "x" * 200))
        history.append(AssistantMessage(content=[TextContent(text="a" * 200)]))
    return history


async def _session_frames(tmp_path: Path) -> list[dict[str, Any]]:
    """Every command, and every frame kind reachable in-process."""

    @tool(read_only=True)
    async def echo(text: str) -> str:
        """Echo."""
        return text

    client = Client()
    install(
        client,
        [
            [say("looking"), *whole_call("echo", '{"text": "hi"}'), finish("tool_use")],
            [say("done"), finish()],
            [say("## Goal\nsummary"), finish()],  # the compaction after that turn
            [say("## Goal\nagain"), finish()],  # the `compact` command
        ],
    )
    agent = Agent(client=client, model="m", tools=ToolRegistry([echo]))
    agent.history = _long_history()
    session = Session.new(tmp_path / "s.jsonl", model="m")
    server = RpcServer(
        agent,
        session=session,
        compaction_threshold=1,
        compaction_keep_recent=120,
        session_dir=tmp_path,
    )

    lines: asyncio.Queue[bytes] = asyncio.Queue()
    frames: list[dict[str, Any]] = []
    buffer = b""

    async def write(data: bytes) -> None:
        nonlocal buffer
        buffer += data
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            frames.append(json.loads(line))

    async def send(cmd: dict[str, Any] | str) -> None:
        await lines.put((cmd if isinstance(cmd, str) else json.dumps(cmd)).encode() + b"\n")

    async def until(kind: str, count: int = 1) -> None:
        for _ in range(400):
            if sum(f.get("type") == kind for f in frames) >= count:
                return
            await asyncio.sleep(0.005)
        raise AssertionError(f"never saw {count} x {kind}")

    task = asyncio.create_task(server.serve(read_line=lines.get, write=write))

    # A full turn: text, a tool call and its result, then compaction.
    await send({"id": "p", "type": "prompt", "message": "go"})
    await until("agent_settled")

    # Every read, and the controls that are safe to run in a row.
    reads = [
        "get_state", "get_commands", "get_messages", "get_last_assistant_text",
        "get_system_prompt", "get_profiles", "list_sessions",
    ]
    for i, name in enumerate(reads):
        await send({"id": f"r{i}", "type": name})
    await send({"id": "n", "type": "set_session_name", "name": "contract"})
    await send({"id": "sp", "type": "set_system_prompt", "prompt": "be terse"})
    await send({"id": "sm", "type": "set_model", "model": "m2"})
    await send({"id": "cp", "type": "compact"})
    await send({"id": "cc", "type": "clear_context"})
    await send({"id": "rl", "type": "reload"})
    await send({"id": "up", "type": "use_profile", "name": "nope"})
    await send({"id": "ns", "type": "new_session", "path": str(tmp_path / "t.jsonl")})
    await send({"id": "os", "type": "open_session", "path": str(tmp_path / "s.jsonl")})
    await send({"id": "fu", "type": "follow_up", "message": "later"})
    await send({"id": "ab", "type": "abort"})
    await send("{not json")
    await send({"id": "x", "type": "nonesuch"})
    await until("response", 1 + len(reads) + 13)

    # A turn that is steered, and the steer delivered: queue updates, and the
    # `user_message` that carries `source` and `queue_id`.
    delivered = asyncio.Event()
    install_gated(client, [say("part"), finish()], delivered)
    await send({"id": "p2", "type": "prompt", "message": "again"})
    await until("assistant_text_delta", 3)
    await send({"id": "st", "type": "steer", "message": "also this"})
    await until("queue_update", 2)
    delivered.set()
    await until("agent_settled", 2)

    # A turn that is aborted: the `error` frame.
    held = asyncio.Event()
    install_gated(client, [say("part")], held)
    await send({"id": "p3", "type": "prompt", "message": "once more"})
    await until("response", 1 + len(reads) + 13 + 3)
    await asyncio.sleep(0.05)
    await send({"id": "ab2", "type": "abort"})
    await until("agent_settled", 3)

    await lines.put(b"")
    held.set()
    await task
    session.close()
    return frames


async def test_every_frame_matches_the_golden_contract(tmp_path: Path) -> None:
    frames = await _session_frames(tmp_path)
    seen = sorted({json.dumps(shape(f), sort_keys=True) for f in frames})
    got = [json.loads(s) for s in seen]

    if os.environ.get("MIDGE_UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(got, indent=2) + "\n")
    want = json.loads(GOLDEN.read_text())

    added = [s for s in got if s not in want]
    removed = [s for s in want if s not in got]
    assert not added and not removed, (
        "The RPC wire contract changed.\n"
        f"new shapes: {json.dumps(added, indent=1)}\n"
        f"gone shapes: {json.dumps(removed, indent=1)}\n"
        "If this is intended: update docs/rpc.md, rerun with MIDGE_UPDATE_GOLDEN=1, and bump "
        "PROTOCOL_VERSION if a client relying on a gone shape would break."
    )
    assert frames[0]["type"] == "ready"


def test_the_reference_names_every_frame_and_command() -> None:
    # Read from the source rather than listed here, so a new frame or command
    # cannot be added without this noticing that the document did not follow.
    src = (ROOT / "src/midge/rpc/wire.py").read_text() + (
        ROOT / "src/midge/rpc/server.py"
    ).read_text()
    frames = set(re.findall(r'"type": "([a-z_]+)"', src))
    commands = set(re.findall(r'case "([a-z_]+)"', src))
    doc = DOC.read_text()

    missing = sorted(n for n in frames | commands if f"`{n}`" not in doc)
    assert not missing, f"docs/rpc.md does not describe: {missing}"
