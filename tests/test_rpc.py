from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from midge.agent import Agent
from midge.client import Client
from midge.config import ProviderConfig
from midge.extensions import load_extensions
from midge.hooks import Hooks, ToolCallEvent, ToolCallResult
from midge.messages import ToolCall, UserMessage
from midge.persistence import Session, read_transcript, session_continuations
from midge.profiles import Profile, ProfileSet
from midge.profiles import validate as validate_profiles
from midge.providers import ModelRegistry
from midge.rpc import RpcServer, event_to_wire
from midge.rpc import server as rpc_server
from midge.skills import Skill, load_skills, skills_prompt
from midge.subagents import bind_subagents
from midge.tools import ToolRegistry, tool
from tests.fakes import finish, install, install_gated, say, tcall


class _Inbox:
    def __init__(self) -> None:
        self._q: asyncio.Queue[bytes] = asyncio.Queue()

    async def feed(self, line: bytes) -> None:
        await self._q.put(line)

    async def feed_text(self, line: str) -> None:
        await self._q.put(line.encode("utf-8"))

    def close(self) -> None:
        self._q.put_nowait(b"")

    async def read_line(self) -> bytes:
        return await self._q.get()


class _Outbox:
    def __init__(self) -> None:
        self.lines: list[dict[str, Any]] = []
        self._buffer = b""

    async def write(self, data: bytes) -> None:
        self._buffer += data
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if line:
                self.lines.append(json.loads(line.decode("utf-8")))


async def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError("predicate did not become true in time")
        await asyncio.sleep(0.005)


def _start_server(agent: Agent) -> tuple[RpcServer, _Inbox, _Outbox, asyncio.Task[None]]:
    server = RpcServer(agent)
    inbox = _Inbox()
    outbox = _Outbox()
    task = asyncio.create_task(
        server.serve(read_line=inbox.read_line, write=outbox.write)
    )
    return server, inbox, outbox, task


async def test_prompt_response_then_events() -> None:
    client = Client()
    install(client, [[say("hi"), finish()]])
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "prompt", "message": "ping"}\n')
    await _wait_for(lambda: any(line.get("type") == "agent_end" for line in outbox.lines))
    inbox.close()
    await task

    types = [line.get("type") for line in outbox.lines]
    assert types[0] == "response"
    assert outbox.lines[0]["id"] == "r1"
    assert outbox.lines[0]["success"] is True
    assert "user_message" in types
    assert "assistant_text_delta" in types
    assert "assistant_message_end" in types
    assert "agent_end" in types


async def test_get_messages_returns_history() -> None:
    client = Client()
    install(client, [[say("hi"), finish()]])
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "prompt", "message": "ping"}\n')
    await _wait_for(lambda: any(line.get("type") == "agent_end" for line in outbox.lines))
    await inbox.feed_text('{"id": "r2", "type": "get_messages"}\n')
    await _wait_for(
        lambda: any(
            line.get("type") == "response" and line.get("id") == "r2"
            for line in outbox.lines
        )
    )
    inbox.close()
    await task

    resp = next(line for line in outbox.lines if line.get("id") == "r2")
    assert resp["success"] is True
    data = resp["data"]
    assert len(data) == 2
    assert data[0]["role"] == "user"
    assert data[1]["role"] == "assistant"


async def test_abort_cancels_in_flight_prompt() -> None:
    client = Client()
    gate = asyncio.Event()
    install_gated(client, [say("part")], gate)
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "prompt", "message": "go"}\n')
    await _wait_for(
        lambda: any(line.get("type") == "assistant_text_delta" for line in outbox.lines)
    )

    await inbox.feed_text('{"id": "r2", "type": "abort"}\n')
    await _wait_for(
        lambda: any(
            line.get("type") == "response" and line.get("id") == "r2"
            for line in outbox.lines
        )
    )
    await _wait_for(
        lambda: any(
            line.get("type") == "error" and line.get("stop_reason") == "aborted"
            for line in outbox.lines
        )
    )

    gate.set()
    inbox.close()
    await task

    abort_resp = next(line for line in outbox.lines if line.get("id") == "r2")
    assert abort_resp["success"] is True
    error_events = [
        line for line in outbox.lines
        if line.get("type") == "error" and line.get("stop_reason") == "aborted"
    ]
    assert len(error_events) == 1, "expected exactly one cancelled-error event"


async def test_abort_when_idle_responds_failure() -> None:
    client = Client()
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "abort"}\n')
    await _wait_for(lambda: any(line.get("id") == "r1" for line in outbox.lines))
    inbox.close()
    await task

    resp = outbox.lines[0]
    assert resp["success"] is False
    assert "no prompt in flight" in resp["error"]


async def test_parse_error_responds_with_command_parse() -> None:
    client = Client()
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text("not valid json\n")
    await _wait_for(lambda: any(line.get("command") == "parse" for line in outbox.lines))
    inbox.close()
    await task

    resp = outbox.lines[0]
    assert resp["type"] == "response"
    assert resp["command"] == "parse"
    assert resp["success"] is False
    assert "id" not in resp


async def test_unknown_command_responds_failure() -> None:
    client = Client()
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "no_such_command"}\n')
    await _wait_for(lambda: any(line.get("id") == "r1" for line in outbox.lines))
    inbox.close()
    await task

    resp = outbox.lines[0]
    assert resp["success"] is False
    assert "unknown command" in resp["error"]


async def test_prompt_missing_message_responds_failure() -> None:
    client = Client()
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "prompt"}\n')
    await _wait_for(lambda: any(line.get("id") == "r1" for line in outbox.lines))
    inbox.close()
    await task

    resp = outbox.lines[0]
    assert resp["success"] is False
    assert "message" in resp["error"]


async def test_prompt_in_flight_is_queued_not_rejected() -> None:
    """Queueing replaced the hard rejection, but the response still says which
    happened — a client should not infer it from whether events follow."""
    client = Client()
    gate = asyncio.Event()
    install_gated(client, [say("part")], gate)
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _start_server(agent)

    first = await _command(inbox, outbox, {"id": "p1", "type": "prompt", "message": "one"})
    assert first["data"] == {"accepted": "started"}

    await _wait_for(lambda: any(x.get("type") == "assistant_text_delta" for x in outbox.lines))
    second = await _command(inbox, outbox, {"id": "p2", "type": "prompt", "message": "two"})

    assert second["success"] is True
    assert second["data"] == {"accepted": "queued"}
    updates = [x for x in outbox.lines if x.get("type") == "queue_update"]
    assert updates[-1]["follow_up"][0]["content"] == "two"

    gate.set()
    inbox.close()
    await task


async def test_no_id_means_no_id_in_response() -> None:
    client = Client()
    agent = Agent(client=client, model="m")
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"type": "abort"}\n')
    await _wait_for(lambda: any(line.get("type") == "response" for line in outbox.lines))
    inbox.close()
    await task

    resp = outbox.lines[0]
    assert "id" not in resp


async def test_tool_call_events_flow_through() -> None:
    @tool
    async def echo(text: str) -> str:
        return f"echoed:{text}"

    client = Client()
    install(
        client,
        [
            [
                tcall(index=0, id="c1", name="echo", args='{"text":"hi"}'),
                finish("tool_use"),
            ],
            [say("ok"), finish()],
        ],
    )
    agent = Agent(client=client, model="m", tools=ToolRegistry([echo]))
    _, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id": "r1", "type": "prompt", "message": "go"}\n')
    await _wait_for(lambda: any(line.get("type") == "agent_end" for line in outbox.lines))
    inbox.close()
    await task

    types = [line.get("type") for line in outbox.lines]
    assert "tool_call_start" in types
    assert "tool_call_end" in types
    assert "tool_execution_start" in types
    assert "tool_result" in types

    tool_result = next(line for line in outbox.lines if line.get("type") == "tool_result")
    assert tool_result["tool_call_id"] == "c1"
    assert tool_result["content"] == "echoed:hi"
    assert tool_result["is_error"] is False


def test_event_to_wire_drops_internal_events() -> None:
    from midge.client import StreamStart, TextEnd, TextStart
    from midge.messages import AssistantMessage

    partial = AssistantMessage()
    assert event_to_wire(StreamStart(partial=partial)) is None
    assert event_to_wire(TextStart(content_index=0, partial=partial)) is None
    assert event_to_wire(TextEnd(content_index=0, content="x", partial=partial)) is None


def test_event_to_wire_text_delta() -> None:
    from midge.client import TextDelta
    from midge.messages import AssistantMessage, TextContent

    partial = AssistantMessage(content=[TextContent(text="hel")])
    wire = event_to_wire(TextDelta(content_index=0, delta="lo", partial=partial))
    assert wire == {"type": "assistant_text_delta", "delta": "lo"}


def test_event_to_wire_tool_call_end() -> None:
    from midge.client import ToolCallEnd
    from midge.messages import AssistantMessage, ToolCall

    tc = ToolCall(id="c1", name="read", arguments={"path": "x"})
    partial = AssistantMessage(content=[tc])
    wire = event_to_wire(
        ToolCallEnd(content_index=0, tool_call=tc, partial=partial)
    )
    assert wire == {
        "type": "tool_call_end",
        "id": "c1",
        "name": "read",
        "arguments": {"path": "x"},
    }


def test_event_to_wire_unicode_preserved() -> None:
    """ensure_ascii=False so unicode survives without \\u escapes."""
    from midge.client import TextDelta
    from midge.messages import AssistantMessage

    partial = AssistantMessage()
    wire = event_to_wire(TextDelta(content_index=0, delta="héllo 🚀", partial=partial))
    assert wire is not None
    line = json.dumps(wire, ensure_ascii=False)
    assert "héllo" in line
    assert "🚀" in line


# ---- state and control commands ----


async def _run_to_completion(inbox: _Inbox, outbox: _Outbox, message: str = "hi") -> None:
    await inbox.feed_text(json.dumps({"id": "p", "type": "prompt", "message": message}) + "\n")
    await _wait_for(lambda: any(x.get("type") == "agent_end" for x in outbox.lines))


async def _command(inbox: _Inbox, outbox: _Outbox, cmd: dict[str, Any]) -> dict[str, Any]:
    await inbox.feed_text(json.dumps(cmd) + "\n")
    await _wait_for(
        lambda: any(x.get("type") == "response" and x.get("id") == cmd["id"] for x in outbox.lines)
    )
    return next(x for x in outbox.lines if x.get("type") == "response" and x.get("id") == cmd["id"])


async def test_get_state_reports_model_and_counts() -> None:
    client = Client()
    install(client, [[say("hi"), finish()]])
    agent = Agent(client=client, model="gpt-4o")
    _server, inbox, outbox, task = _start_server(agent)

    await _run_to_completion(inbox, outbox)
    resp = await _command(inbox, outbox, {"id": "s", "type": "get_state"})

    assert resp["success"] is True
    assert resp["data"] == {
        "model": "gpt-4o",
        "streaming": False,
        "session": None,
        "session_name": None,
        "messages": 2,
    }
    inbox.close()
    await task


async def test_get_state_reports_streaming_while_in_flight() -> None:
    client = Client()
    gate = asyncio.Event()
    install_gated(client, [say("part")], gate)
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id":"p","type":"prompt","message":"hi"}\n')
    await _wait_for(lambda: any(x.get("type") == "assistant_text_delta" for x in outbox.lines))
    resp = await _command(inbox, outbox, {"id": "s", "type": "get_state"})

    assert resp["data"]["streaming"] is True
    gate.set()
    inbox.close()
    await task


async def test_get_last_assistant_text() -> None:
    client = Client()
    install(client, [[say("the answer"), finish()]])
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await _run_to_completion(inbox, outbox)
    resp = await _command(inbox, outbox, {"id": "t", "type": "get_last_assistant_text"})

    assert resp["data"] == {"text": "the answer"}
    inbox.close()
    await task


async def test_get_last_assistant_text_null_when_empty() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _command(inbox, outbox, {"id": "t", "type": "get_last_assistant_text"})

    assert resp["data"] == {"text": None}
    inbox.close()
    await task


async def test_system_prompt_round_trips() -> None:
    agent = Agent(client=Client(), model="m", system_prompt="original")
    _server, inbox, outbox, task = _start_server(agent)

    before = await _command(inbox, outbox, {"id": "g1", "type": "get_system_prompt"})
    assert before["data"]["prompt"] == "original"
    assert before["data"]["base"] == "original"

    setr = await _command(
        inbox, outbox, {"id": "s1", "type": "set_system_prompt", "prompt": "rewritten"}
    )
    assert setr["success"] is True

    after = await _command(inbox, outbox, {"id": "g2", "type": "get_system_prompt"})
    assert after["data"]["prompt"] == "rewritten"
    assert agent.system_prompt == "rewritten"
    inbox.close()
    await task


# Stands in for whatever midge generates and appends. Sourced here from an
# extension rather than a skill so these tests stay about the base/generated
# split; the skills half has its own tests under `reload`.
GENERATED = "Extension guidance: prefer the notes tool."


def _server_with_composed_prompt(agent: Agent) -> tuple[RpcServer, _Inbox, _Outbox, Any]:
    server = RpcServer(agent, base_prompt="You are a coding assistant.", extension_prompt=GENERATED)
    agent.system_prompt = server.controls.compose_prompt()
    inbox, outbox = _Inbox(), _Outbox()
    task = asyncio.create_task(server.serve(read_line=inbox.read_line, write=outbox.write))
    return server, inbox, outbox, task


async def test_set_system_prompt_keeps_the_generated_half() -> None:
    """The composed prompt is base + extension contributions + skills
    catalogue. Setting the base must not delete the rest — a client could not
    put it back, since the composed string is undelimited and the catalogue
    carries absolute paths."""
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_composed_prompt(agent)

    assert GENERATED in (agent.system_prompt or "")

    await _command(
        inbox, outbox, {"id": "s", "type": "set_system_prompt", "prompt": "You are a lawyer."}
    )

    assert agent.system_prompt is not None
    assert agent.system_prompt.startswith("You are a lawyer.")
    assert GENERATED in agent.system_prompt, "the generated half was dropped"
    assert "coding assistant" not in agent.system_prompt
    inbox.close()
    await task


async def test_get_system_prompt_separates_the_halves() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_composed_prompt(agent)

    resp = await _command(inbox, outbox, {"id": "g", "type": "get_system_prompt"})

    # A client can see what it owns and what midge appends, without guessing
    # where one ends and the other begins.
    assert resp["data"]["base"] == "You are a coding assistant."
    assert resp["data"]["appended"] == GENERATED
    assert resp["data"]["prompt"] == agent.system_prompt
    inbox.close()
    await task


async def test_set_system_prompt_takes_effect_on_the_next_turn_only() -> None:
    """`_stream` snapshots the prompt outside its loop, so a mid-run change
    must not alter the turn already in flight.

    The tool blocks on a gate so the prompt is definitely changed *before* the
    run's second provider call — without that the tool finishes first and the
    test passes without exercising anything.
    """
    client = Client()
    captured = install(
        client,
        [
            [
                tcall(index=0, id="c1", name="wait", args="{}"),
                finish("tool_use"),
            ],
            [say("done"), finish()],
            [say("second turn"), finish()],
        ],
    )
    release = asyncio.Event()

    @tool
    async def wait() -> str:
        """Block until the test lets go."""
        await release.wait()
        return "released"

    agent = Agent(client=client, model="m", tools=ToolRegistry([wait]), system_prompt="original")
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id":"p","type":"prompt","message":"go"}\n')
    await _wait_for(lambda: any(x.get("type") == "tool_execution_start" for x in outbox.lines))

    await _command(inbox, outbox, {"id": "s", "type": "set_system_prompt", "prompt": "rewritten"})
    assert agent.system_prompt == "rewritten"
    assert len(captured) == 1, "the second provider call must not have happened yet"

    release.set()
    await _wait_for(lambda: any(x.get("type") == "agent_end" for x in outbox.lines))

    # Both calls of the run in flight used the prompt it started with.
    assert [c["messages"][0]["content"] for c in captured] == ["original", "original"]

    outbox.lines.clear()
    await _run_to_completion(inbox, outbox, "again")
    assert captured[2]["messages"][0]["content"] == "rewritten"
    inbox.close()
    await task


async def test_set_model_changes_the_model() -> None:
    agent = Agent(client=Client(), model="old")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _command(inbox, outbox, {"id": "m", "type": "set_model", "model": "new"})

    assert resp["success"] is True
    assert agent.model == "new"
    inbox.close()
    await task


async def test_set_model_rejects_non_string() -> None:
    agent = Agent(client=Client(), model="old")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _command(inbox, outbox, {"id": "m", "type": "set_model", "model": 7})

    assert resp["success"] is False
    assert "model" in resp["error"]
    assert agent.model == "old"
    inbox.close()
    await task


def _registered(**models: str) -> ModelRegistry:
    return ModelRegistry(
        models=models,
        providers={name: ProviderConfig() for name in set(models.values())},
    )


async def test_set_model_accepts_a_registered_model() -> None:
    agent = Agent(client=Client(registry=_registered(fast="svc", slow="svc")), model="fast")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _command(inbox, outbox, {"id": "m", "type": "set_model", "model": "slow"})

    assert resp["success"] is True
    assert agent.model == "slow"
    inbox.close()
    await task


async def test_set_model_refuses_an_unregistered_model() -> None:
    """The original defect: success, then a 404 on the next turn.

    A refusal has to name the alternatives, because the client asking has no
    other way to find out what it may pick.
    """
    agent = Agent(client=Client(registry=_registered(fast="svc")), model="fast")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _command(inbox, outbox, {"id": "m", "type": "set_model", "model": "made-up"})

    assert resp["success"] is False
    assert "made-up" in resp["error"] and "fast" in resp["error"]
    assert agent.model == "fast"
    inbox.close()
    await task


async def test_an_empty_registry_still_accepts_anything() -> None:
    # Permissive until a user writes a `[models]` table. This is every install
    # that predates the registry.
    agent = Agent(client=Client(), model="old")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _command(inbox, outbox, {"id": "m", "type": "set_model", "model": "whatever"})

    assert resp["success"] is True
    assert agent.model == "whatever"
    inbox.close()
    await task


async def test_set_system_prompt_rejects_non_string() -> None:
    agent = Agent(client=Client(), model="m", system_prompt="keep")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _command(inbox, outbox, {"id": "s", "type": "set_system_prompt"})

    assert resp["success"] is False
    assert agent.system_prompt == "keep"
    inbox.close()
    await task


async def test_new_session_requires_a_path() -> None:
    """Without one there would be no new session, only a silent end to
    persistence — which is what `clear_context` is for."""
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _command(inbox, outbox, {"id": "n", "type": "new_session"})

    assert resp["success"] is False
    assert "path" in resp["error"]
    inbox.close()
    await task


async def test_clear_context_clears_history_and_keeps_recording(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    session = Session.new(path, model="m")
    client = Client()
    install(
        client,
        [
            [say("first"), finish()],
            [say("second"), finish()],
        ],
    )
    agent = Agent(client=client, model="m")
    server = RpcServer(agent, session=session)
    inbox, outbox = _Inbox(), _Outbox()
    task = asyncio.create_task(server.serve(read_line=inbox.read_line, write=outbox.write))

    await _run_to_completion(inbox, outbox, "before")
    session.append_many(agent.history)

    resp = await _command(inbox, outbox, {"id": "c", "type": "clear_context"})

    assert resp["success"] is True
    assert resp["data"]["cleared"] == 2
    assert agent.history == []
    # The log stays open — clearing changes what the model sees, not what was
    # written, and persistence must not silently stop.
    assert resp["data"]["session"] == str(path)
    assert server.session is session

    outbox.lines.clear()
    await _run_to_completion(inbox, outbox, "after")
    session.append_many(agent.history)
    inbox.close()
    await task
    session.close()

    # Everything is still on disk — the file is the record of what happened, so
    # anything reading the JSONL still sees the cleared turns.
    _, entries = read_transcript(path)
    assert any("before" in str(getattr(e, "content", "")) for e in entries)

    # But a resume honours the clear: only what came after it is replayed.
    restored = Session.load(path).messages
    assert not any("before" in str(m.content) for m in restored)
    assert any("after" in str(m.content) for m in restored)


async def test_clear_context_without_a_session_is_fine() -> None:
    client = Client()
    install(client, [[say("hi"), finish()]])
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await _run_to_completion(inbox, outbox)
    resp = await _command(inbox, outbox, {"id": "c", "type": "clear_context"})

    assert resp["success"] is True
    assert resp["data"] == {"cleared": 2, "session": None}
    assert agent.history == []
    inbox.close()
    await task


async def test_new_session_opens_a_fresh_log(tmp_path: Path) -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    path = tmp_path / "fresh.jsonl"
    resp = await _command(
        inbox, outbox, {"id": "n", "type": "new_session", "path": str(path)}
    )

    assert resp["data"] == {"session": str(path)}
    assert path.exists()
    inbox.close()
    await task


# ---- open_session: attaching to an existing transcript (#63) ----
#
# Startup could bind a transcript and hold it until exit; nothing could move a
# running process to another one. That is what makes an excursion a round trip
# instead of a one-way door.


async def _open(inbox: _Inbox, outbox: _Outbox, path: Path) -> dict[str, Any]:
    cmd_id = f"os{len(outbox.lines)}"
    return await _command(inbox, outbox, {"id": cmd_id, "type": "open_session", "path": str(path)})


async def test_open_session_restores_the_conversation(tmp_path: Path) -> None:
    path = tmp_path / "earlier.jsonl"
    with Session.new(path, model="m") as s:
        s.append(UserMessage(content="what did we decide?"))

    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)
    assert agent.history == []

    resp = await _open(inbox, outbox, path)

    assert resp["success"] is True
    assert resp["data"]["messages"] == 1
    assert resp["data"]["reopened"] is True
    assert [str(m.content) for m in agent.history] == ["what did we decide?"]
    inbox.close()
    await task


async def test_open_session_restores_the_base_prompt_but_not_the_model(tmp_path: Path) -> None:
    """The asymmetry: a prompt is part of what the conversation *is*, so a
    reviewer's transcript resumed without it would leave its own history
    misleading. A model is infrastructure, and mid-run the running one is a live
    choice a recorded value must not silently override."""
    path = tmp_path / "reviewer.jsonl"
    with Session.new(path, model="recorded-model", system_prompt="you are adversarial"):
        pass

    agent = Agent(client=Client(), model="running-model", system_prompt="base")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _open(inbox, outbox, path)

    assert agent.model == "running-model"
    assert "you are adversarial" in (agent.system_prompt or "")
    assert resp["data"]["model_differs"] is True
    assert resp["data"]["recorded_model"] == "recorded-model"
    inbox.close()
    await task


async def test_open_session_creates_a_missing_file(tmp_path: Path) -> None:
    """`resume_last` needs open-or-create in one call: a profile's first use in
    a session has nothing to resume, and that is an ordinary first run rather
    than an error for the caller to sequence around."""
    path = tmp_path / "not-yet.jsonl"
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _open(inbox, outbox, path)

    assert resp["success"] is True
    assert path.exists()
    assert resp["data"]["messages"] == 0
    inbox.close()
    await task


async def test_opening_the_current_session_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    session = Session.new(path, model="m")
    agent = Agent(client=Client(), model="m")
    server, inbox, outbox, task = _session_server(agent, session)

    resp = await _open(inbox, outbox, path)

    assert resp["success"] is True
    assert resp["data"]["reopened"] is False
    assert server.session is session, "the open handle was swapped for no reason"
    inbox.close()
    await task


async def test_a_return_writes_no_forward_link(tmp_path: Path) -> None:
    """`continued` means another transcript *started* here. A return starts
    nothing, and recording one would make the chain cyclic — which is what #67
    walks to resolve `resume_last`."""
    root = tmp_path / "root.jsonl"
    excursion = tmp_path / "excursion.jsonl"
    Session.new(excursion, model="m").close()
    session = Session.new(root, model="m")
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _session_server(agent, session)

    await _open(inbox, outbox, excursion)

    assert session_continuations(read_transcript(root)[1]) == []
    assert session_continuations(read_transcript(excursion)[1]) == []
    inbox.close()
    await task


async def test_open_session_is_refused_mid_turn(tmp_path: Path) -> None:
    path = tmp_path / "other.jsonl"
    Session.new(path, model="m").close()
    client = Client()
    install(client, [[say("working"), finish()]])
    agent = Agent(client=client, model="m")
    server, inbox, outbox, task = _start_server(agent)

    # A run that has not settled, exactly as the reload guard tests set up.
    pending = asyncio.create_task(asyncio.Event().wait())
    server._current_run = pending  # type: ignore[assignment]
    resp = await _open(inbox, outbox, path)

    assert resp["success"] is False
    assert "in flight" in resp["error"]
    pending.cancel()
    inbox.close()
    await task


async def test_a_failed_open_leaves_the_server_untouched(tmp_path: Path) -> None:
    directory = tmp_path / "a-directory"
    directory.mkdir()
    original = tmp_path / "s.jsonl"
    session = Session.new(original, model="m")
    agent = Agent(client=Client(), model="m")
    agent.history = [UserMessage(content="keep me")]
    server, inbox, outbox, task = _session_server(agent, session)

    resp = await _open(inbox, outbox, directory)

    assert resp["success"] is False
    assert server.session is session
    assert [str(m.content) for m in agent.history] == ["keep me"]
    inbox.close()
    await task


async def test_new_session_header_stores_the_base_not_the_composed_prompt(
    tmp_path: Path,
) -> None:
    """A resume reads the header back as the durable base and re-appends the
    generated half itself, so storing the composed string would duplicate the
    skills catalogue every time."""
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_composed_prompt(agent)
    assert GENERATED in (agent.system_prompt or "")

    path = tmp_path / "fresh.jsonl"
    await _command(inbox, outbox, {"id": "n", "type": "new_session", "path": str(path)})

    header = Session.load(path).header
    assert header.system_prompt == "You are a coding assistant."
    assert GENERATED not in (header.system_prompt or "")
    inbox.close()
    await task


async def test_failed_new_session_leaves_state_untouched(tmp_path: Path) -> None:
    """Reporting failure from a state it already destroyed is worse than
    failing: the caller has lost the history and the log either way."""
    taken = tmp_path / "taken.jsonl"
    Session.new(taken, model="m").close()

    live = Session.new(tmp_path / "live.jsonl", model="m")
    client = Client()
    install(client, [[say("hi"), finish()]])
    agent = Agent(client=client, model="m")
    server = RpcServer(agent, session=live)
    inbox, outbox = _Inbox(), _Outbox()
    task = asyncio.create_task(server.serve(read_line=inbox.read_line, write=outbox.write))

    await _run_to_completion(inbox, outbox)
    assert len(agent.history) == 2

    resp = await _command(
        inbox, outbox, {"id": "n", "type": "new_session", "path": str(taken)}
    )

    assert resp["success"] is False
    assert len(agent.history) == 2, "history was wiped by a call that failed"
    assert server.session is live, "the log was closed by a call that failed"
    inbox.close()
    await task
    live.close()


# ---- protocol hygiene ----


async def test_unterminated_final_line_is_still_dispatched() -> None:
    """`readline()` returns buffered data without a newline at EOF, so a client
    that forgets the trailing \\n still gets its command run."""
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id":"r1","type":"get_state"}')  # no newline
    inbox.close()
    await task

    responses = [x for x in outbox.lines if x.get("type") == "response"]
    assert [r["id"] for r in responses] == ["r1"]
    assert responses[0]["success"] is True


async def test_blank_lines_are_skipped() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text("\n\n   \n")
    await inbox.feed_text('{"id":"r1","type":"get_state"}\n')
    inbox.close()
    await task

    assert [x["id"] for x in outbox.lines if x.get("type") == "response"] == ["r1"]


async def test_crlf_is_tolerated() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id":"r1","type":"get_state"}\r\n')
    inbox.close()
    await task

    assert [x["id"] for x in outbox.lines if x.get("type") == "response"] == ["r1"]


async def test_non_object_json_is_a_parse_error() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text("[1, 2, 3]\n")
    inbox.close()
    await task

    resp = outbox.lines[0]
    assert resp["command"] == "parse"
    assert resp["success"] is False
    assert "JSON object" in resp["error"]


async def test_eof_cancels_an_in_flight_run() -> None:
    client = Client()
    gate = asyncio.Event()
    install_gated(client, [say("part")], gate)
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id":"p","type":"prompt","message":"hi"}\n')
    await _wait_for(lambda: any(x.get("type") == "assistant_text_delta" for x in outbox.lines))

    inbox.close()
    await task  # `serve`'s finally cancels the run and awaits it

    errors = [x for x in outbox.lines if x.get("type") == "error"]
    assert [e["stop_reason"] for e in errors] == ["aborted"]


# ---- steering, follow-up, settled ----


def _tool_turn(call_id: str = "c1") -> list[Any]:
    return [
        tcall(index=0, id=call_id, name="wait", args="{}"),
        finish("tool_use"),
    ]


def _gated_tool(release: asyncio.Event) -> Any:
    @tool
    async def wait() -> str:
        """Block until the test lets go."""
        await release.wait()
        return "released"

    return wait


async def test_steer_lands_in_the_next_provider_call_of_the_same_run() -> None:
    client = Client()
    captured = install(
        client, [_tool_turn(), [say("done"), finish()]]
    )
    release = asyncio.Event()
    agent = Agent(client=client, model="m", tools=ToolRegistry([_gated_tool(release)]))
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id":"p","type":"prompt","message":"go"}\n')
    await _wait_for(lambda: any(x.get("type") == "tool_execution_start" for x in outbox.lines))

    resp = await _command(inbox, outbox, {"id": "s", "type": "steer", "message": "actually, stop"})
    assert resp["success"] is True
    assert resp["data"]["queue_id"]

    release.set()
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))

    # The steer reached the second call of the *same* run.
    roles = [m["role"] for m in captured[1]["messages"]]
    contents = [str(m.get("content")) for m in captured[1]["messages"]]
    assert "actually, stop" in contents[-1]
    assert roles[-1] == "user"
    inbox.close()
    await task


async def test_steer_never_splits_a_tool_group_on_the_wire() -> None:
    """A user message between an assistant's tool_calls and its results is
    rejected by providers, and `to_openai_messages` does not repair it."""
    client = Client()
    captured = install(
        client, [_tool_turn(), [say("done"), finish()]]
    )
    release = asyncio.Event()
    agent = Agent(client=client, model="m", tools=ToolRegistry([_gated_tool(release)]))
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id":"p","type":"prompt","message":"go"}\n')
    await _wait_for(lambda: any(x.get("type") == "tool_execution_start" for x in outbox.lines))
    await _command(inbox, outbox, {"id": "s", "type": "steer", "message": "steered"})
    release.set()
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))

    wire = captured[1]["messages"]
    roles = [m["role"] for m in wire]

    # Vacuously true if the steer never arrived, so assert it did first.
    assert any("steered" in str(m.get("content")) for m in wire), "the steer was not delivered"
    for i, role in enumerate(roles):
        if role == "tool":
            assert roles[i - 1] in ("assistant", "tool"), (
                f"tool at {i} follows {roles[i - 1]}: {roles}"
            )
    inbox.close()
    await task


async def test_steer_rearms_a_turn_that_answered_in_text() -> None:
    client = Client()
    captured = install(
        client,
        [
            [say("first answer"), finish()],
            [say("second answer"), finish()],
        ],
    )
    agent = Agent(client=client, model="m")
    steering = agent.steering
    _server, inbox, outbox, task = _start_server(agent)
    assert steering is None  # the server supplies one

    # Queue before the run starts: the pre-flight drain picks it up.
    await _command(inbox, outbox, {"id": "s", "type": "steer", "message": "and also this"})
    await inbox.feed_text('{"id":"p","type":"prompt","message":"go"}\n')
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))

    assert len(captured) == 1
    assert "and also this" in str(captured[0]["messages"][-1]["content"])
    inbox.close()
    await task


async def test_follow_up_runs_after_the_current_run() -> None:
    client = Client()
    captured = install(
        client,
        [
            [say("first"), finish()],
            [say("second"), finish()],
        ],
    )
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id":"p","type":"prompt","message":"one"}\n')
    await _command(inbox, outbox, {"id": "f", "type": "follow_up", "message": "two"})
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))

    ends = [x for x in outbox.lines if x.get("type") == "agent_end"]
    settled = [x for x in outbox.lines if x.get("type") == "agent_settled"]
    assert len(ends) == 2, "one agent_end per run"
    assert len(settled) == 1, "one agent_settled per client prompt"
    assert outbox.lines.index(settled[0]) > outbox.lines.index(ends[-1])
    assert len(captured) == 2
    inbox.close()
    await task


async def test_agent_settled_fires_when_the_run_errors() -> None:
    client = Client()
    install(client, [[say(""), finish("error")]])
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await _run_to_completion(inbox, outbox)
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))
    inbox.close()
    await task


async def test_agent_settled_fires_when_aborted() -> None:
    client = Client()
    gate = asyncio.Event()
    install_gated(client, [say("part")], gate)
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id":"p","type":"prompt","message":"hi"}\n')
    await _wait_for(lambda: any(x.get("type") == "assistant_text_delta" for x in outbox.lines))
    await _command(inbox, outbox, {"id": "a", "type": "abort"})
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))

    gate.set()
    inbox.close()
    await task


async def test_abort_clears_the_queues() -> None:
    """pi leaves them, so aborting silently starts a new run from what was
    pending. "Stop" should mean stop."""
    client = Client()
    gate = asyncio.Event()
    install_gated(client, [say("part")], gate)
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await inbox.feed_text('{"id":"p","type":"prompt","message":"hi"}\n')
    await _wait_for(lambda: any(x.get("type") == "assistant_text_delta" for x in outbox.lines))
    await _command(inbox, outbox, {"id": "f", "type": "follow_up", "message": "queued"})

    resp = await _command(inbox, outbox, {"id": "a", "type": "abort"})

    assert [d["content"] for d in resp["data"]["dropped"]] == ["queued"]
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))
    assert len([x for x in outbox.lines if x.get("type") == "agent_end"]) == 0
    gate.set()
    inbox.close()
    await task


async def test_steer_requires_a_message() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _command(inbox, outbox, {"id": "s", "type": "steer"})

    assert resp["success"] is False
    assert "message" in resp["error"]
    inbox.close()
    await task


# ---- get_commands ----


def _write_skill(directory: Path, name: str, *, extra: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: Does {name} things.\n{extra}---\n\n# {name}\n\nbody\n",
        encoding="utf-8",
    )
    return path


def _server_with_skills(agent: Agent, skills: list[Skill]) -> tuple[Any, _Inbox, _Outbox, Any]:
    server = RpcServer(agent, skills=skills)
    inbox, outbox = _Inbox(), _Outbox()
    task = asyncio.create_task(server.serve(read_line=inbox.read_line, write=outbox.write))
    return server, inbox, outbox, task


async def _commands(inbox: _Inbox, outbox: _Outbox) -> list[dict[str, Any]]:
    resp = await _command(inbox, outbox, {"id": "gc", "type": "get_commands"})
    assert resp["success"] is True
    return resp["data"]["commands"]


async def test_builtins_are_listed_without_any_skills() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    commands = await _commands(inbox, outbox)

    names = {c["name"] for c in commands}
    assert "compact" in names and "abort" in names and "set_model" in names
    assert all(c["source"] == "builtin" for c in commands)
    inbox.close()
    await task


async def test_every_entry_carries_the_contract() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    for c in await _commands(inbox, outbox):
        assert set(c) >= {"name", "source", "invoke", "description", "parameters"}
        assert c["invoke"] in ("command", "prompt")
        assert c["description"]
        params = c["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert isinstance(params.get("properties", {}), dict)
        # Provenance is skills-only for now.
        assert ("source_info" in c) == (c["source"] == "skill")
    inbox.close()
    await task


async def test_zero_argument_commands_are_select_and_fire() -> None:
    """Empty `properties` is the signal a consumer uses to decide whether to
    collect input before invoking."""
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    by_name = {c["name"]: c for c in await _commands(inbox, outbox)}

    for name in ("compact", "clear_context", "abort"):
        assert by_name[name]["parameters"]["properties"] == {}, name
    inbox.close()
    await task


async def test_required_and_optional_arguments_are_distinguishable() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    by_name = {c["name"]: c for c in await _commands(inbox, outbox)}

    set_model = by_name["set_model"]["parameters"]
    assert set_model["required"] == ["model"]
    assert set_model["properties"]["model"]["type"] == "string"

    reload = by_name["reload"]["parameters"]
    assert "targets" in reload["properties"]
    assert "targets" not in reload.get("required", [])
    inbox.close()
    await task


async def test_set_model_stays_a_free_string_without_a_registry() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    by_name = {c["name"]: c for c in await _commands(inbox, outbox)}

    assert "enum" not in by_name["set_model"]["parameters"]["properties"]["model"]
    inbox.close()
    await task


async def test_a_registry_turns_set_model_into_an_enum() -> None:
    """So a client renders a picker with nothing hardcoded.

    Same projection `reload` does for its targets: the server already knows the
    answer, and a client that had to carry its own list would go stale.
    """
    agent = Agent(client=Client(registry=_registered(slow="svc", fast="svc")), model="fast")
    _server, inbox, outbox, task = _start_server(agent)

    by_name = {c["name"]: c for c in await _commands(inbox, outbox)}

    assert by_name["set_model"]["parameters"]["properties"]["model"]["enum"] == ["fast", "slow"]
    inbox.close()
    await task


async def test_listed_builtins_are_all_really_dispatchable() -> None:
    """Guards against a command being added to the registry but never wired,
    or renamed on one side only."""
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    commands = await _commands(inbox, outbox)
    inbox.close()
    await task

    source = Path(rpc_server.__file__).read_text(encoding="utf-8")
    for c in commands:
        if c["source"] == "builtin":
            assert f'case "{c["name"]}"' in source, f"{c['name']} is listed but not dispatched"


async def test_skills_are_listed_with_provenance(tmp_path: Path) -> None:
    path = _write_skill(tmp_path / "deploy", "deploy")
    skills = load_skills([tmp_path])
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_skills(agent, skills)

    entry = next(c for c in await _commands(inbox, outbox) if c["source"] == "skill")

    assert entry["name"] == "skill:deploy"
    assert entry["invoke"] == "prompt"
    assert entry["description"] == "Does deploy things."
    assert entry["source_info"]["path"] == str(path.resolve())
    inbox.close()
    await task


async def test_a_non_invocable_skill_is_listed_but_not_advertised(tmp_path: Path) -> None:
    """Hiding a skill from the model's catalogue is exactly the case where an
    explicit command is the only way to reach it — the two surfaces disagree on
    purpose."""
    _write_skill(tmp_path / "manual", "manual", extra="disable-model-invocation: true\n")
    skills = load_skills([tmp_path])
    assert skills_prompt(skills) == ""

    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_skills(agent, skills)

    names = {c["name"] for c in await _commands(inbox, outbox)}
    assert "skill:manual" in names
    inbox.close()
    await task


async def test_command_list_is_stable_across_calls(tmp_path: Path) -> None:
    _write_skill(tmp_path / "deploy", "deploy")
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_skills(agent, load_skills([tmp_path]))

    first = await _commands(inbox, outbox)
    outbox.lines.clear()
    second = await _commands(inbox, outbox)

    assert [c["name"] for c in first] == [c["name"] for c in second]
    inbox.close()
    await task


# ---- /skill: expansion ----


async def test_skill_command_expands_into_the_envelope(tmp_path: Path) -> None:
    _write_skill(tmp_path / "deploy", "deploy")
    client = Client()
    captured = install(client, [[say("ok"), finish()]])
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _server_with_skills(agent, load_skills([tmp_path]))

    await inbox.feed_text('{"id":"p","type":"prompt","message":"/skill:deploy"}\n')
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))

    sent = str(captured[0]["messages"][-1]["content"])
    assert sent.startswith('<skill name="deploy"')
    assert "body" in sent
    assert "/skill:deploy" not in sent
    inbox.close()
    await task


async def test_skill_command_appends_trailing_arguments(tmp_path: Path) -> None:
    _write_skill(tmp_path / "deploy", "deploy")
    client = Client()
    captured = install(client, [[say("ok"), finish()]])
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _server_with_skills(agent, load_skills([tmp_path]))

    await inbox.feed_text('{"id":"p","type":"prompt","message":"/skill:deploy to staging"}\n')
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))

    sent = str(captured[0]["messages"][-1]["content"])
    assert sent.endswith("</skill>\n\nto staging")
    inbox.close()
    await task


async def test_ordinary_text_is_untouched(tmp_path: Path) -> None:
    _write_skill(tmp_path / "deploy", "deploy")
    client = Client()
    captured = install(client, [[say("ok"), finish()]])
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _server_with_skills(agent, load_skills([tmp_path]))

    await inbox.feed_text('{"id":"p","type":"prompt","message":"what does a/b testing mean?"}\n')
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))

    assert str(captured[0]["messages"][-1]["content"]) == "what does a/b testing mean?"
    inbox.close()
    await task


async def test_unknown_skill_fails_and_starts_no_run(tmp_path: Path) -> None:
    _write_skill(tmp_path / "deploy", "deploy")
    agent = Agent(client=Client(), model="m")
    server, inbox, outbox, task = _server_with_skills(agent, load_skills([tmp_path]))

    resp = await _command(inbox, outbox, {"id": "p", "type": "prompt", "message": "/skill:nope"})

    assert resp["success"] is False
    assert "nope" in resp["error"]
    assert server._current_run is None, "a failed expansion must not start a run"
    inbox.close()
    await task


async def test_unknown_skill_in_steer_never_reaches_the_queue(tmp_path: Path) -> None:
    """The failure has to happen at enqueue, which is the point of resolving
    there: it reaches whoever queued it rather than surfacing mid-run."""
    _write_skill(tmp_path / "deploy", "deploy")
    agent = Agent(client=Client(), model="m")
    server, inbox, outbox, task = _server_with_skills(agent, load_skills([tmp_path]))

    resp = await _command(inbox, outbox, {"id": "s", "type": "steer", "message": "/skill:nope"})

    assert resp["success"] is False
    assert server.steering.snapshot() == {"steering": [], "follow_up": []}
    inbox.close()
    await task


async def test_queued_skill_resolves_at_enqueue_not_at_delivery(tmp_path: Path) -> None:
    """Editing the SKILL.md between queueing and delivery must not change what
    is delivered — the user gets what they asked for when they asked."""
    path = _write_skill(tmp_path / "deploy", "deploy")
    agent = Agent(client=Client(), model="m")
    server, inbox, outbox, task = _server_with_skills(agent, load_skills([tmp_path]))

    await _command(inbox, outbox, {"id": "f", "type": "follow_up", "message": "/skill:deploy"})
    path.write_text(
        "---\nname: deploy\ndescription: d\n---\n\nCOMPLETELY DIFFERENT\n", encoding="utf-8"
    )

    queued = server.steering.take_follow_up()
    assert queued is not None
    assert "body" in str(queued.message.content)
    assert "COMPLETELY DIFFERENT" not in str(queued.message.content)
    inbox.close()
    await task


# ---- reload ----


_NOTES_EXT = '''
from midge.tools import tool


@tool
async def notes(text: str) -> str:
    """Record a note."""
    return text
'''

# `read` is what gates the skills catalogue, so a fake one is the lever the
# coupling tests need. Its behaviour is irrelevant; only its name is.
_READ_EXT = '''
from midge.tools import tool


@tool
async def read(path: str) -> str:
    """Read a file."""
    return path
'''


def _write_ext(directory: Path, stem: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.py"
    path.write_text(body, encoding="utf-8")
    return path


def _server_with_sources(
    agent: Agent,
    *,
    extension_sources: list[Path] | None = None,
    skill_sources: list[Path] | None = None,
) -> tuple[RpcServer, _Inbox, _Outbox, asyncio.Task[None]]:
    """Construct the server the way `cli.py` does: load, then hand over the
    same source lists so `reload` repeats exactly that call."""
    profiles = ProfileSet()
    if extension_sources is not None:
        registry, ext_prompt = load_extensions(
            extension_sources, hooks=agent.hooks, profiles=profiles
        )
        agent.tools = registry
        validate_profiles(
            profiles,
            tools=registry,
            hook_names=agent.hooks.source_names() if agent.hooks is not None else set(),
        )
    else:
        ext_prompt = ""
    skills = load_skills(skill_sources) if skill_sources is not None else []
    server = RpcServer(
        agent,
        base_prompt="BASE",
        extension_prompt=ext_prompt,
        skills=skills,
        profiles=profiles,
        extension_sources=extension_sources,
        skill_sources=skill_sources,
    )
    agent.system_prompt = server.controls.compose_prompt()
    inbox, outbox = _Inbox(), _Outbox()
    task = asyncio.create_task(server.serve(read_line=inbox.read_line, write=outbox.write))
    return server, inbox, outbox, task


async def _reload(inbox: _Inbox, outbox: _Outbox, **kw: Any) -> dict[str, Any]:
    cmd_id = f"rl{len(outbox.lines)}"
    return await _command(inbox, outbox, {"id": cmd_id, "type": "reload", **kw})


async def _command_names(inbox: _Inbox, outbox: _Outbox) -> list[str]:
    """`_commands` reuses one id, so it re-reads its own first response. These
    tests enumerate twice around a reload and need a fresh id each time."""
    cmd_id = f"gc{len(outbox.lines)}"
    resp = await _command(inbox, outbox, {"id": cmd_id, "type": "get_commands"})
    assert resp["success"] is True
    return [c["name"] for c in resp["data"]["commands"]]


# --- profiles ---

_PROFILE_EXT = '''
from midge.profiles import Profile

REVIEWER = Profile(
    name="reviewer",
    description="Reviews recent work.",
    prompt="Assume it is wrong.",
    tools=("read",),
)
'''


async def test_get_profiles_returns_what_was_discovered(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    _write_ext(ext, "reader", _READ_EXT)
    path = _write_ext(ext, "reviewer", _PROFILE_EXT)
    agent = Agent(client=Client(), model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])

    resp = await _command(inbox, outbox, {"id": "1", "type": "get_profiles"})

    assert resp["success"] is True
    assert resp["data"]["profiles"] == [
        {
            "name": "reviewer",
            "description": "Reviews recent work.",
            "model": None,
            "tools": ["read"],
            "hooks": [],
            "prompt": "Assume it is wrong.",
            "source": str(path),
        }
    ]
    inbox.close()
    await task


async def test_get_profiles_is_empty_without_any(tmp_path: Path) -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_sources(agent)

    resp = await _command(inbox, outbox, {"id": "1", "type": "get_profiles"})

    assert resp["data"]["profiles"] == []
    inbox.close()
    await task


async def test_reload_picks_up_a_profile_written_after_startup(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    _write_ext(ext, "reader", _READ_EXT)
    agent = Agent(client=Client(), model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])

    _write_ext(ext, "reviewer", _PROFILE_EXT)
    resp = await _reload(inbox, outbox)

    assert resp["data"]["profiles"] == 1
    listed = await _command(inbox, outbox, {"id": "p2", "type": "get_profiles"})
    assert [p["name"] for p in listed["data"]["profiles"]] == ["reviewer"]
    inbox.close()
    await task


async def test_reload_drops_a_profile_whose_tool_went_away(tmp_path: Path) -> None:
    """The validation rule, seen through the one command that can change what
    exists after startup: the profile still declares `read`, but nothing
    provides it any more, so it stops loading rather than loading with less."""
    ext = tmp_path / "ext"
    _write_ext(ext, "reader", _READ_EXT)
    _write_ext(ext, "reviewer", _PROFILE_EXT)
    agent = Agent(client=Client(), model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])

    assert (await _reload(inbox, outbox))["data"]["profiles"] == 1

    (ext / "reader.py").unlink()
    resp = await _reload(inbox, outbox)

    assert resp["data"]["profiles"] == 0
    inbox.close()
    await task


# --- skills ---


async def test_reload_picks_up_a_skill_written_after_startup(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    _write_ext(ext, "reader", _READ_EXT)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_sources(
        agent, extension_sources=[ext], skill_sources=[skills_dir]
    )

    assert "skill:deploy" not in await _command_names(inbox, outbox)

    _write_skill(skills_dir / "deploy", "deploy")
    resp = await _reload(inbox, outbox, targets=["skills"])
    assert resp["success"] is True
    assert resp["data"]["skills"] == 1

    assert "skill:deploy" in await _command_names(inbox, outbox)
    assert "deploy" in (agent.system_prompt or ""), "the catalogue was not recomposed"
    inbox.close()
    await task


async def test_reload_drops_a_deleted_skill(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    _write_ext(ext, "reader", _READ_EXT)
    skills_dir = tmp_path / "skills"
    path = _write_skill(skills_dir / "deploy", "deploy")
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_sources(
        agent, extension_sources=[ext], skill_sources=[skills_dir]
    )
    assert "deploy" in (agent.system_prompt or "")

    path.unlink()
    await _reload(inbox, outbox, targets=["skills"])

    assert "skill:deploy" not in await _command_names(inbox, outbox)
    assert "deploy" not in (agent.system_prompt or "")
    inbox.close()
    await task


async def test_reload_keeps_a_base_prompt_set_over_rpc(tmp_path: Path) -> None:
    """`set_system_prompt` owns the base; reload replaces only what midge
    generates. Losing the operator's prompt to a re-scan would be silent."""
    ext = tmp_path / "ext"
    _write_ext(ext, "reader", _READ_EXT)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_sources(
        agent, extension_sources=[ext], skill_sources=[skills_dir]
    )

    await _command(
        inbox, outbox, {"id": "sp", "type": "set_system_prompt", "prompt": "YOU ARE A POET"}
    )
    _write_skill(skills_dir / "deploy", "deploy")
    await _reload(inbox, outbox)

    assert (agent.system_prompt or "").startswith("YOU ARE A POET")
    assert "deploy" in (agent.system_prompt or "")
    inbox.close()
    await task


# --- extensions ---


async def test_reload_picks_up_a_new_tool(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])
    assert "notes" not in agent.tools

    _write_ext(ext, "notes", _NOTES_EXT)
    resp = await _reload(inbox, outbox, targets=["extensions"])

    assert resp["success"] is True
    assert resp["data"]["tools"] == 1
    assert "notes" in agent.tools
    inbox.close()
    await task


async def test_reload_drops_a_deleted_extension(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    path = _write_ext(ext, "notes", _NOTES_EXT)
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])
    assert "notes" in agent.tools

    path.unlink()
    await _reload(inbox, outbox, targets=["extensions"])

    assert "notes" not in agent.tools
    inbox.close()
    await task


def _hook_ext(log: Path, *, block: bool) -> str:
    """An extension whose hook appends to `log` on every invocation.

    A file rather than a module-level counter: a reload re-imports under a new
    synthetic module name, so anything in-module resets and could not tell a
    second registration apart from a first.
    """
    return f'''
from pathlib import Path

from midge.hooks import ToolCallResult

LOG = Path({str(log)!r})


def register_hooks(hooks):
    def _seen(event, ctx=None):
        with LOG.open("a") as fh:
            fh.write("x")
        return ToolCallResult(block=True, reason="denied") if {block!r} else None

    hooks.on("tool_call", _seen)
'''


_CLEANUP_EXT = '''
from pathlib import Path

LOG = Path({log!r})


def register_hooks(hooks):
    hooks.add_cleanup(lambda: LOG.open("a").write("c"))
'''


async def _fire_tool_call(agent: Agent) -> Any:
    assert agent.hooks is not None
    return await agent.hooks.emit(ToolCallEvent(tool_call=ToolCall(id="t1", name="notes")))


async def test_a_hook_fires_once_after_repeated_reloads(tmp_path: Path) -> None:
    """The regression reload exists to avoid. Tools are *returned* as a fresh
    registry, so they swap cleanly; hooks are *mutated* into a shared `Hooks`,
    so without `clear()` every reload leaves another copy of every handler
    registered and a single tool call is judged N times."""
    ext = tmp_path / "ext"
    log = tmp_path / "hook.log"
    _write_ext(ext, "policy", _hook_ext(log, block=False))
    agent = Agent(client=Client(), model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])

    await _reload(inbox, outbox, targets=["extensions"])
    await _reload(inbox, outbox, targets=["extensions"])
    await _fire_tool_call(agent)

    assert log.read_text() == "x", "the handler was registered more than once"
    inbox.close()
    await task


async def test_reload_applies_an_edited_hook_policy(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    log = tmp_path / "hook.log"
    _write_ext(ext, "policy", _hook_ext(log, block=False))
    agent = Agent(client=Client(), model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])
    assert await _fire_tool_call(agent) is None

    _write_ext(ext, "policy", _hook_ext(log, block=True))
    await _reload(inbox, outbox, targets=["extensions"])

    result = await _fire_tool_call(agent)
    assert isinstance(result, ToolCallResult)
    assert result.block is True
    inbox.close()
    await task


async def test_reload_drops_the_hooks_of_a_deleted_extension(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    log = tmp_path / "hook.log"
    path = _write_ext(ext, "policy", _hook_ext(log, block=True))
    agent = Agent(client=Client(), model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])

    path.unlink()
    await _reload(inbox, outbox, targets=["extensions"])

    assert await _fire_tool_call(agent) is None, "a deleted extension still enforces policy"
    inbox.close()
    await task


async def test_reload_runs_an_extensions_cleanup(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    log = tmp_path / "cleanup.log"
    _write_ext(ext, "resource", _CLEANUP_EXT.format(log=str(log)))
    agent = Agent(client=Client(), model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])
    assert not log.exists()

    await _reload(inbox, outbox, targets=["extensions"])

    assert log.read_text() == "c", "unload did not run the extension's cleanup"
    inbox.close()
    await task


async def test_a_broken_extension_is_skipped_and_the_rest_load(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    _write_ext(ext, "notes", _NOTES_EXT)
    _write_ext(ext, "broken", "this is not python(")
    agent = Agent(client=Client(), model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])

    resp = await _reload(inbox, outbox, targets=["extensions"])

    assert resp["success"] is True
    assert "notes" in agent.tools
    inbox.close()
    await task


# --- the coupling between the two targets ---


async def test_reloading_extensions_can_remove_the_skills_catalogue(tmp_path: Path) -> None:
    """The one place the targets are not independent. The catalogue tells the
    model to open a `SKILL.md`, so it is gated on a `read` tool — and an
    extensions reload can take that tool away without any skill changing."""
    ext = tmp_path / "ext"
    reader = _write_ext(ext, "reader", _READ_EXT)
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir / "deploy", "deploy")
    agent = Agent(client=Client(), model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(
        agent, extension_sources=[ext], skill_sources=[skills_dir]
    )
    assert "deploy" in (agent.system_prompt or "")

    reader.unlink()
    await _reload(inbox, outbox, targets=["extensions"])
    assert "deploy" not in (agent.system_prompt or ""), (
        "the catalogue survived losing the tool it depends on"
    )

    _write_ext(ext, "reader", _READ_EXT)
    await _reload(inbox, outbox, targets=["extensions"])
    assert "deploy" in (agent.system_prompt or ""), "the catalogue did not come back"
    inbox.close()
    await task


# --- targets, refusal, validation ---


async def test_reloading_skills_leaves_the_tool_registry_untouched(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    _write_ext(ext, "notes", _NOTES_EXT)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    agent = Agent(client=Client(), model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(
        agent, extension_sources=[ext], skill_sources=[skills_dir]
    )
    before = agent.tools

    resp = await _reload(inbox, outbox, targets=["skills"])

    assert resp["data"]["targets"] == ["skills"]
    assert agent.tools is before, "a skills reload rebuilt the tool registry"
    inbox.close()
    await task


async def test_reload_is_refused_while_a_run_is_in_flight() -> None:
    client = Client()
    gate = asyncio.Event()
    install_gated(client, [say("part")], gate)
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[])
    before = agent.tools

    await inbox.feed_text('{"id": "p", "type": "prompt", "message": "go"}\n')
    await _wait_for(
        lambda: any(line.get("type") == "assistant_text_delta" for line in outbox.lines)
    )
    resp = await _reload(inbox, outbox, targets=["extensions"])

    assert resp["success"] is False
    assert "in flight" in resp["error"]
    assert agent.tools is before, "a refused reload still swapped the registry"

    gate.set()
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))
    inbox.close()
    await task


async def test_an_unknown_target_is_rejected(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    agent = Agent(client=Client(), model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])
    before = agent.tools

    resp = await _reload(inbox, outbox, targets=["bogus"])

    assert resp["success"] is False
    assert agent.tools is before, "a rejected reload still ran"
    inbox.close()
    await task


async def test_an_unconfigured_target_is_named_in_the_error() -> None:
    """`_start_server` wires no sources, which is how an embedder that built its
    own registry arrives here. Reloading it would silently replace that registry
    with an empty one."""
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _reload(inbox, outbox, targets=["extensions", "skills"])

    assert resp["success"] is False
    assert "extensions" in resp["error"] and "skills" in resp["error"]
    inbox.close()
    await task


async def test_the_bare_form_reloads_only_what_is_wired_up(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir / "deploy", "deploy")
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _server_with_sources(agent, skill_sources=[skills_dir])

    resp = await _reload(inbox, outbox)

    assert resp["success"] is True
    assert resp["data"]["targets"] == ["skills"]
    inbox.close()
    await task


async def test_reload_is_enumerated_with_a_targets_enum() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    commands = await _commands(inbox, outbox)
    entry = next(c for c in commands if c["name"] == "reload")

    assert entry["invoke"] == "command"
    schema = entry["parameters"]
    assert schema["additionalProperties"] is False
    # Optional, so a consumer can render it as select-and-fire or as a
    # multi-select without a second convention.
    assert "targets" not in schema.get("required", [])
    assert set(_enum_values(schema)) == {"skills", "extensions"}
    inbox.close()
    await task


def _enum_values(schema: dict[str, Any]) -> list[str]:
    """Pull the target enum out wherever pydantic put it ($defs or inline)."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("enum"), list):
                found.extend(node["enum"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return found


_SUBAGENT_EXT = '''
from midge.subagents import subagent
from tests.fakes import finish, install, install_gated, say, tcall


@subagent(description="Probe something.", prompt="You are a probe.", tools=())
async def probe(question: str) -> str:
    return question
'''


async def test_a_subagent_still_runs_after_a_reload(tmp_path: Path) -> None:
    """Re-import produces *new* `SubagentTool` instances, so the reload has to
    call `bind_subagents` again. Without it the fresh tool has no runtime and
    the model's first delegation fails."""
    ext = tmp_path / "ext"
    _write_ext(ext, "probe", _SUBAGENT_EXT)
    client = Client()
    install(client, [[say("child says hi"), finish()]])
    agent = Agent(client=client, model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])
    before = agent.tools.get("spawn_probe")

    await _reload(inbox, outbox, targets=["extensions"])

    after = agent.tools.get("spawn_probe")
    assert after is not before, "reload did not re-import the sub-agent tool"
    result = await agent.tools.invoke("spawn_probe", {"question": "why"}, call_id="c1")
    assert "child says hi" in str(result)
    inbox.close()
    await task


async def test_a_nested_agents_activity_reaches_the_wire(tmp_path: Path) -> None:
    """#51: a `spawn_*` call used to take ninety seconds, produce one paragraph
    and say nothing in between. The child's turns still stay out of the
    parent's context — that is what delegating is for — but not out of its
    observability."""
    ext = tmp_path / "ext"
    _write_ext(ext, "probe", _SUBAGENT_EXT)
    client = Client()
    install(
        client,
        [
            # The parent delegates, the child answers, the parent wraps up.
            [tcall(index=0, id="c1", name="spawn_probe", args='{"question":"why"}'),
             finish("tool_use")],
            [say("child says hi"), finish()],
            [say("all done"), finish()],
        ],
    )
    agent = Agent(client=client, model="m", hooks=Hooks())
    _server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])

    await _command(inbox, outbox, {"id": "p", "type": "prompt", "message": "go"})
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))

    nested = [x for x in outbox.lines if x.get("agent")]
    # The envelope rides as a sibling key, so a client that ignores it sees
    # exactly the frames it saw before.
    assert nested[0]["agent"] == {
        "agent": "probe",
        "agent_id": "c1",
        "parent_id": None,
        "depth": 1,
    }
    assert "type" in nested[0], "still an ordinary event, envelope aside"
    inbox.close()
    await task


async def test_top_level_events_carry_no_envelope() -> None:
    client = Client()
    install(client, [[say("hi"), finish()]])
    agent = Agent(client=client, model="m")
    _server, inbox, outbox, task = _start_server(agent)

    await _command(inbox, outbox, {"id": "p", "type": "prompt", "message": "go"})
    await _wait_for(lambda: any(x.get("type") == "agent_settled" for x in outbox.lines))

    events = [x for x in outbox.lines if x.get("type") != "response"]
    assert events, "the run produced events"
    assert all("agent" not in x for x in events)
    inbox.close()
    await task


async def test_a_subagent_follows_new_session_to_the_new_file(tmp_path: Path) -> None:
    """Sub-agents hold the parent session, so `new_session` has to re-bind them.
    Without it the runtime keeps the session just closed, and the next spawn
    writes to a closed file rather than beside the current transcript."""
    ext = tmp_path / "ext"
    _write_ext(ext, "probe", _SUBAGENT_EXT)
    client = Client()
    install(client, [[say("child says hi"), finish()]])
    agent = Agent(client=client, model="m", hooks=Hooks())

    first_path = tmp_path / "first.jsonl"
    first = Session.new(first_path, model="m")
    server, inbox, outbox, task = _server_with_sources(agent, extension_sources=[ext])
    server.session = first
    bind_subagents(agent.tools, client=client, model="m", hooks=agent.hooks, session=first)

    second = tmp_path / "second.jsonl"
    await _command(inbox, outbox, {"id": "n", "type": "new_session", "path": str(second)})
    await agent.tools.invoke("spawn_probe", {"question": "why"}, call_id="c1")

    (link,) = session_continuations(read_transcript(second)[1])
    assert (second.parent / link.path).exists()
    # And nothing landed in the transcript that was closed.
    assert session_continuations(read_transcript(first_path)[1]) == []
    inbox.close()
    await task


# ---- use_profile: the atomic switch (#67) ----
#
# The reason this is one command rather than documented guidance: a client
# hand-orchestrating a switch gets `success: true` from `set_system_prompt`
# while the entire previous toolset and every hook stay active.


@tool(description="Read a file.")
async def _p_read(path: str) -> str:
    return "contents"


@tool(description="Write a file.")
async def _p_write(path: str, text: str) -> str:
    return "ok"


_BUILDER = Profile(
    name="builder",
    description="Builds.",
    prompt="You are a builder.",
    model="gpt-4o",
    tools=("_p_read", "_p_write"),
)
_REVIEWER = Profile(
    name="reviewer",
    description="Reviews.",
    prompt="You are adversarial.",
    model="o3",
    tools=("_p_read",),
)


def _profile_server(
    session: Session | None, *, profiles: tuple[Profile, ...] = (_BUILDER, _REVIEWER), **kw: Any
) -> tuple[RpcServer, _Inbox, _Outbox, asyncio.Task[None]]:
    agent = Agent(
        client=Client(),
        model="start-model",
        tools=ToolRegistry([_p_read, _p_write]),
        system_prompt="base",
        hooks=Hooks(),
    )
    pset = ProfileSet()
    for p in profiles:
        pset.add(p)
    server = RpcServer(agent, session=session, profiles=pset, base_prompt="base", **kw)
    inbox, outbox = _Inbox(), _Outbox()
    task = asyncio.create_task(server.serve(read_line=inbox.read_line, write=outbox.write))
    return server, inbox, outbox, task


async def _use(inbox: _Inbox, outbox: _Outbox, name: str, **kw: Any) -> dict[str, Any]:
    cmd_id = f"up{len(outbox.lines)}"
    return await _command(
        inbox, outbox, {"id": cmd_id, "type": "use_profile", "name": name, **kw}
    )


async def test_use_profile_applies_every_dimension(tmp_path: Path) -> None:
    session = Session.new(tmp_path / "s.jsonl", model="start-model")
    server, inbox, outbox, task = _profile_server(session)

    resp = await _use(inbox, outbox, "reviewer")

    assert resp["success"] is True
    assert server.agent.model == "o3"
    assert "You are adversarial." in (server.agent.system_prompt or "")
    assert sorted(t.name for t in server.agent.tools) == ["_p_read"]
    inbox.close()
    await task


async def test_a_switch_that_cannot_resolve_changes_nothing(tmp_path: Path) -> None:
    """Every dimension or none. A partial switch reporting success is worse
    than no switch."""
    session = Session.new(tmp_path / "s.jsonl", model="start-model")
    server, inbox, outbox, task = _profile_server(session)

    resp = await _use(inbox, outbox, "nonexistent")

    assert resp["success"] is False
    assert server.agent.model == "start-model"
    assert sorted(t.name for t in server.agent.tools) == ["_p_read", "_p_write"]
    assert session.profile is None
    inbox.close()
    await task


async def test_switching_back_widens_the_toolset_again(tmp_path: Path) -> None:
    """The reason the full registry is retained separately: projecting from the
    already-narrowed set would make every switch a ratchet."""
    session = Session.new(tmp_path / "s.jsonl", model="start-model")
    server, inbox, outbox, task = _profile_server(session)

    await _use(inbox, outbox, "reviewer")
    assert sorted(t.name for t in server.agent.tools) == ["_p_read"]
    await _use(inbox, outbox, "builder")

    assert sorted(t.name for t in server.agent.tools) == ["_p_read", "_p_write"]
    inbox.close()
    await task


async def test_the_switch_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    session = Session.new(path, model="start-model")
    _server, inbox, outbox, task = _profile_server(session)

    await _use(inbox, outbox, "reviewer")

    # The applied model, not the declared one — they differ when a profile
    # names no model and keeps whatever is running.
    reloaded = Session.load(path)
    assert (reloaded.profile, reloaded.model) == ("reviewer", "o3")
    inbox.close()
    await task


async def test_use_profile_is_refused_mid_turn(tmp_path: Path) -> None:
    session = Session.new(tmp_path / "s.jsonl", model="start-model")
    server, inbox, outbox, task = _profile_server(session)

    pending = asyncio.create_task(asyncio.Event().wait())
    server._current_run = pending  # type: ignore[assignment]
    resp = await _use(inbox, outbox, "reviewer")

    assert resp["success"] is False
    assert "in flight" in resp["error"]
    assert server.agent.model == "start-model"
    pending.cancel()
    inbox.close()
    await task


async def test_fork_opens_a_linked_transcript(tmp_path: Path) -> None:
    root = tmp_path / "run.jsonl"
    session = Session.new(root, model="start-model")
    server, inbox, outbox, task = _profile_server(session)

    resp = await _use(inbox, outbox, "reviewer", transcript="fork")

    forked = Path(resp["data"]["session"])
    assert forked != root and forked.exists()
    assert server.session is not None and server.session.path == forked
    # Linked per #62, in both directions.
    assert Session.load(forked).header.origin == "profile"
    assert Session.load(forked).header.parent_session == str(root)
    (link,) = session_continuations(read_transcript(root)[1])
    assert (link.path, link.reason) == (forked.name, "profile")
    inbox.close()
    await task


async def test_fork_does_not_touch_history(tmp_path: Path) -> None:
    """ADR Decision 4: a switch changes which file the turns are written to,
    not what the agent still holds. A caller wanting a clean slate composes
    `clear_context` after."""
    session = Session.new(tmp_path / "run.jsonl", model="start-model")
    server, inbox, outbox, task = _profile_server(session)
    server.agent.history = [UserMessage(content="build the parser")]

    await _use(inbox, outbox, "reviewer", transcript="fork")

    assert [str(m.content) for m in server.agent.history] == ["build the parser"]
    inbox.close()
    await task


async def test_resume_last_returns_to_the_thread_it_left(tmp_path: Path) -> None:
    """The round trip the option exists for: build, fork to a reviewer so the
    build conversation cannot cloud the review, then pick the build thread back
    up where it was."""
    root = tmp_path / "run.jsonl"
    session = Session.new(root, model="start-model")
    session.append(UserMessage(content="build the parser"))
    server, inbox, outbox, task = _profile_server(session)

    await _use(inbox, outbox, "builder")
    await _use(inbox, outbox, "reviewer", transcript="fork")
    resp = await _use(inbox, outbox, "builder", transcript="resume_last")

    assert resp["data"]["transcript"] == "resume_last"
    assert Path(resp["data"]["session"]) == root
    assert [str(m.content) for m in server.agent.history] == ["build the parser"]
    assert sorted(t.name for t in server.agent.tools) == ["_p_read", "_p_write"]
    inbox.close()
    await task


async def test_resume_last_falls_back_when_there_is_nothing_to_resume(tmp_path: Path) -> None:
    """A profile's first use in a session is an ordinary first run, not an
    error — a client cannot know whether one has been used before without
    asking. The response says which happened."""
    session = Session.new(tmp_path / "run.jsonl", model="start-model")
    _server, inbox, outbox, task = _profile_server(session)

    resp = await _use(inbox, outbox, "reviewer", transcript="resume_last")

    assert resp["success"] is True
    assert (resp["data"]["requested"], resp["data"]["transcript"]) == ("resume_last", "fork")
    inbox.close()
    await task


async def test_the_configured_fallback_is_honoured(tmp_path: Path) -> None:
    session = Session.new(tmp_path / "run.jsonl", model="start-model")
    _server, inbox, outbox, task = _profile_server(session, resume_fallback="continue")

    resp = await _use(inbox, outbox, "reviewer", transcript="resume_last")

    assert resp["data"]["transcript"] == "continue"
    assert Path(resp["data"]["session"]).name == "run.jsonl"
    inbox.close()
    await task


async def test_a_thread_that_switched_away_is_not_a_candidate(tmp_path: Path) -> None:
    """ADR Decision 5 read backwards: which profile a transcript is under is the
    *last* one recorded in it."""
    session = Session.new(tmp_path / "run.jsonl", model="start-model")
    _server, inbox, outbox, task = _profile_server(session)

    # The root ends up under `reviewer`, so resuming `builder` finds nothing.
    await _use(inbox, outbox, "builder")
    await _use(inbox, outbox, "reviewer")
    resp = await _use(inbox, outbox, "builder", transcript="resume_last")

    assert resp["data"]["transcript"] == "fork"
    inbox.close()
    await task


async def test_a_subagent_transcript_is_never_resumed(tmp_path: Path) -> None:
    """They are delegations, not profile excursions. Resuming one as though it
    were a thread would be wrong rather than untidy."""
    root = tmp_path / "run.jsonl"
    session = Session.new(root, model="start-model")
    child = tmp_path / "run.explore-c1.jsonl"
    with Session.new(
        child, model="m", origin="subagent", parent_session=str(root)
    ) as c:
        c.set_profile(name="reviewer", model="o3", system_prompt="p")
    session.append_continued(path=child.name, reason="subagent")
    _server, inbox, outbox, task = _profile_server(session)

    resp = await _use(inbox, outbox, "reviewer", transcript="resume_last")

    assert Path(resp["data"]["session"]) != child
    inbox.close()
    await task


async def test_profile_names_are_an_enum_in_the_command_schema(tmp_path: Path) -> None:
    """Discovery is the only source of profiles, so an empty set is an empty
    enum rather than 'anything goes'. A name is never a free string."""
    _server, inbox, outbox, task = _profile_server(None)

    commands = await _commands(inbox, outbox)

    switch = next(c for c in commands if c["name"] == "use_profile")
    assert switch["parameters"]["properties"]["name"]["enum"] == ["builder", "reviewer"]
    inbox.close()
    await task


# ---- session naming ----


def _session_server(
    agent: Agent, session: Session | None
) -> tuple[RpcServer, _Inbox, _Outbox, asyncio.Task[None]]:
    server = RpcServer(agent, session=session)
    inbox, outbox = _Inbox(), _Outbox()
    task = asyncio.create_task(server.serve(read_line=inbox.read_line, write=outbox.write))
    return server, inbox, outbox, task


async def test_set_session_name_persists_and_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    session = Session.new(path, model="m")
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _session_server(agent, session)

    resp = await _command(
        inbox, outbox, {"id": "n", "type": "set_session_name", "name": "auth refactor"}
    )
    assert resp["success"] is True
    state = await _command(inbox, outbox, {"id": "s", "type": "get_state"})
    assert state["data"]["session_name"] == "auth refactor"

    inbox.close()
    await task
    session.close()

    assert Session.load(path).name == "auth refactor"


async def test_set_session_name_flattens_newlines(tmp_path: Path) -> None:
    """A newline would split one record across two JSONL lines and corrupt the
    file, so it is stripped rather than rejected."""
    path = tmp_path / "s.jsonl"
    session = Session.new(path, model="m")
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _session_server(agent, session)

    resp = await _command(
        inbox, outbox, {"id": "n", "type": "set_session_name", "name": "a\nb\tc"}
    )

    assert resp["data"]["name"] == "a b c"
    inbox.close()
    await task
    session.close()
    assert len(path.read_text().strip().splitlines()) == 2


async def test_set_session_name_without_a_session_fails() -> None:
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    resp = await _command(inbox, outbox, {"id": "n", "type": "set_session_name", "name": "x"})

    assert resp["success"] is False
    inbox.close()
    await task


async def test_an_empty_session_name_is_rejected(tmp_path: Path) -> None:
    session = Session.new(tmp_path / "s.jsonl", model="m")
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _session_server(agent, session)

    resp = await _command(inbox, outbox, {"id": "n", "type": "set_session_name", "name": "   "})

    assert resp["success"] is False
    inbox.close()
    await task
    session.close()


async def test_set_session_name_is_enumerated_and_round_trips(tmp_path: Path) -> None:
    """Drive it purely from its own schema, with nothing about the command
    hardcoded — the property `get_commands` exists to provide."""
    path = tmp_path / "s.jsonl"
    session = Session.new(path, model="m")
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _session_server(agent, session)

    entry = next(c for c in await _commands(inbox, outbox) if c["name"] == "set_session_name")
    schema = entry["parameters"]
    filled = {k: "from the schema" for k in schema["required"]}
    resp = await _command(inbox, outbox, {"id": "x", "type": entry["name"], **filled})

    assert resp["success"] is True
    inbox.close()
    await task
    session.close()
    assert Session.load(path).name == "from the schema"


# --- a changed identity survives a resume (#57) ----------------------------


async def test_set_model_survives_a_resume(tmp_path: Path) -> None:
    """The defect: `success: true`, then silently reverted on the next resume.

    Nothing on the wire said so, and `set_model` is in BUILTIN_COMMANDS, so
    `get_commands` advertised it to clients with no way to discover the caveat.
    """
    path = tmp_path / "s.jsonl"
    session = Session.new(path, model="original")
    agent = Agent(client=Client(), model="original")
    _server, inbox, outbox, task = _session_server(agent, session)

    resp = await _command(inbox, outbox, {"id": "m", "type": "set_model", "model": "switched"})
    assert resp["success"] is True

    inbox.close()
    await task
    session.close()

    assert Session.load(path).model == "switched"


async def test_set_system_prompt_survives_a_resume(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    session = Session.new(path, model="m", system_prompt="You are a coding assistant.")
    agent = Agent(client=Client(), model="m")
    server = RpcServer(
        agent, session=session, base_prompt="You are a coding assistant.", extension_prompt=GENERATED
    )
    agent.system_prompt = server.controls.compose_prompt()
    inbox, outbox = _Inbox(), _Outbox()
    task = asyncio.create_task(server.serve(read_line=inbox.read_line, write=outbox.write))

    resp = await _command(
        inbox,
        outbox,
        {"id": "p", "type": "set_system_prompt", "prompt": "You are an adversarial reviewer."},
    )
    assert resp["success"] is True

    inbox.close()
    await task
    session.close()

    stored = Session.load(path).system_prompt
    # The base, not the composed prompt: storing the composed string would
    # duplicate the generated half on every resume, and `cli.py` re-appends it.
    assert stored == "You are an adversarial reviewer."
    assert GENERATED not in (stored or "")


async def test_setting_either_without_a_session_still_succeeds() -> None:
    # Persistence is optional; the commands are not gated on it.
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)

    assert (await _command(inbox, outbox, {"id": "1", "type": "set_model", "model": "x"}))[
        "success"
    ] is True
    assert (
        await _command(inbox, outbox, {"id": "2", "type": "set_system_prompt", "prompt": "y"})
    )["success"] is True
    assert agent.model == "x"

    inbox.close()
    await task


async def test_the_response_says_whether_the_change_is_durable(tmp_path: Path) -> None:
    """#57's complaint was that nothing on the wire said so.

    A client cannot see whether a session is attached, and both commands report
    success either way, so the answer has to be in the response.
    """
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _start_server(agent)
    resp = await _command(inbox, outbox, {"id": "1", "type": "set_model", "model": "x"})
    assert resp["data"]["durable"] is False
    inbox.close()
    await task

    session = Session.new(tmp_path / "s.jsonl", model="m")
    agent = Agent(client=Client(), model="m")
    _server, inbox, outbox, task = _session_server(agent, session)
    resp = await _command(inbox, outbox, {"id": "2", "type": "set_model", "model": "x"})
    assert resp["data"]["durable"] is True
    resp = await _command(inbox, outbox, {"id": "3", "type": "set_system_prompt", "prompt": "p"})
    assert resp["data"]["durable"] is True
    inbox.close()
    await task
    session.close()
