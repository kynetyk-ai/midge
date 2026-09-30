from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Static, TextArea

from midge.agent import Agent
from midge.client import Client
from midge.commands import Controls
from midge.config import ProviderConfig
from midge.messages import TextContent, UserMessage
from midge.persistence import Session
from midge.profiles import Profile, ProfileSet
from midge.providers import ModelRegistry
from midge.tui.app import (
    AssistantBubble,
    MidgeCommands,
    PiApp,
    Sidebar,
    StatusLine,
    UserBubble,
)
from tests.fakes import finish, install, install_gated, say, whole_call


def _build_agent(turns: list[list[Any]]) -> Agent:
    client = Client()
    install(client, turns)
    return Agent(client=client, model="m")


def _app(turns: list[list[Any]], **kw: Any) -> PiApp:
    return PiApp(Controls(_build_agent(turns), **kw))


@pytest.mark.asyncio
async def test_app_launches_and_exits_cleanly() -> None:
    agent = _build_agent([])
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.is_running
        await pilot.press("ctrl+d")
    # If we got here without hanging, the app exited cleanly.


@pytest.mark.asyncio
async def test_submit_creates_user_and_assistant_bubbles() -> None:
    agent = _build_agent(
        [[say("hello"), say(" there"), finish()]]
    )
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        # Type into the input area and submit via Enter
        input_widget = app.query_one("#input")
        input_widget.text = "hi"  # type: ignore[attr-defined]
        await pilot.press("enter")

        # Wait for the run worker to complete
        await app.workers.wait_for_complete()
        await pilot.pause()

        user_bubbles = list(app.query(UserBubble))
        assistant_bubbles = list(app.query(AssistantBubble))
        assert len(user_bubbles) == 1
        assert len(assistant_bubbles) == 1
        # The assistant bubble accumulated the streamed deltas
        assert assistant_bubbles[0]._text == "hello there"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_escape_clears_input() -> None:
    agent = _build_agent([])
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input")
        input_widget.text = "draft text"  # type: ignore[attr-defined]
        await pilot.press("escape")
        assert input_widget.text == ""  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["ctrl+o", "alt+enter"])
async def test_newline_keys_insert_without_submitting(key: str) -> None:
    agent = _build_agent([])
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input")
        input_widget.text = "line one"  # type: ignore[attr-defined]
        await pilot.press(key)
        await pilot.pause()

        # Newline got inserted; nothing was submitted
        assert "\n" in input_widget.text  # type: ignore[attr-defined]
        assert len(list(app.query(UserBubble))) == 0


@pytest.mark.asyncio
async def test_enter_on_empty_input_is_noop() -> None:
    agent = _build_agent([])
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert len(list(app.query(UserBubble))) == 0


@pytest.mark.asyncio
async def test_multiline_prompt_submits_via_enter_with_newlines_intact() -> None:
    agent = _build_agent(
        [[say("ok"), finish()]]
    )
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input")
        input_widget.text = "line one\nline two"  # type: ignore[attr-defined]
        await pilot.press("enter")

        await app.workers.wait_for_complete()
        await pilot.pause()

        user_bubbles = list(app.query(UserBubble))
        assert len(user_bubbles) == 1
        # The agent's history should carry both lines
        history = agent.history
        assert len(history) >= 1
        first = history[0]
        assert hasattr(first, "content")
        assert "line one" in str(first.content)
        assert "line two" in str(first.content)


# --- the command surface --------------------------------------------------


def _status(app: PiApp) -> list[str]:
    return [str(s.visual) for s in app.query(StatusLine)]


async def _settle(pilot: Any) -> None:
    """`_on_submit` is async and mounts its result, so one pause is not enough."""
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_a_slash_command_runs_instead_of_prompting() -> None:
    agent = _build_agent([])
    agent.history.extend([UserMessage(content="a"), UserMessage(content="b")])
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/clear_context"
        await pilot.press("enter")
        await _settle(pilot)

        assert agent.history == []
        assert any("cleared 2 messages" in s for s in _status(app))
        assert not list(app.query(UserBubble))


@pytest.mark.asyncio
async def test_a_message_that_merely_starts_with_a_slash_is_a_prompt() -> None:
    # The reason only known names intercept: this is a sentence about a path,
    # and rejecting it as an unknown command would make the input box refuse
    # ordinary English.
    app = _app([[say("ok"), finish()]])
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/etc/hosts is missing"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert [str(b.visual) for b in app.query(UserBubble)] == ["/etc/hosts is missing"]


@pytest.mark.asyncio
async def test_a_command_argument_reaches_the_operation() -> None:
    app = _app([])
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/set_model gpt-4o"
        await pilot.press("enter")
        await _settle(pilot)

        assert app.agent.model == "gpt-4o"
        assert any("model is now gpt-4o" in s for s in _status(app))


@pytest.mark.asyncio
async def test_a_refusal_is_shown_rather_than_raised() -> None:
    app = _app([])
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/set_session_name whatever"
        await pilot.press("enter")
        await _settle(pilot)

        assert any("no session" in s for s in _status(app))
        assert app.is_running


@pytest.mark.asyncio
async def test_the_palette_offers_every_argument_free_builtin() -> None:
    app = _app([])
    async with app.run_test() as pilot:
        await pilot.pause()
        offered = {d for d, _desc, _arg in MidgeCommands(app.screen)._entries()}

        # Free-text commands are absent: a palette entry has nowhere for a path.
        assert {"abort", "compact", "clear_context", "reload"} <= offered
        assert not any(o.startswith("new_session") for o in offered)
        # And `set_model` only once a `[models]` table says what the values are
        # — offering it bare would fire it with no model at all.
        assert not any(o.startswith("set_model") for o in offered)


@pytest.mark.asyncio
async def test_switches_live_in_the_drawer_not_the_command_list() -> None:
    # #115: `set_model` in both places read as two different commands. With a
    # registry the model is a choice, and the drawer is where choices are.
    registry = ModelRegistry(
        models={"a-model": "p", "b-model": "p"},
        providers={"p": ProviderConfig(kind="openai")},
    )
    agent = Agent(client=Client(registry=registry), model="a-model")
    app = PiApp(Controls(agent, profiles=_profiles("builder")))
    async with app.run_test() as pilot:
        await pilot.pause()
        offered = {d for d, _desc, _arg in MidgeCommands(app.screen)._entries()}
        assert not any(o.startswith(("set_model", "use_profile")) for o in offered)

        await pilot.press("ctrl+b")
        await _settle(pilot)
        ids = {i for _label, i in _options(app)}
        assert {"set_model\x00a-model", "set_model\x00b-model", "use_profile\x00builder"} <= ids


@pytest.mark.asyncio
async def test_a_slash_still_switches_the_model() -> None:
    registry = ModelRegistry(
        models={"a-model": "p", "b-model": "p"},
        providers={"p": ProviderConfig(kind="openai")},
    )
    agent = Agent(client=Client(registry=registry), model="a-model")
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/set_model b-model"
        await pilot.press("enter")
        await _settle(pilot)
        assert agent.model == "b-model"


@pytest.mark.asyncio
async def test_typing_mid_turn_queues_rather_than_cancelling() -> None:
    # The old path cancelled the running worker and awaited its teardown so two
    # writers could not interleave into `Agent.history`. Steering removes the
    # second writer instead, and keeps the work already done.
    gate = asyncio.Event()
    client = Client()
    provider = install_gated(client, [say("first"), finish()], gate)
    assert provider is not None
    agent = Agent(client=client, model="m")
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "start"
        await pilot.press("enter")
        await _settle(pilot)
        assert app.busy()

        app.query_one("#input", TextArea).text = "and also this"
        await pilot.press("enter")
        await _settle(pilot)

        assert app.busy(), "the turn must still be running"
        assert agent.steering is not None and agent.steering.pending()
        assert any("queued: and also this" in s for s in _status(app))

        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


@pytest.mark.asyncio
async def test_compact_is_refused_while_a_turn_is_running() -> None:
    # Compaction awaits a provider call before reassigning `agent.history`, so a
    # turn running across that window has its appended messages dropped.
    gate = asyncio.Event()
    client = Client()
    install_gated(client, [say("first"), finish()], gate)
    agent = Agent(client=client, model="m")
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "start"
        await pilot.press("enter")
        await _settle(pilot)

        await app.invoke("compact", None)
        await _settle(pilot)
        assert any("run is in flight" in s for s in _status(app))

        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_command_needing_an_argument_refuses_without_one() -> None:
    app = _app([])
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/set_model"
        await pilot.press("enter")
        await _settle(pilot)

        assert app.agent.model == "m", "an empty argument must not be applied"
        assert any("needs an argument" in s for s in _status(app))


# --- the drawer -----------------------------------------------------------


def _registry(*models: str) -> ModelRegistry:
    return ModelRegistry(
        models=dict.fromkeys(models, "p"), providers={"p": ProviderConfig(kind="openai")}
    )


def _profiles(*names: str) -> ProfileSet:
    profiles = ProfileSet()
    for name in names:
        profiles.add(
            Profile(name=name, prompt="p", tools=(), description=name),
            path=Path(f"{name}.py"),
        )
    return profiles


def _sections(app: PiApp) -> list[str]:
    """Headings and hints: the lines of the drawer that cannot be chosen."""
    bar = app.query_one("#sidebar", Sidebar)
    return [str(o.prompt).strip() for o in bar.options if o.disabled]


def _options(app: PiApp) -> list[tuple[str, str | None]]:
    bar = app.query_one("#sidebar", Sidebar)
    return [(str(o.prompt), o.id) for o in bar.options if not o.disabled]


def _highlighted(app: PiApp) -> str | None:
    bar = app.query_one("#sidebar", Sidebar)
    assert bar.highlighted is not None
    return bar.get_option_at_index(bar.highlighted).id


@pytest.mark.asyncio
async def test_the_drawer_is_closed_until_asked_for() -> None:
    app = _app([])
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#sidebar", Sidebar).has_class("hidden")


@pytest.mark.asyncio
async def test_the_drawer_lists_what_the_agent_could_be(tmp_path: Path) -> None:
    with Session.new(tmp_path / "a.jsonl", model="gpt-4o") as s:
        s.set_name("auth refactor")
    agent = Agent(client=Client(registry=_registry("gpt-4o", "haiku")), model="gpt-4o")
    app = PiApp(Controls(agent, profiles=_profiles("builder"), session_dir=tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("ctrl+b")
        await _settle(pilot)

        assert not app.query_one("#sidebar", Sidebar).has_class("hidden")
        assert _sections(app) == ["model", "profiles", "sessions"]
        labels = [prompt for prompt, _id in _options(app)]
        assert "  auth refactor" in labels
        assert "  builder" in labels
        # The one you are on is marked, which is the thing a modal cannot do.
        assert "● gpt-4o" in labels
        assert "  haiku" in labels


@pytest.mark.asyncio
async def test_a_section_with_nothing_to_offer_says_why() -> None:
    # #115: a vanished model section read as "the model cannot be switched
    # here". It is shown, with the current model and how to list others.
    app = _app([])
    async with app.run_test() as pilot:
        await pilot.press("ctrl+b")
        await _settle(pilot)

        assert _sections(app) == [
            "model",
            "m — list models under [models] in config to switch here",
            "profiles",
            "none — declare a Profile in an extension",
            "sessions",
            "none saved yet",
        ]
        assert _options(app) == []


@pytest.mark.asyncio
async def test_choosing_applies_it_and_closes(tmp_path: Path) -> None:
    agent = Agent(client=Client(), model="m")
    app = PiApp(Controls(agent, profiles=_profiles("builder", "reviewer")))
    async with app.run_test() as pilot:
        await pilot.press("ctrl+b")
        await _settle(pilot)
        await pilot.press("down", "enter")
        await _settle(pilot)

        assert app.controls.profile == "reviewer"
        assert app.query_one("#sidebar", Sidebar).has_class("hidden")
        assert any("profile reviewer" in s for s in _status(app))


@pytest.mark.asyncio
async def test_arrows_run_through_every_section(tmp_path: Path) -> None:
    Session.new(tmp_path / "a.jsonl", model="gpt-4o").close()
    agent = Agent(client=Client(registry=_registry("gpt-4o", "haiku")), model="gpt-4o")
    app = PiApp(Controls(agent, profiles=_profiles("builder"), session_dir=tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("ctrl+b")
        await _settle(pilot)
        seen = [_highlighted(app)]
        for _ in range(3):
            await pilot.press("down")
            seen.append(_highlighted(app))

        assert seen == [
            "set_model\x00gpt-4o",
            "set_model\x00haiku",
            "use_profile\x00builder",
            f"open_session\x00{tmp_path / 'a.jsonl'}",
        ]


@pytest.mark.asyncio
async def test_tab_jumps_to_the_next_section_with_a_choice(tmp_path: Path) -> None:
    # No profiles, so Tab from the models passes over that section's hint.
    Session.new(tmp_path / "a.jsonl", model="gpt-4o").close()
    agent = Agent(client=Client(registry=_registry("gpt-4o", "haiku")), model="gpt-4o")
    app = PiApp(Controls(agent, profiles=_profiles(), session_dir=tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("ctrl+b")
        await _settle(pilot)
        await pilot.press("down", "tab")
        assert _highlighted(app) == f"open_session\x00{tmp_path / 'a.jsonl'}"

        await pilot.press("shift+tab")
        assert _highlighted(app) == "set_model\x00gpt-4o"
        assert app.focused is app.query_one("#sidebar", Sidebar)


@pytest.mark.asyncio
async def test_the_drawer_reflects_a_switch_made_elsewhere() -> None:
    # Rebuilt on every open rather than kept in sync: a `/set_model` typed into
    # the input box has to move the mark.
    agent = Agent(client=Client(registry=_registry("gpt-4o", "haiku")), model="gpt-4o")
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "/set_model haiku"
        await pilot.press("enter")
        await _settle(pilot)
        await pilot.press("ctrl+b")
        await _settle(pilot)

        assert "● haiku" in [prompt for prompt, _id in _options(app)]


@pytest.mark.asyncio
async def test_escape_closes_the_drawer_before_it_clears_the_draft() -> None:
    app = PiApp(Controls(_build_agent([]), profiles=_profiles("builder")))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "a draft"
        await pilot.press("ctrl+b")
        await _settle(pilot)
        await pilot.press("escape")
        await _settle(pilot)

        assert app.query_one("#sidebar", Sidebar).has_class("hidden")
        assert app.query_one("#input", TextArea).text == "a draft"


@pytest.mark.asyncio
async def test_ctrl_c_mid_turn_keeps_the_turn_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    gate = asyncio.Event()
    client = Client()
    install_gated(client, [say("partial")], gate)
    agent = Agent(client=client, model="m")
    session = Session.new(path, model="m")
    app = PiApp(Controls(agent, session=session))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "keep me"
        await pilot.press("enter")
        await _settle(pilot)
        assert app.busy()

        await pilot.press("ctrl+c")
        await app.workers.wait_for_complete()
        await _settle(pilot)

        assert any("[interrupted]" in s for s in _status(app))
    session.close()

    restored = Session.load(path).messages
    assert restored and restored[0].content == "keep me"


@pytest.mark.asyncio
async def test_square_brackets_render_literally_and_the_turn_survives() -> None:
    # #110: `[` in a prompt, a tool argument, a result or a reply is text, not
    # markup — an unbalanced one used to raise MarkupError and kill the turn.
    from midge.tools import ToolRegistry, tool

    @tool
    async def echo(text: str) -> str:
        """Echo."""
        return f"[red]{text}"

    client = Client()
    install(
        client,
        [
            [*whole_call("echo", '{"text": "[bold]x[/bold] and [unclosed"}'), finish("tool_use")],
            [say("[OK] done [unclosed"), finish()],
        ],
    )
    agent = Agent(client=client, model="m", tools=ToolRegistry([echo]))
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "say [dim]hi"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await _settle(pilot)

        texts = [str(w.visual) for w in app.query(Static)]
        assert any("say [dim]hi" in t for t in texts)
        assert any("[OK]" in t and "[red][bold]x[/bold] and [unclosed" in t for t in texts)
        assert any("[OK] done [unclosed" in t for t in texts)
        assert not any("turn failed" in t for t in _status(app))


@pytest.mark.asyncio
async def test_long_tool_arguments_are_truncated_in_the_log() -> None:
    from midge.tools import ToolRegistry, tool

    release = asyncio.Event()

    @tool
    async def write_it(content: str) -> str:
        """Write."""
        await release.wait()  # hold the bubble at "running...", arguments on show
        return "ok"

    client = Client()
    body = "y" * 5000
    install(
        client,
        [
            [*whole_call("write_it", f'{{"content": "{body}"}}'), finish("tool_use")],
            [say("done"), finish()],
        ],
    )
    agent = Agent(client=client, model="m", tools=ToolRegistry([write_it]))
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "go"
        await pilot.press("enter")
        await _settle(pilot)
        running = [str(w.visual) for w in app.query(Static) if "running" in str(w.visual)]
        assert running and all(len(t) < 400 for t in running)
        release.set()
        await app.workers.wait_for_complete()


@pytest.mark.asyncio
async def test_ctrl_c_drops_the_queue_like_abort() -> None:
    # #112: a message typed mid-turn and then abandoned with Ctrl+C used to
    # ride along with the next prompt.
    gate = asyncio.Event()
    client = Client()
    install_gated(client, [say("first")], gate)
    agent = Agent(client=client, model="m")
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        app.query_one("#input", TextArea).text = "start"
        await pilot.press("enter")
        await _settle(pilot)
        app.query_one("#input", TextArea).text = "never mind this"
        await pilot.press("enter")
        await _settle(pilot)
        assert agent.steering is not None and agent.steering.pending()

        await pilot.press("ctrl+c")
        await app.workers.wait_for_complete()
        await _settle(pilot)

        assert not agent.steering.pending()
        assert any("interrupted; 1 queued message(s) dropped" in s for s in _status(app))


@pytest.mark.asyncio
async def test_ctrl_c_when_idle_says_nothing() -> None:
    app = _app([])
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await _settle(pilot)
        assert _status(app) == []


# --- approval: the TUI asks before anything that is not read-only ----------


def _approval_app(calls: list[tuple[str, str]], ran: list[str], **kw: Any) -> PiApp:
    from midge.hooks import Hooks
    from midge.tools import ToolRegistry, tool

    @tool
    async def poke(x: str) -> str:
        ran.append(f"poke {x}")
        return "poked"

    @tool(read_only=True)
    async def peek(x: str) -> str:
        ran.append(f"peek {x}")
        return "seen"

    client = Client()
    first = [
        chunk
        for i, (name, x) in enumerate(calls)
        for chunk in whole_call(name, f'{{"x": "{x}"}}', index=i, id=f"c{i}")
    ]
    install(client, [[*first, finish("tool_use")], [say("done"), finish()]])
    agent = Agent(client=client, model="m", tools=ToolRegistry([poke, peek]), hooks=Hooks())
    return PiApp(Controls(agent), approve_tools=kw.get("approve_tools", True))


async def _submit(app: PiApp, pilot: Any, text: str = "go") -> None:
    app.query_one("#input", TextArea).text = text
    await pilot.press("enter")


async def _until_asked(app: PiApp, pilot: Any) -> Any:
    from midge.tui.app import ApprovalScreen

    for _ in range(200):
        if isinstance(app.screen, ApprovalScreen):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError("the approval prompt never appeared")


def _results(app: PiApp) -> list[str]:
    from midge.messages import ToolResultMessage

    return [
        m.content[0].text
        for m in app.agent.history
        if isinstance(m, ToolResultMessage) and isinstance(m.content[0], TextContent)
    ]


@pytest.mark.asyncio
async def test_approval_y_runs_the_tool() -> None:
    ran: list[str] = []
    app = _approval_app([("poke", "1")], ran)
    async with app.run_test() as pilot:
        await _submit(app, pilot)
        screen = await _until_asked(app, pilot)
        assert screen.call.name == "poke"
        await pilot.press("y")
        await app.workers.wait_for_complete()
    assert ran == ["poke 1"]


@pytest.mark.asyncio
async def test_approval_n_tells_the_model_and_runs_nothing() -> None:
    ran: list[str] = []
    app = _approval_app([("poke", "1")], ran)
    async with app.run_test() as pilot:
        await _submit(app, pilot)
        await _until_asked(app, pilot)
        await pilot.press("n")
        await app.workers.wait_for_complete()
    assert ran == []
    assert _results(app) == ["Denied by the user."]


@pytest.mark.asyncio
async def test_approval_a_stops_asking_about_that_tool() -> None:
    # Two mutating calls in one message: asked once, in order, then trusted.
    ran: list[str] = []
    app = _approval_app([("poke", "1"), ("poke", "2")], ran)
    async with app.run_test() as pilot:
        await _submit(app, pilot)
        await _until_asked(app, pilot)
        await pilot.press("a")
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert ran == ["poke 1", "poke 2"]


@pytest.mark.asyncio
async def test_read_only_tools_are_never_asked_about() -> None:
    from midge.tui.app import ApprovalScreen

    ran: list[str] = []
    app = _approval_app([("peek", "1")], ran)
    async with app.run_test() as pilot:
        await _submit(app, pilot)
        await app.workers.wait_for_complete()
        assert not isinstance(app.screen, ApprovalScreen)
    assert ran == ["peek 1"]


@pytest.mark.asyncio
async def test_approval_off_never_asks() -> None:
    ran: list[str] = []
    app = _approval_app([("poke", "1")], ran, approve_tools=False)
    async with app.run_test() as pilot:
        await _submit(app, pilot)
        await app.workers.wait_for_complete()
    assert ran == ["poke 1"]


@pytest.mark.asyncio
async def test_approval_survives_an_extension_reload() -> None:
    ran: list[str] = []
    app = _approval_app([("poke", "1")], ran)
    async with app.run_test() as pilot:
        assert app.agent.hooks is not None
        await app.agent.hooks.unload_extensions()
        await _submit(app, pilot)
        await _until_asked(app, pilot)
        await pilot.press("n")
        await app.workers.wait_for_complete()
    assert ran == []


@pytest.mark.asyncio
async def test_ctrl_c_while_asked_ends_the_turn_and_the_question() -> None:
    from midge.tui.app import ApprovalScreen

    ran: list[str] = []
    app = _approval_app([("poke", "1")], ran)
    async with app.run_test() as pilot:
        await _submit(app, pilot)
        await _until_asked(app, pilot)
        await pilot.press("ctrl+c")
        await app.workers.wait_for_complete()
        await _settle(pilot)
        assert not isinstance(app.screen, ApprovalScreen)
        assert any("interrupted" in s for s in _status(app))
    assert ran == []


@pytest.mark.asyncio
async def test_a_startup_notice_is_on_screen() -> None:
    app = PiApp(Controls(_build_agent([])), notices=["No API key for openai: set OPENAI_API_KEY"])
    async with app.run_test() as pilot:
        await _settle(pilot)
        assert "[No API key for openai: set OPENAI_API_KEY]" in _status(app)


@pytest.mark.asyncio
async def test_token_counts_add_up_in_the_header() -> None:
    from midge.messages import AssistantMessage, Usage
    from tests.fakes import tokens

    agent = _build_agent([[say("hi"), tokens(input=1200, output=30, cached=1000), finish()]])
    # A resumed session brings its own spend with it.
    agent.history = [
        UserMessage(content="before"),
        AssistantMessage(content=[TextContent(text="x")], usage=Usage(input=800, output=20)),
    ]
    app = PiApp(Controls(agent))
    async with app.run_test() as pilot:
        await _settle(pilot)
        assert app.sub_title == "in 800 · out 20 · cached 0"
        await _submit(app, pilot)
        await app.workers.wait_for_complete()
        await _settle(pilot)
        assert app.sub_title == "in 2.0k · out 50 · cached 1.0k"
