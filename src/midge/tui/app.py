"""Textual TUI for interactive use.

A minimum-viable shell:
- Header (model + status), scrolling conversation log, multi-line input box, Footer.
- User text becomes a UserBubble. Assistant text streams into a single
  AssistantBubble that grows in place. Tool calls/executions render as
  inline status cards.
- Enter submits; Ctrl+O inserts a newline (Alt+Enter too, where the terminal
  sends it). Ctrl+J also submits (most terminals send it for Ctrl+Enter).
- Ctrl+C interrupts the current turn and drops anything queued behind it —
  the same as `/abort`.
- Ctrl+D quits.
- Esc clears the input draft.

No custom widgets, no markdown rendering during streaming (markdown re-parse
on every token is laggy).

The control surface — compact, clear, reload, switch model, switch profile — is
not reimplemented here. It is `midge.commands.Controls`, the same object the RPC
server drives, and both surfaces enumerate the same `BUILTIN_COMMANDS`. That is
what keeps them from drifting: a command added to that table appears in the
palette without this file being edited.

Verbs and nouns get different surfaces (#115). **Ctrl+P** opens the command
list: things you *do* — compact, clear, reload, abort, a skill. **Ctrl+B** opens
the drawer: what the agent *is* — its session, profile and model — with the
current one marked, and the alternatives to switch to. A switch appears in the
drawer only; offering it in both read as two different commands. **A leading
slash** in the input box reaches any command, and is the way to pass one an
argument (`/set_model gpt-4o`, `/new_session path`).

A slash only intercepts when the word after it is a command anyone could invoke.
`/etc/hosts is missing` is a sentence, and treating it as a failed command would
make the input box refuse ordinary English about paths.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Vertical, VerticalScroll
from textual.logging import TextualHandler
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

from midge.agent import Agent, AgentEnd, SteeringQueue, ToolExecutionEnd, ToolExecutionStart
from midge.client import (
    Done,
    Error,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)
from midge.commands import (
    BUILTIN_COMMANDS,
    CompactionEnd,
    CompactionStart,
    Controls,
    Refused,
)
from midge.hooks import Hooks, ToolCallEvent, ToolCallResult
from midge.messages import AssistantMessage, TextContent, ToolCall, Usage
from midge.persistence import Session

_logger = logging.getLogger(__name__)


class _SubmitTextArea(TextArea):
    """TextArea where Enter submits and Ctrl+O inserts a newline.

    Ctrl+O is the advertised key because every terminal sends it as its own
    byte. Alt+Enter is kept, but only works where the terminal sends Option as
    Meta — macOS Terminal and iTerm do not by default, and send a bare Enter
    instead, which submitted a half-typed prompt (#113). Ctrl+J is kept as a
    fallback for terminals that don't deliver a clean Enter keysym.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "submit", "Submit", show=False, priority=True),
        Binding("ctrl+o", "newline", "Newline", show=True, priority=True),
        Binding("alt+enter", "newline", "Newline", show=False, priority=True),
        Binding("ctrl+j", "submit", "Submit", show=False),
    ]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def action_submit(self) -> None:
        text = self.text.strip()
        if not text:
            return
        self.post_message(self.Submitted(text))
        self.clear()

    def action_newline(self) -> None:
        self.insert("\n")


# Every bubble is `markup=False`, for the reason `StatusLine` gives: what lands
# in one is someone else's text. A model's `[bold]` vanished, a tool's `[OK]`
# tag never rendered, and an unbalanced `[` raised MarkupError and killed the
# turn (#110).


class UserBubble(Static):
    DEFAULT_CSS = """
    UserBubble {
        padding: 0 1;
        margin: 1 0 0 0;
        background: $boost;
        border-left: thick $primary;
    }
    """

    def __init__(self, content: str) -> None:
        super().__init__(content, markup=False)


class AssistantBubble(Static):
    DEFAULT_CSS = """
    AssistantBubble {
        padding: 0 1;
        margin: 1 0 0 0;
    }
    """

    def __init__(self) -> None:
        super().__init__("", markup=False)
        self._text = ""

    def append(self, delta: str) -> None:
        self._text += delta
        self.update(self._text)


class ToolCallBubble(Static):
    DEFAULT_CSS = """
    ToolCallBubble {
        padding: 0 1;
        margin: 1 0 0 0;
        background: $surface;
        border-left: thick $accent;
    }
    ToolCallBubble.error {
        border-left: thick $error;
    }
    """

    def __init__(self, content: str) -> None:
        super().__init__(content, markup=False)


def _preview(text: str, limit: int = 200) -> str:
    # A `write` carries the whole file as an argument; the log is not the place.
    return text if len(text) <= limit else text[:limit] + "…"


class StatusLine(Static):
    """A one-line note about what just happened, rendered literally.

    `markup=False` because everything that lands here is someone else's text —
    a model id, a path, a provider's error message — and Textual reads square
    brackets as style tags. `[model is now gpt-4o]` parses as a tag and renders
    as nothing at all, which is the worst way for a status line to fail.
    """

    DEFAULT_CSS = """
    StatusLine { color: $text-muted; padding: 0 1; }
    """

    def __init__(self, content: str) -> None:
        super().__init__(content, markup=False)


class MidgeCommands(Provider):
    """The palette, over the same table the RPC server enumerates."""

    @property
    def _app(self) -> PiApp:
        app = self.app
        assert isinstance(app, PiApp)
        return app

    def _entries(self) -> list[tuple[str, str, str | None]]:
        """(display, description, argument) for everything the list offers.

        Read off the schema rather than a list kept here: a built-in that needs
        no argument is a thing you do, so it is offered; one that needs an
        argument is either a switch — which the drawer offers, with the current
        choice marked — or free text like a path, which has to be typed after a
        slash. Skills are verbs too.
        """
        app = self._app
        out: list[tuple[str, str, str | None]] = []
        for command in BUILTIN_COMMANDS:
            if not app.controls.builtin_schema(command).get("required"):
                out.append((command.name, command.description, None))
        for skill in app.controls.skills:
            out.append((f"skill:{skill.name}", skill.description, None))
        return out

    async def discover(self) -> Hits:
        for display, description, argument in self._entries():
            yield DiscoveryHit(
                display, partial(self._app.invoke, display, argument), help=description
            )

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for display, description, argument in self._entries():
            if (score := matcher.match(display)) > 0:
                yield Hit(
                    score,
                    matcher.highlight(display),
                    partial(self._app.invoke, display, argument),
                    help=description,
                )


class Sidebar(VerticalScroll):
    """What the agent *is*, and what it could be instead.

    The palette is verbs — compact, clear, reload, things you do once. This is
    nouns: the session, the profile and the model are each a set of named
    alternatives with exactly one current, which is a different shape and wants
    a different affordance. `Controls` already reports all three that way, so
    this renders rather than computes.

    It also answers a question the TUI could not previously answer at all. A
    modal shows you the list only while you are choosing and then vanishes;
    docked, it shows what you are on. Before this the only visible state was the
    model in the title bar.

    Every section is always shown. One with nothing to offer says why in a
    line, rather than vanishing: without a `[models]` table midge cannot know
    what the alternatives to the current model are, and a missing section read
    as "the model cannot be switched here" (#115) rather than "none are listed".
    """

    DEFAULT_CSS = """
    Sidebar { dock: left; width: 34; border-right: solid $accent; padding: 0 1; }
    Sidebar.hidden { display: none; }
    Sidebar > .section { color: $text-muted; text-style: bold; padding: 1 0 0 0; }
    Sidebar > OptionList { border: none; background: transparent; height: auto; }
    Sidebar > .hint { color: $text-muted; padding: 0 0 0 2; }
    """

    def rebuild(self, controls: Controls) -> None:
        """Read the current state and redraw. Called on every open.

        Cheaper than staying in sync: sessions appear on disk from other
        processes, and a profile switch changes two sections at once.
        """
        self.remove_children()
        for title, options, empty in (
            ("sessions", _session_options(controls), "none saved yet"),
            (
                "profiles",
                _profile_options(controls),
                "none — declare a Profile in an extension",
            ),
            (
                "model",
                _model_options(controls),
                f"{controls.agent.model} — list models under [models] in config to switch here",
            ),
        ):
            self.mount(Static(title, classes="section"))
            if options:
                self.mount(OptionList(*options))
            else:
                self.mount(Static(empty, classes="hint", markup=False))


# NUL, because the argument half is a filesystem path and everything printable
# can appear in one.
_ARG = "\x00"
_CURRENT = "\u25cf"


def _mark(label: str, *, current: bool) -> str:
    return f"{_CURRENT if current else ' '} {label}"


def _session_options(controls: Controls) -> list[Option]:
    return [
        Option(
            _mark(s["name"] or Path(s["path"]).name, current=bool(s["current"])),
            id=f"open_session{_ARG}{s['path']}",
        )
        for s in controls.session_list()
    ]


def _profile_options(controls: Controls) -> list[Option]:
    return [
        Option(_mark(name, current=name == controls.profile), id=f"use_profile{_ARG}{name}")
        for name in controls.profiles.names()
    ]


def _model_options(controls: Controls) -> list[Option]:
    registry = controls.agent.client.registry
    return [
        Option(_mark(name, current=name == controls.agent.model), id=f"set_model{_ARG}{name}")
        for name in (registry.names() if registry else ())
    ]


class ApprovalScreen(ModalScreen[str]):
    """Asks whether one tool call may run. Dismisses with `once`, `always` or
    `deny`; Escape is a deny, so the way out of a prompt is never a yes."""

    DEFAULT_CSS = """
    ApprovalScreen { align: center middle; }
    ApprovalScreen > Vertical {
        width: 80%; height: auto; max-height: 80%;
        padding: 1 2; background: $panel; border: thick $warning;
    }
    ApprovalScreen .choices { margin-top: 1; color: $text-muted; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "choose('once')", "Allow once"),
        Binding("a", "choose('always')", "Allow this tool for the session"),
        Binding("n", "choose('deny')", "Deny"),
        Binding("escape", "choose('deny')", "Deny", show=False),
    ]

    def __init__(self, call: ToolCall) -> None:
        super().__init__()
        self.call = call

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(f"Allow {self.call.name}?", markup=False)
            yield Static(_preview(str(self.call.arguments), 1500), markup=False)
            yield Static(
                "[y] allow once   [a] always allow this tool   [n] deny",
                markup=False,
                classes="choices",
            )

    def action_choose(self, answer: str) -> None:
        self.dismiss(answer)


def _tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _compaction_note(ev: CompactionEnd) -> str:
    if ev.error is not None:
        return f"[compaction failed: {ev.error}]"
    if ev.cut_index is None:
        return "[compaction: nothing to summarize yet]"
    return (
        f"[compacted: {ev.cut_index} messages summarized; "
        f"history is now {ev.message_count} messages]"
    )


class PiApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #log { height: 1fr; padding: 0 1; }
    #input { height: 6; border-top: solid $accent; }
    """

    COMMANDS: ClassVar[set[type[Provider] | Callable[[], type[Provider]]]] = {MidgeCommands}

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "interrupt", "Interrupt", priority=True),
        # Declared here so Textual does not add its own, labelled "palette": to
        # someone using midge it is the list of commands, and the drawer
        # (Ctrl+B) is where switching happens.
        Binding("ctrl+p", "command_palette", "Commands", show=False, priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Switch to…", priority=True),
        Binding("escape", "clear_input", "Clear input"),
    ]

    def __init__(
        self,
        controls: Controls,
        *,
        approve_tools: bool = False,
        notices: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.controls = controls
        # Shown in the log on mount: things the operator must see before the
        # first prompt, like a missing API key, which a log file would bury.
        self._notices = list(notices)
        # Tokens this session has cost, shown in the header. Seeded from the
        # history a resumed session brings, then added to after every turn.
        self._usage = Usage()
        for m in controls.agent.history:
            self._add_usage(m)
        controls.runner = self
        # Tools the person has said to stop asking about, for this process.
        self._always_allowed: set[str] = set()
        self._approval_lock = asyncio.Lock()
        if approve_tools:
            self._register_approval()
        # Steering has to be a real queue before a turn starts, or a message
        # typed mid-turn has nowhere to land.
        if controls.agent.steering is None:
            controls.agent.steering = SteeringQueue()
        self._current_assistant: AssistantBubble | None = None
        self._tool_bubbles: dict[str, ToolCallBubble] = {}
        self._current_worker: Worker[None] | None = None
        self._dropped_on_interrupt = 0
        self.title = f"midge · {controls.agent.model}"

    def _register_approval(self) -> None:
        """Ask before any tool that is not read-only runs.

        A `tool_call` handler, registered here by the front-end rather than by
        an extension, for three reasons. It has no `source`, so a profile
        cannot switch it off and `reload` does not remove it. It sees every
        call a sub-agent makes too, because a child's hooks reach this same
        registry. And it exists only in the TUI: RPC never registers it,
        because nobody is there to answer.

        It is registered after the extensions loaded at startup, so it runs
        after their handlers — an extension that blocks a call does so before
        anyone is asked. (After a `reload`, re-imported extensions register
        behind it and are asked second; a block still blocks.)
        """
        if self.controls.agent.hooks is None:
            self.controls.agent.hooks = Hooks()
        self.controls.agent.hooks.on("tool_call", self._approve)

    async def _approve(self, event: ToolCallEvent, ctx: Any) -> ToolCallResult | None:
        call = event.tool_call
        tool = self.controls.discovered_tools.get(call.name) or self.agent.tools.get(call.name)
        if tool is not None and tool.read_only:
            return None
        # Decisions for one message are gathered concurrently, so two prompts
        # could otherwise be pushed at once; one at a time, in call order.
        async with self._approval_lock:
            if call.name in self._always_allowed:
                return None
            answer = await self._ask(call)
        if answer == "always":
            self._always_allowed.add(call.name)
        if answer in ("once", "always"):
            return None
        _logger.info("tool_denied_by_user tool=%s id=%s", call.name, call.id)
        return ToolCallResult(block=True, reason="Denied by the user.")

    async def _ask(self, call: ToolCall) -> str:
        answered: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        screen = ApprovalScreen(call)

        def settle(answer: str | None) -> None:
            if not answered.done():
                answered.set_result(answer or "deny")

        await self.push_screen(screen, callback=settle)
        try:
            return await answered
        except asyncio.CancelledError:
            # Ctrl+C while the question is open: the turn is gone, so the
            # question goes with it rather than waiting on a dead worker.
            if screen.is_active:
                screen.dismiss(None)
            raise

    def _add_usage(self, m: Any) -> None:
        if isinstance(m, AssistantMessage) and m.usage is not None:
            u = self._usage
            self._usage = Usage(
                input=u.input + m.usage.input,
                output=u.output + m.usage.output,
                cached=u.cached + m.usage.cached,
            )

    def _show_usage(self) -> None:
        u = self._usage
        if u.input or u.output:
            self.sub_title = (
                f"in {_tokens(u.input)} · out {_tokens(u.output)} · cached {_tokens(u.cached)}"
            )

    @property
    def agent(self) -> Agent:
        return self.controls.agent

    @property
    def session(self) -> Session | None:
        return self.controls.session

    @property
    def compaction_keep_recent(self) -> int:
        return self.controls.compaction_keep_recent

    # `Controls.Runner`: what "a run is in flight" means here is a worker.
    def busy(self) -> bool:
        return self._current_worker is not None and self._current_worker.state is WorkerState.RUNNING

    def cancel(self) -> None:
        if self._current_worker is not None:
            self._current_worker.cancel()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Sidebar(id="sidebar", classes="hidden")
        yield VerticalScroll(id="log")
        yield _SubmitTextArea(id="input", soft_wrap=True)
        yield Footer()

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar", Sidebar)
        if sidebar.has_class("hidden"):
            sidebar.rebuild(self.controls)
            sidebar.remove_class("hidden")
            # Focus follows, or the arrow keys would still be editing the draft.
            lists = sidebar.query(OptionList)
            if lists:
                lists.first().focus()
        else:
            self._close_sidebar()

    def _close_sidebar(self) -> None:
        self.query_one("#sidebar", Sidebar).add_class("hidden")
        self.query_one("#input", _SubmitTextArea).focus()

    @on(OptionList.OptionSelected)
    async def _on_sidebar_choice(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is None:
            return
        name, _, argument = event.option_id.partition(_ARG)
        # Closed first: the choice changes what the panel would show, and
        # leaving it open displaying the old state is worse than dismissing it.
        self._close_sidebar()
        await self.invoke(name, argument)

    def on_mount(self) -> None:
        self.query_one("#input", _SubmitTextArea).focus()
        log = self.query_one("#log", VerticalScroll)
        for notice in self._notices:
            log.mount(StatusLine(f"[{notice}]"))
        if self.agent.history:
            log.mount(StatusLine(f"[resumed: {len(self.agent.history)} prior messages]"))
        self._show_usage()

    def _as_command(self, text: str) -> tuple[str, str] | None:
        """`(name, argument)` if this is a command, else None.

        Only a name from the table intercepts. Anything else starting with a
        slash is a sentence about a path, and refusing it as an unknown command
        would make the input box reject ordinary English.
        """
        if not text.startswith("/"):
            return None
        name, _, argument = text[1:].partition(" ")
        known = {c.name for c in BUILTIN_COMMANDS}
        if name in known:
            return name, argument.strip()
        if name.startswith("skill:") and any(
            s.name == name[len("skill:") :] for s in self.controls.skills
        ):
            return name, argument.strip()
        return None

    @on(_SubmitTextArea.Submitted)
    async def _on_submit(self, message: _SubmitTextArea.Submitted) -> None:
        command = self._as_command(message.value)
        if command is not None:
            await self.invoke(f"{command[0]} {command[1]}".strip(), None)
            return

        if self.busy():
            # Queued, not cancel-and-restart. The old path cancelled the worker
            # and awaited its teardown so two writers could not interleave into
            # `Agent.history` — steering removes the second writer instead, and
            # keeps the work already done in the turn.
            steering = self.agent.steering
            assert steering is not None, "steering is created in __init__"
            log = self.query_one("#log", VerticalScroll)
            try:
                expanded = self.controls.expand(message.value)
            except (KeyError, OSError) as e:
                # A mistyped `/skill:` mid-turn used to raise out of the
                # handler; RPC refuses it, and so does this.
                await log.mount(StatusLine(f"[not queued: {e}]"))
                log.scroll_end(animate=False)
                return
            steering.steer(expanded)
            await log.mount(StatusLine(f"[queued: {message.value}]"))
            log.scroll_end(animate=False)
            return

        self._current_worker = self.run_worker(
            self._run_turn(message.value),
            exclusive=True,
            exit_on_error=False,
        )

    async def invoke(self, display: str, argument: str | None) -> None:
        """Run one command and say what happened.

        `display` is `name` or `name argument` — the palette builds it that way
        so an entry is a thing you can read, and the slash path produces the
        same string. A skill is not a command: it expands to a prompt, which is
        the `invoke: "prompt"` the command table already declares.
        """
        name, _, inline = display.partition(" ")
        value = argument if argument is not None else inline.strip()
        log = self.query_one("#log", VerticalScroll)

        if name.startswith("skill:"):
            self._current_worker = self.run_worker(
                self._run_turn(f"/{display}"), exclusive=True, exit_on_error=False
            )
            return

        try:
            result = await self._dispatch(name, value)
        except Refused as e:
            await log.mount(StatusLine(f"[{name}: {e}]"))
        except (OSError, ValueError) as e:
            _logger.exception("tui_command_failed command=%s", name)
            await log.mount(StatusLine(f"[{name} failed: {e}]"))
        else:
            await log.mount(StatusLine(f"[{result}]"))
            self.title = f"midge · {self.agent.model}"
        log.scroll_end(animate=False)

    async def _dispatch(self, name: str, value: str) -> str:
        """Call the operation and describe it. Refusals propagate."""
        c = self.controls
        # A command whose argument is required cannot run without one. The
        # palette never offers these bare; a slash can be typed bare.
        if not value and name in {
            "set_model",
            "use_profile",
            "set_session_name",
            "set_system_prompt",
            "new_session",
            "open_session",
        }:
            raise Refused(f"{name} needs an argument")
        match name:
            case "abort":
                dropped = c.abort()
                return f"aborted; {len(dropped)} queued message(s) dropped"
            case "compact":
                data = await c.compact()
                if data["summary"] is None:
                    return "nothing to compact"
                return (
                    f"compacted: {data['cut_index']} messages summarized; "
                    f"history is now {data['message_count']} messages"
                )
            case "clear_context":
                return f"cleared {c.clear_context()['cleared']} messages"
            case "reload":
                data = await c.reload()
                return (
                    f"reloaded {', '.join(data['targets']) or 'nothing'} — "
                    f"{data['tools']} tools, {data['skills']} skills"
                )
            case "set_model":
                c.set_model(value)
                return f"model is now {value}"
            case "use_profile":
                data = c.use_profile(value)
                return f"profile {data['profile']} — {len(data['tools'])} tools, {data['model']}"
            case "set_session_name":
                return f"session named {c.set_session_name(value)['name']}"
            case "set_system_prompt":
                c.set_system_prompt(value)
                return "system prompt replaced"
            case "new_session":
                return f"recording to {c.new_session(Path(value))['session']}"
            case "open_session":
                data = c.open_session(Path(value))
                return f"opened {data['session']} — {data['messages']} messages"
        raise Refused(f"unknown command {name!r}")

    async def _run_turn(self, prompt: str) -> None:
        log = self.query_one("#log", VerticalScroll)
        await log.mount(UserBubble(prompt))
        log.scroll_end(animate=False)
        self._current_assistant = None
        self._tool_bubbles = {}

        # Persisting the turn — including one interrupted or broken mid-render —
        # and compacting after it are `Controls.run_turn`'s job, shared with RPC.
        # What is left here is saying what happened.
        compacting = False
        try:
            async with contextlib.aclosing(self.controls.run_turn(prompt)) as events:
                async for ev in events:
                    if isinstance(ev, CompactionStart):
                        compacting = True
                        await log.mount(StatusLine("[compacting context...]"))
                    elif isinstance(ev, CompactionEnd):
                        compacting = False
                        await log.mount(StatusLine(_compaction_note(ev)))
                    else:
                        self._handle_event(ev, log)
                    log.scroll_end(animate=False)
        except asyncio.CancelledError:
            note = "compaction interrupted" if compacting else "interrupted"
            dropped, self._dropped_on_interrupt = self._dropped_on_interrupt, 0
            if dropped:
                note += f"; {dropped} queued message(s) dropped"
            await log.mount(StatusLine(f"[{note}]"))
            log.scroll_end(animate=False)
            raise

    def _handle_event(self, ev: StreamEvent | Any, log: VerticalScroll) -> None:
        if isinstance(ev, TextDelta):
            if self._current_assistant is None:
                self._current_assistant = AssistantBubble()
                log.mount(self._current_assistant)
            self._current_assistant.append(ev.delta)
        elif isinstance(ev, Done | Error):
            self._current_assistant = None
            if isinstance(ev, Error):
                log.mount(StatusLine(f"[error: {ev.message.error_message}]"))
        elif isinstance(ev, ToolCallStart):
            tc = ev.partial.content[ev.content_index]
            assert isinstance(tc, ToolCall)
            bubble = ToolCallBubble(f"⚙ {tc.name}(...)")
            self._tool_bubbles[tc.id] = bubble
            log.mount(bubble)
        elif isinstance(ev, ToolCallEnd):
            bubble = self._tool_bubbles.get(ev.tool_call.id)
            if bubble is not None:
                bubble.update(f"⚙ {ev.tool_call.name}({_preview(str(ev.tool_call.arguments))})")
        elif isinstance(ev, ToolExecutionStart):
            bubble = self._tool_bubbles.get(ev.tool_call.id)
            if bubble is not None:
                args = _preview(str(ev.tool_call.arguments))
                bubble.update(f"⚙ {ev.tool_call.name}({args}) — running...")
        elif isinstance(ev, ToolExecutionEnd):
            bubble = self._tool_bubbles.get(ev.tool_call.id)
            if bubble is None:
                return
            text = ""
            if ev.result.content and isinstance(ev.result.content[0], TextContent):
                text = ev.result.content[0].text
            preview = _preview(text)
            tag = "ERR" if ev.result.is_error else "OK"
            if ev.result.is_error:
                bubble.add_class("error")
            bubble.update(f"⚙ {ev.tool_call.name} → [{tag}] {preview}")
        elif isinstance(ev, AgentEnd):
            self._current_assistant = None
            for m in ev.new_messages:
                self._add_usage(m)
            self._show_usage()

    @on(Worker.StateChanged)
    def _on_worker_state(self, event: Worker.StateChanged) -> None:
        # `exit_on_error=False` otherwise swallows the exception into Textual's
        # internal log and the turn just stops mid-render with no explanation.
        if event.state is not WorkerState.ERROR:
            return
        err = event.worker.error
        _logger.error("tui_turn_failed error=%s", type(err).__name__, exc_info=err)
        log = self.query_one("#log", VerticalScroll)
        log.mount(StatusLine(f"[turn failed: {type(err).__name__}: {err}]"))
        log.scroll_end(animate=False)

    def action_interrupt(self) -> None:
        # Through `Controls.abort`, like `/abort`, so the queue is cleared
        # before the cancel. Cancelling the worker alone left a message typed
        # mid-turn to ride along, unseen, with the next prompt (#112).
        if self.busy():
            self._dropped_on_interrupt = len(self.controls.abort())

    def action_clear_input(self) -> None:
        if not self.query_one("#sidebar", Sidebar).has_class("hidden"):
            self._close_sidebar()
            return
        self.query_one("#input", _SubmitTextArea).clear()


def tui_log_handler(log_file: Path | None = None) -> logging.Handler | None:
    """A log handler safe to install before `App.run()`.

    `logging.StreamHandler` binds `sys.stderr` at construction, so it writes
    straight past Textual's `redirect_stderr` and shreds the display.
    `TextualHandler` resolves the active app per record instead — but it routes
    to the devtools console, visible only under `textual console`, so a log file
    is the practical way to read these. `None` hands that case back to
    `logs.configure`, which opens the file itself.
    """
    return None if log_file else TextualHandler()


def run_tui(
    controls: Controls, *, approve_tools: bool = False, notices: Sequence[str] = ()
) -> None:
    PiApp(controls, approve_tools=approve_tools, notices=notices).run()
