"""Session save/load: append-only JSONL persistence for agent runs.

File layout (one JSON object per line):

    {"type":"header","version":2,"created_at":"...","model":"...","system_prompt":...}
    {"type":"message","data":{...}}
    {"type":"session_info","name":"auth refactor","timestamp":...}
    {"type":"model_change","model":"...","timestamp":...}
    {"type":"identity","system_prompt":"...","timestamp":...}
    {"type":"compaction","summary":"...","cut_index":N,"timestamp":...}
    {"type":"clear","cut_index":N,"timestamp":...}
    {"type":"profile","name":"...","model":"...","system_prompt":"...","timestamp":...}
    {"type":"continued","path":"...","reason":"subagent","timestamp":...}
    {"type":"message","data":{...}}

The first line MUST be the header. A file written by a *newer* build is
rejected; an older one loads, because every entry type it can contain is still
understood.

Anything mutable is expressed as an appended record and replayed on load —
last write wins. That is why renaming a session does not rewrite the header:
a rewrite would break the invariant that makes truncated-tail recovery in
`read_transcript` work, since a partial final line is only recoverable when
nothing else is ever rewritten.

One session can span several files — every sub-agent run writes its own — and
they say so in both directions: a child's header carries `origin` and
`parent_session`, and the parent appends a `continued` record naming the child.
A back-pointer alone would make "find the head of this session" a directory
scan. See `docs/adr/0001-session-profiles.md`, Decision 2.

A session is intentionally simple:
- Linear history within a file, no tree / branching / forks.
- Append-only writes; no rewrite-on-modify, no deletions.
- Tools and extensions are NOT persisted — they're rebuilt from the registry at
  load time, the same way pi-mono works.

Don't open the same session file in two processes simultaneously; results
are undefined.
"""

from __future__ import annotations

import json
import logging
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime
from io import TextIOBase
from pathlib import Path
from typing import IO, Any, Literal

from pydantic import BaseModel, TypeAdapter

from midge.messages import Message, make_summary_message

# 2 added the `clear` and `session_info` records. Bumped for `clear`
# specifically: an older build skips an unknown entry type, and skipping a
# clear silently restores messages the user cleared. Skipping a `session_info`
# would only lose a name, which would not have justified this on its own.
#
# `model_change` and `identity` were added without a bump, by the same test: an
# older build skipping them degrades to "the change did not happen", which is
# exactly what that build did anyway. Nothing is restored that a user discarded.
#
# `continued` and the header's `origin` pass that test too: skipping a
# `continued` leaves a reader with no forward link, which is what every build
# before this one had, and restores nothing.
#
# `profile` passes it too, and more comfortably than most: an older build
# skipping it leaves the model and prompt at whatever the `identity` and
# `model_change` records before it said, which is that build's own reading of
# the session. It never restores anything discarded.
#
# That promise is why `origin` declares its whole vocabulary now rather than
# only `subagent`, the one value with a producer today. Not moving VERSION means
# a newer build's `origin: "profile"` arrives in a file this build must read,
# and a Literal that did not list it would raise on the *header* and strand the
# whole transcript. A value outside the three is a format change that would have
# to move VERSION, precisely because it cannot degrade.
VERSION = 2


_logger = logging.getLogger(__name__)


class SessionHeader(BaseModel):
    type: Literal["header"] = "header"
    version: int = VERSION
    created_at: str
    model: str
    system_prompt: str | None = None
    # What this transcript is, stated rather than inferred. Absent means a root
    # session. Before this existed a sub-agent run was identifiable only by
    # accident — it happened to have `parent_tool_call_id` set — and a profile
    # excursion would have been indistinguishable from one.
    origin: Literal["subagent", "profile", "fork"] | None = None
    # Set on a sub-agent transcript so a file found on its own says which turn
    # of which conversation produced it. `parent_tool_call_id` is the same id
    # that appears on the parent's ToolCall and its ToolResultMessage.
    parent_session: str | None = None
    parent_tool_call_id: str | None = None


class CompactionRecord(BaseModel):
    type: Literal["compaction"] = "compaction"
    summary: str
    cut_index: int
    timestamp: int = 0


class ClearRecord(BaseModel):
    """A compaction with no summary: drop `cut_index` messages, add nothing.

    Written when the user discards context. The messages stay in the file — it
    is the record of what happened — but a reload no longer replays them.
    """

    type: Literal["clear"] = "clear"
    cut_index: int
    timestamp: int = 0


class SessionInfoRecord(BaseModel):
    """A rename. The name lives here rather than on the header because the
    header is written once and never rewritten; the last record wins."""

    type: Literal["session_info"] = "session_info"
    name: str
    timestamp: int = 0


class ModelChangeRecord(BaseModel):
    """The model switched mid-session. Last record wins over the header."""

    type: Literal["model_change"] = "model_change"
    model: str
    timestamp: int = 0


class IdentityRecord(BaseModel):
    """The base system prompt was replaced.

    The **base**, never the composed prompt. What tools and skills exist is a
    fact about this machine right now — `cli.py` recomposes it on every start —
    so storing the composed string would duplicate the skills catalogue on every
    resume and carry absolute paths that may point at another machine.
    """

    type: Literal["identity"] = "identity"
    system_prompt: str
    timestamp: int = 0


class ProfileRecord(BaseModel):
    """The agent was retargeted to a named profile.

    One record rather than an `identity` plus a `model_change`, because a switch
    is one act: reading two entries and inferring they were the same decision is
    exactly what naming a profile exists to avoid.

    It carries the model and prompt **as applied**, not just the name, because a
    profile is a `.py` file that can be edited or deleted afterwards — a
    transcript whose meaning depends on the current contents of a file outside
    it is not an audit trail. Restoring a session whose profile has since
    vanished falls back to these.

    It does **not** carry `tools` or `hooks`, and the line is the one this module
    already draws: tools and extensions are rebuilt from the registry at load,
    never persisted. Model and prompt are *values* and restore themselves;
    tools and hooks are *references* into a registry that may no longer contain
    them, so recording them would not make a vanished profile restorable.
    """

    type: Literal["profile"] = "profile"
    name: str
    model: str
    system_prompt: str
    timestamp: int = 0


class ContinuedRecord(BaseModel):
    """Another transcript of this session started here.

    `parent_session` on the child is a back-pointer, so without this "find the
    head of this session" is a directory scan. This makes the chain walkable
    forwards, and cheaply: one header and the records, never every message.

    `reason` says whether the parent stopped (`profile`, `fork`) or carried on
    (`subagent`) — the same fact as the child's `origin`, duplicated on purpose
    so a walk reads the parent rather than opening every child to classify it.

    `path` is relative to *this* file's directory. Children are always siblings,
    and an absolute path would bake one machine's layout into the audit trail —
    the objection `IdentityRecord` already makes about composed prompts.
    """

    type: Literal["continued"] = "continued"
    path: str
    reason: Literal["subagent", "profile", "fork"]
    timestamp: int = 0


# Everything that is not a message. Spelled as a plain union so it works with
# `isinstance`, which the `Annotated` `Message` alias does not.
#
# A record parsed but left out of this union is not merely unlisted: it falls
# through `fold_history`'s final `else` and is appended to history as though it
# were a message. Adding a record type is two edits.
SessionRecord = (
    CompactionRecord
    | ClearRecord
    | SessionInfoRecord
    | ModelChangeRecord
    | IdentityRecord
    | ContinuedRecord
    | ProfileRecord
)

# What the file literally holds, before the records are folded into history.
TranscriptEntry = Message | SessionRecord


_MESSAGE_ADAPTER: TypeAdapter[Message] = TypeAdapter(Message)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_session_dir() -> Path:
    """A function, not a constant: `Path.cwd()` at import time would freeze
    whatever directory the interpreter started in. Same reason `config_paths`
    and `default_skill_dirs` are functions."""
    return Path.cwd() / ".midge" / "sessions"


def resolve_session_path(
    explicit: Path | None, *, directory: Path | None = None, enabled: bool = True
) -> Path | None:
    """Where this run's transcript goes, or None for no persistence.

    Every entrypoint asks this rather than deciding for itself, so `--session`
    means the same thing in the CLI and in the example agents.

    A relative `--session` resolves *under the session directory*, not under the
    working directory: transcripts are one collection, and a path that sometimes
    means `./run.jsonl` and sometimes `.midge/sessions/run.jsonl` is worse than
    one that always means the latter. An absolute path is the way out.

    An explicit path outranks `enabled=False`, because naming a file is a
    deliberate request and a default-off is only a default.
    """
    root = directory if directory is not None else default_session_dir()
    if explicit is not None:
        return explicit if explicit.is_absolute() else root / explicit
    if not enabled:
        return None
    # Seconds, then four random hex: the stamp sorts a listing chronologically,
    # which is what session discovery will want, and the suffix is because
    # `Session.new` raises on collision rather than retrying — two midge
    # processes started in the same second would otherwise fail at startup.
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return root / f"{stamp}-{secrets.token_hex(2)}.jsonl"


def read_transcript(path: str | Path) -> tuple[SessionHeader, list[TranscriptEntry]]:
    """Read the file as written: every message ever appended, plus the compaction
    records, in order. This is the complete record; `Session.messages` is only
    what the agent still holds in context.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty session file: {p}")

    first = json.loads(lines[0])
    if first.get("type") != "header":
        raise ValueError(
            f"First entry of {p} must be a 'header'; got type={first.get('type')!r}"
        )
    # Only a *newer* file is rejected. An older one loads because this build
    # still understands every entry type it can hold; refusing it would strand
    # sessions for no gain. The asymmetry is the point of the version: a build
    # that would misread a record must decline rather than guess.
    version = first.get("version")
    if not isinstance(version, int) or version > VERSION:
        raise ValueError(
            f"Session version {version} is incompatible "
            f"with this build (expected {VERSION} or older)."
        )
    header = SessionHeader.model_validate(first)

    entries: list[TranscriptEntry] = []
    for i, line in enumerate(lines[1:], start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            # `append` is write-then-flush, not atomic, so a crash mid-write
            # leaves a partial final line. Dropping it recovers the session;
            # a corrupt line anywhere else is real damage and still raises.
            if i == len(lines) - 1:
                _logger.warning("session_trailing_line_truncated path=%s line=%d", p, i + 1)
                break
            raise
        entry_type = raw.get("type")
        if entry_type == "message":
            entries.append(_MESSAGE_ADAPTER.validate_python(raw["data"]))
        elif entry_type == "compaction":
            entries.append(CompactionRecord.model_validate(raw))
        elif entry_type == "clear":
            entries.append(ClearRecord.model_validate(raw))
        elif entry_type == "session_info":
            entries.append(SessionInfoRecord.model_validate(raw))
        elif entry_type == "model_change":
            entries.append(ModelChangeRecord.model_validate(raw))
        elif entry_type == "identity":
            entries.append(IdentityRecord.model_validate(raw))
        elif entry_type == "continued":
            entries.append(ContinuedRecord.model_validate(raw))
        elif entry_type == "profile":
            entries.append(ProfileRecord.model_validate(raw))
        else:
            _logger.warning("session_unknown_entry_type type=%r path=%s", entry_type, p)

    return header, entries


def fold_history(entries: Sequence[TranscriptEntry]) -> list[Message]:
    """Replay the records to rebuild the history the agent actually held.

    Applying them rather than skipping them matters on resume: skipping replayed
    the pre-compaction history and silently undid the compaction, and would do
    the same to a clear.
    """
    messages: list[Message] = []
    for entry in entries:
        if isinstance(entry, CompactionRecord):
            messages = [
                make_summary_message(entry.summary),
                *messages[entry.cut_index :],
            ]
        elif isinstance(entry, ClearRecord):
            messages = messages[entry.cut_index :]
        elif isinstance(entry, SessionRecord):
            continue  # not history; folded by `session_name` / `identity`
        else:
            messages.append(entry)
    return messages


def session_name(entries: Sequence[TranscriptEntry]) -> str | None:
    """The most recent name, or None if the session was never named."""
    return _last(entries, SessionInfoRecord, "name")


def session_model(entries: Sequence[TranscriptEntry]) -> str | None:
    """The most recent model, or None if it was never changed after the header.

    None rather than the header's model: the caller holds the header and can
    decide, and conflating "never changed" with a value would make this look
    like the authority on what the model is.

    A profile switch counts, because it sets the model as much as `set_model`
    does — whichever came last wins, read positionally.
    """
    return _last(entries, (ModelChangeRecord, ProfileRecord), "model")


def session_prompt(entries: Sequence[TranscriptEntry]) -> str | None:
    """The most recent *base* system prompt, or None if it was never replaced."""
    return _last(entries, (IdentityRecord, ProfileRecord), "system_prompt")


def session_profile(entries: Sequence[TranscriptEntry]) -> str | None:
    """The profile this transcript is running under, or None if it never was.

    The *last* one recorded, per ADR Decision 5 — a thread that has since
    switched away is running under whatever it switched to. Read backwards for
    the same reason everything else here is.
    """
    return _last(entries, ProfileRecord, "name")


def session_continuations(entries: Sequence[TranscriptEntry]) -> list[ContinuedRecord]:
    """Every transcript that started from this one, in the order they did.

    Plural, and so not `_last`: a parent has one record per child, and a walk
    needs all of them.
    """
    return [e for e in entries if isinstance(e, ContinuedRecord)]


def session_chain(path: str | Path) -> list[Path]:
    """Every transcript of the session `path` belongs to, root first.

    Up through `parent_session` to the root, then down through the `continued`
    records — which is why both directions exist (#62). A back-pointer alone
    would make this a directory scan, and a session has no directory it owns.

    Unreadable links are skipped rather than raising: a chain is an audit
    convenience, and a caller asking "where else have I been" should not be
    stopped by one deleted file. A `visited` set guards cycles even though
    nothing writes one — a return is deliberately not recorded as a branch, and
    a walk that could hang on a malformed file would be worse than one that
    quietly stops.
    """
    start = Path(path)
    root = start
    seen_up: set[Path] = set()
    while root not in seen_up:
        seen_up.add(root)
        try:
            header, _entries = read_transcript(root)
        except (OSError, ValueError):
            break
        if not header.parent_session:
            break
        root = Path(header.parent_session)

    order: list[Path] = []
    visited: set[Path] = set()
    queue = [root]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        try:
            _header, entries = read_transcript(current)
        except (OSError, ValueError):
            continue
        order.append(current)
        queue.extend(current.parent / link.path for link in session_continuations(entries))
    return order


def _last(
    entries: Sequence[TranscriptEntry],
    kind: type | tuple[type, ...],
    field: str,
) -> str | None:
    """The last record of `kind`, read backwards — last write wins.

    A tuple where more than one record type can set the same value: a model
    comes from either a `model_change` or a `profile`, and which one is
    authoritative is simply which came last.
    """
    for entry in reversed(entries):
        if isinstance(entry, kind):
            return getattr(entry, field)
    return None


class Session:
    def __init__(
        self,
        path: Path,
        *,
        file: IO[str] | TextIOBase,
        header: SessionHeader,
        messages: list[Message],
        name: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        profile: str | None = None,
    ) -> None:
        self.path = path
        self._file = file
        self.header = header
        self.messages = messages
        self.name = name
        # The profile this transcript is running under, or None if it never was
        # retargeted. No header fallback: unlike model and prompt, "no profile"
        # is a real state rather than a value the header supplies.
        self.profile = profile
        # The header records what the session *started* as and is never
        # rewritten. These are what it is *now* — the header's value unless a
        # record superseded it. Read these, not `header.model`.
        self.model = model if model is not None else header.model
        self.system_prompt = (
            system_prompt if system_prompt is not None else header.system_prompt
        )

    @classmethod
    def new(
        cls,
        path: str | Path,
        *,
        model: str,
        system_prompt: str | None = None,
        origin: Literal["subagent", "profile", "fork"] | None = None,
        parent_session: str | None = None,
        parent_tool_call_id: str | None = None,
    ) -> Session:
        p = Path(path)
        if p.exists():
            raise FileExistsError(
                f"Session file already exists: {p}. "
                "Use Session.load() to resume, or open() with a different path."
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        f = p.open("w", encoding="utf-8")
        header = SessionHeader(
            created_at=_now_iso(),
            model=model,
            system_prompt=system_prompt,
            origin=origin,
            parent_session=parent_session,
            parent_tool_call_id=parent_tool_call_id,
        )
        f.write(header.model_dump_json() + "\n")
        f.flush()
        return cls(p, file=f, header=header, messages=[])

    @classmethod
    def load(cls, path: str | Path) -> Session:
        p = Path(path)
        header, entries = read_transcript(p)
        f = p.open("a", encoding="utf-8")
        return cls(
            p,
            file=f,
            header=header,
            messages=fold_history(entries),
            name=session_name(entries),
            model=session_model(entries),
            system_prompt=session_prompt(entries),
            profile=session_profile(entries),
        )

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        model: str,
        system_prompt: str | None = None,
    ) -> Session:
        """Resume the session at `path` if it exists; otherwise create a new one
        with the given header fields. The returned session is open for append.
        """
        p = Path(path)
        if p.exists():
            return cls.load(p)
        return cls.new(p, model=model, system_prompt=system_prompt)

    def append(self, message: Message) -> None:
        entry = {"type": "message", "data": message.model_dump(mode="json")}
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()
        self.messages.append(message)

    def append_many(self, messages: list[Message]) -> None:
        for m in messages:
            self.append(m)

    def append_compaction(self, *, summary: str, cut_index: int) -> None:
        self._append_record({"type": "compaction", "summary": summary, "cut_index": cut_index})
        # Keep the in-memory view identical to what `load` would rebuild.
        self.messages = [make_summary_message(summary), *self.messages[cut_index:]]

    def append_clear(self, *, cut_index: int) -> None:
        """Record that `cut_index` messages were discarded from the front.

        The messages stay in the file — it is the record of what happened, and
        anything reading the JSONL still sees them. What changes is what a
        resume replays.
        """
        self._append_record({"type": "clear", "cut_index": cut_index})
        self.messages = self.messages[cut_index:]

    def set_profile(self, *, name: str, model: str, system_prompt: str) -> None:
        """Record a retargeting, and fold it the way a resume would.

        Sets `self.model` and `self.system_prompt` as well as appending, because
        a profile switch *is* a model and prompt change — leaving them stale
        would make an in-process read disagree with what a reload rebuilds.
        """
        self._append_record(
            {
                "type": "profile",
                "name": name,
                "model": model,
                "system_prompt": system_prompt,
            }
        )
        self.profile = name
        self.model = model
        self.system_prompt = system_prompt

    def append_continued(
        self, *, path: str | Path, reason: Literal["subagent", "profile", "fork"]
    ) -> None:
        """Record that another transcript of this session started here.

        Unlike `set_name` or `set_model` this updates nothing in memory: it is
        an event that happened, not a current value some later write supersedes.

        `path` is stored as given and is expected to be relative to this file's
        directory — `Path.name` for a sibling, which is every case today.
        """
        self._append_record({"type": "continued", "path": str(path), "reason": reason})

    def set_name(self, name: str) -> None:
        self._append_record({"type": "session_info", "name": name})
        self.name = name

    def set_model(self, model: str) -> None:
        self._append_record({"type": "model_change", "model": model})
        self.model = model

    def set_system_prompt(self, system_prompt: str) -> None:
        """Record a new *base* prompt — never the composed one.

        `cli.py` recomposes the extension and skills halves on every start, so
        persisting the composed string would duplicate the catalogue on each
        resume and bake in absolute paths from this machine.
        """
        self._append_record({"type": "identity", "system_prompt": system_prompt})
        self.system_prompt = system_prompt

    def _append_record(self, entry: dict[str, Any]) -> None:
        stamped: dict[str, Any] = {
            **entry,
            "timestamp": int(datetime.now(UTC).timestamp() * 1000),
        }
        self._file.write(json.dumps(stamped, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
