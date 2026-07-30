"""Session save/load: append-only JSONL persistence for agent runs.

File layout (one JSON object per line):

    {"type":"header","version":2,"created_at":"...","model":"...","system_prompt":...}
    {"type":"message","data":{...}}
    {"type":"session_info","name":"auth refactor","timestamp":...}
    {"type":"model_change","model":"...","timestamp":...}
    {"type":"identity","system_prompt":"...","timestamp":...}
    {"type":"compaction","summary":"...","cut_index":N,"timestamp":...}
    {"type":"clear","cut_index":N,"timestamp":...}
    {"type":"message","data":{...}}

The first line MUST be the header. A file written by a *newer* build is
rejected; an older one loads, because every entry type it can contain is still
understood.

Anything mutable is expressed as an appended record and replayed on load —
last write wins. That is why renaming a session does not rewrite the header:
a rewrite would break the invariant that makes truncated-tail recovery in
`read_transcript` work, since a partial final line is only recoverable when
nothing else is ever rewritten.

A session is intentionally simple:
- Linear history, no tree / branching / forks.
- Append-only writes; no rewrite-on-modify, no deletions.
- Tools and extensions are NOT persisted — they're rebuilt from the registry at
  load time, the same way pi-mono works.

Don't open the same session file in two processes simultaneously; results
are undefined.
"""

from __future__ import annotations

import json
import logging
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
VERSION = 2


_logger = logging.getLogger(__name__)


class SessionHeader(BaseModel):
    type: Literal["header"] = "header"
    version: int = VERSION
    created_at: str
    model: str
    system_prompt: str | None = None
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


# Everything that is not a message. Spelled as a plain union so it works with
# `isinstance`, which the `Annotated` `Message` alias does not.
SessionRecord = (
    CompactionRecord | ClearRecord | SessionInfoRecord | ModelChangeRecord | IdentityRecord
)

# What the file literally holds, before the records are folded into history.
TranscriptEntry = Message | SessionRecord


_MESSAGE_ADAPTER: TypeAdapter[Message] = TypeAdapter(Message)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
    """
    return _last(entries, ModelChangeRecord, "model")


def session_prompt(entries: Sequence[TranscriptEntry]) -> str | None:
    """The most recent *base* system prompt, or None if it was never replaced."""
    return _last(entries, IdentityRecord, "system_prompt")


def _last(entries: Sequence[TranscriptEntry], kind: type, field: str) -> str | None:
    """The last record of `kind`, read backwards — last write wins."""
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
    ) -> None:
        self.path = path
        self._file = file
        self.header = header
        self.messages = messages
        self.name = name
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
        `export_html` still renders them. What changes is what a resume replays.
        """
        self._append_record({"type": "clear", "cut_index": cut_index})
        self.messages = self.messages[cut_index:]

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
