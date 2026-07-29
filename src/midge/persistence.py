"""Session save/load: append-only JSONL persistence for agent runs.

File layout (one JSON object per line):

    {"type":"header","version":1,"created_at":"...","model":"...","system_prompt":...}
    {"type":"message","data":{...}}
    {"type":"message","data":{...}}
    {"type":"compaction","summary":"...","cut_index":N,"timestamp":...}
    {"type":"message","data":{...}}

The first line MUST be the header. Hard-fails on version mismatch — this is
v1 of the format; if/when we change it we'll either bump and migrate, or
write a new file.

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

VERSION = 1


_logger = logging.getLogger(__name__)


class SessionHeader(BaseModel):
    type: Literal["header"] = "header"
    version: int = VERSION
    created_at: str
    model: str
    system_prompt: str | None = None


class CompactionRecord(BaseModel):
    type: Literal["compaction"] = "compaction"
    summary: str
    cut_index: int
    timestamp: int = 0


# What the file literally holds, before compactions are folded into history.
TranscriptEntry = Message | CompactionRecord


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
    if first.get("version") != VERSION:
        raise ValueError(
            f"Session version {first.get('version')} is incompatible "
            f"with this build (expected {VERSION})."
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
        else:
            _logger.warning("session_unknown_entry_type type=%r path=%s", entry_type, p)

    return header, entries


def fold_compactions(entries: Sequence[TranscriptEntry]) -> list[Message]:
    """Replay compactions to rebuild the history the agent actually held.

    Applying them rather than skipping them matters on resume: skipping replayed
    the pre-compaction history and silently undid the compaction.
    """
    messages: list[Message] = []
    for entry in entries:
        if isinstance(entry, CompactionRecord):
            messages = [
                make_summary_message(entry.summary),
                *messages[entry.cut_index :],
            ]
        else:
            messages.append(entry)
    return messages


class Session:
    def __init__(
        self,
        path: Path,
        *,
        file: IO[str] | TextIOBase,
        header: SessionHeader,
        messages: list[Message],
    ) -> None:
        self.path = path
        self._file = file
        self.header = header
        self.messages = messages

    @classmethod
    def new(
        cls,
        path: str | Path,
        *,
        model: str,
        system_prompt: str | None = None,
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
        )
        f.write(header.model_dump_json() + "\n")
        f.flush()
        return cls(p, file=f, header=header, messages=[])

    @classmethod
    def load(cls, path: str | Path) -> Session:
        p = Path(path)
        header, entries = read_transcript(p)
        f = p.open("a", encoding="utf-8")
        return cls(p, file=f, header=header, messages=fold_compactions(entries))

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
        entry: dict[str, Any] = {
            "type": "compaction",
            "summary": summary,
            "cut_index": cut_index,
            "timestamp": int(datetime.now(UTC).timestamp() * 1000),
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()
        # Keep the in-memory view identical to what `load` would rebuild.
        self.messages = [make_summary_message(summary), *self.messages[cut_index:]]

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
