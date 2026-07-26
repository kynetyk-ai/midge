"""Personal-notes knowledge-base skill.

A Phase 5 demo skill: shows how a user-authored skill pack retargets the
harness without any change to core code. Storage is a single JSON file
(default `~/.pym-notes/kb.json`; override via `PYM_NOTES_KB`).

Tools:
    add_note(title, content, tags=None) — create a new note.
    search_notes(query, limit=10) — substring search across title/body/tags.
    read_note(title) — return the full body of a note.
    list_notes(tag=None) — list all notes, optionally filtered by tag.
    link_notes(from_title, to_title) — record a directional link.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pym.tools import tool

SYSTEM_PROMPT = """You also have access to a personal-notes knowledge base. Use the notes tools to help the user capture, find, and connect their notes.

Conventions:
- Note titles should be short and descriptive; they are the unique identifier.
- Tags are lowercase single words (e.g., "research", "todo"). Multiple tags are fine.
- Search before adding to avoid duplicates.
- Use link_notes to record relationships between ideas."""


def _kb_path() -> Path:
    return Path(os.getenv("PYM_NOTES_KB", str(Path.home() / ".pym-notes" / "kb.json")))


def _load() -> dict[str, Any]:
    p = _kb_path()
    if not p.exists():
        return {"notes": {}, "links": []}
    return json.loads(p.read_text(encoding="utf-8"))


def _save(data: dict[str, Any]) -> None:
    p = _kb_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _normalize_tags(tags: list[str] | None) -> list[str]:
    return [t.lower().strip() for t in (tags or []) if t and t.strip()]


@tool
async def add_note(
    title: str,
    content: str,
    tags: list[str] | None = None,
) -> str:
    """Save a new note. Returns the canonical slug.

    Args:
        title: short, descriptive title (must be unique).
        content: free-form body text (markdown is fine).
        tags: lowercase single-word tags (optional).
    """
    kb = _load()
    slug = _slug(title)
    if not slug:
        raise ValueError("Title must contain at least one alphanumeric character")
    if slug in kb["notes"]:
        raise ValueError(f"Note {title!r} already exists at slug {slug!r}")
    kb["notes"][slug] = {
        "title": title,
        "content": content,
        "tags": _normalize_tags(tags),
        "created_at": datetime.now(UTC).isoformat(),
    }
    _save(kb)
    return f"Saved note {title!r} (slug: {slug})"


@tool
async def search_notes(query: str, limit: int = 10) -> str:
    """Find notes whose title, body, or tags contain `query` (case-insensitive)."""
    kb = _load()
    q = query.lower()
    hits: list[tuple[str, dict[str, Any]]] = []
    for slug, n in kb["notes"].items():
        haystack = f"{n['title']} {n['content']} {' '.join(n['tags'])}".lower()
        if q in haystack:
            hits.append((slug, n))
    if not hits:
        return f"No notes match {query!r}."
    out = [f"{len(hits)} match{'es' if len(hits) != 1 else ''}:"]
    for slug, n in hits[:limit]:
        tag_str = ", ".join(n["tags"]) if n["tags"] else "(no tags)"
        out.append(f"- {n['title']} [tags: {tag_str}] — slug: {slug}")
    return "\n".join(out)


@tool
async def read_note(title: str) -> str:
    """Return the full body of a note, looked up by title."""
    kb = _load()
    slug = _slug(title)
    n = kb["notes"].get(slug)
    if n is None:
        for s, candidate in kb["notes"].items():
            if candidate["title"].lower() == title.lower():
                slug, n = s, candidate
                break
    if n is None:
        raise KeyError(f"No note titled {title!r}")
    parts = [f"# {n['title']}", f"slug: {slug}"]
    if n.get("tags"):
        parts.append(f"tags: {', '.join(n['tags'])}")
    parts.append("")
    parts.append(n["content"])
    return "\n".join(parts)


@tool
async def list_notes(tag: str | None = None) -> str:
    """List all notes, optionally filtered to those bearing `tag`."""
    kb = _load()
    notes = list(kb["notes"].values())
    if tag is not None:
        normalized = tag.lower().strip()
        notes = [n for n in notes if normalized in n["tags"]]
    if not notes:
        return "No notes yet." if tag is None else f"No notes with tag {tag!r}."
    return "\n".join(
        f"- {n['title']} ({', '.join(n['tags']) or 'no tags'})" for n in notes
    )


@tool
async def link_notes(from_title: str, to_title: str) -> str:
    """Record a directional link from one note to another. Both must exist."""
    kb = _load()
    fs, ts = _slug(from_title), _slug(to_title)
    if fs not in kb["notes"]:
        raise KeyError(f"Note {from_title!r} not found")
    if ts not in kb["notes"]:
        raise KeyError(f"Note {to_title!r} not found")
    link = {"from": fs, "to": ts}
    if link in kb["links"]:
        return f"Link already exists: {from_title!r} → {to_title!r}"
    kb["links"].append(link)
    _save(kb)
    return f"Linked: {from_title!r} → {to_title!r}"
