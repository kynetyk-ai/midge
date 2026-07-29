"""Agent Skills: `SKILL.md` files the model loads on demand.

A skill is a directory holding a `SKILL.md` — YAML frontmatter carrying at
minimum a `description`, followed by markdown instructions. Bundled `scripts/`,
`references/` and `assets/` alongside it are freeform.

The loader records only the metadata, never the body. A catalogue of names,
descriptions and *absolute paths* goes into the system prompt; the model opens
the file itself with `read` when a task matches. That is the whole mechanism —
there is no skill tool, no new message type, and no change to the agent loop.
Storing the path instead of the body is also what makes this survive
compaction: the catalogue lives in `system_prompt`, which the positional cut in
`compaction.py` never touches, while a body that has been read into history is
disposable and re-fetchable from the same path.

A body pulled in by `skill_message` is an ordinary user message, so it *is*
subject to compaction. That is the intended trade-off, not an oversight.

Validation is lenient on purpose: it lets midge ingest skill directories
authored for other harnesses. Everything except a missing description is a
warning that still loads.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import yaml

from midge.messages import UserMessage

_SKILL_FILE = "SKILL.md"
_SKIP_DIRS = frozenset({"node_modules", "__pycache__"})
_MAX_DEPTH = 6
_MAX_NAME = 64
_MAX_DESCRIPTION = 1024
_NAME_RE = re.compile(r"^[a-z0-9-]+$")

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    path: Path
    base_dir: Path
    # The frontmatter key is the negative `disable-model-invocation`; carrying
    # it positively reads better everywhere it is actually checked.
    model_invocable: bool = True


def default_skill_dirs() -> list[Path]:
    """Project skills before personal ones, most specific first.

    A function rather than a module constant like `BUILTIN_TOOL_DIRS`: `cwd()`
    resolved at import time would freeze whatever directory the interpreter
    started in.
    """
    return [
        Path.cwd() / ".midge" / "skills",
        Path.cwd() / ".agents" / "skills",
        Path.home() / ".midge" / "skills",
        Path.home() / ".agents" / "skills",
    ]


def load_skills(sources: Iterable[Path | str]) -> list[Skill]:
    """Discover skills across `sources`, in order. The first to claim a name wins.

    Precedence is a property of the order of this list, never of the order the
    walk happened to reach files in. Callers put the most specific sources
    first — explicit `--skill-dir` paths ahead of `default_skill_dirs()`.
    """
    skills: dict[str, Skill] = {}
    seen: set[Path] = set()

    for raw in sources:
        for file in _sources_for(Path(raw).expanduser()):
            real = file.resolve()
            if real in seen:
                continue
            seen.add(real)
            skill = _load_skill_file(file)
            if skill is None:
                continue
            existing = skills.get(skill.name)
            if existing is not None:
                _logger.warning(
                    "Skill %r at %s shadowed by %s",
                    skill.name,
                    skill.path,
                    existing.path,
                )
                continue
            skills[skill.name] = skill

    return list(skills.values())


def skills_prompt(skills: Iterable[Skill]) -> str:
    """The `<available_skills>` catalogue, or `''` when nothing is invocable.

    Only ever append this when a read-capable tool is registered — advertising
    skills the model has no way to open is worse than saying nothing.
    """
    visible = [s for s in skills if s.model_invocable]
    if not visible:
        return ""

    lines = [
        "The following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill's "
        "directory (the one containing SKILL.md) and use that absolute path in tool calls.",
        "",
        "<available_skills>",
    ]
    for s in visible:
        lines += [
            "  <skill>",
            f"    <name>{escape(s.name)}</name>",
            f"    <description>{escape(s.description)}</description>",
            f"    <location>{escape(str(s.path))}</location>",
            "  </skill>",
        ]
    lines.append("</available_skills>")
    return "\n".join(lines)


def find_skill(skills: Iterable[Skill], name: str) -> Skill | None:
    return next((s for s in skills if s.name == name), None)


def skill_message(
    skills: Iterable[Skill], name: str, instructions: str | None = None
) -> UserMessage:
    """Force a skill: its body, wrapped, as a user message.

    The catalogue relies on the model choosing to open the file, which it does
    not always do. This is the deterministic path. The body is read here rather
    than at discovery time, so it is always current.
    """
    skill = find_skill(skills, name)
    if skill is None:
        raise KeyError(f"No skill named {name!r}")

    body = _split_frontmatter(skill.path.read_text(encoding="utf-8"))[1].strip()
    block = (
        f'<skill name="{skill.name}" location="{skill.path}">\n'
        f"References are relative to {skill.base_dir}.\n\n"
        f"{body}\n"
        "</skill>"
    )
    if instructions:
        block += f"\n\n{instructions}"
    return UserMessage(content=block)


def _sources_for(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix != ".md":
            _logger.warning("Skill source is not a markdown file: %s", path)
            return []
        return [path]
    if path.is_dir():
        return _walk(path, 0)
    _logger.warning("Skill source not found: %s", path)
    return []


def _walk(directory: Path, depth: int) -> list[Path]:
    if depth > _MAX_DEPTH:
        _logger.warning("Skill search stopped at depth %d: %s", _MAX_DEPTH, directory)
        return []

    # A directory holding SKILL.md is a leaf. This is what keeps a bundled
    # references/ tree from being mistaken for a nest of sub-skills.
    skill_file = directory / _SKILL_FILE
    if skill_file.is_file():
        return [skill_file]

    try:
        entries = sorted(directory.iterdir())
    except OSError as e:
        _logger.warning("Cannot read skill directory %s: %s", directory, e)
        return []

    found: list[Path] = []
    for entry in entries:
        if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
            continue
        # `is_dir` follows symlinks and answers False for a broken one.
        if entry.is_dir():
            found.extend(_walk(entry, depth + 1))
    return found


def _split_frontmatter(text: str) -> tuple[str | None, str]:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---"):
        return None, normalized
    end = normalized.find("\n---", 3)
    if end == -1:
        return None, normalized
    return normalized[4:end], normalized[end + 4 :]


def _load_skill_file(path: Path) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        _logger.warning("Skipping skill %s: %s", path, e)
        return None

    raw_frontmatter, _ = _split_frontmatter(text)
    if raw_frontmatter is None:
        _logger.warning("Skipping skill %s: missing YAML frontmatter", path)
        return None

    try:
        parsed = yaml.safe_load(raw_frontmatter)
    except yaml.YAMLError as e:
        _logger.warning("Skipping skill %s: invalid frontmatter: %s", path, e)
        return None
    if not isinstance(parsed, dict):
        _logger.warning("Skipping skill %s: frontmatter is not a mapping", path)
        return None

    description = parsed.get("description")
    # The description is the only thing the model sees before deciding whether
    # to open the file, so a skill without one is unreachable by construction.
    if not isinstance(description, str) or not description.strip():
        _logger.warning("Skipping skill %s: description is required", path)
        return None
    description = description.strip()
    if len(description) > _MAX_DESCRIPTION:
        _logger.warning(
            "Skill %s: description is %d chars (max %d)",
            path,
            len(description),
            _MAX_DESCRIPTION,
        )

    fallback = path.parent.name if path.name == _SKILL_FILE else path.stem
    raw_name = parsed.get("name")
    if raw_name is None:
        name = fallback
    elif isinstance(raw_name, str) and raw_name.strip():
        name = raw_name.strip()
    else:
        # Unquoted `on`/`no`/`yes` parse as bools, not strings.
        _logger.warning("Skill %s: name is not a string, using %r", path, fallback)
        name = fallback

    _warn_about_name(path, name)

    return Skill(
        name=name,
        description=description,
        path=path.resolve(),
        base_dir=path.resolve().parent,
        model_invocable=parsed.get("disable-model-invocation") is not True,
    )


def _warn_about_name(path: Path, name: str) -> None:
    # The spec also requires the name to match the parent directory. That rule
    # is hostile to skill directories shared between harnesses, so it is not
    # enforced here — pi dropped it for the same reason.
    if len(name) > _MAX_NAME:
        _logger.warning("Skill %s: name is %d chars (max %d)", path, len(name), _MAX_NAME)
    if not _NAME_RE.match(name):
        _logger.warning(
            "Skill %s: name %r should be lowercase a-z, 0-9 and hyphens only", path, name
        )
    if name.startswith("-") or name.endswith("-"):
        _logger.warning("Skill %s: name %r should not start or end with a hyphen", path, name)
    if "--" in name:
        _logger.warning("Skill %s: name %r should not contain consecutive hyphens", path, name)


__all__ = [
    "Skill",
    "default_skill_dirs",
    "find_skill",
    "load_skills",
    "skill_message",
    "skills_prompt",
]
