"""Skills: directories of `@tool`-decorated Python files that auto-register.

A "skill" is a `.py` file that defines one or more `Tool` instances at module
top level (typically via `@tool`) and optionally a `SYSTEM_PROMPT` constant
whose contents are appended to the agent's system prompt.

The loader walks each given directory (or single file path), imports each
public `.py` file (skipping `_*.py` and `__init__.py`), and pulls every `Tool`
into a fresh `ToolRegistry`. First-registered wins on name collisions; later
duplicates are warned about and dropped.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

from pym.tools import Tool, ToolRegistry

_BUILTIN_ROOT = Path(__file__).parent
BUILTIN_DIRS: list[Path] = [_BUILTIN_ROOT / "coding"]

_logger = logging.getLogger(__name__)


def load_skills(sources: Iterable[Path | str]) -> tuple[ToolRegistry, str]:
    registry = ToolRegistry()
    prompts: list[str] = []

    for raw in sources:
        path = Path(raw).resolve()
        files = _files_for(path)
        if files is None:
            continue
        for f in files:
            try:
                module = _import_file(f)
            except Exception as e:
                _logger.warning("Failed to load skill %s: %s", f, e)
                continue
            for t in _extract_tools(module):
                if t.name in registry:
                    _logger.warning(
                        "Skill tool %r from %s shadowed by earlier registration",
                        t.name,
                        f,
                    )
                    continue
                registry.add(t)
            sp = getattr(module, "SYSTEM_PROMPT", None)
            if isinstance(sp, str) and sp.strip():
                prompts.append(sp.strip())

    return registry, "\n\n".join(prompts)


def _files_for(path: Path) -> list[Path] | None:
    if path.is_file():
        return [path] if _is_skill_file(path) else []
    if path.is_dir():
        return sorted(p for p in path.iterdir() if _is_skill_file(p))
    _logger.warning("Skill source not found: %s", path)
    return None


def _is_skill_file(p: Path) -> bool:
    return (
        p.suffix == ".py"
        and not p.name.startswith("_")
        and p.name != "__init__.py"
        and p.is_file()
    )


def _import_file(path: Path) -> ModuleType:
    abs_path = path.resolve()
    name = f"_pi_skill_{abs_path.stem}_{abs(hash(str(abs_path)))}"
    spec = importlib.util.spec_from_file_location(name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = abs_path.parent.name
    module.__file__ = str(abs_path)
    sys.modules[name] = module
    sys.path.insert(0, str(abs_path.parent))
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        try:
            sys.path.remove(str(abs_path.parent))
        except ValueError:
            pass
    return module


def _extract_tools(module: ModuleType) -> list[Tool]:
    return [v for v in vars(module).values() if isinstance(v, Tool)]


__all__ = ["BUILTIN_DIRS", "load_skills"]
