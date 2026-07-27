"""Extensions: directories of `@tool`-decorated Python files that auto-register.

An "extension" is a `.py` file that defines one or more `Tool` instances at
module top level (typically via `@tool`) and optionally a `SYSTEM_PROMPT`
constant whose contents are appended to the agent's system prompt.

The loader walks each given directory (or single file path), imports each
public `.py` file (skipping `_*.py` and `__init__.py`), and pulls every `Tool`
into a fresh `ToolRegistry`. First-registered wins on name collisions; later
duplicates are warned about and dropped.

The built-in tools in `midge/tools/coding/` are loaded through this same path —
built-in and user-supplied tools are not special-cased.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

from midge.hooks import Hooks
from midge.tools import Tool, ToolRegistry

_BUILTIN_TOOL_ROOT = Path(__file__).parent / "tools"
BUILTIN_TOOL_DIRS: list[Path] = [_BUILTIN_TOOL_ROOT / "coding"]

_logger = logging.getLogger(__name__)


def load_extensions(
    sources: Iterable[Path | str], *, hooks: Hooks | None = None
) -> tuple[ToolRegistry, str]:
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
                _logger.warning("Failed to load extension %s: %s", f, e)
                continue
            for t in _extract_tools(module):
                if t.name in registry:
                    _logger.warning(
                        "Tool %r from %s shadowed by earlier registration",
                        t.name,
                        f,
                    )
                    continue
                registry.add(t)
            sp = getattr(module, "SYSTEM_PROMPT", None)
            if isinstance(sp, str) and sp.strip():
                prompts.append(sp.strip())
            if hooks is not None:
                _register_hooks(module, f, hooks)

    return registry, "\n\n".join(prompts)


def _register_hooks(module: ModuleType, path: Path, hooks: Hooks) -> None:
    """Call an extension's `register_hooks(hooks)` if it defines one.

    Registrations are tagged with the extension path so a failing handler
    names the file it came from.
    """
    fn = getattr(module, "register_hooks", None)
    if not callable(fn):
        return
    scoped = _SourceScopedHooks(hooks, str(path))
    try:
        fn(scoped)
    except Exception as e:
        _logger.warning("register_hooks failed for %s: %s", path, e)


class _SourceScopedHooks:
    """Thin proxy that stamps `source` onto every registration."""

    def __init__(self, hooks: Hooks, source: str) -> None:
        self._hooks = hooks
        self._source = source

    def on(self, type: str, handler: Any) -> Any:
        return self._hooks.on(type, handler, source=self._source)

    def observe(self, handler: Any) -> Any:
        return self._hooks.observe(handler, source=self._source)

    def add_cleanup(self, fn: Any) -> Any:
        return self._hooks.add_cleanup(fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hooks, name)


def _files_for(path: Path) -> list[Path] | None:
    if path.is_file():
        return [path] if _is_extension_file(path) else []
    if path.is_dir():
        return sorted(p for p in path.iterdir() if _is_extension_file(p))
    _logger.warning("Extension source not found: %s", path)
    return None


def _is_extension_file(p: Path) -> bool:
    return (
        p.suffix == ".py"
        and not p.name.startswith("_")
        and p.name != "__init__.py"
        and p.is_file()
    )


def _import_file(path: Path) -> ModuleType:
    abs_path = path.resolve()
    name = f"_midge_ext_{abs_path.stem}_{abs(hash(str(abs_path)))}"
    spec = importlib.util.spec_from_file_location(name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_tools(module: ModuleType) -> list[Tool]:
    return [v for v in vars(module).values() if isinstance(v, Tool)]


__all__ = ["BUILTIN_TOOL_DIRS", "load_extensions"]
