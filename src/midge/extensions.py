"""Extensions: directories of `@tool`-decorated Python files that auto-register.

An "extension" is a `.py` file that defines one or more `Tool` instances at
module top level (typically via `@tool`) and optionally a `SYSTEM_PROMPT`
constant whose contents are appended to the agent's system prompt.

The loader walks each given directory (or single file path), imports each
public `.py` file (skipping `_*.py` and `__init__.py`), and pulls every `Tool`
into a fresh `ToolRegistry`. First-registered wins on name collisions; later
duplicates are warned about and dropped. A `Profile` instance is collected the
same way when the caller supplies a `ProfileSet` — so one file may declare a
tool, a sub-agent and a profile together, and is imported exactly once.

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
from midge.profiles import Profile, ProfileSet
from midge.tools import Tool, ToolRegistry

_BUILTIN_TOOL_ROOT = Path(__file__).parent / "tools"
BUILTIN_TOOL_DIRS: list[Path] = [_BUILTIN_TOOL_ROOT / "coding"]

_logger = logging.getLogger(__name__)


def load_extensions(
    sources: Iterable[Path | str],
    *,
    hooks: Hooks | None = None,
    profiles: ProfileSet | None = None,
) -> tuple[ToolRegistry, str]:
    """Import every extension file under `sources` and collect what it declares.

    Tools and the prompt contribution come back; hooks and profiles are
    collected into whatever the caller passes in, because a caller that does not
    care about them should not have to acknowledge them. A caller that passes no
    `profiles` simply discovers none — which is every entrypoint that predates
    them.
    """
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
                _logger.warning(
                    "extension_import_failed path=%s error=%s",
                    f,
                    type(e).__name__,
                    exc_info=e,
                )
                continue
            # Extensions log under the same `midge` root, so one env var covers
            # them and the record names the file it came from. An extension that
            # declares its own `log` module-level name keeps it.
            setattr(module, "log", logging.getLogger(f"midge.ext.{f.stem}"))  # noqa: B010
            for t in _extract_tools(module):
                if t.name in registry:
                    _logger.warning("tool_name_shadowed tool=%s path=%s", t.name, f)
                    continue
                registry.add(t)
                _logger.debug("tool_registered tool=%s path=%s", t.name, f)
            if profiles is not None:
                for p in _extract_profiles(module):
                    if p.name in profiles:
                        _logger.warning(
                            "profile_name_shadowed name=%s path=%s winner=%s",
                            p.name,
                            f,
                            profiles.path_of(p.name) or "-",
                        )
                        continue
                    profiles.add(p, path=f)
                    _logger.debug("profile_registered profile=%s path=%s", p.name, f)
            sp = getattr(module, "SYSTEM_PROMPT", None)
            if isinstance(sp, str) and sp.strip():
                prompts.append(sp.strip())
            if hooks is not None:
                _register_hooks(module, f, hooks)
            _logger.debug("extension_loaded path=%s", f)

    _logger.info(
        "extensions_loaded tools=%d prompt_contributions=%d profiles=%d",
        len(registry),
        len(prompts),
        len(profiles) if profiles is not None else 0,
    )
    return registry, "\n\n".join(prompts)


def _register_hooks(module: ModuleType, path: Path, hooks: Hooks) -> None:
    """Call an extension's `register_hooks(hooks)` if it defines one.

    Registrations are tagged with the extension path so a failing handler
    names the file it came from.
    """
    fn = getattr(module, "register_hooks", None)
    if not callable(fn):
        return
    scoped = _SourceScopedHooks(hooks, str(path), path.stem)
    try:
        fn(scoped)
    except Exception as e:
        _logger.warning(
            "extension_register_hooks_failed path=%s error=%s",
            path,
            type(e).__name__,
            exc_info=e,
        )


class _SourceScopedHooks:
    """Thin proxy that stamps `source` and `name` onto every registration.

    Both, because they answer different questions: the path is what a failing
    handler's warning has to name, and the stem is what a profile can write.
    """

    def __init__(self, hooks: Hooks, source: str, name: str) -> None:
        self._hooks = hooks
        self._source = source
        self._name = name

    def on(self, type: str, handler: Any) -> Any:
        return self._hooks.on(type, handler, source=self._source, name=self._name)

    def observe(self, handler: Any) -> Any:
        return self._hooks.observe(handler, source=self._source, name=self._name)

    def add_cleanup(self, fn: Any) -> Any:
        return self._hooks.add_cleanup(fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hooks, name)


def _files_for(path: Path) -> list[Path] | None:
    if path.is_file():
        return [path] if _is_extension_file(path) else []
    if path.is_dir():
        return sorted(p for p in path.iterdir() if _is_extension_file(p))
    _logger.warning("extension_source_not_found path=%s", path)
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


def _extract_profiles(module: ModuleType) -> list[Profile]:
    return [v for v in vars(module).values() if isinstance(v, Profile)]


__all__ = ["BUILTIN_TOOL_DIRS", "load_extensions"]
