# Extensions — patterns to borrow from `pi-mono`

Source:
- `pi-mono/packages/coding-agent/src/core/skills.ts` — discovery + loading + system-prompt formatting
- `pi-mono/packages/coding-agent/src/core/system-prompt.ts:69–73, 162–164` — placement
- Extension loader (registers tools dynamically): the `pi.registerTool(definition)` API in `loader.ts`

> **Terminology update.** This note was written during the original port, when py-mono called
> its extension loader "skills". It now uses `pi-mono`'s vocabulary: **tools** are the
> LLM-callable functions, **extensions** are the `.py` files that register them, and **skill**
> is reserved for the `SKILL.md` standard, which py-mono does not implement. The design
> decisions below are unchanged — only the words are. See the glossary in `CLAUDE.md`.

## Important: pi-mono splits "skills" from "extensions" — we are porting the latter, not the former

In `pi-mono` the two concepts are distinct:

| Concept | Medium | Loading | Tools |
|---|---|---|---|
| **Skill** | Markdown `SKILL.md` with YAML frontmatter | Filesystem scan, presented to model as `<available_skills>` block; model reads SKILL.md on demand via the `read` tool ("progressive disclosure") | None — skills *reference* tools that already exist; they don't add any |
| **Extension** | TypeScript/JS module loaded via `jiti` | Dynamic require | Yes — `pi.registerTool(...)` adds new LLM-callable tools |

Our roadmap's "skills auto-loading" — *"drop a Python file into a temp dir with one `@tool` function; `pym --extension-dir <dir>` picks it up and the model can call it"* — describes the **extension** pattern, not the markdown-SKILL.md pattern. Phase 2 implements exactly that, under the name `extensions` (see `src/pym/extensions.py`).

We are **not** porting:
- Markdown SKILL.md files
- The progressive-disclosure `<available_skills>` block in the system prompt
- The tiered `~/.pi/agent/skills/` + `.pi/skills/` + `.agents/skills/` hierarchy
- gitignore-aware filesystem scanning
- The `/skill:name args` slash-command syntax

SKILL.md support is **deferred, not rejected** — it is purely additive and the `skills` name is
kept free for it.

## Patterns we *do* borrow

### Discovery is filesystem-based and explicit

User points the harness at directories (`--extension-dir`); the harness scans them at startup. Built-in tools live in the package itself (`src/pym/tools/coding/`); user extensions live wherever the user puts them. **Built-in vs. user tools are not special-cased** — they all flow through the same loader.

### Each extension module declares zero or more `@tool` functions

This is the "extension" shape, mapped to our `@tool` decorator. The loader walks the imported module's namespace and grabs every `Tool` instance (the symbol kind, not by special metadata). This avoids implicit module state — `@tool` is a pure transform from function to `Tool`, and the loader pattern-matches at the namespace level.

### Optional system-prompt contribution

An extension module can expose a `SYSTEM_PROMPT` constant (or a function returning a string) that gets concatenated into the agent's system prompt. **Concatenated, not sectioned.** Order: built-in tools first, user extensions after, in load order. No XML wrapping.

This is a simplification of pi-mono — they wrap each skill in `<skill><name>...</name><description>...</description><location>.../SKILL.md</location></skill>`. Ours is just text appended to the system prompt. If an extension wants structure, it puts it in its own contribution.

### No init / teardown hooks

Extensions are stateless per pi-mono: no async setup, no per-extension state, no shared resources. Same for us — Phase 2 keeps it that way. If an extension needs persistent state, it owns that itself (module-level globals, a class).

### Name collisions: first-registered wins, with a diagnostic

When two extensions register a tool with the same `name`, the loader emits a warning and skips the duplicate. Pi-mono does the same (`skills.ts:~250`).

## Translation guide for Python

| TS pattern | Python equivalent |
|---|---|
| `jiti` for dynamic require of `.ts`/`.js` modules | `importlib.util.spec_from_file_location` + `module_from_spec` for arbitrary `.py` paths; or `importlib.import_module(name)` for installed packages |
| `pi.registerTool(definition)` API | `loader` walks the imported module's `vars()` and picks out `Tool` instances (no API call needed; the decorator already produced the right thing) |
| `~/.pi/agent/skills/` vs `.pi/skills/` tiered scan | Just a list of `--extension-dir` paths plus the built-in `src/pym/tools/coding/`. No tiering. |
| Frontmatter metadata | Module-level constants (`NAME`, `SYSTEM_PROMPT`) — convention, not protocol |
| `disable-model-invocation` | Skip — not relevant when extensions are tool-bundles, not markdown |

## What Phase 2 implements

- A freestanding function `load_extensions(paths) -> ToolRegistry` that:
  - Imports every `.py` file under each path (skipping `_*.py` and `__init__.py`'s side effects)
  - Walks each module's namespace for `Tool` instances
  - Aggregates them into a `ToolRegistry`, refusing duplicates
  - Aggregates any `SYSTEM_PROMPT` strings from the modules into a single concatenated prompt addition
- A way to wire this into `Agent` construction so `examples/coding_agent.py` can do:
  ```python
  registry, prompt_addition = load_extensions(["src/pym/tools/coding", *args.extension_dirs])
  agent = Agent(client=..., model=..., tools=registry, system_prompt=BASE_PROMPT + prompt_addition)
  ```
- A CLI flag `--extension-dir DIR` (repeatable) on `examples/coding_agent.py`.

That's the minimum-viable extension loader. ~80 lines. Defer per-extension state, hot-reload, version checks, and slash-command-style invocation.
