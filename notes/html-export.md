# HTML session export — patterns to borrow from `pi-mono`

Source:
- `pi-mono/packages/coding-agent/src/cli/main.ts:461–472` — `--export session.jsonl [output.html]` flag
- `pi-mono/packages/coding-agent/src/export/index.ts` — entrypoint (`exportSessionToHtml`, `exportFromFile`)
- `pi-mono/packages/coding-agent/src/export/template.html` + `template.js` — page shell + client-side rendering
- `pi-mono/packages/coding-agent/src/export/ansi-to-html.ts` — ANSI escape parser

## What we're porting and what we're not

`pi-mono`'s exporter is a full-blown SPA: tree navigator with branching sessions, sidebar, dark/light theme derivation, search, filters, ANSI→HTML for tool output, base64-embedded session JSON for round-trip download, deep links via URL params, marked.js + highlight.js bundled inline, image lightbox, localStorage-persisted sidebar width.

**For Phase 2 we ship the minimum viable version** — a single self-contained HTML file that renders a linear session as a readable transcript. Most of pi-mono's polish is wonderful and most of it can wait.

**In scope (Phase 2):**
- One self-contained `.html` file, no external assets
- Linear message feed: user → assistant text → tool calls → tool results → next user → ...
- Markdown rendering for user and assistant text
- Code blocks with syntax highlighting (Pygments → inline HTML, no JS)
- Tool calls shown as a labeled card with arguments
- Tool results as collapsible blocks (collapsed by default if > N lines), open via a tiny inline `<details>` (no JS needed)
- Inline CSS
- Header showing model + total turn count

**Out of scope (Phase 2), can come back later:**
- Tree / branching navigation
- Sidebar, search, filters
- ANSI escape handling (our coding tools don't emit ANSI by default)
- Dark/light theme derivation
- Image lightbox / image rendering at all (Phase 1 tools don't return images)
- Deep linking / URL-driven state
- Round-trip: downloading the JSONL back from the page
- Compaction / branch markers / model-change badges (Phase 3 concerns)

## Patterns we *do* borrow

### 1. Single self-contained file

All CSS in a `<style>` block. No external fonts, no CDNs, no asset directory. Pi-mono goes further and embeds JS libraries — we don't need any JS in v1; `<details>` handles collapsing natively.

### 2. Static rendering, no client-side JS for v1

Pi-mono renders client-side (template.js), reading a base64-encoded session JSON from a hidden `<script type="application/json">` tag. That's the right pattern for their feature set (tree, search, filters), but for a linear transcript we can just emit the rendered HTML server-side. **No JS, no embedded JSON, no DOM manipulation.** The `.html` file is fully readable as soon as it's open.

If we later want round-trip / re-rendering, embed the session JSON in a `<script type="application/json" id="session-data">` tag (base64-decoded by future JS). Easy to add later, no cost to defer.

### 3. Per-content-type rendering

| Content kind | Rendering |
|---|---|
| `UserMessage(content=str)` | One `<section class="user">` with the text in a `<div class="markdown">` (markdown-rendered) |
| `UserMessage(content=list[...])` | Iterate blocks; text → markdown, image → `<img src="data:..." />` (Phase 2 may skip image case) |
| `AssistantMessage` text blocks | One or more `<div class="markdown">` (markdown-rendered) inside `<section class="assistant">` |
| `AssistantMessage` tool calls | A `<div class="tool-call">` block: tool name + JSON-pretty arguments in a code block |
| `ToolResultMessage` | A `<details>` block summarized as `<summary>tool-name</summary>` with the content inside; `is_error=True` adds an "error" CSS class for red border |

### 4. HTML escaping is non-optional and pervasive

Every piece of text from the session must be `html.escape(..., quote=True)`'d before it lands in the output. Pi-mono uses `.replace()`-based escaping (template.js:591–595, ansi-to-html.ts:63–69). Python: `html.escape`. The model's output contains `<` / `>` / `&` constantly; double-escaping is harmless, missing one is XSS even for "internal" tools.

### 5. Code-block syntax highlighting via the markdown library

If we use `markdown` with the `fenced_code` + `codehilite` extensions, code blocks become `<pre><code class="language-...">` with Pygments-generated `<span class="...">` tokens. Add Pygments' inline CSS once at the top of the page.

Alternative: `mistune` with a custom renderer + Pygments. Slightly more code, slightly faster, identical output.

## Translation guide for Python

| TS pattern | Python equivalent |
|---|---|
| marked.js (markdown → HTML) | `markdown` (with `fenced_code`, `codehilite`, `tables`, `nl2br`) or `mistune` |
| highlight.js (syntax highlighting) | `pygments` — generate inline CSS once via `HtmlFormatter().get_style_defs()` |
| ansi-to-html | Skip in v1. If/when needed: `ansi2html` package or hand-rolled parser. |
| Base64-embedded session JSON | Skip in v1. Add later as `<script type="application/json" id="session-data">{json}</script>` |
| Hamburger sidebar / tree navigator | Don't port. |
| `<details>` collapsibles | Use the HTML element directly, no JS. |
| `marked.parse()` per message | Pre-render at export time, not at view time. Static HTML. |

## What Phase 2 implements

- `src/midge/session.py` (or `src/midge/export.py` — name TBD; if we anticipate session save/load in Phase 3 the file should be `session.py` and the export function lives there as a sibling to save/load):
  - `def export_html(messages: list[Message], *, title: str = "midge session", model: str = "") -> str` — returns the full HTML document as a string.
  - Helpers: `_render_user`, `_render_assistant`, `_render_tool_result`, `_render_tool_call`, `_md(text)`.
- A constant `_PAGE_TEMPLATE` string with placeholders for `{title}`, `{model}`, `{message_count}`, `{pygments_css}`, `{body}`.
- Tests in `tests/test_export.py`:
  - Smoke test: render a fixture with one of each message type → assert HTML is valid (well-formed, escapes preserved, markdown rendered).
  - Tool-error test: `is_error=True` shows up with the error class.
  - XSS-style test: a user message containing `<script>alert(1)</script>` is escaped, not executed.

~150 lines for the exporter + tests. Add `markdown` and `pygments` to `pyproject.toml`.
