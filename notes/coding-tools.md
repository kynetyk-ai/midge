# Coding tools — patterns to borrow from `pi-mono`

Source: `pi-mono/packages/coding-agent/src/core/tools/{read,bash,edit,write,edit-diff,truncate}.ts`

The four coding tools are independently specified — each has its own argument schema, return shape, and error model. They share two utilities: truncation and (for `edit`) line-ending/Unicode-aware diffing.

## General conventions

- **Async-first.** All tools are async. Python port uses `asyncio` throughout.
- **Errors raise.** Tools raise exceptions and the harness catches them and converts to a tool-error result. There is no error-as-result pattern inside the tool.
- **Path resolution is cwd-bound.** All file paths resolve relative to the agent's cwd. Never trust user paths verbatim.
- **AbortSignal everywhere.** Each tool checks abort before and after I/O. In Python this is `asyncio.CancelledError` propagation plus explicit checks on long-running operations.
- **Mutation queue.** Writes to the same file serialize so concurrent edits don't race.

## `read`

- Args: `path` (str, required), `offset` (int, optional, 1-indexed), `limit` (int, optional).
- Limits: 2000 lines OR 50KB, whichever hits first. Head-truncated.
- Edge case: if first line alone exceeds 50KB, return an error suggesting `sed`/`head` rather than a partial first line.
- Image files: detect by MIME type; if model supports vision, return text note + base64 image (resized to max 2000×2000). Otherwise text-only note.
- On truncation, the success message tells the model how to continue: "use offset=N to continue".
- Trim trailing empty lines from rendered output.

## `bash`

- Args: `command` (str, required), `timeout` (int seconds, optional).
- Output limit: 2000 lines OR 50KB. **Tail-truncated** (keep last N), opposite of `read`.
- Above 50KB total output, spill the full output to a temp file and include the path in the result (so the model can read it back if needed).
- stdout and stderr are interleaved into a single stream.
- Spawn detached so grandchildren can survive parent death; kill the entire process group on timeout/abort.
- Streams partial updates during execution (the harness sees a running tail).

## `edit`

- Args: `path` (str), `edits` (list of `{oldText, newText}`).
- All edits matched against the **original** file content, not incrementally — prevents compounding errors.
- Reject overlapping edits.
- **Line-ending preservation.** Detect CRLF vs LF on read; normalize to LF for matching; restore on write.
- **BOM preservation.** Strip before matching, prepend back to output.
- **Fuzzy fallback.** If exact match fails, try with NFKC normalization, smart quotes → ASCII, smart dashes → ASCII, special spaces → regular space, trailing whitespace stripped per line.
- Result includes a unified diff and the 1-indexed first changed line (useful for editor jump).

## `write`

- Args: `path` (str), `content` (str).
- Always overwrites. No existence check.
- Create parent directories with `mkdir -p` semantics.
- UTF-8 output.
- Result message reports byte count (the JS source uses `content.length` which is char count — Python should use `len(content.encode("utf-8"))`).

## What to skip in v1

- Pluggable operations (SSH/remote variants of these tools). Out of scope for our local-only port.
- Spawn hooks for command interception. Add later if a real use case appears.
- Highlight-cache integration with the TUI's edit panel. We're not porting the TUI panel.

## Translation guide for Python

| TS pattern | Python equivalent |
|---|---|
| `AbortSignal` | `asyncio.CancelledError` propagation; explicit checks for long-running loops |
| `fs/promises` | `aiofiles` or `asyncio.to_thread(open, ...)` for small reads |
| `child_process.spawn` | `asyncio.create_subprocess_exec` with `process_group=0` for process-group kill |
| Process-group kill | `os.killpg(os.getpgid(proc.pid), signal.SIGTERM)` |
| `partial-json` | `jiter` (already a transitive dep via `openai`) or hand-rolled buffer-and-retry |
| Unified diff | `difflib.unified_diff` |
| NFKC normalize | `unicodedata.normalize("NFKC", s)` |
