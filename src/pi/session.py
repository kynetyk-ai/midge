"""HTML export for agent sessions.

Renders a list of `Message`s to a single self-contained HTML file. Server-side
rendering only — no client-side JS, no external assets, no embedded session
JSON for round-trip. Tool results use `<details>` for native collapsing.

Markdown is rendered with mistune's `escape=True` so any raw HTML in the input
(including LLM output) is escaped rather than passed through. Code blocks are
highlighted via Pygments; CSS for highlighting is generated and inlined.
"""

from __future__ import annotations

import html
import json
from string import Template
from typing import Any

import mistune
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

from pi.messages import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


class _HighlightRenderer(mistune.HTMLRenderer):
    def block_code(self, code: str, info: str | None = None, **kwargs: Any) -> str:
        lexer = None
        if info:
            try:
                lexer = get_lexer_by_name(info.strip().split()[0], stripall=True)
            except (ClassNotFound, IndexError):
                lexer = None
        if lexer is not None:
            formatter = HtmlFormatter(cssclass="codehilite")
            return highlight(code, lexer, formatter)
        return super().block_code(code, info=info)


_md_render = mistune.create_markdown(
    escape=True,
    renderer=_HighlightRenderer(),
    plugins=["table", "url"],
)


_BASE_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  max-width: 920px;
  margin: 2rem auto;
  padding: 0 1rem;
  line-height: 1.5;
  color: #1a1a1a;
}
header { border-bottom: 1px solid #ddd; margin-bottom: 1.5rem; padding-bottom: 0.5rem; }
header h1 { margin: 0; font-size: 1.4rem; }
header .meta { color: #666; font-size: 0.85rem; margin-top: 0.25rem; }
.msg { padding: 0.75rem 1rem; margin-bottom: 1rem; border-radius: 6px;
       border: 1px solid #e5e5e5; }
.msg .role { font-weight: 600; font-size: 0.75rem; text-transform: uppercase;
             letter-spacing: 0.04em; color: #666; margin-bottom: 0.5rem; }
.msg.user { background: #f6f8fa; }
.msg.assistant { background: #fff; }
.msg.tool-result { background: #fafafa; padding: 0.5rem 0.75rem; }
.msg.tool-result.error { border-color: #d33; background: #fff5f5; }
.msg.tool-result summary { cursor: pointer; font-family: ui-monospace, monospace;
                           font-size: 0.85rem; font-weight: 600; user-select: none; }
.tool-output { background: #f0f0f0; padding: 0.5rem; border-radius: 4px;
               overflow-x: auto; font-size: 0.85rem; max-height: 30rem;
               overflow-y: auto; margin: 0.5rem 0 0; }
.tool-call { background: #f0f4ff; border: 1px solid #d0d8e8; border-radius: 4px;
             padding: 0.5rem 0.75rem; margin: 0.5rem 0; }
.tool-call .tool-name { font-family: ui-monospace, monospace; font-weight: 600;
                        color: #2a4a7a; font-size: 0.85rem; margin-bottom: 0.25rem; }
.tool-args { background: #fff; padding: 0.5rem; margin: 0.25rem 0 0;
             border-radius: 3px; font-size: 0.8rem; overflow-x: auto; }
pre, code { font-family: ui-monospace, "SF Mono", Cascadia, Menlo, Consolas, monospace; }
pre { overflow-x: auto; padding: 0.5rem; background: #f0f0f0; border-radius: 4px; }
:not(pre) > code { background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px;
                   font-size: 0.9em; }
.codehilite { background: #f8f8f8; border-radius: 4px; padding: 0.5rem;
              overflow-x: auto; margin: 0.5rem 0; }
img { max-width: 100%; height: auto; border-radius: 4px; }
"""

_PAGE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>$title</title>
<style>
$base_css
$pygments_css
</style>
</head>
<body>
<header>
  <h1>$title</h1>
  <div class="meta">$meta</div>
</header>
<main>
$body
</main>
</body>
</html>
"""
)


def export_html(
    messages: list[Message],
    *,
    title: str = "pi session",
    model: str = "",
) -> str:
    pygments_css = HtmlFormatter(cssclass="codehilite").get_style_defs(".codehilite")
    body = "\n".join(_render(m) for m in messages)

    meta = f"{len(messages)} message{'s' if len(messages) != 1 else ''}"
    if model:
        meta = f"model: {html.escape(model)} · {meta}"

    return _PAGE.substitute(
        title=html.escape(title),
        meta=meta,
        base_css=_BASE_CSS,
        pygments_css=pygments_css,
        body=body,
    )


def _render(m: Message) -> str:
    if isinstance(m, UserMessage):
        return _render_user(m)
    if isinstance(m, AssistantMessage):
        return _render_assistant(m)
    if isinstance(m, ToolResultMessage):
        return _render_tool_result(m)
    return ""


def _render_user(m: UserMessage) -> str:
    if isinstance(m.content, str):
        body = _md(m.content)
    else:
        parts: list[str] = []
        for block in m.content:
            if isinstance(block, TextContent):
                parts.append(_md(block.text))
            elif isinstance(block, ImageContent):
                parts.append(
                    f'<img src="data:{html.escape(block.mime_type, quote=True)};'
                    f'base64,{html.escape(block.data, quote=True)}" alt="">'
                )
        body = "\n".join(parts)
    return f'<section class="msg user"><div class="role">user</div>{body}</section>'


def _render_assistant(m: AssistantMessage) -> str:
    parts: list[str] = []
    for block in m.content:
        if isinstance(block, TextContent):
            if block.text:
                parts.append(f'<div class="text">{_md(block.text)}</div>')
        elif isinstance(block, ToolCall):
            args_pretty = json.dumps(block.arguments, indent=2, ensure_ascii=False)
            parts.append(
                f'<div class="tool-call">'
                f'<div class="tool-name">{html.escape(block.name)} '
                f'<small>({html.escape(block.id)})</small></div>'
                f'<pre class="tool-args"><code>{html.escape(args_pretty)}</code></pre>'
                f"</div>"
            )
    body = "\n".join(parts) or '<div class="text"><em>(empty)</em></div>'
    role_label = "assistant"
    if m.stop_reason:
        role_label += f" · stop: {html.escape(m.stop_reason)}"
    return (
        f'<section class="msg assistant">'
        f'<div class="role">{role_label}</div>'
        f"{body}"
        f"</section>"
    )


def _render_tool_result(m: ToolResultMessage) -> str:
    text_parts = [c.text for c in m.content if isinstance(c, TextContent)]
    text = "".join(text_parts)
    error_class = " error" if m.is_error else ""
    label = m.tool_name or m.tool_call_id
    if m.is_error:
        label = "⚠ " + label
    return (
        f'<details class="msg tool-result{error_class}">'
        f"<summary>{html.escape(label)}</summary>"
        f'<pre class="tool-output"><code>{html.escape(text)}</code></pre>'
        f"</details>"
    )


def _md(text: str) -> str:
    rendered = _md_render(text)
    return rendered if isinstance(rendered, str) else str(rendered)
