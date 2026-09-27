"""A stand-in for an OpenAI-compatible server, for tests that cross a process.

The in-process fakes (`tests/fakes.py`) replace the provider object, which a
`midge --rpc` subprocess cannot see. This serves the real wire instead —
`POST /v1/chat/completions` answered with server-sent events — so the child
runs its real provider, SDK and HTTP stack against scripted replies.

Standard library only: `http.server` on port 0 in a daemon thread. It records
every request body, so a test can check what history a resumed agent sent.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class FakeOpenAI:
    """`replies` are, in order, either text for the model to say or a tool call
    to make: `{"tool": "add_note", "arguments": {"title": "x", ...}}`."""

    def __init__(self, replies: list[str | dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                fake.bodies.append(json.loads(self.rfile.read(length)))
                reply = fake.replies.pop(0) if fake.replies else "(no reply scripted)"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if isinstance(reply, dict):
                    call = {
                        "index": 0,
                        "id": f"call_{len(fake.bodies)}",
                        "type": "function",
                        "function": {
                            "name": reply["tool"],
                            "arguments": json.dumps(reply["arguments"]),
                        },
                    }
                    steps = (({"tool_calls": [call]}, None), ({}, "tool_calls"))
                else:
                    steps = (({"content": reply}, None), ({}, "stop"))
                for delta, finish in steps:
                    chunk = {
                        "id": "fake",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "fake",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, format: str, *args: Any) -> None:
                pass  # the test's output is not the place for an access log

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}/v1"

    def __enter__(self) -> FakeOpenAI:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
