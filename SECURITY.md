# Security

## Threat model

`midge` is a local CLI / TUI that runs an LLM agent in your shell. The threat surface is small but real:

- **The `bash` tool executes whatever shell command the model emits.** That is its job. If you point the agent at an untrusted model or paste an untrusted prompt, the model can run arbitrary commands as the user that launched `midge`. Treat the agent like a shell session — don't run it as root, don't run it against random servers, and review the prompt and tool calls before approving destructive actions.
- **Session JSONL files contain the verbatim conversation,** including any secrets or paths you mentioned. Treat them like shell history.
- **Custom extensions that hit the network** can be a vector. Validate inputs and prefer narrow APIs over `bash`.

These are intentional design choices, not bugs.

## What I'd consider a real vulnerability

- Crashes or arbitrary code execution triggered by **input from the LLM endpoint** (a malicious server returning crafted streaming chunks, tool-call arguments, or `usage` payloads) that escape the harness's parsing.
- HTML escape misses in `midge.session.export_html` that allow an attacker who controls part of a session to inject script tags into the rendered transcript.
- RPC parse handling in `midge.rpc` that lets a malicious stdin stream crash the process or escape the command dispatcher.
- Path traversal or injection in the `read` / `edit` / `write` tools beyond the usual "the LLM asked for it" surface — i.e., something the model can accidentally trigger via well-formed-looking arguments.
- Session loader (`midge.persistence.Session.load`) parsing issues that let a crafted JSONL file crash or hang the process.

## How to report

Use GitHub's **private vulnerability reporting** for this repository:

1. Open the **Security** tab on the repo.
2. Click **Report a vulnerability**.
3. Fill out the advisory form (description, impact, repro).

This routes the report directly to the maintainer privately; the issue is not visible to anyone else until a fix ships and the advisory is published.

Please don't file public GitHub issues for vulnerabilities.

I'll acknowledge receipt within a few days. This is a personal project, so response time is best-effort — if the issue is severe or actively exploited, please flag that in the advisory.

## Out of scope

- "The agent did a thing I didn't want." That's a prompting / tool-design problem, not a security issue.
- "The bash tool ran the command I asked it to." Working as designed.
- "I committed my session.jsonl to a public repo and now my API key is leaked." That's on you. (See [`.gitignore`](./.gitignore).)
- Issues in upstream dependencies (`openai`, `pydantic`, `mistune`, `pygments`, `textual`) — please report those to their respective projects.

## Disclosure

Once a fix lands, the commit message and (if material) a release note will describe the issue and credit the reporter unless they ask to remain anonymous.
