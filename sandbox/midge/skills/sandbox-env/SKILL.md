---
name: sandbox-env
description: >-
  Describes the Docker Sandbox midge is running in: available package managers
  (uv, Poetry, conda, npm, apt), network rules, and how to persist environment
  variables. Use before installing dependencies, creating environments, running
  tests, or when a network request is refused.
---

# Sandbox environment

You run as user `agent` inside a Docker Sandbox microVM. The project directory
is mounted from the host; changes there are visible to the user immediately.
Everything outside it is private to this sandbox.

## Package managers

Use the one the project already uses. Check for these files first:

| File | Tool | Typical commands |
|---|---|---|
| `poetry.lock` | Poetry | `poetry install`, `poetry run pytest` |
| `uv.lock` or `[tool.uv]` | uv | `uv sync`, `uv run pytest` |
| `environment.yml` | conda | `conda env create -f environment.yml`, `conda run -n <env> pytest` |
| `requirements.txt` only | uv | `uv venv && uv pip install -r requirements.txt` |
| `package.json` | npm | `npm ci`, `npm test` |

- `conda activate` needs an interactive shell. Use `conda run -n <env> <cmd>`.
- System packages: `sudo apt-get install -y <pkg>`.
- Do not install into the system Python with `pip`. Use a project environment.

## Environment variables

Each `bash` call is a new shell. To keep a variable for later calls, append an
export to `/etc/sandbox-persistent.sh`:

    echo 'export MY_VAR=value' >> /etc/sandbox-persistent.sh

## Network

Outbound traffic passes through a host proxy that enforces an allowlist. A
refused connection (HTTP 403 from the proxy, or a connection error to one
host) means the destination is not allowed. Report the host name to the user
and stop; do not try to route around the proxy.

Credentials such as `OPENAI_API_KEY` contain placeholders. The proxy inserts
the real value. Do not print, copy, or test them.
