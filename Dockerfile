# midge as a coding agent, working on a project mounted at /workspace.
#
#   docker build -t midge .
#   docker run -it --rm --detach-keys ctrl-^ \
#     -e OPENAI_API_KEY -v "$PWD":/workspace midge
#
# Arguments after the image name reach `midge` (`--continue`, `--rpc`, ...).
# `--detach-keys` is needed for the TUI: Docker's default detach sequence starts
# with Ctrl+P, which it would otherwise swallow before midge sees it.
#
# The container is the boundary for what the agent can touch. It sees the
# mounted project and nothing else of the host, unless more is mounted.

# --- build: resolve dependencies from poetry.lock -----------------------------
FROM python:3.13-slim AS build

RUN pip install --no-cache-dir poetry==2.2.1

WORKDIR /opt/midge
ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --extras tui --no-root
COPY src ./src
RUN poetry install --only main --extras tui

# --- runtime ------------------------------------------------------------------
FROM python:3.13-slim

# What the `bash` tool is likely to reach for in a coding session. Extend with
# a `FROM midge` image for a project's own toolchain (compilers, node, ...).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git ca-certificates curl ripgrep jq less procps make openssh-client \
 && rm -rf /var/lib/apt/lists/* \
 # The mounted project is owned by the host user, which git otherwise refuses.
 && git config --system --add safe.directory '*'

# Match the host user so files the agent writes into the mount are yours.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid "$GID" midge \
 && useradd --uid "$UID" --gid "$GID" --create-home --shell /bin/bash midge

COPY --from=build /opt/midge /opt/midge
COPY examples /opt/midge/examples

ENV PATH=/opt/midge/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    COLORTERM=truecolor

USER midge
WORKDIR /workspace
ENTRYPOINT ["midge"]
