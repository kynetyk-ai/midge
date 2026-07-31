# A disposable midge, driven over RPC from outside the container.
#
#   docker build -f harness/Dockerfile -t midge-test .
#   docker run -d --name midge --env-file .env midge-test
#
# `git` is not incidental: the workspace is a git repository so the
# `commit-message` skill has real staged changes to describe, and `bash` has a
# command that can genuinely fail.
FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/midge

# Source before fixtures: the layer that changes least often is cached.
COPY pyproject.toml poetry.lock README.md ./
COPY src ./src
# `pytest` is the workspace's, not midge's: the toybox project has a suite, and
# a scenario where `bash` runs it — and makes it fail — needs a runner present.
RUN pip install --no-cache-dir . pytest

# midge's own shipped examples, which are project-agnostic and load as they are.
COPY examples ./examples
COPY harness/fixtures /opt/harness

COPY harness/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# There is no --model flag; the model comes from the environment or the config
# file, and nothing reads a .env, so it is set here.
ENV MIDGE_MODEL=gpt-5.4-mini \
    MIDGE_LOG_LEVEL=DEBUG \
    MIDGE_LOG_FILE=/run/midge/midge.log \
    MIDGE_SESSION_DIR=/run/midge/sessions \
    MIDGE_NOTES_KB=/run/midge/notes.json \
    PYTHONUNBUFFERED=1

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
