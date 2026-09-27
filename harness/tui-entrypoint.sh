#!/bin/sh
# Run the TUI on a real terminal.
#
# Deliberately not the RPC entrypoint with a flag: that one redirects stdin from
# a FIFO and stdout to a file, which is exactly wrong for a program that draws.
# Here stdio is left alone and `docker run -it` supplies the TTY Textual needs.
#
# Logs still go to MIDGE_LOG_FILE. That matters more here than in RPC mode:
# `tui_log_handler` returns None when a log file is configured, so records land
# in the file instead of `TextualHandler`, where they would only be visible
# under `textual console`.
set -eu

RUN=/run/midge
mkdir -p "$RUN/sessions" /work

cp -a /opt/harness/workspace/. /work/
mkdir -p /work/.midge
cp "/opt/harness/${MIDGE_HARNESS_CONFIG:-config.toml}" /work/.midge/config.toml
cd /work
if [ ! -d .git ]; then
    git init -q
    git config user.email harness@example.com
    git config user.name "midge harness"
    git add -A
    git commit -qm "toybox: initial commit"
fi

exec midge "$@"
