#!/bin/sh
# Run midge as PID 1 with a control channel the host can write to repeatedly.
#
# `serve_stdio` shuts down on stdin EOF, and every `docker exec` is a separate
# process — so a per-command writer closing the FIFO would kill the server after
# one command. `sleep infinity` holds a writer open forever so the last-writer
# -closed condition never arrives.
#
# stdout goes to a *file* rather than a second FIFO, so the driver can read from
# a byte offset and never lose frames between calls.
set -eu

RUN=/run/midge
mkdir -p "$RUN/sessions" /work

rm -f "$RUN/in"
mkfifo "$RUN/in"
sleep infinity > "$RUN/in" &

# A fresh copy of the workspace on every container start, which is what makes
# `reset` mean something.
cp -a /opt/harness/workspace/. /work/
mkdir -p /work/.midge
cp /opt/harness/config.toml /work/.midge/config.toml
cd /work
if [ ! -d .git ]; then
    git init -q
    git config user.email harness@example.com
    git config user.name "midge harness"
    git add -A
    git commit -qm "toybox: initial commit"
fi

# `exec` so midge is PID 1 and `docker stop` reaches the SIGTERM/SIGHUP handlers
# `serve_stdio` installs.
exec midge --rpc "$@" < "$RUN/in" >> "$RUN/out" 2>> "$RUN/err"
