"""Writes a marker from a cleanup, so a scenario can see cleanups run at exit.

The marker goes under /run/midge/sessions, which the e2e scenarios mount on the host.
"""

from pathlib import Path


def register_hooks(hooks):
    hooks.add_cleanup(lambda: Path("/run/midge/sessions/cleanup-marker").write_text("ok"))
