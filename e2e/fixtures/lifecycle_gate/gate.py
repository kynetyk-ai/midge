"""Refuses startup from a session_start hook."""

from midge.hooks import CancelResult


def register_hooks(hooks):
    hooks.on("session_start", lambda event, ctx: CancelResult(cancel=True))
