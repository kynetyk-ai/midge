"""Profiles the shipped example cannot express.

`examples/profile_extension` declares `tools=("read", "bash")`, so it cannot
answer the question phase 2 left open: the skills catalogue is only injected
when `read` is in the registry, and every CLI run loads the built-ins. A profile
that projects `read` away is the only way to reach that gate.

`no_hooks` exists to trip `profile_hook_undecided` — it names no hook sources at
all, so loading it beside an extension that registers one is a validation case
rather than a runtime one.
"""

from __future__ import annotations

from midge.profiles import Profile

# Deliberately no `read`: this is what removes the skills catalogue.
BLIND = Profile(
    name="blind",
    description="Can run commands but cannot open a file. Tests the catalogue gate.",
    prompt="You cannot read files directly. Use shell commands only.",
    tools=("bash",),
    hooks={"approve": True},
)

# Names no hook sources, so any registered source is undecided for it.
NO_HOOKS = Profile(
    name="no-hooks",
    description="Declares nothing about hooks. Tests profile_hook_undecided.",
    prompt="You are terse.",
    tools=("read", "bash"),
    hooks={},
)
