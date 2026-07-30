"""A profile: what the agent is, declared once and discovered.

A profile bundles a system prompt, a model, a tool subset and a set of active
hooks under a name, so that "the adversarial reviewer" is one thing rather than
three changes a reader has to correlate. See
`docs/adr/0001-session-profiles.md`.

    midge --extension-dir examples/profile_extension \\
          --extension-dir examples/approval_extension \\
          --profile adversarial-reviewer

Both directories, because this profile names the `approve` hook: a profile that
names a tool, hook or registered model which does not exist is dropped at
startup with a diagnostic, rather than loading and silently granting less than
it claims. Run it without `--extension-dir examples/approval_extension` to see
that refusal.

A profile file is an ordinary extension file — one `.py` may declare a tool, a
sub-agent and a profile together. There is no `--profile-dir`.

Note what this does *not* do yet: **nothing applies a profile.** Discovery,
validation and enumeration (`get_profiles` over RPC) are all that exist today;
switching to a profile at runtime is issue #67.
"""

from __future__ import annotations

from midge.profiles import Profile

ADVERSARIAL = Profile(
    name="adversarial-reviewer",
    description="Reviews work that has just been done, looking for what is wrong with it.",
    # Read-only on purpose. A reviewer that can edit will fix what it finds
    # instead of reporting it, and the report is the product.
    tools=("read", "bash"),
    # Named by the extension file's stem — `examples/approval_extension/approve.py`.
    hooks=("approve",),
    # Omitted, so the profile keeps whatever model the agent is already running.
    # Set it to pin a reviewer to a stronger model than the builder used; with a
    # `[models]` table configured, the name has to be one you registered.
    prompt="""
You are reviewing work that has just been done. Assume it is wrong and find out
how. Cite `path:line` for every claim — a finding without a location is not a
finding. Prefer one defect you can demonstrate over five you suspect. Say
plainly when you have found nothing.
""".strip(),
)
