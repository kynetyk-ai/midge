"""A profile: what the agent is, declared once and discovered.

A profile bundles a system prompt, a model, a tool subset and a set of active
hooks under a name, so that "the adversarial reviewer" is one thing rather than
three changes a reader has to correlate. See
`docs/adr/0001-session-profiles.md`.

    midge --extension-dir examples/profile_extension \\
          --extension-dir examples/approval_extension \\
          --profile adversarial-reviewer

Both directories, because this profile decides the `approve` hook: a profile
that names a tool, hook or registered model which does not exist is dropped at
startup with a diagnostic, rather than loading and silently granting less than
it claims. Run it without `--extension-dir examples/approval_extension` to see
that refusal — and note that the reverse is a refusal too, since loading a hook
source this profile says nothing about is what `profile_hook_undecided`
catches.

A profile file is an ordinary extension file — one `.py` may declare a tool, a
sub-agent and a profile together. There is no `--profile-dir`.

`--profile` applies it at startup. Over RPC, `use_profile` switches at runtime —
every dimension at once, refused mid-turn, and recorded so any message is
attributable to the configuration that produced it:

    {"id":"1","type":"use_profile","name":"adversarial-reviewer",
     "transcript":"fork"}

`transcript` says where the turns go. `continue` stays on this one; `fork` opens
a linked transcript so an excursion's turns do not sit in the original file; and
`resume_last` returns to this session's most recent thread under the named
profile, which is what makes a fork a round trip rather than a one-way door.
"""

from __future__ import annotations

from midge.profiles import Profile

ADVERSARIAL = Profile(
    name="adversarial-reviewer",
    description="Reviews work that has just been done, looking for what is wrong with it.",
    # Read-only on purpose. A reviewer that can edit will fix what it finds
    # instead of reporting it, and the report is the product.
    tools=("read", "bash"),
    # A decision for *every* hook source that is loaded, keyed by the extension
    # file's stem — `examples/approval_extension/approve.py` is `"approve"`.
    # Leaving one out is a validation failure, not a default: omitting a tool
    # makes the agent less capable, but omitting a hook would make it unguarded,
    # so the one dimension where silence grants capability refuses to be silent.
    hooks={"approve": True},
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
