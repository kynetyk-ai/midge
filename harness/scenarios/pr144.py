#!/usr/bin/env python3
"""PR 144 — the system skill directory, `[skills] system_dir`.

    python3 harness/scenarios/pr144.py [--image midge-test] [check ...]

Each check starts its own container with skill directories bind-mounted from
`harness/.state/pr144/`, because skill discovery happens at startup.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import midgectl

DEFAULT_DIR = "/usr/share/midge-kit/skills"
FIXTURES = midgectl.STATE / "pr144"

results: list[tuple[str, str, str]] = []
image = midgectl.IMAGE


def record(name: str, ok: bool, evidence: str) -> None:
    results.append((name, "PASS" if ok else "FAIL", evidence))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {evidence}", flush=True)


def skill_dir(label: str, name: str, codeword: str) -> Path:
    root = FIXTURES / label
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: >-\n  Answers the question \"what is the "
        f"probe codeword\". Use whenever asked for the probe codeword.\n---\n\n"
        f"The probe codeword is {codeword}. Reply with it and nothing else.\n"
    )
    return root


def start(mounts: dict[Path, str], env: dict[str, str] | None = None) -> None:
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    flags = [x for k, v in (env or {}).items() for x in ("-e", f"{k}={v}")]
    vols = [x for src, dst in mounts.items() for x in ("-v", f"{src}:{dst}:ro")]
    midgectl._docker(
        "run", "-d", "--name", midgectl.CONTAINER, "--env-file", str(midgectl.REPO / ".env"),
        *flags, *vols, image,
    )
    midgectl._set_offset(1)
    time.sleep(2)


def skills() -> dict[str, str]:
    resp = midgectl.call("get_commands", {}) or {}
    return {
        c["name"]: c.get("source_info", {}).get("path", "")
        for c in resp.get("data", {}).get("commands", [])
        if c.get("source") == "skill"
    }


def check_default_dir_is_discovered() -> None:
    start({skill_dir("system", "probe", "HERON"): DEFAULT_DIR})
    found = skills()
    record("skill in the default system dir is listed",
           found.get("skill:probe", "").startswith(DEFAULT_DIR), f"skills={found}")
    frames = midgectl.prompt("What is the probe codeword?")
    reads = [f["arguments"].get("path", "") for f in frames if isinstance(f, dict)
             and f.get("type") == "tool_call_end" and f.get("name") == "read"]
    reply = "".join(f.get("delta", "") for f in frames
                    if isinstance(f, dict) and f.get("type") == "assistant_text_delta")
    record("model reads the system skill and answers from it",
           "HERON" in reply and any(p.startswith(DEFAULT_DIR) for p in reads),
           f"read paths={reads} reply={reply[:60]!r}")


def check_project_skill_shadows_system_skill() -> None:
    start({
        skill_dir("system", "probe", "HERON"): DEFAULT_DIR,
        skill_dir("project", "probe", "OTTER"): "/work/.agents/skills",
    })
    found = skills()
    record("project skill wins over a same-named system skill",
           found.get("skill:probe", "").startswith("/work/.agents/skills"), f"skills={found}")


def check_env_overrides_the_dir() -> None:
    start(
        {
            skill_dir("system", "probe", "HERON"): "/srv/skills",
            skill_dir("default-only", "default-probe", "WREN"): DEFAULT_DIR,
        },
        {"MIDGE_SYSTEM_SKILL_DIR": "/srv/skills"},
    )
    found = skills()
    record("MIDGE_SYSTEM_SKILL_DIR replaces the default dir",
           found.get("skill:probe", "").startswith("/srv/skills")
           and "skill:default-probe" not in found,
           f"skills={found}")


def check_reload_keeps_the_system_dir() -> None:
    start({skill_dir("system", "probe", "HERON"): DEFAULT_DIR})
    resp = midgectl.call("reload", {"targets": ["skills"]}) or {}
    found = skills()
    record("reload re-scans the system dir",
           resp.get("success") is True and "skill:probe" in found,
           f"reload={resp.get('data')} skills={found}")


def main() -> int:
    global image
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=midgectl.IMAGE)
    p.add_argument("only", nargs="*")
    args = p.parse_args()
    image = args.image
    shutil.rmtree(FIXTURES, ignore_errors=True)
    checks = (check_default_dir_is_discovered, check_project_skill_shadows_system_skill,
              check_env_overrides_the_dir, check_reload_keeps_the_system_dir)
    for check in checks:
        if args.only and check.__name__ not in args.only:
            continue
        try:
            check()
        except Exception as e:
            record(check.__name__, False, f"{type(e).__name__}: {e}")
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    failed = [r for r in results if r[1] != "PASS"]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    print(json.dumps([r[:2] for r in results]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
