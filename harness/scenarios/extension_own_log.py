#!/usr/bin/env python3
"""An extension that defines its own module-level `log` keeps it.

    python3 harness/scenarios/extension_own_log.py [--image midge-test]
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

EXT = """import logging
from midge.tools import tool

log = logging.getLogger("probe.custom")


@tool(read_only=True)
async def probe() -> str:
    \"\"\"Returns the probe codeword.\"\"\"
    log.warning("probe_called")
    return "KESTREL"
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=midgectl.IMAGE)
    image = p.parse_args().image
    ext_dir = midgectl.STATE / "extension_own_log"
    shutil.rmtree(ext_dir, ignore_errors=True)
    ext_dir.mkdir(parents=True)
    (ext_dir / "probe_ext.py").write_text(EXT)

    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    midgectl._docker(
        "run", "-d", "--name", midgectl.CONTAINER, "--env-file", str(midgectl.REPO / ".env"),
        "-v", f"{ext_dir}:/opt/ext:ro", image, "--extension-dir", "/opt/ext",
    )
    midgectl._set_offset(1)
    time.sleep(2)
    results = []
    try:
        frames = midgectl.prompt("Call the probe tool and reply with what it returns.")
        called = [f["name"] for f in frames
                  if isinstance(f, dict) and f.get("type") == "tool_call_end"]
        # A logger outside the `midge` tree bypasses midge's handler and reaches
        # Python's default stderr handler, which prints the bare message.
        log = midgectl._docker("exec", midgectl.CONTAINER, "cat", f"{midgectl.RUN}/midge.log")
        err = midgectl._docker("exec", midgectl.CONTAINER, "cat", f"{midgectl.RUN}/err")
        in_log = next((ln for ln in log.splitlines() if "probe_called" in ln), "")
        results.append(("model called the extension's tool", "probe" in called,
                        f"tool calls={called}"))
        results.append(("record is not renamed to midge.ext.probe_ext",
                        "midge.ext.probe_ext" not in in_log,
                        in_log[-70:] or "no probe_called line in midge.log"))
        results.append(("record reaches the extension's own logger (stderr)",
                        "probe_called" in err and not in_log,
                        f"probe_called in stderr={'probe_called' in err}"))
    except Exception as e:
        results.append(("scenario", False, f"{type(e).__name__}: {e}"))
    midgectl._docker("rm", "-f", midgectl.CONTAINER, check=False)
    for name, ok, ev in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {ev}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    print(json.dumps([(r[0], "PASS" if r[1] else "FAIL") for r in results]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
