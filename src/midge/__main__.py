"""`python -m midge` — the same entrypoint as the `midge` script.

For environments where the console script is not on PATH, and for tests that
spawn midge as a subprocess with the interpreter they are running under.
"""

from midge.cli import main

main()
