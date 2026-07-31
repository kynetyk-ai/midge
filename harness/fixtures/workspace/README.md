# toybox

A deliberately small project. It exists so an agent has real files to read,
edit and run — not to be useful.

- `src/toybox/settings.py` — the config surface: defaults in a dataclass, one loader.
- `src/toybox/text.py` — string helpers with real edge cases.
- `src/toybox/tally.py` — counting, with one function that is wrong on purpose.
- `tests/` — passes on a clean checkout.

Run the tests with `python -m pytest -q` from the project root.
