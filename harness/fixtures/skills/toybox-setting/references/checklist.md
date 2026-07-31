# Checklist

Work in this order. Each step is checkable before the next.

1. **Name it.** Lower snake case, no prefix. `width`, not `toybox_width`.
2. **Field first**, so the type is decided before anything reads it. Frozen
   dataclass, so a default is required.
3. **Loader second.** Coerce explicitly — `tomllib` gives you whatever the file
   said, and a string where an int was meant should degrade to the default, not
   raise.
4. **Example last**, because writing the prose is when you find out the setting
   is badly named.

## Recurring mistakes

- **A default in two places.** The dataclass field is the default. `load` must
  not repeat the literal; read the field or a module constant.
- **An environment variable that beats the file with no way to see it.**
  `width` reads `TOYBOX_WIDTH` first; if you add another, say so in the example.
- **Raising on bad input.** A malformed value degrades to the default. A typo
  should not stop the program from starting.
- **Forgetting the example.** Nothing enforces it, so it is the one that rots.
