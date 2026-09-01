"""Natural-language turns for bug reports and fixes.

Templated, never model-generated — same invariant as taskswitch. Coding-flavored
phrasing so the domain reads as a defect queue on one file, not a generic state game.
"""

from __future__ import annotations

from random import Random

from .ops import MODULE_NAME, Op, OpKind, SymbolDef

REPORT_TEMPLATES: tuple[str, ...] = (
    "Bug in `{mod}`: `{sym}()` returns {obs} but should return {exp}.",
    "`{sym}()` in `{mod}` is wrong — I get {obs}, expected {exp}.",
    "Found a defect: `{sym}()` gives {obs} when it should be {exp} ({mod}).",
    "Another one on `{mod}`: `{sym}()` still returns {obs}, needs to be {exp}.",
)

FIX_TEMPLATES: tuple[str, ...] = (
    "Fixed `{sym}()` in `{mod}` — it now returns {exp}.",
    "Patch applied: `{sym}()` should return {exp} now.",
    "I changed `{sym}()` so it returns {exp}.",
    "`{sym}()` fix landed — correct value is {exp}.",
)

CONTEXT_TEMPLATES: tuple[str, ...] = (
    "Still working on `{mod}`.",
    "Context: we're in `{mod}` right now.",
    "Back to `{mod}` — same module.",
    "Continuing on `{mod}`.",
)

NOISE_TEMPLATES: tuple[str, ...] = (
    "CI is slow today.",
    "Can we rename this module later?",
    "I'll push after these fixes.",
    "The test suite takes a minute.",
)

FALSE_REPORT_TEMPLATES: tuple[str, ...] = (
    "I thought `{sym}()` was broken but it was my test data.",
    "False alarm on `{sym}()` — works fine on my machine.",
    "Scratch that report for `{sym}()`, not actually a bug.",
)


def system_prompt(symbols: list[SymbolDef]) -> str:
    names = ", ".join(f"`{s.name}()`" for s in symbols)
    return (
        f"You track defects and symbol values for `{MODULE_NAME}`. "
        f"Symbols in scope: {names}. "
        "When the user reports a bug, remember which symbol is wrong and what the "
        "correct value should be. When they say they fixed something, update your "
        "belief. Ignore false alarms and filler. "
        "At the end you will report every symbol's current value and which bugs remain open."
    )


def final_request() -> str:
    return (
        "Report the current state as JSON with exactly two keys: "
        "`symbols` (object mapping symbol name to its current value as a string) and "
        "`open_bugs` (array of symbol names still broken and not yet fixed). "
        "Use only the symbols discussed in this conversation."
    )


def render(op: Op, symbols: list[SymbolDef], rng: Random) -> str:
    by_name = {s.name: s for s in symbols}
    sym_name = str(op.payload.get("symbol", ""))
    sym = by_name.get(sym_name)
    mod = MODULE_NAME

    if op.kind is OpKind.REPORT and sym:
        t = rng.choice(REPORT_TEMPLATES)
        return t.format(mod=mod, sym=sym.name, obs=sym.observed, exp=sym.expected)
    if op.kind is OpKind.FIX and sym:
        t = rng.choice(FIX_TEMPLATES)
        return t.format(mod=mod, sym=sym.name, exp=sym.expected)
    if op.kind is OpKind.CONTEXT:
        return rng.choice(CONTEXT_TEMPLATES).format(mod=mod)
    if op.kind is OpKind.NOISE:
        return rng.choice(NOISE_TEMPLATES)
    if op.kind is OpKind.FALSE_REPORT and sym:
        return rng.choice(FALSE_REPORT_TEMPLATES).format(sym=sym.name)
    return ""


def output_schema(symbols: list[SymbolDef]) -> dict:
    props = {s.name: {"type": "string"} for s in symbols}
    return {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "object",
                "properties": props,
                "required": list(props),
            },
            "open_bugs": {
                "type": "array",
                "items": {"type": "string", "enum": list(props)},
            },
        },
        "required": ["symbols", "open_bugs"],
    }
