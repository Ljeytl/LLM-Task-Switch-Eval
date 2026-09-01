"""Ground-truth oracle for bugqueue.

Applies the same op list the surface renders. REPORT marks a symbol as open; FIX sets
its value to the correct expected value and clears it from open. The final snapshot is
what the model must reproduce at the end of the transcript.
"""

from __future__ import annotations

from typing import Any

from .ops import INERT_KINDS, Op, OpKind, SymbolDef


def ground_truth(ops: list[Op], symbols: list[SymbolDef]) -> dict[str, Any]:
    """Return `{symbols: {name: value}, open_bugs: [names]}` after applying ops."""
    by_name = {s.name: s for s in symbols}
    values: dict[str, str] = {s.name: s.observed for s in symbols}
    open_bugs: set[str] = set()

    for op in ops:
        if op.kind in INERT_KINDS:
            continue
        name = str(op.payload.get("symbol", "")).strip()
        if not name or name not in by_name:
            continue
        sym = by_name[name]
        if op.kind is OpKind.REPORT:
            open_bugs.add(name)
            values[name] = sym.observed
        elif op.kind is OpKind.FIX:
            values[name] = sym.expected
            open_bugs.discard(name)

    return {
        "symbols": dict(values),
        "open_bugs": sorted(open_bugs),
    }
