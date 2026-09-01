"""Ground-truth oracle for bugqueue."""

from __future__ import annotations

from typing import Any

from .ops import INERT_KINDS, Op, OpKind, SymbolDef


def ground_truth(ops: list[Op], symbols: list[SymbolDef]) -> dict[str, Any]:
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

    return {"symbols": dict(values), "open_bugs": sorted(open_bugs)}
