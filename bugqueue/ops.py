"""Operation grammar for the bugqueue domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

MODULE_NAME = "checkout.py"


class OpKind(str, Enum):
    REPORT = "report"
    FIX = "fix"
    CONTEXT = "context"
    NOISE = "noise"
    FALSE_REPORT = "false_report"


INERT_KINDS: frozenset[OpKind] = frozenset({
    OpKind.CONTEXT, OpKind.NOISE, OpKind.FALSE_REPORT,
})


@dataclass(frozen=True)
class Op:
    kind: OpKind
    payload: dict[str, Any]
    idx: int

    @property
    def mutating(self) -> bool:
        return self.kind not in INERT_KINDS


@dataclass(frozen=True)
class SymbolDef:
    name: str
    observed: str
    expected: str


SYMBOL_POOL: tuple[SymbolDef, ...] = (
    SymbolDef("total", "0", "0.0"),
    SymbolDef("tax", "0.10", "0.08"),
    SymbolDef("shipping", "5", "5.99"),
    SymbolDef("discount", "none", "0.0"),
    SymbolDef("item_count", "0", "0"),
    SymbolDef("coupon_applied", "true", "false"),
)
