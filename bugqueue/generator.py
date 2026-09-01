"""Build matched pairs: accumulate vs ticket ordering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Any

from .ops import Op, OpKind, SYMBOL_POOL, SymbolDef
from .state import ground_truth
from .surface import final_request, render, system_prompt


class Ordering(str, Enum):
    ACCUMULATE = "accumulate"
    TICKET = "ticket"


@dataclass
class Conversation:
    seed: int
    ordering: Ordering
    symbols: list[SymbolDef]
    ops: list[Op]
    turns: list[str]
    expected: dict[str, Any]
    token_count: int
    n_bugs: int = 0
    n_noise: int = 0
    n_false: int = 0
    system: str = ""
    final_request: str = ""

    @property
    def cell(self) -> str:
        return f"bugs_{self.n_bugs}_n{self.n_noise}"


def approx_tokens(texts: list[str]) -> int:
    return sum(len(t.split()) for t in texts)


def _subrng(seed: int, *parts: object) -> Random:
    key = "|".join(str(p) for p in (seed, *parts)).encode()
    return Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))


def sample_core_ops(seed: int, n_bugs: int) -> tuple[list[Op], list[SymbolDef]]:
    rng = _subrng(seed, "sample", n_bugs)
    pool = list(SYMBOL_POOL)
    rng.shuffle(pool)
    if n_bugs > len(pool):
        raise ValueError(f"n_bugs={n_bugs} exceeds pool size {len(pool)}")
    symbols = pool[:n_bugs]
    ops: list[Op] = []
    for sym in symbols:
        ops.append(Op(OpKind.REPORT, {"symbol": sym.name}, 0))
        ops.append(Op(OpKind.FIX, {"symbol": sym.name}, 0))
    return ops, symbols


def sample_inert_ops(seed: int, n_noise: int, n_false: int,
                     symbols: list[SymbolDef]) -> list[Op]:
    mix = _subrng(seed, "inert", n_false, n_noise)
    pool = list(SYMBOL_POOL)
    used = {s.name for s in symbols}
    spare = [s for s in pool if s.name not in used] or pool
    out: list[Op] = []
    for _ in range(n_false):
        sym = mix.choice(spare)
        out.append(Op(OpKind.FALSE_REPORT, {"symbol": sym.name}, 0))
    for _ in range(n_noise):
        out.append(Op(OpKind.NOISE, {}, 0))
    if out:
        out.insert(0, Op(OpKind.CONTEXT, {}, 0))
    return out


def order_core(ops: list[Op], ordering: Ordering) -> list[Op]:
    reports = [o for o in ops if o.kind is OpKind.REPORT]
    fixes = [o for o in ops if o.kind is OpKind.FIX]
    if ordering is Ordering.ACCUMULATE:
        return reports + fixes
    fix_by_sym = {o.payload["symbol"]: o for o in fixes}
    out: list[Op] = []
    for r in reports:
        out.append(r)
        sym = r.payload["symbol"]
        if sym in fix_by_sym:
            out.append(fix_by_sym[sym])
    return out


def build_pair(seed: int, n_bugs: int, n_noise: int = 0, n_false: int = 0) -> tuple[Conversation, Conversation]:
    core, symbols = sample_core_ops(seed, n_bugs)
    inert = sample_inert_ops(seed, n_noise, n_false, symbols)
    base_ops = inert + core
    expected = ground_truth(base_ops, symbols)
    sys_p = system_prompt(symbols)
    final = final_request()

    def render_ops(seq: list[Op]) -> list[str]:
        seen: dict[tuple, int] = {}
        turns: list[str] = []
        for o in seq:
            ident = (o.kind.value, tuple(sorted(o.payload.items())))
            n = seen.get(ident, 0)
            seen[ident] = n + 1
            turns.append(render(o, symbols, _subrng(seed, "render", *ident, n)))
        return turns

    convs: list[Conversation] = []
    for ordering in (Ordering.ACCUMULATE, Ordering.TICKET):
        seq = inert + order_core(core, ordering)
        seq = [Op(o.kind, o.payload, i) for i, o in enumerate(seq)]
        turns = render_ops(seq)
        convs.append(Conversation(
            seed=seed, ordering=ordering, symbols=list(symbols), ops=seq,
            turns=turns, expected=expected,
            token_count=approx_tokens([sys_p, *turns, final]),
            n_bugs=n_bugs, n_noise=n_noise, n_false=n_false,
            system=sys_p, final_request=final,
        ))

    acc, ticket = convs
    if acc.expected != ticket.expected:
        raise AssertionError(f"seed={seed}: ground truth differs between orderings")
    return acc, ticket
