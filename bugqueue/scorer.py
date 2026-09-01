"""Diff model answer against bugqueue oracle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .runner import RunResult


class Failure(str, Enum):
    WRONG_VALUE = "wrong_value"
    OPEN_BUG = "open_bug"
    CLOSED_BUG = "closed_bug"
    EXTRA_OPEN = "extra_open"
    FORMAT = "format"


@dataclass
class Score:
    joint_correct: bool
    failures: list[tuple[Failure, str]] = field(default_factory=list)
    n_expected: int = 0

    @property
    def failure_kinds(self) -> list[str]:
        return [f.value for f, _ in self.failures]


def _norm(s: Any) -> str:
    return str(s).strip().lower()


def score(result: RunResult) -> Score:
    conv = result.conversation
    exp_syms = {k: _norm(v) for k, v in conv.expected["symbols"].items()}
    exp_open = {_norm(x) for x in conv.expected["open_bugs"]}

    if not result.parse_ok or result.parsed is None:
        detail = result.error or (result.raw_final[:120] if result.raw_final else "empty")
        return Score(joint_correct=False, failures=[(Failure.FORMAT, detail)],
                     n_expected=len(exp_syms) + len(exp_open))

    raw = result.parsed
    rep_syms_raw = raw.get("symbols") if isinstance(raw, dict) else None
    rep_open_raw = raw.get("open_bugs") if isinstance(raw, dict) else None
    if not isinstance(rep_syms_raw, dict):
        return Score(joint_correct=False, failures=[(Failure.FORMAT, "missing symbols")],
                     n_expected=len(exp_syms))

    rep_syms = {k: _norm(v) for k, v in rep_syms_raw.items()}
    rep_open = {_norm(x) for x in (rep_open_raw or []) if str(x).strip()}

    failures: list[tuple[Failure, str]] = []

    for name, exp_val in exp_syms.items():
        if name not in rep_syms:
            failures.append((Failure.WRONG_VALUE, f"{name}: missing"))
        elif rep_syms[name] != exp_val:
            failures.append((Failure.WRONG_VALUE, f"{name}: {rep_syms[name]} != {exp_val}"))

    for name in sorted(exp_open):
        if name not in rep_open:
            failures.append((Failure.CLOSED_BUG, f"{name} should be open"))
    for name in sorted(rep_open):
        if name not in exp_open:
            failures.append((Failure.EXTRA_OPEN, f"{name} falsely open"))
        elif name in exp_open and name in rep_syms and rep_syms.get(name) == exp_syms.get(name):
            failures.append((Failure.OPEN_BUG, f"{name} open but value correct"))

    joint = not failures
    return Score(joint_correct=joint, failures=failures,
                 n_expected=len(exp_syms) + len(exp_open))
