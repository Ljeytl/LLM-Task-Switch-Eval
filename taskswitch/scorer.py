"""Diff the model's final answer against the oracle, and classify every discrepancy.

Pure by design: `score` touches no model and no disk, so the whole corpus can be
re-scored from cache without re-running inference. That matters because the taxonomy
is the part most likely to need revision after looking at real failures.

**Normalisation.** Comparison is case-insensitive and whitespace-trimmed, and times are
canonicalised to HH:MM. The question is whether the model tracked the state, not
whether it capitalised it, and grading `"Milk" != "milk"` as a tracking failure would
inflate the headline number with formatting noise. Everything normalised away is
recorded so the choice is auditable.

**Which failure dominates is more interesting than the accuracy number**, and it comes
free from the diff:

  DROPPED        an operation never landed anywhere
  MISATTRIBUTED  an operation landed in the wrong task's state
  ABSORBED       ungrounded content entered state
  FORMAT         the response did not parse or validate

FORMAT is counted separately and excluded from state accuracy, so a formatting problem
is never reported as a tracking problem.

ABSORBED covers three distinguishable cases and the detail string keeps them apart:

  false_assert   a planted counterfactual the model treated as real -- the instrumented
                 probe
  stale          a value that WAS true earlier and was never retracted, sitting beside
                 the correct current value. Almost always a dropped REMOVE or UPDATE
  hallucination  content that was never mentioned at all

They share a bucket because all three are ungrounded content in the final state, but
they are very different bugs. Collapsing `stale` into `hallucination` would be actively
misleading -- a model that fails to retract is not a model that invents, and the fixes
have nothing in common. This distinction was found by the grounding audit in
`tools/audit_taxonomy.py`, not by reading accuracy numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .ops import OpKind, TaskKind
from .runner import RunResult

_TIME = re.compile(r"^\s*(\d{1,2})\s*[:.]\s*(\d{2})\s*$")


class Failure(str, Enum):
    DROPPED = "dropped"
    MISATTRIBUTED = "misattributed"
    ABSORBED = "absorbed"
    FORMAT = "format"


@dataclass
class Score:
    joint_correct: bool
    per_task_correct: dict[str, bool] | None      # None when the response did not parse
    failures: list[tuple[Failure, str]] = field(default_factory=list)
    n_expected: int = 0
    n_reported: int = 0

    @property
    def failure_kinds(self) -> list[str]:
        return [f.value for f, _ in self.failures]


def _norm(s: Any) -> str:
    return str(s).strip().lower()


def _norm_time(s: Any) -> str:
    """Canonicalise `9:00`, `09.00`, ` 9:00 ` to `09:00`. Formatting is not the thing
    being measured."""
    m = _TIME.match(str(s))
    if not m:
        return _norm(s)
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def _is_shopping(slot: str) -> bool:
    """Slot keys are `<kind>_<index>`, so the kind is the prefix."""
    return slot.split("_")[0] == TaskKind.SHOPPING.value


def _canon(task: str, entries: Any) -> set[tuple[str, ...]]:
    """A task slot's state as a comparable set of tuples."""
    if not isinstance(entries, list):
        return set()
    if _is_shopping(task):
        return {(_norm(e),) for e in entries if str(e).strip()}
    out: set[tuple[str, ...]] = set()
    for e in entries:
        if isinstance(e, dict):
            t, n = e.get("time", ""), e.get("title", "")
            if str(n).strip():
                out.add((_norm_time(t), _norm(n)))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            # Defensive: a positional pair has no intrinsic order. Decide by which
            # element looks like a clock time rather than by position, so a model that
            # emits [title, time] is never scored as having lost the meeting.
            a, b = e[0], e[1]
            if _TIME.match(str(a)):
                out.add((_norm_time(a), _norm(b)))
            elif _TIME.match(str(b)):
                out.add((_norm_time(b), _norm(a)))
            else:
                out.add((_norm_time(a), _norm(b)))
    return out


def _identity(task: str, entry: tuple[str, ...]) -> str:
    """The handle a user would use to name this entry.

    For shopping that is the item; for the schedule it is the title, not the time --
    a meeting moved to the wrong hour is the same meeting tracked badly, which is a
    different failure from the meeting having vanished.
    """
    return entry[0] if _is_shopping(task) else entry[1]


def score(result: RunResult) -> Score:
    """Pure function of (parsed answer, expected answer). No model calls, no I/O."""
    conv = result.conversation
    expected = {t: _canon(t, v) for t, v in conv.expected.items()}

    if not result.parse_ok or result.parsed is None:
        detail = result.error or (result.raw_final[:120] if result.raw_final else "empty response")
        return Score(joint_correct=False, per_task_correct=None,
                     failures=[(Failure.FORMAT, detail)],
                     n_expected=sum(len(v) for v in expected.values()))

    reported_raw = (result.parsed.model_dump()
                    if hasattr(result.parsed, "model_dump") else dict(result.parsed))
    actual = {t: _canon(t, reported_raw.get(t)) for t in expected}

    # Content the conversation planted as counterfactual and that must NOT appear.
    false_ids = {_norm(o.payload.get("item") or o.payload.get("title", ""))
                 for o in conv.ops if o.kind is OpKind.FALSE_ASSERT}

    failures: list[tuple[Failure, str]] = []
    per_task: dict[str, bool] = {}

    for task, exp in expected.items():
        act = actual[task]
        per_task[task] = exp == act
        missing, extra = exp - act, act - exp
        elsewhere_actual = {i for other, s in actual.items() if other != task
                            for i in (_identity(other, e) for e in s)}
        elsewhere_expected = {i for other, s in expected.items() if other != task
                              for i in (_identity(other, e) for e in s)}
        here_actual = {_identity(task, e) for e in act}
        here_expected = {_identity(task, e) for e in exp}

        for m in sorted(missing):
            ident = _identity(task, m)
            if ident in elsewhere_actual:
                failures.append((Failure.MISATTRIBUTED, f"{task}:{ident} -> other task"))
            elif ident in here_actual:
                failures.append((Failure.DROPPED, f"{task}:{ident} wrong value"))
            else:
                failures.append((Failure.DROPPED, f"{task}:{ident}"))

        for e in sorted(extra):
            ident = _identity(task, e)
            if ident in {_identity(task, x) for x in missing}:
                continue                                  # already counted as wrong value
            if ident in false_ids:
                failures.append((Failure.ABSORBED, f"{task}:{ident} false_assert"))
            elif ident in here_expected:
                # The entity is real and is also reported correctly -- this is a second,
                # stale copy of it. Almost always a REMOVE or UPDATE that never landed,
                # leaving the pre-update value behind alongside the current one.
                # Calling this a hallucination would be flatly wrong: the model did not
                # invent the value, it failed to retract it. Different bug, different fix.
                failures.append((Failure.ABSORBED, f"{task}:{ident} stale"))
            elif ident in elsewhere_expected:
                failures.append((Failure.MISATTRIBUTED, f"{task}:{ident} from other task"))
            else:
                failures.append((Failure.ABSORBED, f"{task}:{ident} hallucination"))

    return Score(
        joint_correct=all(per_task.values()),
        per_task_correct=per_task,
        failures=failures,
        n_expected=sum(len(v) for v in expected.values()),
        n_reported=sum(len(v) for v in actual.values()),
    )
