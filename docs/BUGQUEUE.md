# Bugqueue: sequential related defects

**Status:** scaffold + demo. No sweep results yet.

## The usage pattern

During a coding task, a user reports several related bugs in quick succession — before
the assistant has fully resolved the first one. Context muddles; the model stutters,
confuses which defect is still open, or applies a fix to the wrong symbol.

`taskswitch` measures **concurrent** state across different tasks (lists vs calendars).
`bugqueue` measures **sequential defect accumulation** within one module.

## What it measures

Given N related bugs in `checkout.py`, the accuracy cost of **piling all reports before
any fix** versus **report-fix pacing**:

```
ACCUMULATE   R₁ R₂ R₃ R₄  F₁ F₂ F₃ F₄     (pile reports, then fix)
TICKET       R₁ F₁ R₂ F₂ R₃ F₃ R₄ F₄     (one ticket at a time)
```

Same bugs. Same module. Same number of operations. Mechanical oracle on symbol values
and which defects remain open.

Unlike `taskswitch`, the two orderings are **not** token-matched — REPORT and FIX turns
are different strings. The comparison is matched *work*, not matched tokens. See
[`LIMITATIONS.md`](LIMITATIONS.md) (bugqueue section).

## Domain

A small module with named functions (`total`, `tax`, `shipping`, …). Each bug is a
wrong return value for one symbol. The oracle tracks:

- `symbols`: correct values after all FIX ops in transcript order
- `open_bugs`: symbols still reported-but-not-fixed at end of transcript

Grading is a dict diff — no judge model.

## Quickstart

```bash
.venv/bin/python -m pytest tests/test_bugqueue.py -q
.venv/bin/python run_bugs.py --demo    # 3 pairs, local Ollama
```

## Relation to taskswitch

| | taskswitch | bugqueue |
|---|---|---|
| Concurrent tasks | 2–4 (shopping + calendar) | 1 module |
| Manipulation | blocked vs interleaved | accumulate vs ticket |
| Token match | yes (prefilled acks) | no (different turn text) |
| Misattribution | across task slots | wrong symbol / stale bug |
| Runner | `run.py` | `run_bugs.py` |

Shared: `taskswitch/stats.py` for McNemar, Wilson, bootstrap.

## Next

See `prep/ROADMAP.md` (local) for calibration, full grid, provider sweep, and optional
real-code execution domain.
