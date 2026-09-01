# Bugqueue: sequential related defects

**Status:** scaffold + demo. No sweep results committed yet.

## The usage pattern

During a coding task, a user reports several related bugs in quick succession before
the assistant has fully resolved the first one. `taskswitch` measures **concurrent**
tasks; `bugqueue` measures **sequential defect accumulation** within one module.

## Manipulation

| ordering | pattern |
|---|---|
| **accumulate** | R₁ R₂ R₃ R₄ → F₁ F₂ F₃ F₄ |
| **ticket** | R₁ F₁ R₂ F₂ R₃ F₃ R₄ F₄ |

Same bugs, same module (`checkout.py`), mechanical oracle. Not token-matched — see
[`LIMITATIONS.md`](LIMITATIONS.md) §12.

## Quickstart

```bash
.venv/bin/python -m pytest tests/test_bugqueue.py -q
.venv/bin/python run_bugs.py --demo
.venv/bin/python run_bugs.py --config configs/bugs.yaml
```
