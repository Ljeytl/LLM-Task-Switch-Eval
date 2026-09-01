#!/usr/bin/env python3
"""Measure outcome-level determinism, and repair a hole in the negative control.

The sweep's `ctrl_1task` cell reports delta +0.0pp and 0/25 discordance, which reads as
strong evidence. It is not evidence at all. With one task there is nothing to interleave,
so `order_ops` returns the same sequence for both orderings and the two conversations
render **byte-identical** prompts. The response cache is keyed on the rendered prompt, so
the second call is a cache hit on the first. One inference happens, its result is compared
to itself, and the delta is 0.0 by construction.

What the control does establish is a real generator property -- that a one-task
conversation genuinely has no ordering degrees of freedom -- and that the pipeline runs
end to end. What it cannot establish is that the *model* is stable, because the model was
only asked once.

This script asks twice. It runs both halves of each one-task pair with the cache disabled,
so two independent inference calls receive the identical prompt, and reports how often
they agree. Disagreement is model nondeterminism, and it bounds how much of any measured
delta could be noise rather than ordering.

    python tools/determinism_check.py --pairs 15 --model qwen2.5-coder:7b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taskswitch.generator import build_pair                     # noqa: E402
from taskswitch.ops import TaskKind                             # noqa: E402
from taskswitch.runner import resolve_model, run_conversation   # noqa: E402
from taskswitch.scorer import score                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--n-ops", type=int, default=6)
    ap.add_argument("--n-noise", type=int, default=40)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    model = resolve_model(a.model, num_ctx=a.num_ctx)
    print(f"determinism check: {model.name} ({model.digest})")
    print(f"{a.pairs} one-task pairs, cache DISABLED -- identical prompt asked twice\n")

    same_outcome = same_state = 0
    rows = []
    for k in range(a.pairs):
        blocked, inter = build_pair(a.seed0 + k, [TaskKind.SHOPPING],
                                    a.n_ops, 2, a.n_noise)
        # Assert the premise rather than trusting it: if these ever differ, the script is
        # measuring ordering, not determinism, and the number would be meaningless.
        assert blocked.turns == inter.turns, "one-task orderings must render identically"

        r1 = run_conversation(blocked, model, constrained=True, use_cache=False)
        r2 = run_conversation(inter, model, constrained=True, use_cache=False)
        s1, s2 = score(r1), score(r2)

        outcome = s1.joint_correct == s2.joint_correct
        state = r1.parsed == r2.parsed
        same_outcome += outcome
        same_state += state
        rows.append({"seed": a.seed0 + k, "same_outcome": outcome, "same_state": state,
                     "joint_1": s1.joint_correct, "joint_2": s2.joint_correct})
        print(f"  seed {a.seed0+k}  outcome {'=' if outcome else 'DIFFER'}"
              f"   state {'=' if state else 'DIFFER'}")

    n = a.pairs
    print(f"\nidentical outcome: {same_outcome}/{n} ({same_outcome/n:.1%})")
    print(f"identical state:   {same_state}/{n} ({same_state/n:.1%})")
    if same_outcome == n:
        print("\nOutcomes are stable under repetition at this n. The measured deltas are "
              "not explained by run-to-run variation.")
    else:
        flip = (n - same_outcome) / n
        print(f"\n{flip:.1%} of conversations flip outcome on an IDENTICAL prompt. Any "
              f"delta smaller than roughly this is indistinguishable from noise.")
    if a.out:
        Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
