#!/usr/bin/env python3
"""Audit the failure taxonomy two ways.

The taxonomy is the only part of the pipeline with judgement in it. The accuracy number
is a dict comparison and cannot be argued with; whether a discrepancy is "dropped" or
"misattributed" can be.

**Pass 1 — mechanical grounding (no judgement, fully automated).** Every failure the
scorer emits must correspond to a real difference between expected and reported. This
catches a classifier that invents failures or mislabels which task an entity belongs to.
It cannot tell you whether a *bucket* is the right one, only that the underlying
discrepancy is real.

**Pass 2 — blind sample for human review.** With an explicit `--output`, writes a sample
with the automatic label withheld, so a human can classify from the evidence and
agreement can be computed afterwards with `--score`. Reported honestly: until a human
fills it in, this project has a *grounding* check, not an *agreement* number.

    python tools/audit_taxonomy.py results/sweep.jsonl        # grounding, read-only
    python tools/audit_taxonomy.py results/sweep.jsonl \
        --output results/audit_sample.jsonl                   # also create sample
    python tools/audit_taxonomy.py --score results/audit_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BUCKETS = ["dropped", "misattributed", "absorbed", "format"]


def canon_shopping(xs):
    return {str(x).strip().lower() for x in (xs or []) if str(x).strip()}


def canon_schedule(xs):
    out = set()
    for e in xs or []:
        if isinstance(e, dict):
            out.add(str(e.get("title", "")).strip().lower())
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            a, b = str(e[0]).strip().lower(), str(e[1]).strip().lower()
            out.add(b if ":" in a else a)
    return {x for x in out if x}


def is_shopping(slot: str) -> bool:
    """Slot keys are `<kind>_<index>` (`shopping_0`), so the kind is the prefix.

    Matching the bare string `"shopping"` silently routed every slot to the schedule
    canonicaliser after the v2 slot refactor -- and because that returns an empty set for
    a list of plain strings, whole tasks looked stateless. Caught by tests/test_tools.py.
    """
    return slot.split("_")[0] == "shopping"


def identities(blob):
    """Every entity identity present, by task, in either an expected or reported blob."""
    out = {}
    for task, v in (blob or {}).items():
        out[task] = canon_shopping(v) if is_shopping(task) else canon_schedule(v)
    return out


def entries(blob):
    """Full (value, identity) entries by task, so a value mismatch is visible.

    An identity-only view cannot see "the meeting is there but at the wrong hour",
    which is a real discrepancy and a real failure.
    """
    out = {}
    for task, v in (blob or {}).items():
        if is_shopping(task):
            out[task] = {("", str(x).strip().lower()) for x in (v or []) if str(x).strip()}
        else:
            s = set()
            for e in v or []:
                if isinstance(e, dict):
                    s.add((str(e.get("time", "")).strip(), str(e.get("title", "")).strip().lower()))
                elif isinstance(e, (list, tuple)) and len(e) >= 2:
                    a, b = str(e[0]).strip(), str(e[1]).strip().lower()
                    s.add((a, b) if ":" in a else (str(e[1]).strip(), a.lower()))
            out[task] = s
    return out


#: Suffixes the scorer appends after the entity identity. Stripped by longest-first so
#: "from other task" is not mistaken for a bare identity ending in "task".
_SUFFIXES = (" -> other task", " from other task", " wrong value",
             " false_assert", " hallucination", " stale")


def split_detail(detail: str) -> tuple[str, str, str]:
    """`"schedule:one-on-one wrong value"` -> `("schedule", "one-on-one", "wrong value")`.

    Splitting on whitespace would truncate every multi-word entity -- "olive oil"
    became "olive", and the grounding check then looked up a name that never existed.
    """
    task, _, rest = detail.partition(":")
    for suf in sorted(_SUFFIXES, key=len, reverse=True):
        if rest.endswith(suf):
            return task, rest[:-len(suf)].strip(), suf.strip()
    return task, rest.strip(), ""


def grounding_pass(rows):
    """Assert every emitted failure names a discrepancy that actually exists."""
    checked = ungrounded = 0
    problems = []
    for r in rows:
        if not r["parse_ok"] or not r.get("failure_details"):
            continue
        exp, got = identities(r["expected"]), identities(r["reported"])
        exp_e, got_e = entries(r["expected"]), entries(r["reported"])
        for detail in r["failure_details"]:
            checked += 1
            task, ident, suffix = split_detail(detail)
            if not ident:
                continue
            in_exp, in_got = ident in exp.get(task, set()), ident in got.get(task, set())
            exp_here = {e for e in exp_e.get(task, set()) if e[1] == ident}
            got_here = {e for e in got_e.get(task, set()) if e[1] == ident}
            got_elsewhere = any(ident in got.get(t, set()) for t in got if t != task)
            exp_elsewhere = any(ident in exp.get(t, set()) for t in exp if t != task)
            if suffix == "wrong value":
                # Present on both sides under the same identity, but the value differs.
                real = in_exp and in_got and exp_here != got_here
            elif suffix == "stale":
                # A second, out-of-date copy: the identity is expected AND reported,
                # and the reported side carries an entry the expected side does not.
                real = in_exp and in_got and bool(got_here - exp_here)
            elif suffix in ("false_assert", "hallucination"):
                real = in_got and not in_exp
            elif suffix == "-> other task":
                real = bool(exp_here - got_here) and got_elsewhere
            elif suffix == "from other task":
                real = bool(got_here - exp_here) and exp_elsewhere
            else:                                    # plain dropped
                real = in_exp and not in_got
            if not real:
                ungrounded += 1
                problems.append((r["seed"], r["ordering"], detail))
    return checked, ungrounded, problems[:10]


def blind_sample(rows, n=50, seed=0):
    rng = random.Random(seed)
    cands = [r for r in rows if r["parse_ok"] and r.get("failure_details")]
    rng.shuffle(cands)
    out = []
    for r in cands:
        for kind, detail in zip(r["failures"], r["failure_details"]):
            out.append({
                "case_id": f"{r['model']}|{r['seed']}|{r['ordering']}|{detail}",
                "expected": r["expected"], "reported": r["reported"],
                "discrepancy": detail,
                "buckets_available": BUCKETS,
                "human_label": None,        # <-- fill this in
                "_auto_label": kind,        # withheld from the reviewer by convention
            })
            if len(out) >= n:
                return out
    return out


def score_agreement(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    labelled = [r for r in rows if r.get("human_label")]
    if not labelled:
        print(f"No human labels in {path}. Fill `human_label` on each row, then re-run.")
        return
    agree = sum(1 for r in labelled if r["human_label"] == r["_auto_label"])
    print(f"agreement {agree}/{len(labelled)} = {agree/len(labelled):.1%}")
    for r in labelled:
        if r["human_label"] != r["_auto_label"]:
            print(f"  DISAGREE auto={r['_auto_label']:<14} human={r['human_label']:<14} "
                  f"{r['discrepancy']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="results/sweep.jsonl")
    parser.add_argument("--output", type=Path,
                        help="write a blind review sample (never written by default)")
    parser.add_argument("--score", type=Path,
                        help="score a completed blind review sample")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.score:
        score_agreement(args.score)
        return
    src = Path(args.source)
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    checked, ungrounded, problems = grounding_pass(rows)
    print(f"PASS 1 grounding: {checked - ungrounded}/{checked} emitted failures "
          f"correspond to a real expected-vs-reported discrepancy")
    for s, o, d in problems:
        print(f"   ungrounded: seed={s} {o} {d}")

    if args.output:
        sample = blind_sample(rows)
        args.output.write_text("\n".join(json.dumps(r) for r in sample) + "\n")
        print(f"PASS 2: wrote {len(sample)} blind cases to {args.output}")
        print("        fill `human_label` on each, then: "
              f"python tools/audit_taxonomy.py --score {args.output}")


if __name__ == "__main__":
    main()
