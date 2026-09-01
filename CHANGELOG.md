# Changelog

## v2 — per-instance task identity

### Added
- **Task slots** (`Op.slot`, `slot_key`, `assign_slots`). State is keyed per instance, so
  two shopping lists are two independent states. Lifts the two-task cap and makes the
  task-count arm deliverable for the first time. (D17, reverses D14)
- **Disjoint vocabularies per slot** (`surface.vocabulary`): grocery / hardware /
  pharmacy / garden items, work / personal / team / family calendars. This is what keeps
  misattribution detectable between two lists of the *same kind* — the diagnostic the
  original two-kind design got for free.
- **Task-count arm** in `configs/main.yaml`: `tasks_3`, `tasks_4`.
- **`tests/test_surface.py`, `tests/test_runner.py`, `tests/test_plots.py`** — 126 new
  tests covering three modules that previously had none. Suite is now 518.

### Changed
- **Nested noise pools.** `n_noise` removed from the mix-RNG key; one pool per slot with
  each condition taking a prefix, so longer length conditions are strict supersets.
  Required two further fixes: the extras list was shuffled (reassigning positions when it
  grew) and rendering walked one shared RNG in list order (changing paraphrases after any
  insertion). Rendering is now per-op, keyed on op identity. (LIMITATIONS 4b)
- Every surface template names its target list, since with two lists of one kind an
  unnamed turn is ambiguous — and an ambiguous turn makes the answer key unanswerable.
- `parsed` is now a plain slot-keyed dict; `FinalState` cannot express N slots.

### Known
- **The calibration is stale.** `n_ops = 6` was measured on v1 templates. v2 difficulty
  moved non-uniformly -- the control got harder (0.600 -> 0.320) while the primary got
  easier (0.467 -> 0.640) and landed in the intended band. Template wording and noise
  nesting changed together, so nothing is attributable. Re-calibrate before the next
  sweep. See LIMITATIONS 4.
- **Task count and ops-per-task move inversely**, since `n_ops` is total. The arm measures
  splitting fixed work across more states, not the cost of adding a state. See
  LIMITATIONS 4c.
- v1 results are not reproducible from v2 code and are archived under `results/v1/` with
  their provenance.


## Unreleased — initial build

### Added
- `ops.py`, `state.py` — operation grammar and the ground-truth oracle, with the two
  load-bearing invariants asserted in tests: inert ops mutate nothing, and blocked vs
  interleaved orderings of one op list produce identical ground truth.
- `surface.py` — ≥4 paraphrase templates per (task, kind), seed-randomised.
- `generator.py` — op sampling, ordering, and token-matched pairing.
- `runner.py` — Ollama client with prefilled acks, response cache, and runtime token
  verification via `prompt_eval_count`.
- `scorer.py` — pure diff plus the four-bucket failure taxonomy.
- `stats.py` — Wilson, exact-binomial McNemar, paired bootstrap, clustered SE.
- `plots.py`, `run.py`, `configs/main.yaml`.
- Full doc set under `docs/`.

### Changed direction
- **`n_ops` is now total per conversation, not per task.** Per-task would have made
  4-task conversations twice as long as 2-task ones, confounding task count with
  context length. (D4)
- **Context length is varied with noise turns, not more operations.** Driving length
  with operations would confound length with state-update load. (D5)
- **Independent RNG per (seed, task, purpose).** A shared RNG meant that changing
  `n_noise` silently changed the operations too. Caught by a test. (D6)
- **Output schema uses named keys.** Positional `[a, b]` pairs were ambiguous, and the
  model resolved the ambiguity *differently per condition* — which would have
  manufactured a fictitious switch cost. (D7)
- **Prefilled assistant acknowledgements** rather than model-generated turns, making
  the token match exact by construction and the sweep ~10× cheaper. (D3)
- **Expenses task cut**; two task types, not three. (D2)
- **Task-count arm (2/3/4 tasks) cut entirely.** Task identity is `TaskKind` and there
  are only two kinds, so a third task repeated one and merged into its state; blocked
  ordering also emitted its turns twice. Caught by the token-match assertion. Replaced
  with a 1-task negative control, and guarded so it cannot recur silently. (D14)
- **`n_ops` lowered from 24 to 6**, from the measured difficulty curve. At 24 the
  primary model sits at 0.00 blocked accuracy — no room for a delta in either direction.

### Environment findings
- `gemma4:12b` reasons by default and returned *empty content* under a 64-token budget.
  `think=False` is mandatory; it also cut latency 16 s → 3.1 s. (D13)
- `qwen3.6` (36B/23 GB) and `qwen3-coder` (30.5B MoE/18 GB) do not fit alongside a KV
  cache on 24 GB. Usable models are `qwen2.5-coder:7b` and `gemma4:12b`. (D12)
- Prefill runs ~300 tok/s, generation ~16 tok/s. Cost is dominated by output length,
  which is why the noise-based length arm is nearly free.
- Pinned Python 3.12 via `uv`; the system default is 3.14, where scipy/matplotlib wheel
  coverage is still thin.

- **`stale` split out of `hallucination`** in the ABSORBED bucket. A value that was
  true earlier and never retracted is a dropped REMOVE, not an invention. Found by the
  grounding audit; `--rescore` confirmed it changed labels and 0 outcomes. (D15)
- **`tools/audit_taxonomy.py`** added: mechanical grounding check plus a blind sample
  for human review.

- **`runner.extract_state`** added: free-form runs get a second, schema-constrained
  extraction call. Without it the constrained-vs-free-form check measured JSON emission
  rather than tracking and reported a fictitious +66.7pp effect. (D16)

- Constrained-decoding check now reports honestly: 0.667 vs 0.583 at n=12 is a
  one-conversation difference, flagged as underpowered rather than presented as an
  effect.

### Fixed
- `analyse` now excludes token-drifted pairs, which `run_cell`'s comment already claimed
  it did but the code did not.
- `--rescore` crashed on a dict-iteration bug that yielded model-name strings where rows
  were expected.
- `--config` alone now implies `--sweep`, matching the documented usage; argparse
  previously rejected it and the first sweep launch silently did nothing.

### Known methodological gap
- Difficulty was calibrated over `n_ops` with `n_noise` pinned at 0, but noise drives
  difficulty hard on its own (blocked accuracy 0.90 -> 0.47 when 40 noise turns are
  added). The primary cell therefore landed below the calibrated band. Calibration
  should sweep the (`n_ops`, `n_noise`) pair. Recorded in LIMITATIONS.md §4.

### Next
- **Nested noise pools**: draw one max-size noise pool per seed and take the first N per
  condition, so length conditions are strict supersets. Currently `n_noise` is part of
  the mix-RNG key, so each condition draws a fresh placement and between-cell comparison
  carries placement variance. See LIMITATIONS.md §4b.
- **Two-dimensional calibration** over (`n_ops`, `n_noise`), since noise turns out
  to drive difficulty as hard as operation count.
- **Per-instance task identity**, to make a real task-count arm possible. This is the
  question the project set out to answer and currently does not.
- Generative-turn variant (the model replies to each turn) — the most valuable next
  experiment, at ~10× the compute.
- Same-family scaling ladder to replace the descriptive-only cross-model comparison.
- A second domain, to test whether any of this generalises.
- External task state with embedding-based routing (originally "Phase 2"), noted as a
  cleaner replication of a published result rather than a new finding.
