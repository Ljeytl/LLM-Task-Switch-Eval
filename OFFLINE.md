# Working offline

Everything in this repo runs with **no network**. Inference is local via Ollama; nothing
calls a hosted API. This file is the checklist for confirming that before you lose
connectivity.

## Pre-flight (run this while you still have wifi)

```bash
./tools/preflight.sh
```

It verifies the venv, the models, the response cache, and runs every offline command
end to end. If it prints `ALL CHECKS PASSED`, you are good to fly.

## What works with no network

| command | needs network? | notes |
|---|---|---|
| `.venv/bin/python -m pytest tests/ -q` | no | 355 tests, no model required |
| `.venv/bin/python run.py --analyse` | no | pure stats + plots over `results/sweep.jsonl` |
| `.venv/bin/python run.py --rescore` | no | re-scores from the response cache, zero inference |
| `.venv/bin/python tools/audit_taxonomy.py` | no | pure data |
| `.venv/bin/python tools/power_analysis.py` | no | simulation |
| `.venv/bin/python tools/make_readme_results.py` | no | renders `results/RESULTS.md` |
| `.venv/bin/python run.py --demo` | no | **local** Ollama inference |
| `.venv/bin/python run.py --config configs/main.yaml` | no | **local** Ollama inference |
| `.venv/bin/python tools/make_walkthrough.py` | no | **local** Ollama inference |

The only network dependencies in the whole project are `uv pip install` (already done,
`.venv/` is on disk) and `ollama pull` (already done, models are on disk).

## Ollama must be running

Local, but it is a daemon. If commands hang or error with a connection refused:

```bash
ollama serve &          # if it is not already running
ollama list             # should show qwen2.5-coder:7b and gemma4:12b
```

On macOS the Ollama app starts the daemon automatically when it is open. **Keep the app
open**, or run `ollama serve` in a spare terminal.

## The response cache is the important asset

`results/cache/` holds every model response from the sweep, keyed on the exact prompt
and generation parameters. It is **gitignored** (it is large and machine-local) but it is
on this machine, and it is what makes the offline workflow cheap:

- `--rescore` replays the entire 480-conversation corpus through the scorer with **zero**
  inference. Change the taxonomy, re-run, see the effect in seconds.
- `--analyse` needs only `results/sweep.jsonl`.

Do not delete `results/cache/` before the flight. If you do, everything still works, it
just has to re-run inference.

## Reading the docs offline

All documentation is plain Markdown in this repo — no site build, no external assets.
Start with `README.md`, then:

| you want to | read |
|---|---|
| understand the idea | `docs/DESIGN.md` |
| see one conversation end to end | `docs/WALKTHROUGH.md` |
| know why the numbers are trustworthy | `docs/EXPERIMENT.md` |
| navigate the code | `docs/ARCHITECTURE.md` |
| defend the statistics | `docs/STATS.md` |
| answer "why not X?" | `docs/DECISIONS.md` |
| know what is broken | `docs/LIMITATIONS.md` |
| prepare for the walkthrough | `docs/TALK.md` |
| see the numbers | `results/RESULTS.md`, `results/POWER.md` |

Citation links in the docs point at arxiv and will not resolve offline. Everything else
is local.

## Highest-value offline work

In rough order of payoff, and all of it runs on this machine:

1. **Raise `n_pairs` on the primary cell.** `results/POWER.md` says n=30 gives ~20% power
   against a 14-point effect; 120 pairs gets you to ~79%. Edit `n_pairs` in
   `configs/main.yaml` and re-run — the cache means already-computed pairs are free, so
   only the new seeds cost inference.
2. **Fill in `results/audit_sample.jsonl`.** 50 blind failure cases with `human_label`
   set to `null`. Label them, then
   `python tools/audit_taxonomy.py --score results/audit_sample.jsonl` gives a real
   inter-rater agreement number, which the project currently does not have.
3. **Nested noise pools** (`LIMITATIONS.md` §4b). Roughly ten lines in
   `generator._subrng`; makes the length arm properly nested. Needs a full re-run.
4. **Read `docs/TALK.md` and argue with it.** It is the walkthrough spine plus written
   answers to the questions the brief says will be asked.
