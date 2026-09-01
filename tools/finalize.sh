#!/usr/bin/env bash
# Post-sweep assembly. Everything here is derived from results/sweep.jsonl and the
# response cache, so it is cheap to re-run and cannot drift from the data.
set -euo pipefail
PY=${PY:-.venv/bin/python}

echo "==> 1. Re-score with the current taxonomy (no inference; reads the cache)"
$PY run.py --rescore --results results/sweep.jsonl | tail -n 12

echo "==> 2. Taxonomy grounding audit"
$PY tools/audit_taxonomy.py results/sweep.jsonl

echo "==> 3. Render the Results section from the data"
$PY tools/make_readme_results.py > /dev/null && echo "wrote results/RESULTS.md"

echo "==> 4. Regenerate the end-to-end walkthrough from a real pair"
$PY tools/make_walkthrough.py

echo "==> 5. Regenerate the power analysis from the observed discordance"
$PY tools/power_analysis.py results/sweep.jsonl > results/POWER.md
echo "wrote results/POWER.md"

echo "==> 6. Commit a small sample so a reviewer who runs nothing still sees output"
# Stratified, not `head`: rows are written cell by cell, so the first N are all one
# condition from one model, and both halves of every pair are kept together.
$PY tools/make_sample.py results/sweep.jsonl results/sample.jsonl

echo "==> 7. Test suite"
$PY -m pytest tests/ -q | tail -n 2
