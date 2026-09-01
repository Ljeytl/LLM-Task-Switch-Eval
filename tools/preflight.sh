#!/usr/bin/env bash
# Confirm the whole project runs with no network. Run this BEFORE going offline.
set -uo pipefail
PY=.venv/bin/python
fail=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

echo "== 1. Python environment =="
[ -x "$PY" ] && ok "venv present ($($PY --version 2>&1))" || bad "no .venv -- run: uv venv --python 3.12 .venv"
$PY -c "import ollama,pydantic,numpy,scipy,matplotlib,yaml,tqdm" 2>/dev/null \
  && ok "all dependencies importable" || bad "missing dependencies -- run: uv pip install --python $PY -e '.[dev]'"
$PY -c "import taskswitch" 2>/dev/null && ok "taskswitch importable" || bad "package not importable"

echo "== 2. Ollama daemon and models =="
# Capture the list ONCE. Calling `ollama list` inside the loop was flaky and reported a
# model missing that inference then used successfully two checks later -- a preflight
# that cries wolf is worse than none.
if MODELS=$(ollama list 2>/dev/null | awk 'NR>1{print $1}'); then
  ok "ollama daemon reachable"
  for m in qwen2.5-coder:7b gemma4:12b; do
    if printf '%s\n' "$MODELS" | grep -qxF "$m"; then ok "model present: $m"
    else bad "model MISSING: $m -- run: ollama pull $m"; fi
  done
else
  bad "ollama not reachable -- start the Ollama app, or run: ollama serve &"
fi

echo "== 3. Data assets =="
# v2 results land in results/; v1 is archived under results/v1/. Either is enough to work
# offline, so accept a fallback rather than failing while a sweep is still running.
if [ -s results/sweep.jsonl ]; then
  ok "v2 sweep data ($(wc -l < results/sweep.jsonl | tr -d ' ') rows)"
elif [ -s results/v1/sweep.jsonl ]; then
  ok "v1 sweep data ($(wc -l < results/v1/sweep.jsonl | tr -d ' ') rows) -- v2 sweep not finished"
else
  bad "no sweep data in results/ or results/v1/"
fi
n_cache=$(ls results/cache 2>/dev/null | wc -l | tr -d ' ')
n_rows=$( [ -s results/sweep.jsonl ] && wc -l < results/sweep.jsonl | tr -d ' ' || echo 0)
if [ "$n_cache" -ge "$n_rows" ] && [ "$n_cache" -gt 0 ]; then
  ok "response cache ($n_cache entries covers $n_rows rows) -- --rescore needs no inference"
elif pgrep -f "run.py --config" >/dev/null 2>&1; then
  ok "response cache filling ($n_cache entries); a sweep is still running"
else
  bad "response cache ($n_cache entries) does not cover $n_rows rows; --rescore will re-run inference"
fi
for f in RESULTS.md POWER.md sample.jsonl dumbbell.png; do
  if [ -s "results/$f" ]; then ok "present: results/$f"
  elif [ -s "results/v1/$f" ]; then ok "present: results/v1/$f (v1 archive)"
  else bad "missing: $f in results/ and results/v1/"; fi
done
[ -s docs/WALKTHROUGH.md ] && ok "present: docs/WALKTHROUGH.md" || bad "missing: docs/WALKTHROUGH.md"

echo "== 4. Offline commands actually run =="
n_tests=$($PY -m pytest tests/ -q --collect-only 2>/dev/null | grep -oE "^[0-9]+ tests" | head -1)
$PY -m pytest tests/ -q >/dev/null 2>&1 && ok "pytest (${n_tests:-suite} passing)" || bad "pytest failed"
DATA=results/sweep.jsonl; [ -s "$DATA" ] || DATA=results/v1/sweep.jsonl
$PY run.py --analyse --results "$DATA" >/dev/null 2>&1 && ok "run.py --analyse" || bad "run.py --analyse failed"
$PY tools/audit_taxonomy.py "$DATA" >/dev/null 2>&1 && ok "tools/audit_taxonomy.py" || bad "audit failed"
$PY tools/power_analysis.py "$DATA" >/dev/null 2>&1 && ok "tools/power_analysis.py" || bad "power analysis failed"

echo "== 5. Local inference (one real model call) =="
if $PY -c "
import ollama,sys
try:
    r=ollama.chat(model='qwen2.5-coder:7b',messages=[{'role':'user','content':'say ok'}],
                  options={'num_predict':4,'temperature':0},think=False)
    sys.exit(0 if r['message']['content'] else 1)
except Exception as e:
    print(e); sys.exit(1)
" >/dev/null 2>&1; then ok "local inference works"; else bad "local inference failed"; fi

echo
if [ "$fail" -eq 0 ]; then printf '\033[32mALL CHECKS PASSED — safe to go offline.\033[0m\n'
else printf '\033[31mSOME CHECKS FAILED — fix the FAIL lines above before flying.\033[0m\n'; exit 1; fi
