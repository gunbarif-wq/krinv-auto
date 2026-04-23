#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_name="auto150_3m_1y_excl50_real_fetch"

selected_summary_path="$repo_dir/data/chart_retrain/$run_name/selected_summary.json"
summary_path="$repo_dir/data/chart_retrain/$run_name/summary.json"
stdout_path="$repo_dir/logs/fetch_150_1y_real.stdout.txt"
runner_log_path="$repo_dir/logs/fetch_150_1y_real.runner.txt"
data_dir="$repo_dir/data/chart_retrain/$run_name"

get_total() {
  local py
  py="${PYTHON:-python3}"
  if [[ -f "$selected_summary_path" ]]; then
    "$py" - <<PY 2>/dev/null || echo 150
import json
from pathlib import Path
p=Path(r"""$selected_summary_path""")
try:
  d=json.loads(p.read_text(encoding="utf-8"))
  print(len(d.get("selected_symbols", []) or []))
except Exception:
  print(150)
PY
  else
    echo 150
  fi
}

get_completed() {
  local py
  py="${PYTHON:-python3}"
  if [[ -f "$summary_path" ]]; then
    "$py" - <<PY 2>/dev/null || echo 0
import json
from pathlib import Path
p=Path(r"""$summary_path""")
try:
  d=json.loads(p.read_text(encoding="utf-8"))
  per=d.get("per_symbol", {}) or {}
  print(len(list(per.keys())) if isinstance(per, dict) else 0)
except Exception:
  print(0)
PY
  else
    echo 0
  fi
}

last_line() {
  local path="$1"
  if [[ -f "$path" ]]; then
    tail -n 1 "$path" 2>/dev/null || true
  fi
}

while true; do
  total="$(get_total)"
  done_cnt="$(get_completed)"
  pct="0.0"
  if [[ "$total" -gt 0 ]]; then
    pct="$(python3 - <<PY 2>/dev/null || echo 0.0
total=int("$total"); done_cnt=int("$done_cnt")
print(round(100.0*done_cnt/total,1) if total>0 else 0.0)
PY
)"
  fi

  size_mb="0.0"
  if [[ -d "$data_dir" ]]; then
    size_bytes="$(du -sb "$data_dir" 2>/dev/null | awk '{print $1}' || echo 0)"
    size_mb="$(python3 - <<PY 2>/dev/null || echo 0.0
print(round(float("$size_bytes")/1024/1024,1))
PY
)"
  fi

  clear || true
  echo "Fetch Progress: ${done_cnt}/${total} (${pct}%)"
  echo "Data Dir Size: ${size_mb} MB  (${data_dir})"
  runner="$(last_line "$runner_log_path")"
  stdout="$(last_line "$stdout_path")"
  [[ -n "$runner" ]] && echo "Runner: $runner"
  [[ -n "$stdout" ]] && echo "Stdout: $stdout"
  echo "Time: $(date '+%F %T')"
  sleep 5
done

