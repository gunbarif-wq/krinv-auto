#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

run_name="auto150_3m_1y_excl50_real_fetch"
exclude_summary_path="$repo_dir/data/chart_retrain/live50_3m_6m_excl30_real_fetch/summary.json"
selected_summary_path="$repo_dir/data/chart_retrain/$run_name/selected_summary.json"
summary_path="$repo_dir/data/chart_retrain/$run_name/summary.json"
stdout_path="$repo_dir/logs/fetch_150_1y_real.stdout.txt"
stderr_path="$repo_dir/logs/fetch_150_1y_real.stderr.txt"
runner_log_path="$repo_dir/logs/fetch_150_1y_real.runner.txt"

mkdir -p "$repo_dir/logs" "$repo_dir/data/chart_retrain/$run_name"

load_env() {
  local env_path="$repo_dir/.env"
  if [[ ! -f "$env_path" ]]; then
    return
  fi
  # Parse .env safely on Linux even if it contains BOM/CRLF.
  # Supports simple KEY=VALUE lines (no shell expansions).
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Strip CR (Windows line endings)
    line="${line%$'\r'}"
    # Strip UTF-8 BOM if present at the beginning of file/line
    line="${line#"$'\ufeff'"}"
    # Skip blanks/comments
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    # Only accept KEY=VALUE
    if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      key="${line%%=*}"
      val="${line#*=}"
      # Remove surrounding quotes if present
      if [[ "$val" =~ ^\".*\"$ ]]; then
        val="${val:1:${#val}-2}"
      elif [[ "$val" =~ ^\'.*\'$ ]]; then
        val="${val:1:${#val}-2}"
      fi
      export "$key=$val"
    fi
  done <"$env_path"
}

read_json_int() {
  local py code
  py="${PYTHON:-python3}"
  code="$1"
  "$py" - <<PY 2>/dev/null || echo 0
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
try:
  d=json.loads(p.read_text(encoding="utf-8"))
except Exception:
  print(0); raise SystemExit
${code}
PY
}

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

ensure_selected_summary() {
  if [[ -f "$selected_summary_path" ]]; then
    return
  fi
  load_env
  : "${KIS_APP_KEY:?KIS_APP_KEY missing in .env}"
  : "${KIS_APP_SECRET:?KIS_APP_SECRET missing in .env}"
  local py
  py="${PYTHON:-python3}"
  "$py" -u user_only_strategy/build_selected_summary_150.py \
    --base-url "https://openapi.koreainvestment.com:9443" \
    --app-key "$KIS_APP_KEY" \
    --app-secret "$KIS_APP_SECRET" \
    --exclude-summary "$exclude_summary_path" \
    --out-summary "$selected_summary_path" \
    --count 150 \
    >>"$stdout_path" 2>>"$stderr_path"
}

while true; do
  ensure_selected_summary
  total="$(get_total)"
  done_cnt="$(get_completed)"
  if [[ "$total" -gt 0 && "$done_cnt" -ge "$total" ]]; then
    printf '[runner] completed %s/%s at %s\n' "$done_cnt" "$total" "$(date '+%F %T')" >>"$runner_log_path"
    break
  fi

  printf '[runner] launch %s/%s at %s\n' "$done_cnt" "$total" "$(date '+%F %T')" >>"$runner_log_path"
  "${PYTHON:-python3}" -u user_only_strategy/retrain_chart_classifier_live.py \
    --base-url "https://openapi.koreainvestment.com:9443" \
    --quote-base-url "https://openapi.koreainvestment.com:9443" \
    --use-selected-summary "$selected_summary_path" \
    --bar-minutes 3 \
    --fetch-bdays 252 \
    --exclude-recent-bdays 0 \
    --fetch-only \
    --retry-count 2 \
    --max-day-errors-per-symbol 12 \
    --max-consecutive-day-errors 5 \
    --resume \
    --model-name "$run_name" \
    >>"$stdout_path" 2>>"$stderr_path" || true

  sleep 10
done
