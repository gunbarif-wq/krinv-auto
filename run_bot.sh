#!/usr/bin/env bash
set -euo pipefail

cd /home/opc/krinv-auto

exec /usr/bin/python3 -u user_only_strategy/monday_custom_timing_bot.py \
  --bar-minutes 3 \
  --max-universe 50 \
  --max-positions 2 \
  --sync-holdings \
  --holdings-sync-interval-sec 30 \
  --market-open-hhmm 900 \
  --market-close-hhmm 1530 \
  --refresh-start-hhmm 900 \
  --refresh-end-hhmm 1530 \
  --refresh-interval-min 120 \
  --empty-refresh-interval-min 120 \
  --scan-interval-sec 5 \
  --max-cycles 1000000 \
  --log-file logs/user_only_strategy_signals.txt

