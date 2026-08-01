#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT='/Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents'
LOG_DIR="$PROJECT_ROOT/harness_data/logs"
CRON_LINE="*/5 * * * * cd $PROJECT_ROOT && .venv/bin/python -m tradingagents.harness.market_warning.runner >> harness_data/logs/market_warning.log 2>&1"

printf '%s\n' "$CRON_LINE"

if [[ "${1:-}" != "--yes" ]]; then
  printf 'Install this market-warning cron entry? [y/N] '
  read -r answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) printf 'Cancelled.\n'; exit 0 ;;
  esac
fi

existing="$(crontab -l 2>/dev/null || true)"
if printf '%s\n' "$existing" | grep -Fqx "$CRON_LINE"; then
  printf 'Already installed.\n'
  exit 0
fi

mkdir -p "$LOG_DIR"

{
  if [[ -n "$existing" ]]; then
    printf '%s\n' "$existing"
  fi
  printf '%s\n' "$CRON_LINE"
} | crontab -

printf 'Installed.\n'
