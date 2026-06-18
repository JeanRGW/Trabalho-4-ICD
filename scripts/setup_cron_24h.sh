#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_monitor_daily.sh"
CRON_EXPRESSION="${CRON_EXPRESSION:-0 3 * * *}"

chmod +x "$RUNNER"

CURRENT_CRON="$(crontab -l 2>/dev/null || true)"
FILTERED_CRON="$(printf "%s\n" "$CURRENT_CRON" | grep -vF "$RUNNER" || true)"

{
  printf "%s\n" "$FILTERED_CRON"
  printf "%s %s\n" "$CRON_EXPRESSION" "$RUNNER"
} | crontab -

echo "Cron configurado: $CRON_EXPRESSION $RUNNER"
