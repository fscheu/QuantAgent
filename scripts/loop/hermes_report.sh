#!/usr/bin/env bash
# Hermes cron entry 2/2 (--no-agent): prints the result of the last loop run for Telegram.
# Silent (empty stdout) when no run was launched since the last report.
set -uo pipefail
STATE="${LOOP_STATE_DIR:-/home/azureuser/.local/state/quantagent-loop}"
[ -f "$STATE/last-launch.txt" ] || exit 0
if [ -f "$STATE/running.pid" ] && kill -0 "$(cat "$STATE/running.pid")" 2>/dev/null; then
  echo "⏳ QuantAgent loop: la corrida lanzada el $(cat "$STATE/last-launch.txt") sigue en curso"
  exit 0
fi
if [ -s "$STATE/last-result.txt" ]; then
  cat "$STATE/last-result.txt"
else
  echo "⚠ QuantAgent loop: la corrida del $(cat "$STATE/last-launch.txt") terminó sin salida. Logs en $STATE/"
fi
mv "$STATE/last-launch.txt" "$STATE/last-launch.reported"
