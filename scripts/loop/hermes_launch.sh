#!/usr/bin/env bash
# Hermes cron entry 1/2 (--no-agent): must finish in < 120 s (Hermes cron script timeout).
# Checks pause flag and review gate synchronously; if the loop may work, starts the long-running
# wrapper detached and returns. Stdout is delivered to Telegram verbatim.
# Runs the wrapper as committed on origin/main, copied to a temp file, so a deploy that resets the
# main checkout mid-run cannot change the script under bash's feet.
set -uo pipefail
export PATH="/home/azureuser/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
REPO="/home/azureuser/repos/projects/QuantAgent"
STATE="${LOOP_STATE_DIR:-/home/azureuser/.local/state/quantagent-loop}"
mkdir -p "$STATE"

git -C "$REPO" fetch -q --prune origin || { echo "⚠ QuantAgent loop: git fetch falló"; exit 0; }
if git -C "$REPO" cat-file -e origin/main:loop/PAUSE 2>/dev/null; then
  echo "⏸ QuantAgent loop: pausado (existe loop/PAUSE en main)"
  exit 0
fi
if [ -f "$STATE/running.pid" ] && kill -0 "$(cat "$STATE/running.pid")" 2>/dev/null; then
  echo "⏳ QuantAgent loop: sigue corriendo la entrega anterior (pid $(cat "$STATE/running.pid"))"
  exit 0
fi

TMP="$(mktemp -d)"
git -C "$REPO" archive origin/main loop scripts/loop | tar -x -C "$TMP"
GATE="$(cd "$TMP" && python3 scripts/loop/gate.py)"
rc=$?
if [ "$rc" -ne 0 ]; then
  python3 -c 'import json,sys; print(json.loads(sys.argv[1])["message"])' "$GATE" 2>/dev/null || echo "⚠ QuantAgent loop: gate sin salida válida"
  rm -rf "$TMP"
  exit 0
fi

rm -f "$STATE/last-result.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$STATE/last-launch.txt"
setsid nohup bash -c "bash '$TMP/scripts/loop/run_nightly.sh' >'$STATE/last-result.txt' 2>&1; rm -rf '$TMP' '$STATE/running.pid'" >/dev/null 2>&1 &
echo $! >"$STATE/running.pid"
WHAT="$(python3 -c '
import json,sys
d=json.loads(sys.argv[1]); t=d.get("ticket","")
if d.get("mode")=="cambio": print(f"aplica el cambio pedido en {t}")
elif d.get("previous")=="escalar": print(f"{t} queda escalado a Fede y toma el siguiente ticket del lote")
elif t: print(f"cierra {t} y toma el siguiente ticket del lote")
else: print("toma el primer ticket de la cola")' "$GATE")"
echo "▶ QuantAgent loop: arrancó la corrida ($WHAT)."
