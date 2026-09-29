#!/usr/bin/env bash
# Nightly loop wrapper (PLAN-CONTINUACION.md §3.5). Launched by a Hermes cron in --no-agent mode;
# its stdout is delivered to Telegram verbatim, so it prints only short status lines.
# Order: pause flag -> review gate -> fresh worktree -> headless Claude Code -> size check -> cleanup.
set -uo pipefail
REPO="${LOOP_REPO:-/home/azureuser/repos/projects/QuantAgent}"
RUN="$(date -u +%Y%m%dT%H%M%SZ)"
WT="/tmp/ai-loop/$RUN"
export BEADS_NO_DAEMON=1

git -C "$REPO" fetch -q --prune origin || { echo "⚠ QuantAgent loop: git fetch falló"; exit 0; }
if git -C "$REPO" cat-file -e origin/main:loop/PAUSE 2>/dev/null; then
  echo "⏸ QuantAgent loop: pausado (existe loop/PAUSE en main)"
  exit 0
fi

mkdir -p /tmp/ai-loop
git -C "$REPO" worktree add -q --detach "$WT" origin/main >/dev/null 2>&1 || { echo "⚠ QuantAgent loop: no pude crear el worktree"; exit 0; }
cleanup() { cd /; git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1; }
trap cleanup EXIT
cd "$WT" || exit 0
# shellcheck disable=SC1091
source loop/config.env
mkdir -p "$LOOP_LOG_DIR"
LOG="$LOOP_LOG_DIR/$RUN.log"

GATE="$(python3 scripts/loop/gate.py)"
rc=$?
if [ "$rc" -ne 0 ]; then
  python3 -c 'import json,sys; print(json.loads(sys.argv[1])["message"])' "$GATE" 2>/dev/null || echo "⚠ QuantAgent loop: gate sin salida válida"
  exit 0
fi

MODE="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("mode",""))' "$GATE")"
if [ "$MODE" = "cambio" ]; then
  BRANCH="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["branch"])' "$GATE")"
  git fetch -q origin "$BRANCH" && git checkout -q -B "$BRANCH" "origin/$BRANCH" || { echo "⚠ QuantAgent loop: no pude abrir $BRANCH"; exit 0; }
fi

# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
PROMPT="$(cat loop/PROMPT.md)

CONTEXTO DEL GATE (JSON):
$GATE

Worktree de esta corrida: $WT (ya estás parado acá, rama base origin/main o la rama del PR en modo cambio)."

STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
timeout "${LOOP_TIMEOUT_MIN}m" claude -p "$PROMPT" \
  --permission-mode acceptEdits \
  --allowedTools "Bash" "Read" "Edit" "Write" "Glob" "Grep" "Agent" "TodoWrite" \
  --disallowedTools "Bash(git push origin main:*)" "Bash(git push --force:*)" "Bash(git push -f:*)" "Bash(git branch -D:*)" \
  >"$LOG" 2>&1
crc=$?

PR_JSON="$(gh pr list --label "$LOOP_LABEL" --state all --limit 1 --json number,url,body,createdAt,updatedAt,headRefName 2>/dev/null)"
NEW="$(python3 -c '
import json,sys
prs=json.loads(sys.argv[1] or "[]"); started=sys.argv[2]
p=prs[0] if prs else None
print(p["headRefName"] if p and max(p["createdAt"],p["updatedAt"])>=started else "")' "$PR_JSON" "$STARTED")"

if [ -z "$NEW" ]; then
  echo "⚠ QuantAgent loop: la corrida terminó sin entrega (claude exit $crc). Log: $LOG"
  exit 0
fi

git fetch -q origin "$NEW" && git checkout -q --detach "origin/$NEW"
SIZE="$(scripts/loop/diff_size.sh origin/main)" || gh pr edit "$NEW" --add-label excede-limite >/dev/null 2>&1
python3 -c '
import json,sys
p=json.loads(sys.argv[1])[0]
head=[line for line in p["body"].splitlines() if line.strip()][:3]
print("\n".join(head)); print(sys.argv[2]); print(p["url"])' "$PR_JSON" "$SIZE"
