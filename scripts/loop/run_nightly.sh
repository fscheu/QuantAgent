#!/usr/bin/env bash
# Nightly loop wrapper (PLAN-CONTINUACION.md §3.5). Launched by a Hermes cron in --no-agent mode;
# its stdout is delivered to Telegram verbatim, so it prints only short status lines.
# Order: pause flag -> review gate (open lot + reviewed delivery) -> fresh worktree on the lot branch ->
# headless agent (claude | agy) -> size check -> cleanup.
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
HOOKS=""
cleanup() { cd /; git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1; [ -n "$HOOKS" ] && rm -rf "$HOOKS"; }
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

# Agent for this run: the one-shot override file wins over LOOP_AGENT and is consumed here, after the
# gate passed, so a WAIT night does not burn it.
AGENT="${LOOP_AGENT:-claude}"
if [ -s "$LOOP_LOG_DIR/next-agent" ]; then
  AGENT="$(tr -d '[:space:]' <"$LOOP_LOG_DIR/next-agent")"
  rm -f "$LOOP_LOG_DIR/next-agent"
fi
case "$AGENT" in
  claude) AGENT_DESC="claude" ;;
  agy) AGENT_DESC="agy/$LOOP_AGY_MODEL" ;;
  *) echo "⚠ QuantAgent loop: agente desconocido '$AGENT' (válidos: claude, agy). No se lanzó nada."; exit 0 ;;
esac

MODE="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("mode",""))' "$GATE")"
# Deliveries start from the lot branch and their PRs target it (§3.7); only Fede's lot merge reaches main.
BASE="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["base"])' "$GATE")"
case "$BASE" in
  "$LOOP_LOT_PREFIX"?*) ;;
  *) echo "⚠ QuantAgent loop: la base '$BASE' no es una rama de lote ($LOOP_LOT_PREFIX*). No se lanzó nada."; exit 0 ;;
esac
if [ "$MODE" = "cambio" ]; then
  BRANCH="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["branch"])' "$GATE")"
  git fetch -q origin "$BRANCH" && git checkout -q -B "$BRANCH" "origin/$BRANCH" || { echo "⚠ QuantAgent loop: no pude abrir $BRANCH"; exit 0; }
else
  git fetch -q origin "$BASE" && git checkout -q --detach "origin/$BASE" || { echo "⚠ QuantAgent loop: no pude abrir $BASE"; exit 0; }
fi

# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
# Tests that need Postgres run against the loop's own DB, never the dev DB (same layout as CI).
export DATABASE_URL="$LOOP_TEST_DATABASE_URL"
python -m alembic upgrade head >>"$LOG" 2>&1 || { echo "⚠ QuantAgent loop: alembic upgrade falló sobre la base de tests. Log: $LOG"; exit 0; }
PROMPT="$(cat loop/PROMPT.md)

CONTEXTO DEL GATE (JSON):
$GATE

Rama base de esta corrida (el lote): $BASE
Worktree de esta corrida: $WT (ya estás parado acá, sobre origin/$BASE o sobre la rama del PR en modo cambio)."

# Push guard for whichever agent runs: a pre-push hook that only this process tree sees (git reads
# core.hooksPath from the GIT_CONFIG_* env). The repo's other hooks stay active through symlinks.
HOOKS="$(mktemp -d)"
LOOP_ORIG_HOOKS="$(git rev-parse --path-format=absolute --git-common-dir)/hooks"
for h in "$LOOP_ORIG_HOOKS"/*; do
  case "$h" in *.sample|*.backup|*/pre-push) ;; *) [ -x "$h" ] && ln -s "$h" "$HOOKS/" ;; esac
done
# The guard comes from next to this script (origin/main), not from the worktree: in modo cambio the
# worktree is on the PR branch. Without the guard nothing is launched.
cp "$SELF_DIR/push_guard.sh" "$HOOKS/pre-push" && chmod +x "$HOOKS/pre-push" || { echo "⚠ QuantAgent loop: no pude instalar la barrera de push. No se lanzó nada."; exit 0; }
export LOOP_ORIG_HOOKS GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0="$HOOKS"

STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
case "$AGENT" in
  claude)
    timeout "${LOOP_TIMEOUT_MIN}m" claude -p "$PROMPT" \
      --permission-mode acceptEdits \
      --allowedTools "Bash" "Read" "Edit" "Write" "Glob" "Grep" "Agent" "TodoWrite" \
      --disallowedTools "Bash(git push origin main:*)" "Bash(git push --force:*)" "Bash(git push -f:*)" "Bash(git branch -D:*)" \
      >"$LOG" 2>&1 ;;
  agy)
    # No --dangerously-skip-permissions: what agy may run comes from the allow/deny rules in
    # ~/.gemini/antigravity-cli/settings.json (docs/loop/SETUP-HERMES.md). A command outside them is
    # auto-denied in headless mode and the run ends without a delivery.
    timeout "${LOOP_TIMEOUT_MIN}m" agy -p "$PROMPT" --model "$LOOP_AGY_MODEL" --mode accept-edits \
      --print-timeout "${LOOP_TIMEOUT_MIN}m" >"$LOG" 2>&1 ;;
esac
crc=$?
unset GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0

PR_JSON="$(gh pr list -R "$LOOP_GITHUB_REPO" --label "$LOOP_LABEL" --state all --limit 1 --json number,url,body,createdAt,updatedAt,headRefName 2>/dev/null)"
NEW="$(python3 -c '
import json,sys
prs=json.loads(sys.argv[1] or "[]"); started=sys.argv[2]
p=prs[0] if prs else None
print(p["headRefName"] if p and max(p["createdAt"],p["updatedAt"])>=started else "")' "$PR_JSON" "$STARTED")"

if [ -z "$NEW" ]; then
  echo "⚠ QuantAgent loop: la corrida terminó sin entrega ($AGENT_DESC exit $crc). Log: $LOG"
  exit 0
fi
# Label the delivery with the agent that produced it, to compare agents across PRs.
gh label create "agent:$AGENT" -R "$LOOP_GITHUB_REPO" --color ededed >/dev/null 2>&1
gh pr edit -R "$LOOP_GITHUB_REPO" "$NEW" --add-label "agent:$AGENT" >/dev/null 2>&1

git fetch -q origin "$NEW" && git checkout -q --detach "origin/$NEW"
SIZE="$(scripts/loop/diff_size.sh "origin/$BASE")" || gh pr edit -R "$LOOP_GITHUB_REPO" "$NEW" --add-label excede-limite >/dev/null 2>&1
python3 -c '
import json,sys
p=json.loads(sys.argv[1])[0]
head=[line for line in p["body"].splitlines() if line.strip()][:3]
print("\n".join(head)); print(sys.argv[2]); print(p["url"])' "$PR_JSON" "$SIZE agent=$AGENT_DESC"
