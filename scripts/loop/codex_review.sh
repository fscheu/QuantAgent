#!/usr/bin/env bash
# Mechanical first-pass review of one loop PR by Codex (PLAN-CONTINUACION.md §3.7). It reports; the PM decides.
# Launched by a Hermes cron in --no-agent mode (must return in < 120 s), never from inside the PM session:
# Codex needs its own sandbox and cannot build it inside another one.
# Usage: codex_review.sh [<pr> [<model>]]. Without arguments it consumes $LOOP_LOG_DIR/next-review ("<pr> [model]").
# Stdout is delivered to Telegram verbatim. The report lands in $LOOP_LOG_DIR/review-<pr>.md and as a PR comment.
set -uo pipefail
export PATH="/home/azureuser/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH" BEADS_NO_DAEMON=1
REPO="${LOOP_REPO:-/home/azureuser/repos/projects/QuantAgent}"
git -C "$REPO" fetch -q --prune origin || { echo "⚠ revisión Codex: git fetch falló"; exit 0; }
TMP="$(mktemp -d)"
git -C "$REPO" archive "${LOOP_SRC_REF:-origin/main}" loop scripts/loop | tar -x -C "$TMP"
# shellcheck disable=SC1091
source "$TMP/loop/config.env"
STATE="$LOOP_LOG_DIR"; mkdir -p "$STATE"
if [ $# -eq 0 ] && [ -s "$STATE/next-review" ]; then
  # shellcheck disable=SC2046
  set -- $(cat "$STATE/next-review"); rm -f "$STATE/next-review"
fi
PR="${1:-}"; MODEL="${2:-$LOOP_CODEX_MODEL}"
case "$PR" in ''|*[!0-9]*) echo "⚠ revisión Codex: falta el número de PR"; rm -rf "$TMP"; exit 0 ;; esac
case " $LOOP_CODEX_MODELS " in *" $MODEL "*) ;; *) echo "⚠ revisión Codex: modelo desconocido '$MODEL' (válidos: $LOOP_CODEX_MODELS)"; rm -rf "$TMP"; exit 0 ;; esac
if [ -f "$STATE/review.pid" ] && kill -0 "$(cat "$STATE/review.pid")" 2>/dev/null; then
  echo "⏳ revisión Codex: sigue corriendo la anterior"; rm -rf "$TMP"; exit 0
fi
INFO="$(gh pr view "$PR" -R "$LOOP_GITHUB_REPO" --json headRefName,baseRefName,state -q '[.headRefName,.baseRefName,.state]|@tsv')" \
  || { echo "⚠ revisión Codex: no pude leer el PR #$PR"; rm -rf "$TMP"; exit 0; }
read -r BRANCH BASE PRSTATE <<<"$INFO"
case "$BASE" in "$LOOP_LOT_PREFIX"*) ;; *) echo "⚠ revisión Codex: #$PR no va contra una rama de lote ($BASE)"; rm -rf "$TMP"; exit 0 ;; esac
[ "$PRSTATE" = "OPEN" ] || { echo "⚠ revisión Codex: #$PR no está abierto"; rm -rf "$TMP"; exit 0; }
TICKET="QuantAgent-${BRANCH##*QuantAgent-}"
WT="/tmp/ai-review/$PR-$(date -u +%H%M%S)"; mkdir -p /tmp/ai-review
git -C "$REPO" worktree add -q --detach "$WT" "origin/$BRANCH" >/dev/null 2>&1 || { echo "⚠ revisión Codex: no pude crear el worktree"; rm -rf "$TMP"; exit 0; }
OUT="$STATE/review-$PR.md"; rm -f "$OUT"
WEB_CHANGED=0
git -C "$WT" diff --quiet "origin/$BASE"...HEAD -- web/ || WEB_CHANGED=1
{
  cat "$TMP/loop/REVIEW-PROMPT.md"
  printf '\nCONTEXTO\nPR: #%s · rama: %s · base: origin/%s · ticket: %s · modelo: %s\n\nTICKET (bd show)\n' "$PR" "$BRANCH" "$BASE" "$TICKET" "$MODEL"
  (cd "$REPO" && bd show "$TICKET" 2>&1)
  printf '\nCUERPO DEL PR (el revisor no tiene red: no uses gh)\n'
  gh pr view "$PR" -R "$LOOP_GITHUB_REPO" --json body -q .body 2>&1
} >"$WT/.review-prompt.txt"
cat >"$TMP/run.sh" <<RUN
cd '$WT' && source '$REPO/.venv/bin/activate'
export DATABASE_URL='sqlite:///$WT/try.db' QUANTAGENT_SNAPSHOT_DIR='$LOOP_SNAPSHOT_DIR'
python -c 'from quantagent.database import init_db; init_db()'
# E2E runs through try.sh; preserve its status and screenshots before its worktree is removed.
E2E_REPORT='No aplica: el diff no toca web/.'
if [ '$WEB_CHANGED' = 1 ]; then
  E2E_ARTIFACTS='$STATE/review-$PR-e2e-artifacts'
  rm -rf "\$E2E_ARTIFACTS"
  timeout '${LOOP_REVIEW_TIMEOUT_MIN}m' bash '$WT/scripts/loop/try.sh' '$BRANCH' -- "cd web && npm ci && { npm run e2e; status=\\\$?; cp -R e2e-artifacts '\$E2E_ARTIFACTS'; exit \\\$status; }" >'$STATE/review-$PR-e2e.log' 2>&1
  E2E_STATUS=\$?
  E2E_REPORT="npm run e2e vía try.sh: exit \$E2E_STATUS; screenshots: \$E2E_ARTIFACTS"
  printf '\\nWEB: %s\\nMirá los screenshots de 390px antes de reportar; un exit distinto de 0 impide aprobar.\\n' "\$E2E_REPORT" >>.review-prompt.txt
fi
timeout '${LOOP_REVIEW_TIMEOUT_MIN}m' '${LOOP_CODEX_BIN:-codex}' exec -m '$MODEL' -s workspace-write -c shell_environment_policy.inherit=all \\
  -o '$OUT' - <.review-prompt.txt >'$STATE/review-$PR.log' 2>&1
[ -s '$OUT' ] || echo 'Codex no entregó reporte (ver review-$PR.log).' >'$OUT'
# The gate tells PM: from R: by line prefix: a report line must never start with either.
sed -i -E 's/^(PM|R):/> \\1:/' '$OUT'; sed -i '1i CODEX: $TICKET · #$PR · modelo $MODEL · reporte mecánico, no decide\\n' '$OUT'
printf '\\nE2E: %s\\n' "\$E2E_REPORT" >>'$OUT'
if [ '$WEB_CHANGED' = 1 ]; then tail -n 15 '$STATE/review-$PR-e2e.log' >>'$OUT'; fi
[ -n '${LOOP_REVIEW_DRY:-}' ] || gh pr comment '$PR' -R '$LOOP_GITHUB_REPO' --body-file '$OUT' >/dev/null
cd /; git -C '$REPO' worktree remove --force '$WT' >/dev/null 2>&1; rm -rf '$TMP' '$STATE/review.pid'
RUN
setsid nohup bash "$TMP/run.sh" >/dev/null 2>&1 &
echo $! >"$STATE/review.pid"
echo "▶ revisión Codex: arrancó sobre #$PR ($TICKET) con $MODEL. Reporte: $OUT y comentario en el PR."
