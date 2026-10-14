#!/usr/bin/env bash
# Delivery step for Codex as loop implementer (QuantAgent-dhlu). Its sandbox keeps .git read-only even
# with --add-dir, so Codex cannot branch, commit or push. It leaves .loop-delivery/ in the worktree and
# this script, run by run_nightly.sh outside the sandbox and under the push guard, does the git part.
# Usage (cwd = the run's worktree): codex_deliver.sh <base-branch> <mode: nuevo|cambio>
set -u
BASE="$1"; MODE="$2"; D=.loop-delivery
: "${LOOP_GITHUB_REPO:?codex_deliver: falta LOOP_GITHUB_REPO}"  # checked before any push
fail() { echo "codex_deliver: $1"; exit 1; }
[ -s "$D/commit-msg" ] || fail "sin $D/commit-msg: Codex no dejó entrega"
MSG="$(cat "$D/commit-msg")"
BODY="$(cat "$D/pr-body.md" 2>/dev/null)"
LABELS="loop"
for l in $(cat "$D/labels" 2>/dev/null); do
  case "$l" in bloqueo | particion) LABELS="$LABELS,$l" ;; esac
done
if [ "$MODE" != "cambio" ]; then
  ID="$(tr -d '[:space:]' <"$D/ticket" 2>/dev/null)"
  printf '%s' "$ID" | grep -qE '^QuantAgent-[A-Za-z0-9.]+$' || fail "ticket inválido en $D/ticket: '$ID'"
  [ -n "$BODY" ] || fail "sin $D/pr-body.md"
  git checkout -q -b "loop/$ID" || fail "no pude crear loop/$ID"
fi
BRANCH="$(git branch --show-current)"
[ -n "$BRANCH" ] || fail "el worktree no está en una rama"
rm -rf "$D"
git add -A && git commit -q -m "$MSG" || fail "nada para commitear"
git push -q -u origin "$BRANCH" || fail "push de $BRANCH rechazado"
if [ "$MODE" = "cambio" ]; then
  gh pr comment "$BRANCH" -R "$LOOP_GITHUB_REPO" --body "${BODY:-$MSG}"
else
  gh pr create -R "$LOOP_GITHUB_REPO" --base "$BASE" --head "$BRANCH" --label "$LABELS" \
    --title "$(printf '%s\n' "$MSG" | head -1)" --body "$BODY"
fi
