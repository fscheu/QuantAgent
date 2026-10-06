#!/usr/bin/env bash
# pre-push hook installed by run_nightly.sh for the loop agent's process (QuantAgent-l4n).
# It is the agent-independent guard for PROMPT.md's hard rules: no push to main, no branch deletion,
# no non-fast-forward (forced) push. main has no branch protection, so this must not depend on the
# agent CLI's own permission system. After the checks it chains to the repo's own pre-push hook (beads).
zero=0000000000000000000000000000000000000000
INPUT="$(cat)"
while read -r _lref lsha rref rsha; do
  [ -z "${rref:-}" ] && continue
  if [ "$rref" = "refs/heads/main" ]; then
    echo "loop guard: push a main bloqueado" >&2; exit 1
  fi
  if [ "$lsha" = "$zero" ]; then
    echo "loop guard: borrar ramas remotas bloqueado" >&2; exit 1
  fi
  if [ "$rsha" != "$zero" ] && ! git merge-base --is-ancestor "$rsha" "$lsha" 2>/dev/null; then
    echo "loop guard: push forzado (no fast-forward) bloqueado" >&2; exit 1
  fi
done <<<"$INPUT"
if [ -n "${LOOP_ORIG_HOOKS:-}" ] && [ -x "$LOOP_ORIG_HOOKS/pre-push" ]; then
  printf '%s\n' "$INPUT" | "$LOOP_ORIG_HOOKS/pre-push" "$@"
fi
