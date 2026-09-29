#!/usr/bin/env bash
# Measure a delivery against the loop size limit (PLAN-CONTINUACION.md §3.2).
# Usage: scripts/loop/diff_size.sh [base-ref]   (default origin/main)
# Exit 0 within limit, 1 over limit. Whole-file deletions add 0 lines and are listed.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source loop/config.env
BASE="${1:-origin/main}"
set -f
excl=()
for g in $LOOP_EXCLUDE; do excl+=(":(exclude,glob)$g"); done
set +f
lines=0
files=0
deleted=()
while IFS=$'\t' read -r add del path; do
  [ -z "${path:-}" ] && continue
  if ! git cat-file -e "HEAD:$path" 2>/dev/null; then
    deleted+=("$path")
    continue
  fi
  [ "$add" = "-" ] && add=0
  [ "$del" = "-" ] && del=0
  lines=$((lines + add + del))
  files=$((files + 1))
done < <(git diff --no-renames --numstat "$BASE...HEAD" -- . "${excl[@]}")
echo "lines=$lines/$LOOP_MAX_LINES files=$files/$LOOP_MAX_FILES deleted=$(IFS=,; echo "${deleted[*]:-none}")"
if [ "$lines" -gt "$LOOP_MAX_LINES" ] || [ "$files" -gt "$LOOP_MAX_FILES" ]; then
  exit 1
fi
