#!/usr/bin/env bash
# Run a command on a PR branch in a throwaway worktree, with an isolated SQLite DB.
# Meant to be pasted from a phone into the VM terminal. Never touches the main checkout.
# Usage: scripts/loop/try.sh <branch> -- '<command>'
set -uo pipefail
REPO="$(dirname "$(git -C "$(dirname "$0")" rev-parse --path-format=absolute --git-common-dir)")"
branch="${1:?usage: try.sh <branch> -- '<command>'}"
shift
[ "${1:-}" = "--" ] && shift
git -C "$REPO" fetch -q origin "$branch" || { echo "try.sh: branch not found: $branch"; exit 2; }
WT="$(mktemp -d /tmp/loop-try.XXXXXX)"
git -C "$REPO" worktree add -q --detach "$WT" "origin/$branch" >/dev/null 2>&1 || { echo "try.sh: worktree failed"; exit 2; }
trap 'git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1' EXIT
cd "$WT"
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
export DATABASE_URL="sqlite:///$WT/try.db"
python -c "from quantagent.database import init_db; init_db()"
echo "── $branch @ $(git rev-parse --short HEAD) ──"
bash -c "$*"
