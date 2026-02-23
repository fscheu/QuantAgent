#!/bin/bash
# Create a new git worktree with isolated database
# Usage: ./scripts/create_worktree.sh <branch-name> [base-branch]

set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/create_worktree.sh <branch-name> [base-branch]"
    echo ""
    echo "Examples:"
    echo "  ./scripts/create_worktree.sh feature/new-feature"
    echo "  ./scripts/create_worktree.sh hotfix/bug-123 main"
    exit 1
fi

BRANCH_NAME=$1
BASE_BRANCH=${2:-main}
REPO_ROOT=$(git rev-parse --show-toplevel)
REPO_NAME=$(basename "$REPO_ROOT")
WORKTREE_DIR="${REPO_ROOT}/../${REPO_NAME}-${BRANCH_NAME//\//-}"

echo "Creating worktree for branch: $BRANCH_NAME"
echo "Base branch: $BASE_BRANCH"
echo "Worktree directory: $WORKTREE_DIR"
echo ""

# Check if branch exists remotely
if git ls-remote --heads origin "$BRANCH_NAME" | grep -q "$BRANCH_NAME"; then
    echo "Branch exists remotely, checking out..."
    git worktree add "$WORKTREE_DIR" "$BRANCH_NAME"
else
    echo "Creating new branch from $BASE_BRANCH..."
    git worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR" "$BASE_BRANCH"
fi

echo "✓ Worktree created"

# Setup database for this worktree
cd "$WORKTREE_DIR"
bash ./scripts/setup_worktree_db.sh

echo ""
echo "✓ Worktree ready!"
echo ""
echo "To use this worktree:"
echo "  cd $WORKTREE_DIR"
echo "  source venv_wsl/bin/activate"
echo ""
echo "When done, remove with:"
echo "  git worktree remove $WORKTREE_DIR"
echo "  PGPASSWORD=password psql -h localhost -U postgres -c \"DROP DATABASE quantagent_dev_${BRANCH_NAME//\//_};\""
