#!/bin/bash
# Setup isolated database for a worktree
# Usage: ./scripts/setup_worktree_db.sh [worktree_path]

set -e

# Get worktree path or use current directory
WORKTREE_PATH=${1:-.}
cd "$WORKTREE_PATH"

# Get branch name
BRANCH_NAME=$(git branch --show-current)
if [ -z "$BRANCH_NAME" ]; then
    echo "Error: Not on a branch"
    exit 1
fi

# Sanitize branch name for database name (remove special chars)
DB_SUFFIX=$(echo "$BRANCH_NAME" | sed 's/[^a-zA-Z0-9]/_/g')
DB_NAME="quantagent_dev_${DB_SUFFIX}"

echo "Setting up database for worktree: $WORKTREE_PATH"
echo "Branch: $BRANCH_NAME"
echo "Database: $DB_NAME"

# Create .env.local with database URL
cat > .env.local <<EOF
# Worktree-specific database configuration
# Auto-generated for branch: $BRANCH_NAME
DATABASE_URL=postgresql://postgres:password@localhost:5432/${DB_NAME}
EOF

echo "✓ Created .env.local"

# Create database if it doesn't exist
echo "Creating database ${DB_NAME}..."
PGPASSWORD=password psql -h localhost -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'" | grep -q 1 || \
    PGPASSWORD=password psql -h localhost -U postgres -c "CREATE DATABASE ${DB_NAME};"

echo "✓ Database created"

# Load environment and run migrations
if [ -f "venv_wsl/bin/activate" ]; then
    source venv_wsl/bin/activate
    echo "Running migrations..."
    python -m alembic upgrade head
    echo "✓ Migrations applied"
else
    echo "⚠ Virtual environment not found. Run migrations manually:"
    echo "  source venv_wsl/bin/activate"
    echo "  python -m alembic upgrade head"
fi

echo ""
echo "✓ Worktree database setup complete!"
echo "  Database: $DB_NAME"
echo "  Config: .env.local"
echo ""
echo "To verify:"
echo "  PGPASSWORD=password psql -h localhost -U postgres -d ${DB_NAME} -c '\\dt'"
