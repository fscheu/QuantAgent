# Development Scripts

## Worktree Management

### Quick Start

```bash
# Create new feature worktree with isolated database
./scripts/create_worktree.sh feature/my-feature

# Work on it
cd ../QuantAgent-feature-my-feature
source venv_wsl/bin/activate

# When done
git worktree remove ../QuantAgent-feature-my-feature
PGPASSWORD=password psql -h localhost -U postgres -c "DROP DATABASE quantagent_dev_feature_my_feature;"
```

### Scripts

#### `create_worktree.sh <branch-name> [base-branch]`

Creates a new git worktree with isolated database.

**Examples:**
```bash
./scripts/create_worktree.sh feature/logging
./scripts/create_worktree.sh hotfix/bug-123 main
```

**What it does:**
1. Creates git worktree at `../<repo-name>-<branch-name>`
2. Creates PostgreSQL database `quantagent_dev_<sanitized_branch_name>`
3. Creates `.env.local` with database URL override
4. Runs Alembic migrations

#### `setup_worktree_db.sh [worktree-path]`

Sets up isolated database for existing worktree.

**Examples:**
```bash
# Setup in current directory
./scripts/setup_worktree_db.sh

# Setup in specific worktree
./scripts/setup_worktree_db.sh ../QuantAgent-feature-x
```

**What it does:**
1. Detects branch name
2. Creates database `quantagent_dev_<branch_name>`
3. Creates `.env.local`
4. Runs migrations

## Why Database Isolation?

**Problem:** Multiple worktrees sharing one database causes:
- Conflicting Alembic migrations
- Schema mismatches between branches
- Data contamination
- Impossible to rollback individual features

**Solution:** Each worktree gets its own database via `.env.local` override.

See: [docs/workflows/worktree-database-isolation.md](../docs/workflows/worktree-database-isolation.md)

## Common Tasks

### List all worktrees
```bash
git worktree list
```

### List all databases
```bash
PGPASSWORD=password psql -h localhost -U postgres -l | grep quantagent
```

### Cleanup orphaned databases
```bash
# List worktrees
git worktree list

# Compare with databases
PGPASSWORD=password psql -h localhost -U postgres -l | grep quantagent

# Drop orphaned databases
PGPASSWORD=password psql -h localhost -U postgres -c "DROP DATABASE quantagent_dev_old_feature;"
```

### Check which database you're using
```bash
python -c "from quantagent.settings import DATABASE_URL; print(DATABASE_URL)"
```

### Verify migrations
```bash
source venv_wsl/bin/activate
python -m alembic current
python -m alembic history
```
