# Worktree Database Isolation

## Problem

Working with multiple git worktrees against a **single shared database** causes serious problems:

### Issues with Shared Database:
1. **Conflicting migrations**: Each branch may have different Alembic migrations
2. **Schema inconsistency**: Code expects different schemas in different branches
3. **Single `alembic_version`**: Can only track one migration state at a time
4. **Data contamination**: Test data from one branch leaks into others
5. **Impossible rollbacks**: Can't downgrade one branch without affecting others

## Solution: One Database Per Worktree

Each git worktree gets its own isolated PostgreSQL database.

```
Main worktree    → quantagent_dev
feature/logging  → quantagent_dev_feature_logging
feature/positions→ quantagent_dev_feature_positions
```

## Setup

### 1. Initial Configuration

The system is already configured:
- ✅ `.env` - shared defaults (committed)
- ✅ `.env.local` - worktree-specific overrides (gitignored)
- ✅ `quantagent/settings.py` - loads both files (`.env.local` overrides `.env`)

### 2. Create Worktree with Database

Use the provided script:

```bash
# Create worktree and setup isolated database
./scripts/create_worktree.sh feature/new-feature

# This will:
# 1. Create git worktree at ../QuantAgent-feature-new-feature
# 2. Create database quantagent_dev_feature_new_feature
# 3. Create .env.local with DATABASE_URL override
# 4. Run Alembic migrations
```

### 3. Work in Worktree

```bash
cd ../QuantAgent-feature-new-feature
source venv_wsl/bin/activate

# Your code uses the isolated database automatically
python -m alembic upgrade head
python quantagent/backtesting/run_backtest.py
```

### 4. Cleanup When Done

```bash
# Remove worktree
git worktree remove ../QuantAgent-feature-new-feature

# Drop database
PGPASSWORD=password psql -h localhost -U postgres \
  -c "DROP DATABASE quantagent_dev_feature_new_feature;"
```

## Manual Setup (Alternative)

If you prefer manual setup:

```bash
# 1. Create worktree
git worktree add ../QuantAgent-feature-name feature/name

# 2. Create database
PGPASSWORD=password psql -h localhost -U postgres \
  -c "CREATE DATABASE quantagent_dev_feature_name;"

# 3. Create .env.local in the worktree
cd ../QuantAgent-feature-name
cat > .env.local <<EOF
DATABASE_URL=postgresql://postgres:password@localhost:5432/quantagent_dev_feature_name
EOF

# 4. Run migrations
source venv_wsl/bin/activate
python -m alembic upgrade head
```

## Best Practices

### ✅ DO:
- Create a new database for each worktree
- Use `.env.local` for worktree-specific config
- Run migrations in each worktree independently
- Clean up databases when removing worktrees

### ❌ DON'T:
- Share a database between worktrees
- Commit `.env.local` to git
- Apply migrations from one worktree to another's database
- Mix data between development databases

## Migration Workflow

### Creating Migrations

```bash
# In your feature worktree
cd ../QuantAgent-feature-X
source venv_wsl/bin/activate

# Make model changes
vim quantagent/models.py

# Generate migration
python -m alembic revision --autogenerate -m "add new feature"

# Apply locally
python -m alembic upgrade head

# Test your feature
pytest tests/

# Commit
git add alembic/versions/*.py quantagent/models.py
git commit -m "feat: add new feature with migration"
```

### Merging Branches

When merging branches with migrations:

```bash
# After git merge, check for divergent heads
alembic history

# If multiple heads exist (EXPECTED after merge):
# Rev: abc123 (head)
# Rev: def456 (head)

# Create merge migration
python -m alembic merge abc123 def456 -m "merge migrations"

# Apply merge
python -m alembic upgrade head

# Commit merge migration
git add alembic/versions/*_merge_*.py
git commit -m "chore: merge alembic migration heads"
```

### Syncing Worktrees After Merge

After main branch gets a merge:

```bash
# In each worktree, update to latest
cd ../QuantAgent-worktree-X
git pull origin main

# Apply new migrations
python -m alembic upgrade head
```

## Troubleshooting

### "Multiple heads present" Error

This is **normal** after merging branches with migrations.

**Solution**: Create merge migration (see "Merging Branches" above)

### "Table already exists" Error

Your database state doesn't match `alembic_version`.

**Solution**:
```bash
# Check which tables exist
PGPASSWORD=password psql -h localhost -U postgres -d your_db -c "\dt"

# Check alembic version
python -m alembic current

# If table exists but migration not applied:
python -m alembic stamp <revision_id>
```

### Accidentally Used Wrong Database

If you ran migrations in the wrong database:

```bash
# Downgrade the wrong database
python -m alembic downgrade <previous_revision>

# Or drop and recreate
PGPASSWORD=password psql -h localhost -U postgres \
  -c "DROP DATABASE wrong_db;" \
  -c "CREATE DATABASE wrong_db;"
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Main Worktree                                    │
│ /repos_local/QuantAgent                          │
│ Branch: main                                     │
│                                                  │
│ .env         → DATABASE_URL=.../quantagent_dev  │
│ .env.local   → (not present, uses .env)         │
│                                                  │
│ DB: quantagent_dev                               │
│   alembic_version: abc123                       │
│   tables: orders, trades, logs, ...             │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Feature Worktree A                               │
│ /repos_local/QuantAgent-feature-logging          │
│ Branch: feature/logging                          │
│                                                  │
│ .env         → DATABASE_URL=.../quantagent_dev  │
│ .env.local   → DATABASE_URL=.../dev_logging     │
│                                                  │
│ DB: quantagent_dev_feature_logging              │
│   alembic_version: def456                       │
│   tables: orders, trades, logs, ...             │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Feature Worktree B                               │
│ /repos_local/QuantAgent-feature-positions        │
│ Branch: feature/positions                        │
│                                                  │
│ .env         → DATABASE_URL=.../quantagent_dev  │
│ .env.local   → DATABASE_URL=.../dev_positions   │
│                                                  │
│ DB: quantagent_dev_feature_positions            │
│   alembic_version: ghi789                       │
│   tables: orders, trades, active_positions, ... │
└─────────────────────────────────────────────────┘
```

## Alternative: Docker Containers (Advanced)

For complete isolation, use Docker per worktree:

```yaml
# docker-compose.worktree.yml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: quantagent
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5433:5432"  # Different port per worktree
```

This requires more resources but provides complete isolation.
