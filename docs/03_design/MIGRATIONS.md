# Database Migrations with Alembic

This document describes how to manage database migrations for the QuantAgent trading system using Alembic.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- SQLAlchemy >= 2.0.0
- Alembic >= 1.12.0
- psycopg2-binary >= 2.9.0 (PostgreSQL driver)

### 2. Configure PostgreSQL Connection

**⚠️ IMPORTANT:** Never hardcode credentials in `alembic.ini`. Always use environment variables.

#### ⭐ Option A: Docker (RECOMMENDED for Windows)

If you don't have PostgreSQL installed locally and don't want to install it:

```bash
# 1. Start PostgreSQL in Docker (database created automatically)
docker-compose up -d

# 2. Wait for container to be healthy
docker-compose ps

# 3. Configure with setup script
python setup_postgres.py
# Select: yes (use docker-compose PostgreSQL)
```

This will:
- Start a PostgreSQL 15 container with database `quantagent_dev` pre-created
- Save connection string to `.env` file
- No local PostgreSQL installation needed!

**Why Docker?**
- No installation required on Windows
- Isolated environment
- Easy to start/stop
- database created automatically

#### Option B: Interactive Setup (Manual PostgreSQL)

```bash
python setup_postgres.py
# Select: no (use manual PostgreSQL)
```

Then:
- Enter your PostgreSQL host, port, user, password, database name
- Saves connection string to `.env` file
- Requires PostgreSQL to be installed and running locally

#### Option C: Manual .env Configuration

Create or update `.env` file in the project root:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/quantagent_dev
```

Then load the environment variables before running migrations.

#### Option D: Set Environment Variable Directly

```bash
# Linux/Mac
export DATABASE_URL="postgresql://user:password@localhost:5432/quantagent_dev"

# Windows PowerShell
$env:DATABASE_URL = "postgresql://user:password@localhost:5432/quantagent_dev"

# Windows Command Prompt
set DATABASE_URL=postgresql://user:password@localhost:5432/quantagent_dev
```

**How it works:**
- `alembic/env.py` automatically reads the `DATABASE_URL` environment variable
- If `DATABASE_URL` is set, it takes priority over `alembic.ini`
- This keeps credentials out of version control

### 3. Verify PostgreSQL is Ready

#### For Docker:

```bash
# Check if container is healthy
docker-compose ps
# Status should show "healthy" for db service

# View logs if there are issues
docker-compose logs db

# Connect to the database
docker-compose exec db psql -U postgres -d quantagent_dev
# You'll see: quantagent_dev=#
# Exit with: \q
```

#### For Local PostgreSQL:

```bash
# Verify database exists
psql -U postgres -l | grep quantagent_dev

# Or create it if needed
createdb -U postgres quantagent_dev
```

### 4. Load Environment Variables

Before running migrations, ensure the environment variable is set:

**Windows PowerShell:**
```powershell
# If using docker-compose (recommended):
$env:DATABASE_URL = "postgresql://postgres:password@localhost:5432/quantagent_dev"

# Or load from .env:
Get-Content .env | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
  }
}
```

**Linux/Mac:**
```bash
export $(cat .env | xargs)
```

### 5. Run Migrations

Apply all pending migrations to your database:

```bash
python -m alembic upgrade head
```

This will:
- Connect to the database configured via `DATABASE_URL` (PostgreSQL recommended). If you configured SQLite, it will create the file `quantagent.db`.
- Create all required tables: `orders`, `fills`, `positions`, `signals`, `trades`, `market_data`
- Create all indexes for optimal query performance

### 3. Verify Database Setup

Check the current migration revision:

```bash
python -m alembic current
```

View migration history:

```bash
python -m alembic history
```


## Creating New Migrations

### Automatic Migration Generation

When you modify the models, generate a migration automatically:

```bash
python -m alembic revision --autogenerate -m "Your migration description"
```

This will:
1. Detect differences between models and current database schema
2. Generate a migration script in `alembic/versions/`
3. Create an `upgrade()` function for applying changes
4. Create a `downgrade()` function for reverting changes

**Example:**
```bash
python -m alembic revision --autogenerate -m "Add risk_level column to trades"
```

### Manual Migration Creation

For complex migrations, create a blank migration script:

```bash
python -m alembic revision -m "Custom migration description"
```

Then manually edit the generated file in `alembic/versions/` to implement your changes.

## Applying and Reverting Migrations

### Apply All Pending Migrations

```bash
python -m alembic upgrade head
```

### Apply Specific Number of Migrations

```bash
python -m alembic upgrade +2  # Apply next 2 migrations
```

### Downgrade to Previous Migration

```bash
python -m alembic downgrade -1  # Revert last migration
```

### Downgrade All Migrations

```bash
python -m alembic downgrade base  # Remove all migrations
```

### Go to Specific Revision

```bash
python -m alembic upgrade <revision_id>
```

## Using the Migration Helper

Planned convenience helper (coming soon):

```bash
# NOTE: The following helper module is planned but not yet available in the repo.
# Use Alembic commands directly for now (see sections above).

# Intended usage once implemented:
python -m quantagent.migrations_helper upgrade
python -m quantagent.migrations_helper create "Your migration description"
python -m quantagent.migrations_helper current
python -m quantagent.migrations_helper history
python -m quantagent.migrations_helper downgrade
```

## Testing Migrations

### Run the Migration Test Suite

```bash
# Option A: plain Python
python tests/test_migrations.py

# Option B: pytest (recommended)
pytest -q tests/test_migrations.py
```

This will:
1. Create all database tables
2. Insert sample data into each table
3. Verify data insertion and retrieval
4. Display counts of inserted records
5. Clean up test data

### Expected Output

```
==================================================
ALEMBIC MIGRATION TEST
==================================================
Creating all tables...
✓ Tables created successfully

Inserting market data...
✓ Market data inserted: BTC 1h candle
  ID: 1, Close: 45200.00
...
==================================================
✓ ALL TESTS PASSED
==================================================
```

## Environment Variables

Configure the database connection via environment variables:

```bash
# PostgreSQL (Recommended for production)
export DATABASE_URL="postgresql://user:password@localhost:5432/quantagent"

# PostgreSQL with different host/port
export DATABASE_URL="postgresql://user:password@db.example.com:5432/quantagent"

# MySQL (alternative)
export DATABASE_URL="mysql+pymysql://user:password@localhost:3306/quantagent"

# SQLite (development only)
export DATABASE_URL="sqlite:///./quantagent.db"
```

The default is PostgreSQL. Use the `setup_postgres.py` script for interactive configuration.

## Index Strategy

The schema includes strategic indexes for optimal query performance:

### Orders Table
- `idx_symbol_created_at`: Fast lookup by symbol and timestamp range
- `idx_status_symbol`: Fast filtering by order status

### Fills Table
- `idx_order_filled_at`: Fast lookup of fills by order and time

### Signals Table
- `idx_symbol_generated_at`: Fast lookup of signals by symbol and timestamp
- `idx_symbol_signal`: Fast filtering by signal type per symbol

### Trades Table
- `idx_symbol_opened_at`: Fast lookup of trades by symbol and opening time
- `idx_symbol_closed_at`: Fast lookup of closed trades

### MarketData Table
- `idx_symbol_timeframe_timestamp`: Fast lookup of OHLCV candles
- `idx_timestamp`: Fast range queries on timestamps

## Best Practices

1. **Always use migrations** - Never manually modify the database schema
2. **Test migrations** - Run `pytest -q tests/test_migrations.py` (or `python tests/test_migrations.py`) after creating new migrations
3. **Version control** - Commit migration files to git
4. **Descriptive messages** - Use clear, descriptive migration messages
5. **Atomic changes** - Keep migrations focused on a single logical change
6. **Review changes** - Review generated migrations before applying them
7. **Backup first** - Back up your database before applying migrations to production

## Troubleshooting

### "Target metadata has changed" error

This means your models don't match the database schema. Run:

```bash
python -m alembic revision --autogenerate -m "Fix schema mismatch"
python -m alembic upgrade head
```

### Database locked (SQLite)

If you get "database is locked" errors, ensure:
1. No other processes are accessing the database
2. For development, use WAL mode: `sqlite:///./quantagent.db?journal_mode=WAL`

### Cannot import models

Ensure the project is installed:
```bash
pip install -e .
```

## File Structure

```
quantagent/
├── database.py              # SQLAlchemy configuration
├── models.py                # ORM models (Order, Fill, etc.)
└── (migrations_helper.py)   # Planned CLI helper for migrations

alembic/
├── versions/
│   └── <revision>_*.py      # Migration scripts
├── env.py                   # Alembic environment configuration
├── script.py.mako           # Migration template
└── README

alembic.ini                  # Alembic configuration file
tests/test_migrations.py     # Migration test script
```

## References

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy ORM Tutorial](https://docs.sqlalchemy.org/en/20/orm/)
- [QuantAgent README](../README.md)
