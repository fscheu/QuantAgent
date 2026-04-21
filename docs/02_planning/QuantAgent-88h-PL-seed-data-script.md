# QuantAgent-88h — Planning: Seed Data Script

**Issue ID:** QuantAgent-88h  
**Title:** Create seed data script for DEV and QA databases  
**Type:** Task  
**Priority:** 2

---

## Objective

Create `scripts/seed_dev.py` to populate DEV and QA databases with reproducible, realistic test data.

---

## Tasks

### Task 1: Script Structure and CLI Setup
**Estimate:** 0.5h

**What:**
- Create `scripts/seed_dev.py` file
- Import required modules (argparse, os, sys, datetime, decimal)
- Import SQLAlchemy and models (from quantagent)
- Implement `parse_args()` function
  - `--db-url` argument (default: os.getenv("DATABASE_URL"))
  - `--reset` flag (action="store_true")
- Implement `main()` function skeleton
- Add `if __name__ == "__main__"` entry point

**Example structure:**
```python
#!/usr/bin/env python3
"""Seed QuantAgent database with test data for DEV/QA."""

import argparse
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal

def parse_args():
    # ... argparse setup
    pass

def main():
    args = parse_args()
    # ... main logic
    pass

if __name__ == "__main__":
    main()
```

**How to validate:**
```bash
python scripts/seed_dev.py --help
# Should show usage with --db-url and --reset options
```

**Dependencies:** None

---

### Task 2: Database Connection and Truncate Logic
**Estimate:** 0.75h

**What:**
- Implement `setup_database(db_url)` function
  - Create engine with `create_engine(db_url, echo=False)`
  - Create Session with `sessionmaker(bind=engine)`
  - Return session and engine
- Implement `truncate_tables(session)` function
  - Define TRUNCATE_ORDER list (8 tables in reverse FK order)
  - Loop through tables and execute `TRUNCATE TABLE {table} RESTART IDENTITY CASCADE`
  - Commit transaction
- Add error handling for connection failures
- Test connection before proceeding

**Truncate order:**
```python
TRUNCATE_ORDER = [
    "active_positions",
    "trades",
    "fills",
    "orders",
    "signals",
    "backtest_runs",
    "strategy_configs",
    "market_data",
]
```

**How to validate:**
```bash
# Test database connection
python -c "
from scripts.seed_dev import setup_database
import os
session, engine = setup_database(os.getenv('DATABASE_URL'))
print('Connected successfully')
"

# Test truncate (manually)
python -c "
from scripts.seed_dev import setup_database, truncate_tables
import os
session, engine = setup_database(os.getenv('DATABASE_URL'))
truncate_tables(session)
print('Truncated successfully')
"
```

**Dependencies:** Task 1

---

### Task 3: Master Data Generation (Strategy Configs)
**Estimate:** 0.5h

**What:**
- Implement `create_strategy_configs(session)` function
- Create 4 StrategyConfig instances:
  1. RSI Oversold/Overbought (kind="combined")
  2. MACD Crossover (kind="combined")
  3. Triple Screen (kind="combined")
  4. Default Risk Management (kind="risk")
- Each config must have:
  - Unique name
  - Valid kind enum value
  - json_config with realistic parameters
  - version = 1
- Use `session.bulk_save_objects(configs)`
- Return count of created configs

**Example config:**
```python
StrategyConfig(
    name="RSI Oversold/Overbought",
    kind="combined",
    json_config={
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "position_size_pct": 0.05,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04
    },
    version=1
)
```

**How to validate:**
```bash
# After running seed
psql quantagent -c "SELECT name, kind FROM strategy_configs"
# Should show 4 configs
```

**Dependencies:** Task 2

---

### Task 4: Market Data Download and Loading
**Estimate:** 1h

**What:**
- Import `yfinance as yf` and `pandas as pd`
- Implement `load_market_data(session, symbol, interval, period="6mo")` function
  - Call `yf.download(symbol, period=period, interval=interval, progress=False)`
  - Handle empty DataFrame (print warning, return 0)
  - Skip rows with NaN values
  - Convert DataFrame rows to MarketData instances
  - Set fields: symbol, timeframe, timestamp, open, high, low, close, volume
  - Use `session.bulk_save_objects(records)`
  - Return count of inserted records
- Call for 3 datasets:
  - BTC-USD, 4h, 6mo
  - AAPL, 1d, 6mo
  - SPY, 1d, 6mo
- Add try/except for yfinance errors

**How to validate:**
```bash
# Test download (standalone)
python -c "
import yfinance as yf
df = yf.download('BTC-USD', period='6mo', interval='4h', progress=False)
print(f'Downloaded {len(df)} rows for BTC-USD')
"

# After seed
psql quantagent -c "SELECT symbol, timeframe, COUNT(*) FROM market_data GROUP BY symbol, timeframe"
# Should show BTC-USD (4h), AAPL (1d), SPY (1d) with counts
```

**Dependencies:** Task 2

---

### Task 5: Helper Functions for Scenarios
**Estimate:** 0.5h

**What:**
- Implement `get_recent_price(session, symbol="BTC-USD", timeframe="4h")` function
  - Query latest MarketData record for symbol/timeframe
  - Return close price as float
  - Fallback to 50000.0 if no data
- Implement `get_timestamp_sequence()` utility
  - Generate logical timestamp progression for scenarios
  - Return dict with signal_time, order_time, fill_time, close_time
- Import all required enums from models:
  - TradeSignal, OrderSide, OrderStatus, OrderType, Environment, ExitPolicy

**How to validate:**
```bash
# Test helper (after market_data loaded)
python -c "
from scripts.seed_dev import setup_database, get_recent_price
import os
session, _ = setup_database(os.getenv('DATABASE_URL'))
price = get_recent_price(session, 'BTC-USD', '4h')
print(f'Recent BTC price: {price}')
"
```

**Dependencies:** Task 4

---

### Task 6: Scenario Generators (Scenarios 1-5)
**Estimate:** 2h

**What:**
Implement 5 scenario generator functions:

#### `scenario_1_winning_trade(session, entry_price)`
- Create Signal (LONG, confidence 0.85)
- Create Order (BUY, FILLED, linked to signal)
- Create Fill (matches order quantity)
- Create Trade (closed, pnl > 0, pnl_pct > 0)
- Create ActivePosition (is_active=False, close_reason="take_profit")
- All with logical timestamps

#### `scenario_2_losing_trade(session, entry_price)`
- Similar to scenario 1, but:
  - Signal: SHORT
  - Exit price < entry price (loss)
  - pnl < 0
  - close_reason="stop_loss"

#### `scenario_3_open_trade(session, entry_price)`
- Signal + Order + Fill
- Trade: opened_at set, closed_at=NULL, pnl=NULL
- ActivePosition: is_active=True, closed_at=NULL

#### `scenario_4_signal_only(session)`
- Signal: NEUTRAL, confidence 0.60
- No order, fill, or trade created

#### `scenario_5_cancelled_order(session, entry_price)`
- Signal: LONG
- Order: status=CANCELLED, filled_quantity=0
- No fill or trade

**Common patterns:**
- Use `session.add()` + `session.flush()` to get IDs for FK relationships
- Use `Decimal()` for prices and quantities
- Calculate pnl = (exit_price - entry_price) * quantity
- Use timedelta for realistic timestamp progression

**How to validate:**
```bash
# After seed, run AC-5 queries
psql quantagent -c "SELECT * FROM trades WHERE pnl > 0 LIMIT 1"  # Scenario 1
psql quantagent -c "SELECT * FROM trades WHERE pnl < 0 LIMIT 1"  # Scenario 2
psql quantagent -c "SELECT * FROM active_positions WHERE is_active = TRUE LIMIT 1"  # Scenario 3
```

**Dependencies:** Task 5

---

### Task 7: Backtest Scenario Generators (Scenarios 6-7)
**Estimate:** 1.5h

**What:**

#### `scenario_6_complete_backtest(session, entry_price)`
- Create BacktestRun with calculated metrics:
  - name, timeframe, assets, start_date, end_date
  - config_snapshot (JSON dict)
  - total_trades=12, win_rate=0.58, profit_factor=1.4
  - sharpe_ratio=1.2, max_drawdown=-0.08, total_pnl=4500.00
- Create 12 ActivePosition records:
  - 7 winners (close_reason="take_profit")
  - 5 losers (close_reason="stop_loss")
  - All linked to backtest_run_id
  - All is_active=False, closed_at set
  - environment=Environment.BACKTEST
  - Varied timestamps (spread over 30 days)

#### `scenario_7_in_progress_backtest(session, entry_price)`
- Create BacktestRun without metrics:
  - All metric fields = NULL
  - created_at recent
- Create 3 ActivePosition records:
  - 2 closed (is_active=False)
  - 1 open (is_active=True)
  - Linked to backtest_run_id

**How to validate:**
```bash
# Scenario 6
psql quantagent -c "
SELECT id, total_trades, win_rate 
FROM backtest_runs 
WHERE total_trades IS NOT NULL
"

psql quantagent -c "
SELECT COUNT(*) 
FROM active_positions 
WHERE backtest_run_id = (SELECT id FROM backtest_runs WHERE total_trades IS NOT NULL LIMIT 1)
"
# Expected: 12

# Scenario 7
psql quantagent -c "
SELECT id, total_trades 
FROM backtest_runs 
WHERE total_trades IS NULL
"
```

**Dependencies:** Task 5

---

### Task 8: Main Orchestration and Summary Reporting
**Estimate:** 0.75h

**What:**
- Implement `print_summary(counts, elapsed_time)` function
  - Print formatted table of record counts
  - Show execution time
  - Display "Database seeded successfully" message
- Update `main()` function:
  - Parse args
  - Setup database
  - If --reset: truncate tables
  - Track start time
  - Call strategy configs generator, store count
  - Call market data loaders, store count
  - Get recent price for scenarios
  - Call scenario generators 1-7, store counts
  - Commit transaction
  - Calculate elapsed time
  - Print summary
  - Handle errors (rollback, print message, exit 1)

**Summary format:**
```
========================================
Seed data generation complete!
========================================
Execution time: 45.2s
========================================
Records inserted:
  strategy_configs: 4
  market_data: 1340
  signals: 19
  orders: 17
  fills: 15
  trades: 14
  active_positions: 15
  backtest_runs: 2
========================================
Database seeded successfully.
```

**How to validate:**
```bash
python scripts/seed_dev.py --reset
# Check output matches format above
```

**Dependencies:** Tasks 3-7

---

### Task 9: Error Handling and Edge Cases
**Estimate:** 0.5h

**What:**
- Add database connection error handling
  - Catch SQLAlchemy connection errors
  - Print helpful message ("Check DATABASE_URL or --db-url")
  - Exit with code 1
- Add yfinance error handling
  - Catch download exceptions
  - Print warnings (not errors)
  - Continue with other data generation
- Add transaction rollback on errors
  - Wrap main logic in try/except
  - Rollback session on any exception
  - Re-raise with stack trace
- Validate DATABASE_URL is provided
  - Check if db_url is None
  - Print error if missing
  - Exit with code 1

**How to validate:**
```bash
# Test missing DATABASE_URL
unset DATABASE_URL
python scripts/seed_dev.py
# Expected: Error message + exit code 1

# Test invalid URL
python scripts/seed_dev.py --db-url postgresql://invalid:invalid@localhost:9999/test
# Expected: Connection error + exit code 1
```

**Dependencies:** Task 8

---

### Task 10: Testing and Documentation
**Estimate:** 1h

**What:**
- Test full execution on local DEV database
  - Run `python scripts/seed_dev.py --reset`
  - Verify all acceptance criteria (AC-1 through AC-6)
  - Run all validation SQL queries
  - Check summary output
- Test idempotency
  - Run script twice with --reset
  - Compare table counts (should be identical)
- Add docstrings to all functions
  - Module docstring at top
  - Function docstrings with Args/Returns
- Add inline comments for complex logic
  - Timestamp calculations
  - PnL calculations
  - FK relationship setups
- Update `scripts/README.md` if exists
  - Add section on seed_dev.py usage

**How to validate:**
```bash
# Full test suite
python scripts/seed_dev.py --reset
# Run all AC queries from AC document

# Idempotency test
python scripts/seed_dev.py --reset
COUNT1=$(psql quantagent -t -c "SELECT COUNT(*) FROM market_data")
python scripts/seed_dev.py --reset
COUNT2=$(psql quantagent -t -c "SELECT COUNT(*) FROM market_data")
[ "$COUNT1" == "$COUNT2" ] && echo "Idempotent ✓"

# Code quality
pylint scripts/seed_dev.py  # Optional
```

**Dependencies:** Tasks 1-9

---

## Total Estimate

**9 hours** (10 tasks)

**Breakdown:**
- Setup and infrastructure: 1.75h (Tasks 1-2)
- Data generation logic: 5.5h (Tasks 3-7)
- Integration and polish: 1.75h (Tasks 8-10)

---

## Execution Order

1. **Task 1** (CLI setup) — Foundation
2. **Task 2** (DB connection + truncate) — Infrastructure
3. **Task 3** (Strategy configs) — First data generation
4. **Task 4** (Market data) — Real data download
5. **Task 5** (Helpers) — Utilities for scenarios
6. **Task 6** (Scenarios 1-5) — Core trading scenarios
7. **Task 7** (Scenarios 6-7) — Backtest scenarios
8. **Task 8** (Orchestration + summary) — Integration
9. **Task 9** (Error handling) — Robustness
10. **Task 10** (Testing + docs) — Validation

---

## Risks & Mitigations

### Risk 1: yfinance API Changes or Downtime
**Description:** yfinance API fails or returns unexpected data

**Mitigation:**
- Graceful error handling (warnings, not failures)
- Continue with other data generation
- Provide clear warnings in output
- Fallback prices if market data unavailable

**Probability:** Low  
**Impact:** Medium (seed still works, just no market data)

---

### Risk 2: Foreign Key Constraint Violations
**Description:** Incorrect FK setup causes database errors

**Mitigation:**
- Use session.flush() to get IDs before creating dependent records
- Test FK relationships in Task 10
- Clear documentation of FK dependencies

**Probability:** Low (with careful implementation)  
**Impact:** High (breaks seed script)

---

### Risk 3: Time Zone / Timestamp Issues
**Description:** Timestamp math errors or timezone inconsistencies

**Mitigation:**
- Use datetime.utcnow() consistently
- Clear timedelta calculations
- Test timestamp logic

**Probability:** Low  
**Impact:** Low (cosmetic, doesn't break functionality)

---

### Risk 4: Decimal Precision Errors
**Description:** Float/Decimal conversion issues in prices/quantities

**Mitigation:**
- Use Decimal() for all monetary values
- Convert floats to strings before Decimal: `Decimal(str(price))`
- Test calculations

**Probability:** Low  
**Impact:** Medium (incorrect PnL calculations)

---

## Testing Strategy

### Unit-Level Testing (Per Task)
- Task 3: Verify 4 strategy configs created
- Task 4: Verify market data downloaded for 3 symbols
- Task 6: Verify each scenario creates expected records
- Task 7: Verify backtest scenarios create correct structure

### Integration Testing (Task 10)
- Run full script with --reset
- Verify all acceptance criteria
- Run all SQL validation queries
- Check summary output

### Regression Testing
- Run script multiple times (idempotency)
- Compare counts and data structure
- Ensure consistent results

---

## Rollback Plan

If issues discovered after implementation:

1. **Revert script:** Remove `scripts/seed_dev.py`
2. **Manual cleanup:** Truncate tables if needed
3. **Fix issues:** Update script based on findings
4. **Re-test:** Run validation suite again

**Rollback command:**
```bash
git rm scripts/seed_dev.py
git commit -m "Rollback: Remove seed script (needs fixes)"
```

---

## Success Criteria

- [ ] All 10 tasks completed
- [ ] Script runs without errors
- [ ] All acceptance criteria pass (AC-1 through AC-6)
- [ ] All data integrity checks pass
- [ ] Execution time < 2 minutes
- [ ] Code documented with docstrings and comments
- [ ] Idempotency verified
- [ ] Error handling tested

---

## Next Steps After Implementation

1. **Update README:** Add usage examples to `scripts/README.md`
2. **CI Integration:** Consider adding seed to CI pipeline (optional)
3. **Extend Scenarios:** Add more complex scenarios as needed
4. **Caching:** Add market data caching to speed up repeated runs
5. **Generate Fixtures:** Use seeded data to create pytest fixtures

---

## Documentation Files

**Created:**
- `docs/01_requirements/QuantAgent-88h-RQ-seed-data-script.md` ✓
- `docs/03_design/QuantAgent-88h-DS-seed-data-script.md` ✓
- `docs/05_acceptance_tests/QuantAgent-88h-AC-seed-data-script.md` ✓
- `docs/02_planning/QuantAgent-88h-PL-seed-data-script.md` (this file) ✓

**To Create:**
- `scripts/seed_dev.py` (pending implementation)

---

## Final Checklist

Before starting implementation:
- [ ] DATABASE_URL environment variable is set
- [ ] PostgreSQL database is running
- [ ] yfinance is installed (`pip install yfinance`)
- [ ] SQLAlchemy models are up to date

During implementation:
- [ ] Follow execution order (Tasks 1-10)
- [ ] Test each task before moving to next
- [ ] Commit incrementally (don't wait until end)

After implementation:
- [ ] Run full acceptance test suite
- [ ] Verify idempotency
- [ ] Update Beads status
- [ ] Add comment with results
