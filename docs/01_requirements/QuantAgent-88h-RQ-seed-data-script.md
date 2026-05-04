# QuantAgent-88h — Requirements: Seed Data Script for DEV/QA

**Issue ID:** QuantAgent-88h  
**Title:** Create seed data script for DEV and QA databases  
**Type:** Task  
**Priority:** 2  
**Labels:** openclaw:design_approved testing dx

---

## Objective

Create a reproducible seed data script (`scripts/seed_dev.py`) that populates DEV and QA databases with realistic test data, enabling developers and automated testing to work with meaningful datasets instead of empty databases.

---

## Background

### Current Problem
- Developers start with empty databases
- Each environment has different (or no) test data
- Difficult to reproduce bugs that require specific data states
- Manual data creation is time-consuming and error-prone
- Automated development agents struggle without realistic data

### Impact
- Slower development velocity
- Inconsistent testing environments
- Hard to validate complex workflows (backtests, position tracking, etc.)
- QA environment lacks representative data

---

## Scope

### In Scope
1. **Create `scripts/seed_dev.py`** with:
   - CLI arguments for database URL and reset flag
   - Idempotent execution with `--reset`
   - Master data generation (strategy configs)
   - Market data download and loading (yfinance)
   - Transactional scenario generation (7 specific scenarios)
   - Summary reporting

2. **Data Categories:**
   - **Masters**: `strategy_configs` (3-4 configurations)
   - **Base Data**: `market_data` (BTC/4h, AAPL/1d, SPY/1d for 180 days)
   - **Transactional**: Full workflow scenarios (signals → orders → fills → trades → active_positions)

3. **7 Required Scenarios:**
   1. Winning trade (complete chain, PnL > 0)
   2. Losing trade (complete chain, PnL < 0)
   3. Open trade (active position, is_active=True)
   4. Signal without order (NEUTRAL signal, no execution)
   5. Cancelled order (no fills)
   6. Complete backtest run (10+ trades, metrics calculated)
   7. In-progress backtest (no metrics yet)

### Out of Scope
- Modifying existing database models
- Adding new dependencies (use existing: yfinance, SQLAlchemy)
- Generating data for `logs` table (operational data, not business data)
- Creating independent `fills` records (always associated with orders)
- Performance optimization (acceptable if takes 1-2 minutes)
- Generating data for production environments

---

## Requirements

### FR-1: Command-Line Interface
**Description:** Script accepts CLI arguments for configuration

**Requirements:**
- `--db-url <URL>`: Database connection string (default: read from `DATABASE_URL` env var)
- `--reset`: Flag to truncate tables before inserting (idempotent mode)
- Exit code 0 on success, non-zero on error

**Example usage:**
```bash
# Use env var for DB URL, no reset
python scripts/seed_dev.py

# Explicit DB URL, reset existing data
python scripts/seed_dev.py --reset --db-url postgresql://user:pass@localhost:5432/quantagent_dev

# QA environment
python scripts/seed_dev.py --reset --db-url postgresql://qa:qapass@localhost:5433/quantagent_qa
```

---

### FR-2: Idempotent Execution
**Description:** Running script multiple times with `--reset` produces consistent state

**Requirements:**
- With `--reset`: truncate tables in correct order (respecting foreign keys), then insert
- Without `--reset`: append data (may create duplicates, warn user)
- Truncation order must be reverse of foreign key dependencies

**Truncation order:**
```sql
active_positions → trades → fills → orders → signals → backtest_runs → strategy_configs → market_data
```

---

### FR-3: Master Data Generation
**Description:** Create static configuration data

**Requirements:**
- **`strategy_configs`**: 3-4 strategy configurations:
  1. RSI strategy (kind="combined", RSI thresholds)
  2. MACD strategy (kind="combined", MACD parameters)
  3. Triple Screen strategy (kind="combined", multiple indicators)
  4. Default risk management (kind="risk", portfolio sizing rules)

- Each config must have:
  - `name`: Descriptive name (e.g., "RSI Oversold/Overbought")
  - `kind`: "portfolio", "risk", or "combined"
  - `json_config`: Valid JSON with strategy parameters
  - `version`: 1 (initial version)

**Validation:**
- Configs must be valid JSON
- Names must be unique
- Configs should match real strategy patterns used in code

---

### FR-4: Market Data Download
**Description:** Fetch real historical market data via yfinance

**Requirements:**
- **Assets and timeframes:**
  - BTC: 4h timeframe, last 180 days
  - AAPL: 1d timeframe, last 180 days
  - SPY: 1d timeframe, last 180 days

- **Process:**
  1. Use `yf.download(ticker, period="6mo", interval=interval)`
  2. Convert DataFrame to `MarketData` records
  3. Bulk insert into database

- **Data validation:**
  - Ensure timestamp, open, high, low, close, volume are present
  - Skip records with missing data (yfinance sometimes has gaps)
  - Result should be 500+ total records across all assets

---

### FR-5: Transactional Scenario Generation
**Description:** Create realistic trading workflow data

**Requirements:**

#### Scenario 1: Winning Trade (Closed)
- Signal: LONG, confidence 0.85
- Order: BUY, status=FILLED
- Fill: Quantity matched, price = entry price
- Trade: closed_at populated, pnl > 0, pnl_pct > 0
- ActivePosition: is_active=False, closed_at populated, close_reason="take_profit"

#### Scenario 2: Losing Trade (Closed)
- Signal: SHORT, confidence 0.75
- Order: SELL, status=FILLED
- Fill: Quantity matched
- Trade: closed_at populated, pnl < 0, pnl_pct < 0
- ActivePosition: is_active=False, closed_at populated, close_reason="stop_loss"

#### Scenario 3: Open Trade (Active)
- Signal: LONG, confidence 0.80
- Order: BUY, status=FILLED
- Fill: Quantity matched
- Trade: opened_at populated, closed_at=NULL, pnl=NULL
- ActivePosition: is_active=True, closed_at=NULL

#### Scenario 4: Signal Without Execution
- Signal: NEUTRAL, confidence 0.60
- No order created
- No fills, trades, or positions

#### Scenario 5: Cancelled Order
- Signal: LONG, confidence 0.70
- Order: BUY, status=CANCELLED, filled_quantity=0
- No fills or trades

#### Scenario 6: Complete Backtest Run
- BacktestRun: 10+ trades in active_positions
- Metrics calculated: win_rate, sharpe_ratio, max_drawdown, profit_factor, total_pnl
- ActivePositions: Mix of winning/losing trades, all closed (is_active=False)
- All positions linked to backtest_run_id

#### Scenario 7: In-Progress Backtest
- BacktestRun: created_at recent
- Metrics: all NULL (total_trades, win_rate, etc.)
- ActivePositions: 2-3 positions created, some active, some closed

**Data consistency rules:**
- All `environment` fields = "dev" (or "backtest" for backtest scenarios)
- Timestamps must be logical (signal before order before fill before trade close)
- Foreign key relationships must be valid
- Quantities and prices must be realistic (use market_data prices)

---

### FR-6: Summary Reporting
**Description:** Print summary of inserted data

**Requirements:**
- After successful execution, print:
  - Count of records inserted per table
  - Total execution time
  - Any warnings (e.g., missing market data)

**Example output:**
```
Seed data generation complete!
========================================
Execution time: 45.2s
========================================
Records inserted:
  strategy_configs: 4
  market_data: 587
  signals: 7
  orders: 6
  fills: 5
  trades: 5
  active_positions: 15
  backtest_runs: 2
========================================
Database seeded successfully.
```

---

## Acceptance Criteria

### AC-1: DEV Database Execution
**Given** the DEV database is running and accessible  
**When** executing `python scripts/seed_dev.py --reset`  
**Then**:
- Script completes without errors (exit code 0)
- All tables are populated
- Summary is printed

### AC-2: QA Database Execution
**Given** the QA database is running at custom URL  
**When** executing `python scripts/seed_dev.py --reset --db-url postgresql://qa_user:qa_pass@localhost:5433/quantagent_qa`  
**Then**:
- Script completes without errors
- QA database is populated with same data structure

### AC-3: Market Data Volume
**Given** script has completed successfully  
**When** querying `SELECT COUNT(*) FROM market_data`  
**Then** result is > 500 rows (approximately 180 days × 3 assets × variable bars)

### AC-4: All Tables Populated
**Given** script has completed successfully  
**When** querying each transactional table  
**Then** each has at least 1 record:
- `strategy_configs`: ≥ 3
- `market_data`: ≥ 500
- `signals`: ≥ 7
- `orders`: ≥ 5
- `fills`: ≥ 3
- `trades`: ≥ 3
- `active_positions`: ≥ 10
- `backtest_runs`: ≥ 2

### AC-5: Scenario Validation
**Given** script has completed successfully  
**When** querying for specific scenarios  
**Then** all 7 scenarios are present and queryable:
1. `SELECT * FROM trades WHERE pnl > 0 AND closed_at IS NOT NULL LIMIT 1` → 1 row
2. `SELECT * FROM trades WHERE pnl < 0 AND closed_at IS NOT NULL LIMIT 1` → 1 row
3. `SELECT * FROM active_positions WHERE is_active = TRUE LIMIT 1` → 1 row
4. `SELECT * FROM signals WHERE signal = 'NEUTRAL' AND order_id IS NULL LIMIT 1` → 1 row
5. `SELECT * FROM orders WHERE status = 'CANCELLED' LIMIT 1` → 1 row
6. `SELECT * FROM backtest_runs WHERE total_trades IS NOT NULL LIMIT 1` → 1 row
7. `SELECT * FROM backtest_runs WHERE total_trades IS NULL LIMIT 1` → 1 row

### AC-6: Summary Reporting
**Given** script has completed successfully  
**When** reviewing console output  
**Then** summary includes:
- Count for each table
- Execution time
- "Database seeded successfully" message

---

## Constraints

- **No model changes**: Use existing SQLAlchemy models as-is
- **No new dependencies**: Use yfinance (already in pyproject.toml)
- **Respect foreign keys**: Maintain referential integrity
- **Realistic data**: Use actual market data, realistic timestamps
- **Idempotent with --reset**: Same result every time

---

## Non-Functional Requirements

### NFR-1: Performance
- Execution time < 2 minutes on typical development machine
- Bulk inserts preferred over row-by-row

### NFR-2: Error Handling
- Clear error messages if database unreachable
- Graceful handling of yfinance API failures
- Rollback on errors (SQLAlchemy transactions)

### NFR-3: Maintainability
- Code is well-commented
- Scenario generation is modular (separate functions per scenario)
- Easy to add new scenarios in future

---

## Definition of Done

- [ ] `scripts/seed_dev.py` created and executable
- [ ] All 6 acceptance criteria pass
- [ ] Script runs without errors on DEV database
- [ ] Script runs without errors on QA database (if available)
- [ ] Code reviewed for best practices
- [ ] Documentation updated (this file + inline comments)
