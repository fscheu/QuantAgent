# QuantAgent-88h — Acceptance Criteria: Seed Data Script

**Issue ID:** QuantAgent-88h  
**Title:** Create seed data script for DEV and QA databases  
**Type:** Task

---

## AC-1: DEV Database Execution

**Given** the DEV database is running and accessible via `DATABASE_URL` env var  
**When** executing:
```bash
python scripts/seed_dev.py --reset
```
**Then**:
- Script exits with code 0 (success)
- No exceptions or errors printed to console
- Summary report is displayed

**Verification:**
```bash
# Run script
python scripts/seed_dev.py --reset
echo "Exit code: $?"  # Should be 0

# Check output contains summary
# Should see:
# Records inserted:
#   strategy_configs: 4
#   market_data: 587
#   ...
```

---

## AC-2: QA Database Execution

**Given** a QA database is running at `postgresql://qa_user:qa_pass@localhost:5433/quantagent_qa`  
**When** executing:
```bash
python scripts/seed_dev.py --reset --db-url postgresql://qa_user:qa_pass@localhost:5433/quantagent_qa
```
**Then**:
- Script completes successfully
- QA database is populated with same data structure as DEV

**Verification:**
```bash
# Run against QA
python scripts/seed_dev.py --reset --db-url postgresql://qa_user:qa_pass@localhost:5433/quantagent_qa

# Verify data in QA
PGPASSWORD=qa_pass psql -h localhost -p 5433 -U qa_user -d quantagent_qa \
  -c "SELECT COUNT(*) FROM market_data"
# Should show 500+
```

---

## AC-3: Market Data Volume

**Given** script has completed successfully  
**When** querying:
```sql
SELECT COUNT(*) FROM market_data;
```
**Then** count > 500

**Detailed verification:**
```sql
-- Overall count
SELECT COUNT(*) FROM market_data;
-- Expected: ~1300-1400 (180 days × 3 assets)

-- By symbol
SELECT symbol, timeframe, COUNT(*) 
FROM market_data 
GROUP BY symbol, timeframe;
-- Expected:
-- BTC-USD, 4h, ~1080 rows
-- AAPL, 1d, ~130 rows
-- SPY, 1d, ~130 rows

-- Date range
SELECT symbol, MIN(timestamp), MAX(timestamp) 
FROM market_data 
GROUP BY symbol;
-- Expected: ~6 months of data
```

---

## AC-4: All Tables Populated

**Given** script has completed successfully  
**When** querying each table  
**Then** minimum record counts:

```sql
-- Strategy configs: at least 3
SELECT COUNT(*) FROM strategy_configs;
-- Expected: ≥ 3

-- Market data: at least 500
SELECT COUNT(*) FROM market_data;
-- Expected: ≥ 500

-- Signals: at least 7 (one per scenario)
SELECT COUNT(*) FROM signals;
-- Expected: ≥ 7

-- Orders: at least 5 (scenarios 1,2,3,5 have orders)
SELECT COUNT(*) FROM orders;
-- Expected: ≥ 5

-- Fills: at least 3 (scenarios 1,2,3 have fills)
SELECT COUNT(*) FROM fills;
-- Expected: ≥ 3

-- Trades: at least 3 (scenarios 1,2,3 have trades)
SELECT COUNT(*) FROM trades;
-- Expected: ≥ 3

-- Active positions: at least 10 (scenarios 1,2,3 + backtest positions)
SELECT COUNT(*) FROM active_positions;
-- Expected: ≥ 10

-- Backtest runs: at least 2 (scenarios 6,7)
SELECT COUNT(*) FROM backtest_runs;
-- Expected: ≥ 2
```

**Verification script:**
```bash
psql quantagent -c "
SELECT 
  'strategy_configs' AS table_name, COUNT(*) AS count FROM strategy_configs
UNION ALL SELECT 'market_data', COUNT(*) FROM market_data
UNION ALL SELECT 'signals', COUNT(*) FROM signals
UNION ALL SELECT 'orders', COUNT(*) FROM orders
UNION ALL SELECT 'fills', COUNT(*) FROM fills
UNION ALL SELECT 'trades', COUNT(*) FROM trades
UNION ALL SELECT 'active_positions', COUNT(*) FROM active_positions
UNION ALL SELECT 'backtest_runs', COUNT(*) FROM backtest_runs
ORDER BY table_name;
"
```

---

## AC-5: Scenario Validation

**Given** script has completed successfully  
**When** querying for each specific scenario  
**Then** all 7 scenarios are present and queryable

### Scenario 1: Winning Trade (Closed)
```sql
-- Trade with positive PnL, closed
SELECT id, symbol, pnl, pnl_pct, closed_at 
FROM trades 
WHERE pnl > 0 AND closed_at IS NOT NULL 
LIMIT 1;

-- Expected: 1 row with pnl > 0, closed_at populated
```

### Scenario 2: Losing Trade (Closed)
```sql
-- Trade with negative PnL, closed
SELECT id, symbol, pnl, pnl_pct, closed_at 
FROM trades 
WHERE pnl < 0 AND closed_at IS NOT NULL 
LIMIT 1;

-- Expected: 1 row with pnl < 0, closed_at populated
```

### Scenario 3: Open Trade (Active)
```sql
-- Active position (is_active = true, not closed)
SELECT id, symbol, side, entry_price, is_active, closed_at 
FROM active_positions 
WHERE is_active = TRUE 
LIMIT 1;

-- Expected: 1 row with is_active = true, closed_at IS NULL
```

### Scenario 4: Signal Without Execution
```sql
-- Signal without order
SELECT id, symbol, signal, confidence 
FROM signals 
WHERE signal = 'neutral' 
  AND id NOT IN (SELECT trigger_signal_id FROM orders WHERE trigger_signal_id IS NOT NULL)
LIMIT 1;

-- Expected: 1 row, signal = 'neutral', no associated order
```

### Scenario 5: Cancelled Order
```sql
-- Order with status = CANCELLED
SELECT id, symbol, side, status, filled_quantity 
FROM orders 
WHERE status = 'cancelled' 
LIMIT 1;

-- Expected: 1 row with status = 'cancelled', filled_quantity = 0
```

### Scenario 6: Complete Backtest Run
```sql
-- Backtest run with calculated metrics
SELECT id, name, total_trades, win_rate, sharpe_ratio, total_pnl 
FROM backtest_runs 
WHERE total_trades IS NOT NULL 
LIMIT 1;

-- Expected: 1 row with all metrics populated (not NULL)

-- Verify linked active positions
SELECT COUNT(*) 
FROM active_positions 
WHERE backtest_run_id = (
  SELECT id FROM backtest_runs WHERE total_trades IS NOT NULL LIMIT 1
);
-- Expected: ≥ 10 positions linked to this backtest
```

### Scenario 7: In-Progress Backtest
```sql
-- Backtest run without metrics (in progress)
SELECT id, name, total_trades, win_rate, total_pnl 
FROM backtest_runs 
WHERE total_trades IS NULL 
LIMIT 1;

-- Expected: 1 row with metrics = NULL

-- Should still have some positions
SELECT COUNT(*) 
FROM active_positions 
WHERE backtest_run_id = (
  SELECT id FROM backtest_runs WHERE total_trades IS NULL LIMIT 1
);
-- Expected: ≥ 2 positions
```

---

## AC-6: Summary Reporting

**Given** script has completed successfully  
**When** reviewing console output  
**Then** summary includes:
- "Records inserted:" header
- Count for each table (strategy_configs, market_data, signals, etc.)
- Execution time in seconds
- "Database seeded successfully" message

**Example expected output:**
```
Downloading market data...
  BTC-USD (4h): 1080 records
  AAPL (1d): 130 records
  SPY (1d): 130 records

Generating scenarios...
  ✓ Scenario 1: Winning trade
  ✓ Scenario 2: Losing trade
  ✓ Scenario 3: Open trade
  ✓ Scenario 4: Signal only
  ✓ Scenario 5: Cancelled order
  ✓ Scenario 6: Complete backtest
  ✓ Scenario 7: In-progress backtest

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

---

## Edge Cases

### EC-1: No DATABASE_URL and No --db-url
**Given** `DATABASE_URL` env var is not set  
**And** `--db-url` is not provided  
**When** running script  
**Then**:
- Script prints error: "Error: No database URL provided"
- Script exits with code 1
- No database operations attempted

**Verification:**
```bash
unset DATABASE_URL
python scripts/seed_dev.py
# Expected: Error message + exit code 1
```

---

### EC-2: Database Connection Failure
**Given** database URL points to unreachable server  
**When** running script  
**Then**:
- Script prints connection error
- Script exits with code 1
- No partial data is inserted

**Verification:**
```bash
python scripts/seed_dev.py --db-url postgresql://invalid:invalid@localhost:9999/doesnotexist
# Expected: Connection error + exit code 1
```

---

### EC-3: yfinance Download Failure
**Given** internet connection is unavailable or yfinance API fails  
**When** script attempts to download market data  
**Then**:
- Script prints warning for failed downloads
- Script continues with other data generation
- Summary shows 0 market_data records (with warning)

**Verification:**
```bash
# Simulate failure by using invalid ticker
# (Modify code temporarily to use "INVALIDTICKER")
python scripts/seed_dev.py --reset
# Expected: Warning message, continues with scenarios
```

---

### EC-4: Idempotency Test
**Given** database has been seeded once  
**When** running script again with `--reset`  
**Then**:
- Tables are truncated
- Fresh data is inserted
- Final state is identical to first run

**Verification:**
```bash
# First run
python scripts/seed_dev.py --reset
COUNT1=$(psql quantagent -t -c "SELECT COUNT(*) FROM market_data")

# Second run
python scripts/seed_dev.py --reset
COUNT2=$(psql quantagent -t -c "SELECT COUNT(*) FROM market_data")

# Should be equal
[ "$COUNT1" == "$COUNT2" ] && echo "Idempotent ✓" || echo "Not idempotent ✗"
```

---

### EC-5: Run Without --reset (Append Mode)
**Given** database already has data  
**When** running script without `--reset`  
**Then**:
- Existing data is preserved
- New data is appended
- May create duplicate records (warning shown)

**Verification:**
```bash
python scripts/seed_dev.py --reset
COUNT1=$(psql quantagent -t -c "SELECT COUNT(*) FROM signals")

python scripts/seed_dev.py  # No --reset
COUNT2=$(psql quantagent -t -c "SELECT COUNT(*) FROM signals")

# COUNT2 should be > COUNT1 (approximately double)
```

---

## Data Integrity Checks

### DIC-1: Foreign Key Integrity
**Verify:** All foreign key relationships are valid

```sql
-- Orders reference valid signals
SELECT COUNT(*) FROM orders 
WHERE trigger_signal_id IS NOT NULL 
  AND trigger_signal_id NOT IN (SELECT id FROM signals);
-- Expected: 0 (no orphaned references)

-- Fills reference valid orders
SELECT COUNT(*) FROM fills 
WHERE order_id NOT IN (SELECT id FROM orders);
-- Expected: 0

-- Trades reference valid orders
SELECT COUNT(*) FROM trades 
WHERE order_id IS NOT NULL 
  AND order_id NOT IN (SELECT id FROM orders);
-- Expected: 0

-- Active positions reference valid entities
SELECT COUNT(*) FROM active_positions 
WHERE (trade_id IS NOT NULL AND trade_id NOT IN (SELECT id FROM trades))
   OR (signal_id IS NOT NULL AND signal_id NOT IN (SELECT id FROM signals))
   OR (backtest_run_id IS NOT NULL AND backtest_run_id NOT IN (SELECT id FROM backtest_runs));
-- Expected: 0
```

---

### DIC-2: Timestamp Logical Order
**Verify:** Timestamps follow logical progression

```sql
-- Signal before order
SELECT COUNT(*) FROM orders o
JOIN signals s ON o.trigger_signal_id = s.id
WHERE o.created_at < s.generated_at;
-- Expected: 0 (order should be after signal)

-- Order before fill
SELECT COUNT(*) FROM fills f
JOIN orders o ON f.order_id = o.id
WHERE f.filled_at < o.created_at;
-- Expected: 0 (fill should be after order)

-- Trade entry before exit
SELECT COUNT(*) FROM trades
WHERE closed_at IS NOT NULL AND closed_at < opened_at;
-- Expected: 0
```

---

### DIC-3: Status Consistency
**Verify:** Status fields are consistent with related data

```sql
-- FILLED orders should have fills
SELECT COUNT(*) FROM orders 
WHERE status = 'filled' 
  AND id NOT IN (SELECT DISTINCT order_id FROM fills);
-- Expected: 0 (all FILLED orders have fills)

-- CANCELLED orders should have no fills
SELECT COUNT(*) FROM orders 
WHERE status = 'cancelled' 
  AND id IN (SELECT order_id FROM fills);
-- Expected: 0

-- Active positions with is_active=TRUE should not have closed_at
SELECT COUNT(*) FROM active_positions 
WHERE is_active = TRUE AND closed_at IS NOT NULL;
-- Expected: 0
```

---

## Performance Criteria

### P-1: Execution Time
**Given** running on typical development machine  
**When** script executes  
**Then** total time < 2 minutes (120 seconds)

**Measurement:**
```bash
time python scripts/seed_dev.py --reset
# Expected: real time < 2m0s
```

---

### P-2: Database Load
**Given** script is running  
**When** monitoring database connections  
**Then** uses single connection (no connection leaks)

**Verification:**
```sql
-- While script is running, check active connections
SELECT COUNT(*) FROM pg_stat_activity 
WHERE datname = 'quantagent';
-- Expected: 1 connection from script
```

---

## Manual Test Procedure

### Setup
1. Ensure PostgreSQL is running
2. Ensure DATABASE_URL is set or prepare --db-url
3. Activate Python virtual environment

### Test 1: Fresh Database Seed
```bash
# Truncate all tables (manual cleanup)
psql quantagent -c "TRUNCATE TABLE active_positions CASCADE"

# Run seed script
python scripts/seed_dev.py --reset

# Verify all ACs (run all SQL queries above)
```

### Test 2: Idempotency
```bash
# Run twice
python scripts/seed_dev.py --reset
python scripts/seed_dev.py --reset

# Compare table counts (should be identical)
psql quantagent -c "SELECT COUNT(*) FROM market_data"
```

### Test 3: Custom Database URL
```bash
# Use explicit URL (if QA DB available)
python scripts/seed_dev.py --reset --db-url postgresql://qa:qapass@localhost:5433/quantagent_qa
```

### Test 4: Error Handling
```bash
# Test invalid URL
python scripts/seed_dev.py --db-url postgresql://invalid:invalid@localhost:9999/test
# Should fail gracefully with error message
```

---

## Definition of Done (Testing Checklist)

- [ ] AC-1: DEV database execution successful
- [ ] AC-2: QA database execution successful (if QA available)
- [ ] AC-3: Market data count > 500
- [ ] AC-4: All tables have minimum records
- [ ] AC-5: All 7 scenarios present and queryable
- [ ] AC-6: Summary reporting displays correctly
- [ ] EC-1: No DATABASE_URL handling works
- [ ] EC-2: Connection failure handling works
- [ ] EC-3: yfinance failure handling works
- [ ] EC-4: Idempotency verified
- [ ] EC-5: Append mode works (without --reset)
- [ ] DIC-1: Foreign key integrity verified
- [ ] DIC-2: Timestamp logic verified
- [ ] DIC-3: Status consistency verified
- [ ] P-1: Execution time < 2 minutes
- [ ] P-2: No connection leaks
- [ ] All manual tests passed
