# QuantAgent-88h — Design: Seed Data Script for DEV/QA

**Issue ID:** QuantAgent-88h  
**Title:** Create seed data script for DEV and QA databases  
**Type:** Task

---

## Design Overview

Create `scripts/seed_dev.py` — a Python script that populates DEV and QA databases with reproducible, realistic test data across all business tables. The script uses yfinance for market data and generates synthetic but realistic trading scenarios.

---

## Architecture

### Script Structure

```
scripts/seed_dev.py
├─ CLI argument parsing (argparse)
├─ Database connection setup
├─ Truncate functions (--reset mode)
├─ Master data generation
│  └─ create_strategy_configs()
├─ Market data download
│  └─ load_market_data(symbol, interval, period)
├─ Scenario generators (7 functions)
│  ├─ scenario_1_winning_trade()
│  ├─ scenario_2_losing_trade()
│  ├─ scenario_3_open_trade()
│  ├─ scenario_4_signal_only()
│  ├─ scenario_5_cancelled_order()
│  ├─ scenario_6_complete_backtest()
│  └─ scenario_7_in_progress_backtest()
├─ Summary reporting
└─ Main execution orchestration
```

---

## Technical Approach

### 1. CLI Argument Parsing

```python
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Seed QuantAgent database with test data"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("DATABASE_URL"),
        help="Database connection URL (default: DATABASE_URL env var)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate tables before seeding (idempotent mode)"
    )
    return parser.parse_args()
```

---

### 2. Database Setup

```python
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from quantagent.models import Base

def setup_database(db_url):
    engine = create_engine(db_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session, engine
```

---

### 3. Truncate Logic (--reset)

**Truncation order (reverse FK dependencies):**
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

def truncate_tables(session):
    """Truncate all tables in FK-safe order."""
    for table in TRUNCATE_ORDER:
        session.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
    session.commit()
```

**Why this order:**
- `active_positions` references `trades`, `signals`, `backtest_runs`
- `trades` references `orders`
- `fills` references `orders`
- `orders` references `signals`
- No circular dependencies if we follow this order

**Alternative (simpler but PostgreSQL-specific):**
```python
session.execute(text("TRUNCATE TABLE active_positions CASCADE"))
# CASCADE handles all dependencies automatically
```

---

### 4. Strategy Configs Generation

**Example configurations:**

```python
def create_strategy_configs(session):
    configs = [
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
        ),
        StrategyConfig(
            name="MACD Crossover",
            kind="combined",
            json_config={
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "position_size_pct": 0.05,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.05
            },
            version=1
        ),
        StrategyConfig(
            name="Triple Screen",
            kind="combined",
            json_config={
                "timeframes": ["1d", "4h", "1h"],
                "trend_confirm": True,
                "position_size_pct": 0.1,
                "risk_per_trade": 0.02
            },
            version=1
        ),
        StrategyConfig(
            name="Default Risk Management",
            kind="risk",
            json_config={
                "max_daily_loss_pct": 0.05,
                "max_position_pct": 0.15,
                "base_position_pct": 0.05
            },
            version=1
        ),
    ]
    session.bulk_save_objects(configs)
    session.commit()
    return len(configs)
```

---

### 5. Market Data Download

**Implementation:**

```python
import yfinance as yf
from datetime import datetime
from quantagent.models import MarketData

def load_market_data(session, symbol, interval, period="6mo"):
    """Download and load market data for a symbol."""
    
    # Download data
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    
    if df.empty:
        print(f"Warning: No data for {symbol} at {interval}")
        return 0
    
    # Convert to MarketData records
    records = []
    for timestamp, row in df.iterrows():
        if pd.isna(row['Close']):  # Skip rows with missing data
            continue
        
        records.append(MarketData(
            symbol=symbol,
            timeframe=interval,
            timestamp=timestamp.to_pydatetime(),
            open=float(row['Open']),
            high=float(row['High']),
            low=float(row['Low']),
            close=float(row['Close']),
            volume=float(row['Volume']),
        ))
    
    # Bulk insert
    session.bulk_save_objects(records)
    session.commit()
    
    return len(records)

# Usage
count = 0
count += load_market_data(session, "BTC-USD", "4h", "6mo")
count += load_market_data(session, "AAPL", "1d", "6mo")
count += load_market_data(session, "SPY", "1d", "6mo")
```

**Data volume estimate:**
- BTC 4h, 180 days: ~1080 bars (180 days × 6 bars/day)
- AAPL 1d, 180 days: ~130 bars (weekdays only)
- SPY 1d, 180 days: ~130 bars
- **Total: ~1340 bars** (exceeds 500 requirement)

---

### 6. Scenario Generation Functions

#### Helper: Get Recent Market Data

```python
def get_recent_price(session, symbol="BTC-USD", timeframe="4h"):
    """Get a recent close price for realistic trade data."""
    record = session.query(MarketData).filter_by(
        symbol=symbol, timeframe=timeframe
    ).order_by(MarketData.timestamp.desc()).first()
    
    if record:
        return float(record.close)
    return 50000.0  # Fallback if no market data
```

#### Scenario 1: Winning Trade

```python
from decimal import Decimal
from datetime import datetime, timedelta

def scenario_1_winning_trade(session):
    """Create a complete winning trade chain."""
    
    entry_price = get_recent_price(session)
    exit_price = entry_price * 1.05  # 5% profit
    quantity = Decimal("0.1")
    
    # Signal
    signal = Signal(
        symbol="BTC-USD",
        signal=TradeSignal.LONG,
        confidence=0.85,
        timeframe="4h",
        environment=Environment.PAPER,
        generated_at=datetime.utcnow() - timedelta(hours=4)
    )
    session.add(signal)
    session.flush()  # Get signal.id
    
    # Order
    order = Order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=quantity,
        status=OrderStatus.FILLED,
        filled_quantity=quantity,
        average_fill_price=Decimal(str(entry_price)),
        environment=Environment.PAPER,
        trigger_signal_id=signal.id,
        created_at=signal.generated_at + timedelta(minutes=5),
        filled_at=signal.generated_at + timedelta(minutes=6)
    )
    session.add(order)
    session.flush()
    
    # Fill
    fill = Fill(
        order_id=order.id,
        quantity=quantity,
        price=Decimal(str(entry_price)),
        commission=Decimal("0.001"),
        filled_at=order.filled_at
    )
    session.add(fill)
    
    # Trade
    pnl = (Decimal(str(exit_price)) - Decimal(str(entry_price))) * quantity
    pnl_pct = float((pnl / (Decimal(str(entry_price)) * quantity)) * 100)
    
    trade = Trade(
        symbol="BTC-USD",
        order_id=order.id,
        entry_price=Decimal(str(entry_price)),
        exit_price=Decimal(str(exit_price)),
        quantity=quantity,
        side=OrderSide.BUY,
        pnl=pnl,
        pnl_pct=pnl_pct,
        environment=Environment.PAPER,
        opened_at=order.filled_at,
        closed_at=order.filled_at + timedelta(hours=8)
    )
    session.add(trade)
    session.flush()
    
    # ActivePosition (closed)
    active_pos = ActivePosition(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        entry_price=Decimal(str(entry_price)),
        stop_loss=Decimal(str(entry_price * 0.98)),
        take_profit=Decimal(str(exit_price)),
        quantity=quantity,
        decision_timestamp=signal.generated_at,
        exit_policy=ExitPolicy.SL_TP_ONLY,
        prediction_horizon=3,
        candles_direction=[1, 1, 1],
        trade_id=trade.id,
        signal_id=signal.id,
        is_active=False,
        closed_at=trade.closed_at,
        close_reason="take_profit",
        accuracy=1.0,
        environment=Environment.PAPER
    )
    session.add(active_pos)
    session.commit()
```

**Similar patterns for scenarios 2-7** (omitted for brevity, but follow same structure with different parameters)

---

### 7. Backtest Scenario Generation

**Scenario 6: Complete Backtest**

```python
def scenario_6_complete_backtest(session):
    """Generate a backtest run with 10+ trades and calculated metrics."""
    
    # Create backtest run
    backtest = BacktestRun(
        name="BTC 4H Strategy Backtest",
        timeframe="4h",
        assets=["BTC-USD"],
        start_date=datetime.utcnow() - timedelta(days=30),
        end_date=datetime.utcnow() - timedelta(days=1),
        config_snapshot={
            "strategy": "RSI Oversold/Overbought",
            "initial_capital": 100000,
            "position_size_pct": 0.05
        },
        total_trades=12,
        win_rate=0.58,  # 58% win rate
        profit_factor=1.4,
        sharpe_ratio=1.2,
        max_drawdown=-0.08,  # -8% max drawdown
        total_pnl=Decimal("4500.00")
    )
    session.add(backtest)
    session.flush()
    
    # Create 12 active_positions (7 winners, 5 losers)
    entry_price = get_recent_price(session)
    
    for i in range(12):
        is_winner = i < 7
        exit_mult = 1.03 if is_winner else 0.97
        
        # Create signal, order, trade chain...
        # (similar to scenario 1/2, but linked to backtest_run_id)
        
        active_pos = ActivePosition(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            entry_price=Decimal(str(entry_price)),
            stop_loss=Decimal(str(entry_price * 0.98)),
            take_profit=Decimal(str(entry_price * 1.03)),
            quantity=Decimal("0.05"),
            decision_timestamp=datetime.utcnow() - timedelta(days=30-i),
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=[1, 1, 1] if is_winner else [-1, -1, -1],
            backtest_run_id=backtest.id,
            is_active=False,
            closed_at=datetime.utcnow() - timedelta(days=29-i),
            close_reason="take_profit" if is_winner else "stop_loss",
            accuracy=1.0 if is_winner else 0.0,
            environment=Environment.BACKTEST
        )
        session.add(active_pos)
    
    session.commit()
```

---

## Design Decisions

### Decision 1: yfinance vs Static CSV

**Chosen:** yfinance (real-time download)

**Rationale:**
- Always fresh data (recent 6 months)
- No need to commit large CSV files to repo
- Matches production data sources

**Trade-off:**
- Requires internet connection
- Download takes ~10-30s
- yfinance API can occasionally fail

**Mitigation:**
- Graceful error handling
- Fallback message if download fails
- Cache could be added later if needed

---

### Decision 2: Bulk Insert vs Row-by-Row

**Chosen:** Bulk insert with `session.bulk_save_objects()`

**Rationale:**
- Much faster for large datasets (market_data ~1000+ rows)
- Single transaction
- Cleaner code

**Trade-off:**
- Doesn't auto-populate IDs (need `session.flush()` for FK relationships)

**Implementation:**
- Use bulk for market_data (no FKs)
- Use `add()` + `flush()` for transactional scenarios (need IDs for FKs)

---

### Decision 3: Environment Field Values

**Chosen:** `"dev"` for paper trading scenarios, `"backtest"` for backtest scenarios

**Rationale:**
- Matches real usage patterns
- Allows filtering by environment in queries
- Scenarios 1-5 use `Environment.PAPER` ("paper" → "dev" in issue description means paper trading mode)
- Scenarios 6-7 use `Environment.BACKTEST`

**Note:** Issue says "dev", but model enum has "paper"/"backtest"/"prod". Using "paper" for development trading scenarios.

---

### Decision 4: Timestamp Generation

**Chosen:** Use `datetime.utcnow()` with relative offsets

**Rationale:**
- Always generates recent data
- Logical time progression (signal → order → fill → trade close)
- Avoids hardcoding dates that become stale

**Example:**
```python
signal_time = datetime.utcnow() - timedelta(hours=4)
order_time = signal_time + timedelta(minutes=5)
fill_time = order_time + timedelta(minutes=1)
close_time = fill_time + timedelta(hours=8)
```

---

## Alternative Approaches Considered

### ❌ Approach 1: Single Monolithic Function

**Why rejected:**
- Hard to maintain
- Difficult to test individual scenarios
- Poor code organization

**Better:** Separate function per scenario (modular, testable)

---

### ❌ Approach 2: Load from JSON/CSV Files

**Why rejected:**
- Requires maintaining static files
- Data becomes stale
- More complexity (parsing, validation)

**Better:** Generate data programmatically (always fresh, self-contained)

---

### ❌ Approach 3: Use Faker for Synthetic Data

**Why rejected:**
- Faker doesn't generate realistic financial data
- Market data must be real (yfinance)
- Trading scenarios require specific structures

**Better:** Custom generators with realistic values

---

## Error Handling Strategy

### Database Connection Errors
```python
try:
    session, engine = setup_database(db_url)
except Exception as e:
    print(f"Error connecting to database: {e}")
    print("Please check DATABASE_URL or --db-url parameter")
    sys.exit(1)
```

### yfinance Download Failures
```python
try:
    df = yf.download(symbol, ...)
    if df.empty:
        print(f"Warning: No data for {symbol}")
        return 0
except Exception as e:
    print(f"Error downloading {symbol}: {e}")
    return 0
```

### Transaction Rollback on Errors
```python
try:
    # Generate all data
    create_strategy_configs(session)
    load_market_data(...)
    # ... scenarios
    session.commit()
except Exception as e:
    session.rollback()
    print(f"Error during seed: {e}")
    raise
```

---

## Testing Strategy

### Manual Testing
```bash
# Test on local DEV database
python scripts/seed_dev.py --reset

# Verify data
psql quantagent -c "SELECT COUNT(*) FROM market_data"
psql quantagent -c "SELECT COUNT(*) FROM active_positions WHERE is_active = true"

# Test idempotency (run twice)
python scripts/seed_dev.py --reset
python scripts/seed_dev.py --reset
# Should produce identical state
```

### Validation Queries
```sql
-- Scenario 1: Winning trade
SELECT * FROM trades WHERE pnl > 0 AND closed_at IS NOT NULL LIMIT 1;

-- Scenario 3: Open trade
SELECT * FROM active_positions WHERE is_active = TRUE LIMIT 1;

-- Scenario 6: Complete backtest
SELECT * FROM backtest_runs WHERE total_trades IS NOT NULL LIMIT 1;
```

---

## Open Questions

None — design is comprehensive and ready for implementation.
