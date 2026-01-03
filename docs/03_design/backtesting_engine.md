# Backtesting Engine - Technical Documentation

## Overview

The QuantAgent backtesting engine validates trading strategies by simulating execution on historical data. It combines data caching, analysis execution, trade simulation, and performance metrics calculation.

**Location**: `quantagent/backtesting/` and `quantagent/data/`

**Key Components**:
- `Backtest` - Main backtesting orchestrator
- `DataProvider` - Data caching layer (10x faster backtesting)
- `BacktestMetrics` - Performance metrics dataclass

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Backtest Engine                         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌──────────────┐
│ DataProvider  │     │ TradingGraph  │     │ OrderManager │
│ (Caching)     │     │ (Analysis)    │     │ (Execution)  │
└───────────────┘     └───────────────┘     └──────────────┘
        │                     │                     │
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌──────────────┐
│   MarketData  │     │    Signals    │     │    Trades    │
│   (Cache DB)  │     │   (Analysis)  │     │    (P&L)     │
└───────────────┘     └───────────────┘     └──────────────┘
```

### Workflow

1. **Date Range Iteration** - Loop through historical dates at specified timeframe
2. **Data Fetching** - DataProvider fetches OHLC data (from cache or API)
3. **Analysis Execution** - TradingGraph executes multi-agent analysis
4. **Trade Simulation** - OrderManager simulates order execution with slippage
5. **Portfolio Tracking** - PortfolioManager updates positions and P&L
6. **Metrics Calculation** - Calculate win rate, Sharpe, drawdown, etc.
7. **Result Persistence** - Store BacktestRun, Signals, Trades in database

---

## Components

### 1. Backtest Class

**File**: `quantagent/backtesting/backtest.py`

Main orchestrator for backtesting flow.

#### Initialization

```python
from quantagent.backtesting.backtest import Backtest

backtest = Backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    assets=['BTC', 'SPX'],
    timeframe='4h',
    initial_capital=100000.0,
    config={
        'base_position_pct': 0.05,
        'max_daily_loss_pct': 0.05,
        'max_position_pct': 0.10,
        'slippage_pct': 0.01
    },
    use_checkpointing=True
)
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `start_date` | datetime | Backtest start date | Required |
| `end_date` | datetime | Backtest end date | Required |
| `assets` | List[str] | Asset symbols to backtest | Required |
| `timeframe` | str | Analysis timeframe (1h, 4h, 1d, etc.) | Required |
| `initial_capital` | float | Starting portfolio value | Required |
| `config` | Dict | Portfolio/risk/model configuration | `{}` |
| `db_session` | Session | Database session (creates if None) | `None` |
| `use_checkpointing` | bool | Enable LangGraph checkpointing | `False` |

#### Methods

**`run(name: Optional[str] = None) -> BacktestMetrics`**

Execute backtest and return metrics.

```python
metrics = backtest.run(name="Q1 2024 Backtest")

print(f"Win Rate: {metrics.win_rate:.2%}")
print(f"Total P&L: ${metrics.total_pnl:,.2f}")
```

**`get_equity_curve() -> pd.DataFrame`**

Get equity curve as DataFrame.

```python
equity_df = backtest.get_equity_curve()
equity_df.to_csv('equity_curve.csv')
```

---

### 2. DataProvider Class

**File**: `quantagent/data/provider.py`

Cache-aside pattern for market data. Checks database first, fetches from yfinance API if missing, stores fetched data.

#### Pattern

```
Request → Check DB → Found? → Return
                  ↓ Not Found
              Fetch API → Store DB → Return
```

#### Usage

```python
from quantagent.data.provider import DataProvider
from quantagent.database import SessionLocal

provider = DataProvider(SessionLocal())

df = provider.get_ohlc(
    symbol='BTC',
    timeframe='1h',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)
```

#### Key Features

- **Gap Detection** - Identifies missing date ranges in cached data
- **Deduplication** - Avoids storing duplicate records
- **Symbol Mapping** - Maps custom symbols to yfinance format (BTC → BTC-USD)
- **Timeframe Conversion** - Converts timeframes to yfinance intervals

#### Performance

| Scenario | Time | Description |
|----------|------|-------------|
| **Cold Cache** | ~3 min | First run, fetches from API |
| **Warm Cache** | ~10 sec | Second run, reads from DB |
| **Speedup** | **18x** | Cache vs API |

#### Cache Management

```python
# Get cache statistics
stats = provider.get_cache_stats(symbol='BTC')
print(f"Total records: {stats['total_records']}")

# Clear cache
provider.clear_cache(symbol='BTC')
```

---

### 3. BacktestMetrics Dataclass

**File**: `quantagent/backtesting/backtest.py`

Performance metrics container.

#### Fields

| Field | Type | Description | Formula |
|-------|------|-------------|---------|
| `total_trades` | int | Total trades executed | Count |
| `winning_trades` | int | Trades with P&L > 0 | Count |
| `losing_trades` | int | Trades with P&L < 0 | Count |
| `win_rate` | float | Percentage of winning trades | winning_trades / total_trades |
| `profit_factor` | float | Ratio of wins to losses | sum(wins) / abs(sum(losses)) |
| `sharpe_ratio` | float | Risk-adjusted return | (return - rf) / volatility * sqrt(periods) |
| `max_drawdown` | float | Worst peak-to-trough decline | max((peak - trough) / peak) |
| `total_pnl` | float | Total profit/loss | sum(all trade P&L) |
| `avg_win` | float | Average winning trade | sum(wins) / winning_trades |
| `avg_loss` | float | Average losing trade | sum(losses) / losing_trades |
| `largest_win` | float | Largest single win | max(wins) |
| `largest_loss` | float | Largest single loss | min(losses) |
| `total_return_pct` | float | Total return percentage | (total_pnl / initial_capital) * 100 |

---

## Database Schema

### BacktestRun

Stores backtest configuration and results.

```python
class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    timeframe = Column(String(10), nullable=False)
    assets = Column(JSON, nullable=False)  # ["BTC", "SPX", ...]
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)

    # Configuration snapshot (immutable)
    config_snapshot = Column(JSON, nullable=False)

    # Results (populated after run)
    total_trades = Column(Integer)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    total_pnl = Column(Numeric(18, 8))

    created_at = Column(DateTime, default=datetime.utcnow)
```

**Config Snapshot Structure**:

```json
{
  "initial_capital": 100000.0,
  "base_position_pct": 0.05,
  "max_daily_loss_pct": 0.05,
  "max_position_pct": 0.10,
  "slippage_pct": 0.01,
  "model_provider": "openai",
  "model_name": "gpt-4o-mini",
  "temperature": 0.1,
  "use_checkpointing": true
}
```

### Signal (Modified for Backtesting)

Signals generated during backtest have `environment = 'backtest'`.

```python
class Signal(Base):
    # ... existing fields ...

    # Environment tag
    environment = Column(Enum(Environment), nullable=False, default=Environment.BACKTEST)

    # Provenance - link to order created from this signal
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=True)

    # Model metadata
    model_provider = Column(String(50))
    model_name = Column(String(100))
    temperature = Column(Float)

    # Checkpoint integration
    thread_id = Column(String(100))
    checkpoint_id = Column(String(100))
```

### Order (Modified for Backtesting)

Orders created during backtest have `environment = 'backtest'` and link to triggering signal.

```python
class Order(Base):
    # ... existing fields ...

    # Environment tag
    environment = Column(Enum(Environment), nullable=False, default=Environment.BACKTEST)

    # Provenance - link to signal that triggered this order
    trigger_signal_id = Column(Integer, ForeignKey("signals.id"), nullable=True)

    # Relationships
    trigger_signal = relationship("Signal", foreign_keys=[trigger_signal_id])
```

**Provenance Flow**:
```
Analysis → Signal (id=123, environment=backtest)
             ↓
         Order (id=456, trigger_signal_id=123, environment=backtest)
             ↓
         Trade (id=789, order_id=456, environment=backtest)
```

### Trade (Modified for Backtesting)

Trades executed during backtest have `environment = 'backtest'` and link to order.

```python
class Trade(Base):
    # ... existing fields ...

    # Environment tag
    environment = Column(Enum(Environment), nullable=False, default=Environment.BACKTEST)

    # Link to order
    order_id = Column(Integer, ForeignKey("orders.id"))
```

---

## Configuration

### Portfolio & Risk Config

```python
config = {
    # Position Sizing
    'base_position_pct': 0.05,  # 5% of portfolio per trade

    # Risk Management
    'max_daily_loss_pct': 0.05,  # 5% max daily loss (circuit breaker)
    'max_position_pct': 0.10,  # 10% max position size

    # Execution
    'slippage_pct': 0.01,  # 1% slippage simulation (±2% total)

    # Model
    'agent_llm_provider': 'openai',
    'agent_llm_model': 'gpt-4o-mini',
    'agent_llm_temperature': 0.1
}
```

### Timeframes

Supported timeframes:
- `1m`, `5m`, `15m`, `30m` - Intraday (high frequency)
- `1h`, `4h` - Medium frequency (recommended for MVP)
- `1d` - Daily
- `1w`, `1mo` - Long-term

---

## Usage Examples

### Basic Backtest

```python
from datetime import datetime, timedelta
from quantagent.backtesting.backtest import Backtest

# 90-day backtest
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

backtest = Backtest(
    start_date=start_date,
    end_date=end_date,
    assets=['BTC'],
    timeframe='4h',
    initial_capital=100000.0
)

metrics = backtest.run(name="BTC 90-Day Test")

print(f"Win Rate: {metrics.win_rate:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Total Return: {metrics.total_return_pct:.2f}%")
```

### Multi-Asset Backtest

```python
backtest = Backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    assets=['BTC', 'SPX', 'CL'],  # Multiple assets
    timeframe='1h',
    initial_capital=100000.0
)

metrics = backtest.run(name="Q1 Multi-Asset")
```

### With Custom Config

```python
custom_config = {
    'base_position_pct': 0.02,  # Conservative 2% sizing
    'max_daily_loss_pct': 0.03,  # Strict 3% daily loss limit
    'max_position_pct': 0.05,  # Max 5% per position
    'slippage_pct': 0.015,  # 1.5% slippage
    'agent_llm_provider': 'anthropic',
    'agent_llm_model': 'claude-3-haiku',
    'agent_llm_temperature': 0.05
}

backtest = Backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    assets=['BTC'],
    timeframe='4h',
    initial_capital=100000.0,
    config=custom_config
)

metrics = backtest.run(name="Conservative BTC Strategy")
```

### With Checkpointing

```python
# Enable state persistence for long backtests
backtest = Backtest(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1),  # 1 year
    assets=['BTC'],
    timeframe='1h',
    initial_capital=100000.0,
    use_checkpointing=True  # Resume if interrupted
)

metrics = backtest.run(name="1-Year BTC Backtest")
```

### Querying Provenance

```python
from quantagent.models import Signal, Order, Trade, Environment

# Get all backtest orders with their triggering signals
orders = db.query(Order).filter(
    Order.environment == Environment.BACKTEST
).all()

for order in orders:
    signal = order.trigger_signal
    print(f"Order {order.id} triggered by Signal {signal.id}")
    print(f"  Model: {signal.model_provider}/{signal.model_name}")
    print(f"  Confidence: {signal.confidence}")

# Get all signals and their resulting orders
signals = db.query(Signal).filter(
    Signal.environment == Environment.BACKTEST
).all()

for signal in signals:
    if signal.order_id:
        print(f"Signal {signal.id} → Order {signal.order_id}")
```

---

## Performance Metrics

### Viability Criteria (MVP)

Strategy is considered viable if:
- **Win Rate** ≥ 40%
- **Sharpe Ratio** ≥ 1.0
- **Max Drawdown** ≤ 15%

### Interpretation

| Metric | Good | Acceptable | Poor |
|--------|------|-----------|------|
| Win Rate | ≥ 50% | 40-50% | < 40% |
| Profit Factor | ≥ 2.0 | 1.5-2.0 | < 1.5 |
| Sharpe Ratio | ≥ 1.5 | 1.0-1.5 | < 1.0 |
| Max Drawdown | ≤ 10% | 10-15% | > 15% |

---

## Metrics Calculation Details

### Sharpe Ratio

```python
# Annualized Sharpe Ratio
returns = equity_curve.pct_change()
excess_return = returns.mean() - (risk_free_rate / periods_per_year)
sharpe = (excess_return / returns.std()) * sqrt(periods_per_year)
```

**Periods per year**:
- 1h: 252 × 6.5 = 1,638 (trading days × hours per day)
- 4h: 252 × 1.625 = 410
- 1d: 252
- 1w: 52

### Max Drawdown

```python
# Percentage-based drawdown
running_max = equity_curve.expanding().max()
drawdown = (equity_curve - running_max) / running_max
max_drawdown = abs(drawdown.min())
```

### Profit Factor

```python
# Ratio of gross profit to gross loss
total_wins = sum(pnl for pnl in trades if pnl > 0)
total_losses = abs(sum(pnl for pnl in trades if pnl < 0))
profit_factor = total_wins / total_losses
```

---

## Testing

### Unit Tests

**DataProvider**: `tests/test_data_provider.py` (20 tests)
- Structure validation
- Cache-aside pattern
- Gap detection
- Symbol/timeframe mapping
- Edge cases

**Backtest**: `tests/test_backtest.py` (30+ tests)
- Initialization
- Date range generation
- Metrics calculation (all formulas)
- Decision parsing
- Configuration snapshot

### Integration Tests

**Full Flow**: `tests/test_backtest_integration.py` (8 tests)
- End-to-end backtest execution
- Data fetching → Analysis → Execution → Metrics
- Risk manager integration
- Config snapshot reproducibility
- Equity curve tracking

### Running Tests

```bash
# All backtesting tests
pytest tests/test_data_provider.py tests/test_backtest.py -v

# Integration tests
pytest tests/test_backtest_integration.py -v
```

---

## Best Practices

### 1. Data Caching

Always use DataProvider for historical data:
- First run: ~3 minutes (fetches from API)
- Subsequent runs: ~10 seconds (reads from cache)

### 2. Timeframe Selection

Recommended timeframes by asset type:
- **Crypto** (BTC): 1h or 4h
- **Stocks** (SPX): 4h or 1d
- **Commodities** (CL): 4h

### 3. Date Ranges

Minimum backtest periods:
- **Development**: 1 month (quick validation)
- **Testing**: 3 months (strategy validation)
- **Production**: 1 year (robustness check)

### 4. Configuration Management

Use StrategyConfig model to persist profiles:

```python
from quantagent.models import StrategyConfig

# Create profile
config = StrategyConfig(
    name="conservative_crypto",
    kind="combined",
    json_config={
        'base_position_pct': 0.03,
        'max_daily_loss_pct': 0.02,
        # ...
    }
)
db.add(config)
db.commit()

# Load profile
config = db.query(StrategyConfig).filter_by(name="conservative_crypto").first()
backtest = Backtest(config=config.json_config, ...)
```

### 5. Checkpointing

Enable for long backtests:
- Survives crashes
- Resume from last checkpoint
- Full execution history

---

## Troubleshooting

### Issue: "No data available"

**Cause**: Symbol not in cache, API fetch failed

**Solution**:
```python
# Check cache status
stats = provider.get_cache_stats(symbol='BTC')
print(stats)

# Clear and re-fetch
provider.clear_cache(symbol='BTC')
backtest.run()
```

### Issue: "Backtest too slow"

**Cause**: Cold cache, many API calls

**Solution**:
```python
# Pre-populate cache
from quantagent.data.provider import DataProvider
provider = DataProvider(db)

for symbol in ['BTC', 'SPX', 'CL']:
    provider.get_ohlc(symbol, '1h', start, end)

# Run backtest (now fast)
backtest.run()
```

### Issue: "Zero trades executed"

**Cause**: Risk manager rejecting all trades

**Solution**:
```python
# Check config
print(backtest.config)

# Relax limits temporarily for testing
config['max_position_pct'] = 0.50  # 50%
config['max_daily_loss_pct'] = 0.20  # 20%
```

---

## References

- **Code**: `quantagent/backtesting/backtest.py`
- **Tests**: `tests/test_backtest.py`, `tests/test_backtest_integration.py`
- **Example**: `examples/run_backtest.py`
- **Requirements**: `docs/01_requirements/trading_system_requirements.md`
- **Data Caching**: `docs/03_technical/data_caching_architecture.md`
