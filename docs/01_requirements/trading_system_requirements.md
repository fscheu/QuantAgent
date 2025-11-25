# Trading System Requirements

## Functional Requirements for Phase 1 MVP

### Overview

Transform QuantAgent from a demo analysis tool into an **automated paper trading system** that:
- Analyzes OHLC data automatically
- Executes trades in simulated environment
- Validates strategy through backtesting
- Tracks portfolio and performance metrics

**Focus**: Paper trading + backtesting validation (no real broker integration yet)

---

## Core Requirements by Tier

### 🔴 TIER 1: CRITICAL

Requirements without which the system cannot function as a trading system.

#### 1.1 Portfolio Management & Position Tracking
**What**: Track current positions (qty, entry price, current price, P&L)

**Scope**:
- Store position state (in-memory during MVP)
- Calculate unrealized P&L real-time
- Track capital allocation
- Calculate portfolio value

**MVP Deliverable**:
```python
class PortfolioManager:
    positions: Dict[symbol] → {qty, avg_cost, current_price, pnl}
    cash: float
    get_total_value() → float
    get_unrealized_pnl() → float
```

**Success Criteria**:
- ✅ Positions accurate vs. trades executed
- ✅ P&L calculations 100% correct
- ✅ Portfolio value = cash + position values

---

#### 1.2 Risk Management System (RMS)
**What**: Validate trades before execution and monitor post-execution

**Pre-Trade Checks**:
- Sufficient capital available
- Position size within limits (max 10% per trade)
- Daily loss limit not exceeded (max 5% per day)

**Post-Trade Monitoring**:
- Track daily P&L
- Circuit breaker if daily loss > limit
- Position limit enforcement

**MVP Deliverable**:
```python
class RiskManager:
    validate_trade(symbol, qty, price) → (bool, reason)
    check_circuit_breaker() → (bool, reason)
```

**Success Criteria**:
- ✅ No trades executed that violate risk rules
- ✅ Circuit breaker stops all trading if limit hit
- ✅ All rejections logged with reason

---

#### 1.3 Paper Broker & Order Execution
**What**: Execute buy/sell orders in simulated environment

**Scope**:
- Place MARKET orders only (MVP)
- Simulate realistic fills (2% slippage)
- Track order status (PENDING → FILLED)
- Return fill price and quantity

**MVP Deliverable**:
```python
class PaperBroker:
    place_order(Order) → filled_Order
    get_positions() → Dict
    get_balance() → float
```

**Success Criteria**:
- ✅ 100% order execution rate (no rejections)
- ✅ Fills within simulated slippage
- ✅ Portfolio updated on fill

---

#### 1.4 Database Persistence
**What**: Store all trades, orders, signals, analysis results

**Schema**:
- `orders` - Order details (symbol, side, qty, price, status)
- `fills` - Fill details (order_id, fill_price, fill_qty, timestamp)
- `positions` - Current positions (symbol, qty, avg_cost)
- `signals` - Analysis signals (symbol, timeframe, decision, reason, timestamp)
- `trades` - Closed trades (symbol, entry_price, exit_price, pnl)

**MVP Deliverable**:
- SQLite database with above schema
- Insert trades on execution
- Query trades/signals for backtesting

**Success Criteria**:
- ✅ All trades persisted to database
- ✅ Queries work correctly
- ✅ No data loss on restart

---

### 🟠 TIER 2: ESSENTIAL

Requirements needed for MVP to be useful.

#### 2.1 Backtesting Framework
**What**: Run analysis on historical data and measure performance

**Scope**:
- Loop through historical dates
- Execute analysis on each date (like live)
- Compare decision vs actual price 4h later
- Calculate metrics: win rate, profit factor, Sharpe ratio

**MVP Deliverable**:
```python
class Backtest:
    run(start_date, end_date, assets) → results
    results = {
        "total_trades": int,
        "win_rate": float,
        "profit_factor": float,
        "total_pnl": float,
        "max_drawdown": float
    }
```

**Success Criteria**:
- ✅ Backtest completes without errors
- ✅ Metrics calculated correctly
- ✅ Win rate ≥ 40% (viability threshold)

---

#### 2.2 Paper Trading Scheduler
**What**: Run analysis and execute trades automatically at intervals

**Scope**:
- Trigger analysis every N hours (default: 1 hour)
- Execute decision if signal present
- Log all activities

**MVP Deliverable**:
```python
class TradingScheduler:
    start() → runs analysis hourly
    stop() → stops scheduler
```

**Success Criteria**:
- ✅ Analysis runs at scheduled times
- ✅ Trades execute automatically
- ✅ System stable for 24h+ of testing

---

#### 2.3 Data Caching Layer
**What**: Cache market data locally to speed up backtesting and reduce API calls

**Scope**:
- Store OHLC data in database by symbol/timeframe
- Check DB first before API call
- Fallback to yfinance if data missing
- Store fetched data for future use

**MVP Deliverable**:
```python
class DataProvider:
    get_ohlc(symbol, timeframe, start_date, end_date) → DataFrame
    # Returns cached if available, fetches + caches if not
```

**Success Criteria**:
- ✅ Backtesting 10x faster (local DB queries)
- ✅ API calls reduced significantly
- ✅ Reproducible results (same data every run)

---

#### 2.4 Logging & Monitoring
**What**: Record all system events with sufficient detail for debugging

**Scope**:
- Log every decision (why made)
- Log every order (placed, filled, rejected)
- Log every error with stacktrace
- Searchable by time, symbol, event type

**MVP Deliverable**:
- Structured logging to files
- Rotation daily
- JSON format for parseability

**Success Criteria**:
- ✅ Can find "all BTC trades on 2024-11-25"
- ✅ Can find "all risk rejections"
- ✅ Can replay any day's activity

---

### 🟡 TIER 3: IMPORTANT (Phase 1 or Phase 2)

#### 3.1 Configuration Management
**What**: Externalize settings without code changes

**Config Options**:
- Assets to analyze (["BTC", "SPX", "CL"])
- Analysis frequency (hours between runs)
- Risk limits (max loss, position size)
- LLM provider selection

**MVP Deliverable**:
- YAML config file
- Environment variables for secrets
- Validation at startup

---

#### 3.2 Dashboard Monitoring
**What**: Web interface to see system status

**Pages**:
- Dashboard: P&L, positions, key metrics
- Backtest: Run backtest, view results
- Trades: Historical trades table
- Logs: Recent events

**MVP Deliverable**:
- Streamlit app (fast to build)
- Real-time metrics updates
- Backtesting results viewer

---

## Non-Functional Requirements

### Performance
- ✅ Analysis latency: < 30 seconds per asset
- ✅ Database queries: < 100ms
- ✅ Backtest on 3 months: < 5 minutes

### Reliability
- ✅ Uptime: > 99% during testing
- ✅ No data loss (transactions)
- ✅ Graceful error handling

### Portability
- ✅ Docker containerization (optional deployment)
- ✅ Works on different machines
- ✅ No hardcoded paths

---

## Out of Scope (Phase 1)

❌ Real broker integration (Phase 2)
❌ Real-time WebSocket feeds (Phase 2)
❌ Advanced risk models (VaR, Greeks)
❌ Multi-strategy architecture
❌ Production UI (use Streamlit MVP)
❌ Mobile app

---

## Success Criteria (MVP Phase 1)

**Analysis Engine**:
- ✅ All 4 agents working (Indicator, Pattern, Trend, Decision)
- ✅ Generates LONG/SHORT/HOLD decisions

**Paper Trading**:
- ✅ Executes orders automatically
- ✅ Portfolio tracks positions correctly
- ✅ Risk limits enforced

**Backtesting**:
- ✅ Win rate ≥ 40%
- ✅ Sharpe ratio ≥ 1.0
- ✅ Max drawdown ≤ 15%

**Operations**:
- ✅ Runs 24h+ without errors
- ✅ All trades logged to database
- ✅ Dashboard shows real-time metrics

---

## Acceptance Criteria by Component

### Portfolio Manager
```
GIVEN a portfolio with $100k initial capital
WHEN executing a BUY order for 0.1 BTC @ $42,000
THEN portfolio.positions["BTC"].qty = 0.1
AND portfolio.positions["BTC"].avg_cost = 42,000
AND portfolio.cash = 57,800
AND portfolio.get_total_value() = 100,000 (assuming price stable)
```

### Risk Manager
```
GIVEN risk limits: max_loss=5%, max_position=10%
WHEN requesting trade for 15% of capital
THEN risk_manager.validate_trade() returns (False, "Position too large")
AND trade is NOT executed
```

### Paper Broker
```
GIVEN market price = $42,000 for BTC
WHEN placing MARKET BUY order for 0.1 BTC
THEN order.filled_price is between $41,160 and $42,840 (±2% slippage)
AND order.status = "FILLED"
```

### Backtester
```
GIVEN historical data 2024-01-01 to 2024-11-25
WHEN running backtest on BTC
THEN results include:
  - total_trades: integer > 0
  - win_rate: 40-60%
  - profit_factor: > 1.0
  - max_drawdown: < 20%
```

---

## Data Model

### Core Entities

**Order**
```
id: int (PK)
order_id: str (unique)
symbol: str
side: "BUY" | "SELL"
qty: float
type: "MARKET"
price: float (nullable)
status: "PENDING" | "FILLED" | "CANCELLED"
created_at: datetime
filled_at: datetime (nullable)
```

**Fill**
```
id: int (PK)
order_id: str (FK)
filled_qty: float
filled_price: float
filled_at: datetime
```

**Position**
```
id: int (PK)
symbol: str (unique)
qty: float
avg_cost: float
current_price: float
updated_at: datetime
```

**Signal**
```
id: int (PK)
symbol: str
timeframe: str (1h, 4h, 1d)
decision: "LONG" | "SHORT" | "HOLD"
confidence: float (0-1)
reason: str (1000 char max)
created_at: datetime
```

---

## API Contracts

### Backtest API
```
POST /api/backtest
{
  "start_date": "2024-01-01",
  "end_date": "2024-11-25",
  "assets": ["BTC", "SPX"],
  "timeframe": "1h"
}

Response:
{
  "total_trades": 45,
  "win_rate": 0.42,
  "profit_factor": 1.35,
  "total_pnl": 2340.50,
  "max_drawdown": 0.12,
  "sharpe_ratio": 1.15,
  "trades": [...]
}
```

### Portfolio API
```
GET /api/portfolio

Response:
{
  "cash": 97500.50,
  "positions": {
    "BTC": {
      "qty": 0.1,
      "avg_cost": 42000,
      "current_price": 41500,
      "pnl": -50
    }
  },
  "total_value": 100000.50,
  "unrealized_pnl": 150.00,
  "daily_pnl": 500.00
}
```

---

## Open Questions

- [ ] Initial capital for paper trading? ($10k recommended)
- [ ] Assets to focus on? (BTC, SPX, Oil recommended)
- [ ] Analysis frequency? (1h recommended)
- [ ] Max position size % of capital? (10% recommended)
- [ ] Max daily loss %? (5% recommended)

