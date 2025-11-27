# Phase 1 Roadmap: MVP Paper Trading System

**Duration**: 8-10 weeks
**Goal**: Fully automated paper trading system with validated strategy performance
**Key Focus**: Backtesting + Paper Trading validation (not real broker integration)

---

## Timeline Overview

```
Week 1-2:   Database + Core Infrastructure (+ Docker setup)
Week 3-4:   Portfolio + Risk Management
Week 5-6:   Paper Broker + Order Execution
Week 7-8:   Backtesting Engine + Data Caching
Week 9-10:  Streamlit Dashboard + Integration Testing
```

**Total MVP**: 8-10 weeks (vs 12 in original plan)

---

## Phase 1 Architecture (What We're Building)

```
Data Layer
  ├─ yfinance (live)
  └─ Local DB cache (historical)
          ↓
Analysis Engine (REFACTORED QuantAgent)
  ├─ Indicator Subgraph (structured output)
  ├─ Pattern Subgraph (structured output)
  ├─ Trend Subgraph (structured output)
  └─ Decision Agent (consumes structured data)
          ↓
Backtester (NEW)
  ├─ Loop through dates
  ├─ Execute analysis
  ├─ Measure performance
  └─ Validate strategy
          ↓
Paper Trader (NEW)
  ├─ Schedule hourly
  ├─ Execute decisions
  ├─ Track portfolio
  └─ Monitor risk
          ↓
Database (NEW)
  ├─ Persist trades
  ├─ Store signals
  ├─ Checkpoints (LangGraph state history)
  └─ Audit trail
          ↓
Dashboard (Streamlit)
  ├─ Live metrics
  ├─ Backtest results
  └─ Trade history
```

**LangGraph Improvements Integrated**:
- ✅ Agents use `create_agent` pattern (built-in reliability)
- ✅ Structured outputs (Pydantic models) instead of strings
- ✅ Agents as subgraphs (better separation of concerns)
- ✅ Proper ToolNode usage (cleaner tool execution)
- ✅ Checkpointing (state persistence for long backtests)

---

## Week-by-Week Breakdown

### Week 1-2: Foundation

#### Tasks

**1.1 Project Setup**
- [x] Create subdirectories: `quantagent/{portfolio,risk,trading,backtesting}`
- [x] Create `requirements-dev.txt` with new dependencies
- [x] Setup pytest framework with conftest.py
- [ ] Create GitHub Actions CI/CD workflow (runs tests on push)
- [x] Create Dockerfile + docker-compose.yml (lightweight)

**Tools/Dependencies to Add**:
```
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.12.0
pytest==7.4.3
APScheduler==3.10.4
streamlit==1.28.0
```

**1.2 Database Setup**
- [x] Create PostgreSQL database locally
- [x] Setup Alembic for migrations
- [x] Define SQLAlchemy models (Order, Fill, Position, Signal, Trade, MarketData)
- [x] Create initial migration script
- [x] Test migrations work
 - [x] Add environment tagging to operational tables (orders, trades, signals/analyses, positions)
 - [x] Add provenance links: `orders.trigger_signal_id` and `signals.order_id`
 - [x] Add analysis metadata in signals: `thread_id`, `checkpoint_id`, `state_snapshot` (fallback), `model_provider`, `model_name`, `temperature`, `agent_version`, `graph_version`
 - [ ] Persist StrategyConfig (portfolio/risk/combined) and validate Portfolio profile supports Universe list (fixed instruments)

**1.3 Docker Setup**
- [x] Create `Dockerfile` (Python 3.11 + dependencies)
- [x] Create `docker-compose.yml` (PostgreSQL + app)
- [x] Document: "For dev, run `docker-compose up -d db`"
- [ ] Test on clean machine (colleague runs your docker-compose)

**1.4 Testing Infrastructure**
- [x] Create `tests/conftest.py` with fixtures
- [x] Write first unit test (models)
- [x] Setup pytest configuration
- [ ] Configure GitHub Actions to run tests

**1.5 LangGraph Improvements (Reliability & Architecture)**

*Improvement #1: Refactor agents to use `create_agent` pattern*
- [x] Analyze current `indicator_agent.py`, `pattern_agent.py`, `trend_agent.py`
- [x] Replace manual `chain = prompt | llm.bind_tools()` with `create_agent()`
- [x] Remove manual tool call loops (handled by `create_agent`)
- [x] Test: Each agent produces same output as before
- [x] Benefit: Built-in reliability, checkpointing support, less code

*Improvement #5: Add LangGraph checkpointing for state persistence*
- [x] Setup `AsyncPostgresSaver` (uses same PostgreSQL DB)
- [x] Add checkpointer to graph compilation: `graph.compile(checkpointer=checkpointer)`
- [x] Implement `thread_id` based state resumption for backtesting
- [x] Test: Stop backtest mid-way, resume from checkpoint
- [x] Benefit: Long backtests survive crashes, full execution history


**Deliverables**:
- ✅ Database running
- ✅ SQLAlchemy models defined
- ✅ First pytest passes
- ✅ GitHub Actions green on push
- ✅ Dockerfile working
- ✅ teammates can run `docker-compose up -d db`
- ✅ `create_agent` refactoring complete
- ✅ Checkpointing integrated with DB
- ✅ Agents more reliable with built-in features
 - ✅ StrategyConfig model persisted; Portfolio profile includes Universe list (fixed instruments)
 - ✅ Environment tagging and basic provenance fields present in schema

---

### Week 3-4: Core State Management

#### Tasks

**2.1 Position Sizer (NEW)**
```python
class PositionSizer:
    def __init__(self, base_position_pct: float = 0.05):
        # base_position_pct: percentage of portfolio per trade (5% default)

    def calculate_size(
        symbol: str,
        signal_confidence: float,  # 0-1 (0.5=low, 1.0=high)
        current_price: float,
        portfolio_value: float
    ) → float:
        # Returns: qty to buy/sell
        # Logic: position_value = portfolio_value * base_position_pct * signal_confidence
        #        qty = position_value / current_price
```

**Implementation Details**:
- [x] Create `quantagent/trading/position_sizer.py`
- [x] Implement confidence-adjusted sizing formula
- [x] Add unit tests:
  - [x] 50% confidence → 2.5% position size
  - [x] 100% confidence → 5% position size
  - [x] Both BUY and SELL sizing
- [x] Test never exceeds 10% portfolio limit

**2.2 Portfolio Manager**
```python
class PortfolioManager:
    positions: Dict[symbol] → {qty, avg_cost, current_price, pnl}
    cash: float

    def execute_trade(order: Order) → Trade:
        # ONLY updates state (validation already done by RiskManager)

    def get_total_value() → float
    def get_unrealized_pnl() → float
    def get_daily_pnl() → float
    def update_prices(prices: Dict) → None
```

**Implementation Details**:
- [x] Modify existing PortfolioManager to remove pre-trade validation
- [x] Add get_daily_pnl() method for RiskManager circuit breaker checks
- [x] Ensure execute_trade() handles both BUY and SELL correctly
- [x] Unit tests:
  - [x] execute_trade() BUY updates positions/cash correctly
  - [x] execute_trade() SELL updates positions/cash correctly
  - [x] get_total_value() = cash + sum(position values)
  - [x] get_daily_pnl() returns correct daily P&L
  - [x] **NO validation tests** (that's RiskManager's job)

**2.3 Risk Manager**
```python
class RiskManager:
    def __init__(self, portfolio: PortfolioManager, max_daily_loss_pct: float = 0.05):
        self.portfolio = portfolio
        self.max_daily_loss_pct = max_daily_loss_pct
        self.daily_pnl_tracker = {}  # Reset daily

    def validate_trade(
        symbol: str,
        qty: float,
        price: float
    ) → Tuple[bool, Optional[str]]:
        # Called by OrderManager BEFORE broker.place_order()
        # Checks (in order):
        # 1. Capital available: cash >= trade_value
        # 2. Position limit: trade_value <= 10% of portfolio_value
        # 3. Daily loss: current_daily_pnl >= -5% of portfolio_value
        # 4. Circuit breaker: not already triggered
        # 5. Position conflict: (e.g., no SHORT if already LONG)
        # Returns: (is_valid, reason_if_invalid)

    def get_daily_pnl() → float:
        # Returns: sum of all trades today

    def on_trade_executed(trade: Trade) → None:
        # Post-execution: update daily P&L tracking
```

**Implementation Details**:
- [x] Create `quantagent/trading/risk_manager.py`
- [x] Implement 5-point validation check
- [x] Add daily P&L tracking (reset at midnight)
- [x] Add circuit breaker flag
- [x] Unit tests:
  - [x] Insufficient capital → rejects with reason
  - [x] Position > 10% → rejects with reason
  - [x] Daily loss > 5% → rejects with reason
  - [x] Circuit breaker active → rejects with reason
  - [x] Valid trade → returns (True, None)
  - [x] on_trade_executed() updates daily P&L correctly
- [x]Integration test:
  - [x] validate_trade() called BEFORE broker (simulate in test)

**2.4 Order Manager (Orchestrator - NEW)**
```python
class OrderManager:
    def __init__(
        self,
        position_sizer: PositionSizer,
        risk_manager: RiskManager,
        broker: PaperBroker,
        portfolio: PortfolioManager,
        db: SessionLocal
    ):
        # All dependencies injected

    def execute_decision(
        self,
        symbol: str,
        decision: str,          # "LONG" | "SHORT" | "HOLD"
        confidence: float,      # 0-1
        current_price: float
    ) → Optional[Order]:
        # Returns: filled Order if executed, None if rejected
        # Flow:
        # 1. If HOLD → return None
        # 2. Size = position_sizer.calculate_size(...)
        # 3. Valid = risk_manager.validate_trade(symbol, size, price)
        #    - If False → log rejection, return None
        # 4. Order = Order(symbol, side, size, price)
        # 5. FilledOrder = broker.place_order(order)
        # 6. Trade = portfolio.execute_trade(filled_order)
        # 7. risk_manager.on_trade_executed(trade)
        # 8. db.add(trade) → db.commit()
        # 9. Return filled_order
```

**Implementation Details**:
- [x] Create `quantagent/trading/order_manager.py`
- [x] Orchestrate flow correctly (Size → Validate → Execute → Update → Log)
- [x] All logging of rejections/executions
- [x] Unit tests:
  - [x] HOLD decision → returns None
  - [x] LONG with valid trade → executes (size, validate, broker, portfolio, db)
  - [x] LONG with invalid trade → rejects (validated, returns None, no broker call)
  - [x] SELL with valid trade → executes correctly
  - [x] Verify correct order of operations (size before validate before broker)
- [x] Integration test:
  - [x] Full end-to-end: LONG decision → fills → portfolio updated → db logged

**2.5 Unit Tests & Integration**
- [ ] PositionSizer unit tests (70%+ coverage)
- [ ] Portfolio Manager unit tests (70%+ coverage)
- [x] Risk Manager unit tests (70%+ coverage)
- [ ] Order Manager unit tests (70%+ coverage)
- [x] Integration test: Decision → Size → Validate → Execute → Update → Log
- [x] Test: Trade is rejected if validation fails (never reaches broker)
- [x] Test: Trade is accepted if validation passes (all steps complete)

**2.6 Strategy Assembler (Factory)**
```python
class StrategyAssembler:
    def build_components(snapshot, environment):
        # Returns: portfolio, sizer, risk, broker, order_manager wired from snapshot
```

Tasks:
- [ ] Implement StrategyAssembler to construct components from a StrategyConfig snapshot
- [ ] Ensure Order creation persists `environment` and `trigger_signal_id` (provenance)
- [ ] Reuse assembler in Backtest and (later) Scheduler

**2.5 LangGraph Improvements (Agent Architecture Refactoring)**

*Improvement #2: Convert agents to subgraphs with parallelization for independent analysis*

**Phase 2a: Subgraph Architecture** (Sequential foundation)
- [ ] Create `IndicatorSubgraph` (from `indicator_agent.py`)
  - [ ] Define subgraph-specific state schema (`IndicatorSubgraphState`)
  - [ ] Break agent logic into reasoning + tool execution nodes
  - [ ] Compile as independent graph (testable in isolation)
- [ ] Create `PatternSubgraph` (from `pattern_agent.py`)
  - [ ] Define state schema (`PatternSubgraphState`)
  - [ ] Encapsulate vision LLM + K-line chart analysis
  - [ ] Compile as independent graph
- [ ] Create `TrendSubgraph` (from `trend_agent.py`)
  - [ ] Define state schema (`TrendSubgraphState`)
  - [ ] Encapsulate vision LLM + trendline analysis
  - [ ] Compile as independent graph
- [ ] Update `graph_setup.py` to compose subgraphs as nodes
- [ ] Test: Each subgraph works independently + in parent graph
- [ ] Benefit: Cleaner parent graph, easier testing, better code organization

**Phase 2b: Parallelization (Performance optimization)**
- [ ] Analyze agent independence: Pattern & Trend both depend only on initial `kline_data` + Indicator output
  - [ ] Reference: [LangGraph Parallelization Pattern](https://docs.langchain.com/oss/python/langgraph/workflows-agents#parallelization)
- [ ] Update parent graph edges to enable fan-out/fan-in:
  ```python
  # Sequential: START → Indicator (initial processing)
  builder.add_edge(START, "Indicator")

  # Parallel: Indicator → [Pattern, Trend] (both independent)
  builder.add_edge("Indicator", "Pattern")
  builder.add_edge("Indicator", "Trend")

  # Convergence: [Pattern, Trend] → Decision (aggregator)
  builder.add_edge("Pattern", "Decision")
  builder.add_edge("Trend", "Decision")
  ```
- [ ] Test: Pattern and Trend execute in parallel (verify with thread/timing logs)
- [ ] Benchmark: Latency reduction from ~6-9s → ~5-7s
- [ ] Benefit: 40-50% faster analysis cycle (critical for real-time trading)

*Improvement #4: Use LangGraph's ToolNode for tool execution*
- [ ] Replace manual tool call handling in Indicator subgraph
- [ ] Use `ToolNode` from `langgraph.prebuilt`
- [ ] Create conditional edges: "reason" → "tools" (if tool calls) or "end"
- [ ] Test: Tools execute correctly, results flow back to LLM
- [ ] Benefit: Eliminates boilerplate, standard pattern, cleaner code

**Deliverables**:
- ✅ PortfolioManager class (in-memory + DB persistence)
- ✅ RiskManager class with validation rules
- ✅ Unit tests (70%+ coverage)
- ✅ Integration tests pass
- ✅ Agents refactored as subgraphs
- ✅ ToolNode properly integrated
- ✅ Architecture significantly cleaner (~40% code reduction)

---

### Week 5-6: Trading Execution - PaperBroker & Full Integration

#### Tasks

**3.1 Abstract Broker Interface**
```python
class Broker(ABC):
    @abstractmethod
    def place_order(order: Order) → Order:
        # Only receives VALIDATED orders (no validation here)

    def cancel_order(order_id: str) → bool
    def get_balance() → float
    def get_positions() → Dict
```

**3.2 Paper Broker Implementation**
```python
class PaperBroker(Broker):
    def place_order(order: Order) → Order:
        # 1. Simulate 2% slippage
        #    - BUY: filled_price = current_price * 1.01
        #    - SELL: filled_price = current_price * 0.99
        # 2. Mark order as FILLED
        # 3. Return filled order with actual fill_price and fill_qty
        # NOTE: Order already validated by RiskManager, just execute
```

**Implementation Details**:
- [x] Create `quantagent/trading/paper_broker.py`
- [x] Implement realistic slippage simulation (±2%)
- [x] Implement order status transitions (PENDING → FILLED)
- [x] Unit tests:
  - [x] BUY order with slippage: fill_price = price * 1.01
  - [x] SELL order with slippage: fill_price = price * 0.99
  - [x] Order status transitions correctly
  - [x] Returns filled order with correct fill_price and filled_qty
- [x] **No validation** (risk checks already done upstream)

**3.3 Full End-to-End Integration Test**
```
Analysis (Decision + Confidence)
    → PositionSizer.calculate_size()
    → RiskManager.validate_trade() [REJECT if invalid]
    → OrderManager.execute_decision()
        → PaperBroker.place_order() [only if validated]
        → PortfolioManager.execute_trade()
        → RiskManager.on_trade_executed()
        → Database.add(trade)
    → Dashboard shows updated portfolio
```

- [x] Test: LONG decision with valid trade → executes all steps
- [x] Test: SHORT decision with invalid trade → rejected at validation, broker never called
- [x] Test: Portfolio correctly updated after execution
- [x] Test: Trade correctly logged to database
- [x] Test: Daily P&L tracking correct
- [x] Test: Circuit breaker stops all trades if triggered

**3.4 Database Schema Verification**
- [x] Verify Order table has all fields (symbol, side, qty, price, status, created_at, filled_at, trigger_signal_id, environment)
- [x] Verify Trade table has all fields (symbol, entry_price, exit_price, qty, side, opened_at, closed_at, pnl, environment)
- [x] Verify Signal table has provenance fields (trigger_signal_id, model_provider, model_name)
- [x] Migration generated correctly for any schema updates

**Deliverables**:
- ✅ PaperBroker executes orders realistically (slippage simulation)
- ✅ OrderManager (implemented in Week 3-4) orchestrates all steps
- ✅ Full end-to-end flow working: Size → Validate → Execute → Update → Log
- ✅ All integration tests passing
- ✅ Trades never reach broker if validation fails
- ✅ Portfolio updates correctly on execution
- ✅ Database schema supports all operations

---

### Week 7-8: Backtesting Engine

#### Tasks

**4.1 Backtest Framework**
```python
class Backtest:
    def run(start_date, end_date, assets) → results
    def calculate_metrics() → {
        total_trades: int,
        win_rate: float,
        profit_factor: float,
        sharpe_ratio: float,
        max_drawdown: float,
        total_pnl: float
    }
```

**4.2 Data Caching Layer** (Week 3 addition)
```python
class DataProvider:
    def get_ohlc(symbol, timeframe, start_date, end_date) → DataFrame:
        # 1. Check DB for existing data
        # 2. Fetch missing ranges from yfinance
        # 3. Store in DB
        # 4. Return complete dataset
```

**4.3 Backtest Loop**
- [x] Loop through historical dates
- [x] Fetch data for each date
- [x] Execute analysis (same agents as live)
- [x] Compare decision vs actual price 4h later
- [x] Record result
 - [x] Persist backtest run setup (config snapshot, model params, assets/time ranges)
 - [x] Persist generated analyses with model metadata and checkpoint references
 - [ ] Use Universe from Portfolio profile by default when assets are unspecified; BacktestRun stores the final assets list

**4.4 Metrics Calculation**
- [x] Win rate: % of winning trades
- [x] Profit factor: sum(wins) / abs(sum(losses))
- [x] Sharpe ratio: (return - risk_free) / volatility
- [x] Max drawdown: worst peak-to-trough decline
- [x] Total P&L: sum of all trade P&L

**4.5 LangGraph Improvements (Structured Outputs for Decision Quality)**

*Improvement #3: Use Pydantic models for agent outputs*
- [x] Create `IndicatorReport` Pydantic model
  - [x] Fields: `macd`, `macd_signal`, `rsi`, `rsi_level`, `trend_direction`, `confidence`
- [ x Create `PatternReport` Pydantic model
  - [x] Fields: `patterns_detected`, `primary_pattern`, `confidence`, `breakout_probability`
- [x] Create `TrendReport` Pydantic model
  - [x] Fields: `support_level`, `resistance_level`, `trend_direction`, `trend_strength`
- [x] Update agents to return Pydantic models instead of strings
- [x] Update `decision_agent.py` to consume structured data (no string parsing)
- [x] Test: Decision logic more reliable with type-safe access
- [x] Benefit: Type validation, easier decision parsing, fewer bugs in decision agent

**Deliverables**:
- ✅ Backtest runs on 3-4 months historical data
- ✅ Metrics calculated correctly
- ✅ Data caching working (backtesting 10x faster)
- ✅ Results show strategy viability (win rate ≥ 40%)
- ✅ Pydantic models enforce output structure
- ✅ Decision agent simplified (no string parsing)
- ✅ Type-safe state transitions
 - ✅ Backtest run records full setup (config snapshot, model settings)
 - ✅ Replay execution can reuse stored analyses without new LLM calls
 - ✅ Support multiple model variants per (symbol, timeframe, timestamp)

---

### Week 9-10: Scheduler + Dashboard + Integration

#### Tasks

**5.1 APScheduler Integration**
```python
class TradingScheduler:
    def start() → starts hourly analysis
    def analyze_asset(symbol) → executes decision if signal
    def stop() → stops scheduler
```

**5.2 Streamlit Dashboard** (UI choice for MVP)
```
├─ Tab 1: Dashboard
│   ├─ Key metrics (P&L, positions, win rate)
│   ├─ Equity curve chart
│   └─ Recent trades table
│
├─ Tab 2: Backtest
│   ├─ Run backtest (date range, assets)
│   └─ Display metrics + trades
│
├─ Tab 3: Trades History
│   └─ All trades with P&L
│
└─ Tab 4: Logs
    └─ Recent system events
```

Additional MVP UI tasks (per UI requirements):
- [ ] Configuration tab: list/create/edit StrategyConfig (JSON), set defaults per environment; model preset management
- [ ] Analyses tab: filterable table (env/symbol/timeframe/date), details with reports, charts, checkpoint/thread, model metadata, order link
- [ ] Backtesting tab: create run (generate-only vs generate+execute), runs table with status/progress/metrics, run details with logs and cancel
- [ ] Replay tab: select backtest_run, pick profile(s), execute replay; compare metrics (side-by-side) and overlay equity curves
- [ ] Orders & Positions tab (paper): orders with trigger_signal and provenance; positions with unrealized P&L
- [ ] Global environment filter and auto-refresh on heavy tabs
- [ ] Docs: link to `docs/01_requirements/ui_streamlit_mvp_requirements.md`
 - [ ] Configuration tab: manage Universe (multi-select symbols) in Portfolio profile; snapshot preview

**5.3 Integration Testing**
- [ ] Full end-to-end: Analysis → Risk Check → Execution → Portfolio Update → Database → Dashboard
- [ ] Stress test: 100+ trades execution
- [ ] 24h+ continuous operation test
- [ ] Scheduler reliability

**5.4 Documentation**
- [ ] Setup instructions (with docker-compose)
- [ ] How to run backtest
- [ ] How to run scheduler
- [ ] Configuration options

**Deliverables**:
- ✅ Scheduler runs hourly analysis
- ✅ Trades execute automatically
- ✅ Streamlit dashboard live
- ✅ System stable 24h+
- ✅ Full documentation
- ✅ UI supports configuration, analyses exploration, backtest/replay flows, and environment filtering
 - ✅ UI supports Universe management in Portfolio profile and optional asset override per backtest
 - ✅ Dashboard/queries filterable by environment (backtest, paper)

---

## Key Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Database** | PostgreSQL | Production-ready, handles scaling, supports LangGraph checkpointing |
| **Broker (MVP)** | Paper trading mock | Fast validation, no real capital risk |
| **Scheduler** | APScheduler | Simple, single-process, sufficient for MVP |
| **Data Caching** | Local DB cache | Backtesting 10x faster, reproducible |
| **UI Framework** | Streamlit | Fast development (3-4 days), Python-only |
| **Docker** | Lightweight setup | Portability + corporate environment friendly |
| **Testing** | pytest + CI/CD | Safety on changes, reproducible builds |
| **Agent Framework** | `create_agent` (LangChain v1) | Built-in reliability, checkpointing, less manual code |
| **Agent Architecture** | Subgraphs | Clear separation of concerns, independent testing, team scalability |
| **Tool Execution** | `ToolNode` (LangGraph) | Standard pattern, automatic tool handling, cleaner code |
| **Agent Outputs** | Pydantic models | Type safety, validation, easier decision logic, no string parsing |
| **State Persistence** | LangGraph Checkpointing | Long backtest resilience, execution history, debugging |

---

## Definition of Done: Phase 1 MVP

### Trading Architecture Complete
- ✅ PositionSizer: Calculates trade size based on confidence
- ✅ RiskManager: Validates trades BEFORE execution
- ✅ OrderManager: Orchestrates Size → Validate → Execute → Update → Log
- ✅ PaperBroker: Executes validated orders with slippage
- ✅ PortfolioManager: Updates state (no pre-trade validation)

### System Works Automatically
- [ ] Runs analysis every 1 hour
- [ ] Calculates trade size per signal confidence
- [ ] Validates each trade against risk limits
- [ ] Places orders without manual intervention (if valid)
- [ ] Rejects invalid orders with logged reasons
- [ ] Tracks all positions accurately
- [ ] Enforces all risk limits (capital, position size, daily loss, circuit breaker)
- [ ] Logs all activities to database (orders, trades, rejections)

### Risk Management Enforced
- [ ] No trades executed without validation
- [ ] Insufficient capital → rejects
- [ ] Position > 10% of portfolio → rejects
- [ ] Daily loss > 5% → rejects
- [ ] Circuit breaker triggered → stops all trades
- [ ] All rejections logged with specific reason

### Strategy Validated
- [ ] Backtest on 3-4 months data
- [ ] Win rate ≥ 40%
- [ ] Sharpe ratio ≥ 1.0
- [ ] Max drawdown ≤ 15%

### Operations Reliable
- [ ] Uptime ≥ 99% over 48h test
- [ ] Zero database inconsistencies
- [ ] Graceful error handling (errors logged, system continues)
- [ ] Analysis latency ≤ 30 seconds
- [ ] Order execution ≤ 2 seconds
- [ ] Daily P&L tracking accurate

### Code Quality
- [ ] PositionSizer test coverage ≥ 70%
- [ ] RiskManager test coverage ≥ 70%
- [ ] OrderManager test coverage ≥ 70%
- [ ] PaperBroker test coverage ≥ 70%
- [ ] All tests passing
- [ ] CI/CD pipeline green
- [ ] Code reviewed

### Documentation Complete
- [ ] Setup instructions (README)
- [ ] Architecture documented (trading components)
- [ ] Risk management rules documented
- [ ] Order execution flow documented
- [ ] Configuration guide (risk profiles)
- [ ] API endpoints documented
- [ ] Troubleshooting guide

---

## Success Metrics

```
✅ Can say: "System analyzed BTC, SPX, Oil every hour for 7 days"
✅ Can say: "Executed 35 trades automatically (rejected 5 invalid trades)"
✅ Can say: "All trades sized based on signal confidence (2.5% to 5% of capital)"
✅ Can say: "No trades executed that violated risk limits (capital, position, daily loss)"
✅ Can say: "Backtest shows 42% win rate"
✅ Can say: "Dashboard shows all metrics in real-time"
✅ Can say: "Anyone can clone and run with docker-compose"
```

---

## What's NOT in Phase 1

❌ Real broker integration
❌ Real-time WebSocket feeds
❌ Advanced UI (Streamlit is MVP-grade)
❌ Multi-strategy portfolio
❌ Machine learning optimization
❌ Production risk models (VaR, Greeks)
❌ 24/7 live trading (test only)
❌ Human-in-the-loop approvals (order execution is fully automated)
❌ RAG/historical pattern learning (Phase 2 feature)
❌ Conditional agent routing (Phase 2 optimization)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Strategy unprofitable | Extensive backtesting validates before live paper trading |
| Database corruption | Automated backups + immutable transaction logs |
| Scheduler reliability | Logs every execution, monitoring alerts |
| API rate limits | Data caching reduces yfinance calls significantly |
| Team knowledge gap | Well-documented code + setup instructions |

---

## Next: Phase 2

After MVP validation (Week 10), decide:

**IF backtest metrics are good (win rate ≥40%, Sharpe ≥1.0)**:
→ Proceed to Phase 2: Real broker integration + Advanced features

**IF backtest metrics are poor**:
→ Iterate on strategy (different indicators/timeframes)
→ Re-backtest
→ Only proceed when validated

**Phase 2 also includes remaining LangGraph improvements**:
- ☐ Improvement #6: Human-in-the-loop middleware (for real broker trades)
- ☐ Improvement #7: RAG for historical pattern learning
- ☐ Improvement #8: Conditional agent routing (adaptive strategy)
- ☐ Improvement #9: Middleware suite (audit, logging, monitoring)
- ☐ Improvement #10: Streaming support for real-time dashboards

Phase 2 roadmap: `phase2_roadmap.md`

