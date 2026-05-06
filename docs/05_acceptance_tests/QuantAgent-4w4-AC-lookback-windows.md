# QuantAgent-4w4 — Acceptance Criteria: Backtest Lookback Windows

**Issue:** QuantAgent-4w4  
**Type:** Acceptance Tests  
**Created:** 2026-05-06  

---

## AC1 — `TradingStrategy.required_history_bars` property exists with default 30

**Given** `TradingStrategy` is imported from `quantagent.strategy.base`  
**When** any existing concrete strategy is instantiated (e.g. `LLMAgentStrategy`, `TripleScreenStrategy`)  
**Then**:
- `strategy.required_history_bars` returns `30`
- No AttributeError or TypeError is raised

**Testable:** `pytest tests/ -k "test_4w4_required_history_bars_default"`

---

## AC2 — Subclass can override `required_history_bars`

**Given** a custom strategy subclass that overrides `required_history_bars` to return `300`  
**When** the property is accessed  
**Then** the value is `300`, not `30`

**Testable:** `pytest tests/ -k "test_4w4_required_history_bars_override"`

---

## AC3 — `Backtest._analyze_and_trade()` no longer hardcodes 30 days

**Given** `quantagent/backtesting/backtest.py` is read  
**When** searching for `lookback_days = 30`  
**Then** the literal `lookback_days = 30` is absent from `_analyze_and_trade()`

**Testable (static):** `grep -n "lookback_days = 30" quantagent/backtesting/backtest.py` returns no output

---

## AC4 — Engine requests calendar days proportional to `required_history_bars`

**Given** a mock strategy with `required_history_bars = 300` and timeframe `1d`  
**When** `Backtest._analyze_and_trade()` is called for a given `current_date`  
**Then** `data_provider.get_ohlc()` is called with `start_date` at least `300 * (365/252)` ≈ 434 calendar days before `current_date`

**Implementation note:** Use `unittest.mock.patch` on `data_provider.get_ohlc` and capture the `start_date` argument.

**Testable:** `pytest tests/ -k "test_4w4_engine_requests_sufficient_bars"`

---

## AC5 — `_bars_to_calendar_days` conversion is correct for `1d` timeframe

**Given** `bars = 252` and `timeframe = "1d"`  
**When** `_bars_to_calendar_days(252)` is called  
**Then** the result is `365` (i.e. `ceil(252 × 365 / 252)`)

**Given** `bars = 303` and `timeframe = "1d"`  
**When** `_bars_to_calendar_days(303)` is called  
**Then** the result is `ceil(303 × 365 / 252) = 439`

**Testable:** `pytest tests/ -k "test_4w4_bars_to_calendar_days"`

---

## AC6 — Minimum data guard uses `required_history_bars`

**Given** a mock strategy with `required_history_bars = 300`  
**When** `data_provider.get_ohlc()` returns a DataFrame with 10 rows  
**Then** `_analyze_and_trade()` logs a warning containing the actual count (`10`) and returns without executing a trade

**Testable:** `pytest tests/ -k "test_4w4_insufficient_data_guard"`

---

## AC7 — Backward compatibility: default strategies unaffected

**Given** an existing test suite in `tests/test_backtest.py`, `tests/test_backtest_integration.py`,
`tests/test_backtest_market_hours.py`, `tests/test_backtest_phase4_metrics.py`  
**When** the full test suite is run after the 4w4 changes  
**Then** all previously passing tests continue to pass

**Testable:** `pytest tests/test_backtest*.py -v`

---

## AC8 — No `Insufficient data` warnings for a strategy with sufficient lookback

**Given** a strategy with `required_history_bars = 30` (default)  
**And** the data provider supplies 30+ bars for each analysis date  
**When** the backtest runs  
**Then** no `Insufficient data` warning is logged

*(This is a regression guard; the current behavior with `lookback_days = 30` must be preserved.)*

**Testable:** Verify via log capture in `pytest tests/ -k "test_4w4_no_spurious_warnings"`

---

## Manual / Integration AC (not automated in CI)

### AC9 — b8r backtest reference run (unblocks QuantAgent-b8r)

**Given** `FiftyTwoWeekHighStrategy` (from b8r branch) overrides `required_history_bars` to `303`  
**And** AAPL OHLCV daily data for 2021-01-01 → 2023-12-31 is present in the database  
**When** a backtest runs over 2022-01-01 → 2023-12-31 with `timeframe="1d"` and `FiftyTwoWeekHighStrategy`  
**Then**:
- Zero systematic `Insufficient data` warnings from lookback shortage
- At least one LONG signal is evaluated (may be 0 trades if filters don't pass, but strategy logic is exercised)
- Backtest completes without crash

**Note:** This AC is the manual gate for QuantAgent-b8r tech lead review; it is not part of the 4w4 unit test suite.
