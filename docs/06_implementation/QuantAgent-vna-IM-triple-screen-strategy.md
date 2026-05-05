# QuantAgent-vna — Implementation Notes: Triple Screen Strategy

**Issue:** QuantAgent-vna  
**Type:** Implementation Notes  
**Created:** 2026-05-05  

---

## What changed

- Added `quantagent/strategy/triple_screen_strategy.py` with a concrete `TripleScreenStrategy(TradingStrategy)` implementation.
- Exported `TripleScreenStrategy` from `quantagent/strategy/__init__.py`.
- Added `tests/test_triple_screen_strategy.py` covering:
  - Screen 1 trend classification
  - Screen 2 stochastic activation
  - Screen 3 breakout trigger
  - combined `generate_signal()` behaviour
  - backtest smoke coverage for the reference profile
- Adjusted `quantagent/backtesting/backtest.py` so custom non-LLM strategies receive the full OHLC record sequence (`list[dict]`) instead of the truncated agent-oriented payload used by `LLMAgentStrategy`.

## Why the backtest change was necessary

`Backtest._analyze_and_trade()` previously passed `format_ohlcv_for_agents(df)` to every strategy. That payload is designed for the LLM graph path and trims the history to ~45 candles.

Triple Screen needs a larger lookback (`weekly_bars * (trend_ema_period + 1) + stoch_k_period + stoch_d_period`, 78 candles with defaults) and expects standard OHLC records. Without this change, the strategy could pass unit tests in isolation but never receive enough history inside the live backtest loop.

The implementation keeps the existing LLM path unchanged:

- `LLMAgentStrategy` → still receives `format_ohlcv_for_agents(df)`
- concrete `TradingStrategy` implementations → now receive `df.to_dict(orient="records")`

## Reference backtest profile

```python
from datetime import datetime

from quantagent.backtesting.backtest import Backtest
from quantagent.strategy.triple_screen_strategy import TripleScreenStrategy

strategy = TripleScreenStrategy()
backtest = Backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    assets=["BTC-USD"],
    timeframe="4h",
    initial_capital=100_000.0,
    strategy=strategy,
)
metrics = backtest.run(name="QuantAgent-vna-reference")
```

## Verification run executed in this implementation phase

Primary strategy test suite:

```bash
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/test_triple_screen_strategy.py -v
```

Backtest compatibility regression subset:

```bash
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest \
  tests/test_backtest_position_monitor.py \
  -k 'accepts_custom_strategy or active_position_prevents_strategy_invocation' -v
```

Observed result:

- `tests/test_triple_screen_strategy.py`: **24 passed**
- backtest regression subset: **2 passed**
- reference profile smoke (`test_backtest_reference_profile_completes`): completed with finite PnL and no crash

## Quality gates

Executed:

```bash
ruff check --fix .
ruff check quantagent/backtesting/backtest.py quantagent/strategy/__init__.py quantagent/strategy/triple_screen_strategy.py tests/test_triple_screen_strategy.py
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m compileall -q .
```

Outcome:

- targeted Ruff check on touched files: **passed**
- compileall: **passed**
- repo-wide `ruff check --fix .`: **not fully clean due pre-existing unrelated issues** in legacy Alembic/tests files outside this ticket's scope

## Files touched

- `quantagent/backtesting/backtest.py`
- `quantagent/strategy/__init__.py`
- `quantagent/strategy/triple_screen_strategy.py`
- `tests/test_triple_screen_strategy.py`
