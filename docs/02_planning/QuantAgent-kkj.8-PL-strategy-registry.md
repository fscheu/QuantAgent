# QuantAgent-kkj.8 — Planning: Strategy Registry & Parametrized Scheduler

**Issue ID:** QuantAgent-kkj.8
**Title:** Crear strategy registry y parametrizar scheduler para selección de estrategia
**Type:** Feature
**Priority:** 1

---

## Objective

Create a central strategy registry as the source of truth for strategy selection across the system, and fix the `TradingScheduler` so it can use any `TradingStrategy` (not just `LLMAgentStrategy`).

---

## Current State Analysis

### Scheduler hardcoding (scheduler.py:57,65)

```python
strategy: Optional[LLMAgentStrategy] = None  # wrong type — should be TradingStrategy
self.strategy = strategy or LLMAgentStrategy(self.trading_graph)  # ignores passed strategy
```

### thread_id incompatibility (scheduler.py:291-297)

```python
signal = self.strategy.generate_signal(
    kline_data, symbol, self.config.timeframe, current_price,
    thread_id=thread_id,  # <-- TypeError for deterministic strategies
)
```

`RSIMeanReversionStrategy.generate_signal`, `FiftyTwoWeekHighStrategy.generate_signal`, and `TripleScreenStrategy.generate_signal` do not accept `thread_id`. This must be fixed before non-LLM strategies can work in the scheduler.

### Strategy parameters (from source)

| Strategy | Key params |
|---|---|
| `RSIMeanReversionStrategy` | rsi_period=14, oversold_threshold=30.0, overbought_threshold=70.0, stop_loss_pct=0.02, take_profit_pct=0.03, trailing_stop_pct=0.05 |
| `FiftyTwoWeekHighStrategy` | lookback_days=252, proximity_threshold=0.98, trend_ma_period=50, volume_ma_period=20, volume_factor=1.5, stop_loss_pct=0.05, take_profit_pct=0.15, trailing_stop_pct=0.08 |
| `TripleScreenStrategy` | weekly_bars=5, trend_ema_period=13, stoch_k_period=5, stoch_d_period=3, stoch_oversold=20.0, stoch_overbought=80.0, stop_loss_pct=0.02, take_profit_pct=0.04, trailing_stop_pct=0.05 |
| `LLMAgentStrategy` | trading_graph (injected dependency — not a user param) |

---

## Implementation Tasks

### Task 1: Create `quantagent/strategy/registry.py`
**Estimate:** 1h

**What:**

```python
# quantagent/strategy/registry.py

from typing import Any, Dict, Type
from .base import TradingStrategy

STRATEGY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "RSIMeanReversionStrategy": {
        "cls": ...,  # imported below
        "type": "deterministic",
        "display_name": "RSI Mean Reversion",
        "description": "Buys on RSI oversold, sells on RSI overbought.",
        "min_bars": 15,
        "params": {
            "rsi_period": {"type": int, "default": 14, "description": "RSI calculation period"},
            "oversold_threshold": {"type": float, "default": 30.0, "description": "RSI buy threshold"},
            "overbought_threshold": {"type": float, "default": 70.0, "description": "RSI sell threshold"},
            "stop_loss_pct": {"type": float, "default": 0.02, "description": "Stop loss %"},
            "take_profit_pct": {"type": float, "default": 0.03, "description": "Take profit %"},
            "trailing_stop_pct": {"type": float, "default": 0.05, "description": "Trailing stop %"},
        },
    },
    "FiftyTwoWeekHighStrategy": {
        "cls": ...,
        "type": "deterministic",
        "display_name": "52-Week High Momentum",
        "description": "LONG breakout above 52-week high with trend and volume confirmation.",
        "min_bars": 290,
        "params": {
            "lookback_days": {"type": int, "default": 252, "description": "Lookback window (days)"},
            "proximity_threshold": {"type": float, "default": 0.98, "description": "Min ratio to 52w high"},
            "trend_ma_period": {"type": int, "default": 50, "description": "Trend SMA period"},
            "volume_ma_period": {"type": int, "default": 20, "description": "Volume MA period"},
            "volume_factor": {"type": float, "default": 1.5, "description": "Volume breakout multiplier"},
            "stop_loss_pct": {"type": float, "default": 0.05, "description": "Stop loss %"},
            "take_profit_pct": {"type": float, "default": 0.15, "description": "Take profit %"},
            "trailing_stop_pct": {"type": float, "default": 0.08, "description": "Trailing stop %"},
        },
    },
    "TripleScreenStrategy": {
        "cls": ...,
        "type": "deterministic",
        "display_name": "Triple Screen (Elder)",
        "description": "Alexander Elder's three-filter system: trend + oscillator + breakout.",
        "min_bars": 80,
        "params": {
            "weekly_bars": {"type": int, "default": 5, "description": "Higher-TF aggregation size"},
            "trend_ema_period": {"type": int, "default": 13, "description": "Trend EMA period"},
            "stoch_k_period": {"type": int, "default": 5, "description": "Stochastic %K period"},
            "stoch_d_period": {"type": int, "default": 3, "description": "Stochastic %D period"},
            "stoch_oversold": {"type": float, "default": 20.0, "description": "Stochastic oversold level"},
            "stoch_overbought": {"type": float, "default": 80.0, "description": "Stochastic overbought level"},
            "stop_loss_pct": {"type": float, "default": 0.02, "description": "Stop loss %"},
            "take_profit_pct": {"type": float, "default": 0.04, "description": "Take profit %"},
            "trailing_stop_pct": {"type": float, "default": 0.05, "description": "Trailing stop %"},
        },
    },
    "LLMAgentStrategy": {
        "cls": ...,
        "type": "llm",
        "display_name": "LLM Agent (Multi-Agent Graph)",
        "description": "Multi-agent LangGraph pipeline. Requires LLM API. Has token cost.",
        "min_bars": 30,
        "params": {},  # trading_graph is injected — not a user-configurable param
    },
}


def get_strategy_registry() -> Dict[str, Dict[str, Any]]:
    """Return the full strategy registry."""
    return STRATEGY_REGISTRY


def get_strategy_names() -> list[str]:
    """Return list of registered strategy names."""
    return list(STRATEGY_REGISTRY.keys())


def build_strategy(name: str, **kwargs) -> TradingStrategy:
    """
    Instantiate a strategy by name with provided params.

    For LLMAgentStrategy, `trading_graph` must be in kwargs.
    For deterministic strategies, all params are optional (defaults apply).

    Raises:
        KeyError: if name not in registry
        TypeError: if required args missing (e.g. trading_graph for LLM)
    """
    entry = STRATEGY_REGISTRY[name]
    cls = entry["cls"]
    return cls(**kwargs)
```

Import note: Use lazy imports inside the dict to avoid circular imports — or import at module level after the concrete classes are imported. The `__init__.py` approach (importing registry after all classes) is the safest pattern.

**Files changed:**
- `quantagent/strategy/registry.py` (new)
- `quantagent/strategy/__init__.py` (add `STRATEGY_REGISTRY`, `get_strategy_registry`, `get_strategy_names`, `build_strategy` to imports and `__all__`)

---

### Task 2: Add `describe()` classmethod to `TradingStrategy` base
**Estimate:** 30min

**What:**

```python
# quantagent/strategy/base.py — add to TradingStrategy class

@classmethod
def describe(cls) -> dict:
    """Return strategy metadata. Override in subclasses."""
    return {
        "name": cls.__name__,
        "display_name": cls.__name__,
        "type": "unknown",
        "description": "",
    }
```

Each concrete strategy overrides this to return `type: "deterministic"` or `"llm"` and a proper display name. This keeps the registry authoritative but allows strategies to self-describe when needed.

**Files changed:**
- `quantagent/strategy/base.py`
- `quantagent/strategy/rsi_strategy.py`
- `quantagent/strategy/fifty_two_week_high_strategy.py`
- `quantagent/strategy/triple_screen_strategy.py`
- `quantagent/strategy/llm_agent_strategy.py`

---

### Task 3: Fix `TradingScheduler` strategy type and hardcoding
**Estimate:** 30min

**What — `scheduler.py` changes:**

```python
# Change import:
from quantagent.strategy.base import TradingStrategy  # already imported as StrategyTradingSignal alias

# Change __init__ signature (line 57):
strategy: Optional[TradingStrategy] = None,  # was Optional[LLMAgentStrategy]

# Line 65 — keep LLM as default (backward compatible):
self.strategy = strategy if strategy is not None else LLMAgentStrategy(self.trading_graph)
```

**Critical: Fix thread_id incompatibility (lines 291-297):**

```python
# In _process_asset(), replace:
signal = self.strategy.generate_signal(
    kline_data, symbol, self.config.timeframe, current_price,
    thread_id=thread_id,
)

# With:
import inspect
sig = inspect.signature(self.strategy.generate_signal)
if "thread_id" in sig.parameters:
    signal = self.strategy.generate_signal(
        kline_data, symbol, self.config.timeframe, current_price,
        thread_id=thread_id,
    )
else:
    signal = self.strategy.generate_signal(
        kline_data, symbol, self.config.timeframe, current_price,
    )
```

Alternative (simpler): Add `**kwargs` to deterministic strategies' `generate_signal` signatures. This is cleaner if the implementer prefers it.

**Files changed:**
- `quantagent/trading/scheduler.py`

---

### Task 4: Tests
**Estimate:** 1h

**Test file:** `tests/test_strategy_registry.py` (new) + additions to `tests/test_scheduler.py` or similar.

**Required test cases:**

1. **Registry completeness**: `get_strategy_registry()` returns dict with exactly the 4 expected keys.
2. **Registry structure**: Each entry has `cls`, `type`, `display_name`, `params`, `min_bars`.
3. **build_strategy RSI**: `build_strategy("RSIMeanReversionStrategy", rsi_period=20)` returns `RSIMeanReversionStrategy` with `rsi_period=20`.
4. **build_strategy defaults**: `build_strategy("RSIMeanReversionStrategy")` returns instance with default params.
5. **No side effects**: `from quantagent.strategy import STRATEGY_REGISTRY` does not raise.
6. **Scheduler uses RSI**: `TradingScheduler(..., strategy=RSIMeanReversionStrategy())` → `scheduler.strategy` is an `RSIMeanReversionStrategy` instance.
7. **Scheduler no LLM override**: When strategy is explicitly passed, it must NOT be overridden by `LLMAgentStrategy`.
8. **process_asset no TypeError**: Mock `_process_asset` flow with `RSIMeanReversionStrategy` — no `TypeError` from `thread_id`.

---

## File Change Summary

| File | Action | Notes |
|---|---|---|
| `quantagent/strategy/registry.py` | **New** | Central registry dict + factory functions |
| `quantagent/strategy/__init__.py` | **Modify** | Add registry exports to `__all__` |
| `quantagent/strategy/base.py` | **Modify** | Add `describe()` classmethod |
| `quantagent/strategy/rsi_strategy.py` | **Modify** | Override `describe()` |
| `quantagent/strategy/fifty_two_week_high_strategy.py` | **Modify** | Override `describe()` |
| `quantagent/strategy/triple_screen_strategy.py` | **Modify** | Override `describe()` |
| `quantagent/strategy/llm_agent_strategy.py` | **Modify** | Override `describe()` |
| `quantagent/trading/scheduler.py` | **Modify** | Fix type hint, hardcoding, thread_id compat |
| `tests/test_strategy_registry.py` | **New** | Registry + scheduler tests |

---

## Risks

| Risk | Mitigation |
|---|---|
| Circular imports in registry.py | Import concrete classes at module level after they are defined; or use `TYPE_CHECKING` guard |
| `thread_id` fix breaks existing tests | Use `inspect.signature` approach — zero change to existing behavior for LLMAgentStrategy |
| LLMAgentStrategy in registry triggers LLM connection | Verify `LLMAgentStrategy` class import does NOT connect to LLM at import time (it doesn't — trading_graph is injected) |
| Registry min_bars for TripleScreen | `weekly_bars*(trend_ema_period+1) + stoch_k_period + stoch_d_period` = `5*(13+1)+5+3 = 78`. Registry uses 80 (safe margin). |

---

## Implementation Order

1. Task 1 (registry.py) — independent, no deps
2. Task 2 (describe()) — independent
3. Task 3 (scheduler fix) — depends on registry existing (for import)
4. Task 4 (tests) — last, depends on all above

Tasks 1 and 2 can be done in parallel.

---

## Validation

```bash
# After implementation:
cd /home/azureuser/repos/projects/QuantAgent
source .venv/bin/activate

# Smoke import
python -c "from quantagent.strategy import STRATEGY_REGISTRY, get_strategy_registry, build_strategy; print(list(STRATEGY_REGISTRY.keys()))"

# Compile check
python -m compileall -q quantagent/strategy/ quantagent/trading/scheduler.py

# Tests
pytest tests/test_strategy_registry.py -v
```
