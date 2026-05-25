# QuantAgent-kkj.8 — Requirements: Strategy Registry & Parametrized Scheduler

**Issue ID:** QuantAgent-kkj.8
**Title:** Crear strategy registry y parametrizar scheduler para selección de estrategia
**Type:** Feature
**Priority:** 1
**Parent:** QuantAgent-kkj (M2 Milestone)
**Blocks:** QuantAgent-kkj.9 (UI strategy selector)

---

## Problem Statement

`TradingScheduler` is hardcoded to use `LLMAgentStrategy` regardless of what strategy is passed:

```python
# quantagent/trading/scheduler.py:57,65
strategy: Optional[LLMAgentStrategy] = None  # wrong type hint
self.strategy = strategy or LLMAgentStrategy(self.trading_graph)  # ignores explicit strategy
```

There is no central registry of available strategies and their configurable parameters, which blocks:
- UI strategy selection (kkj.9)
- Non-LLM paper trading pilots
- Dynamic parameter configuration

Additionally, `_process_asset` calls `generate_signal(..., thread_id=thread_id)`, but deterministic strategies do not accept `thread_id`, which would cause a `TypeError` if a deterministic strategy were passed.

---

## Functional Requirements

### FR-1: Strategy Registry

A module `quantagent/strategy/registry.py` must exist with:

1. A `STRATEGY_REGISTRY` dict (or equivalent) mapping strategy name strings to metadata.
2. Each entry must include at minimum:
   - `cls`: the Python class
   - `type`: `"deterministic"` or `"llm"`
   - `display_name`: human-readable name
   - `description`: one-line description
   - `params`: dict of configurable parameters, each with `type`, `default`, and `description`
   - `min_bars`: minimum OHLCV bars required
3. A `get_strategy_registry()` function that returns the registry dict.
4. A `get_strategy_names()` function returning a list of strategy name strings.
5. A `build_strategy(name, **kwargs)` factory function that instantiates a strategy by name with provided params.

All 4 strategies must be registered: `RSIMeanReversionStrategy`, `FiftyTwoWeekHighStrategy`, `TripleScreenStrategy`, `LLMAgentStrategy`.

The module must be importable from `quantagent.strategy` without side effects (no LLM connections, no DB access at import time).

### FR-2: TradingStrategy.describe() classmethod

`TradingStrategy` (base class) must gain an optional `@classmethod describe() -> dict` that returns:
- `name`: class name
- `display_name`: human-readable name
- `type`: `"deterministic"` or `"llm"`
- `description`: short description

Each concrete strategy should override `describe()`. Default implementation in base class raises `NotImplementedError` or returns a minimal stub.

### FR-3: TradingScheduler — remove LLMAgentStrategy hardcoding

`TradingScheduler.__init__` must:
1. Change the type hint of the `strategy` parameter from `Optional[LLMAgentStrategy]` to `Optional[TradingStrategy]`.
2. Keep `LLMAgentStrategy` as the default when `strategy=None` (backward compatible).
3. When an explicit `TradingStrategy` is passed, use it — do not override with LLM.

### FR-4: Fix thread_id incompatibility

`_process_asset` calls `self.strategy.generate_signal(..., thread_id=thread_id)`. Deterministic strategies do not accept `thread_id`. The fix must:
- Only pass `thread_id` when the strategy is `LLMAgentStrategy` (or any strategy that accepts it), OR
- Accept and silently ignore `thread_id` in deterministic strategies via `**kwargs`.

The recommended approach: pass `thread_id` via `**kwargs` and only include it in the call when the strategy supports it (checked via `hasattr` or by inspecting `generate_signal` signature).

### FR-5: Tests

Tests must cover:
1. `TradingScheduler` instantiated with `RSIMeanReversionStrategy` uses that strategy (not LLM).
2. `_process_asset` does not raise `TypeError` when called with a deterministic strategy.
3. `get_strategy_registry()` returns all 4 strategies.
4. `build_strategy("RSIMeanReversionStrategy", rsi_period=20)` returns an `RSIMeanReversionStrategy` with `rsi_period=20`.
5. `from quantagent.strategy import STRATEGY_REGISTRY` imports cleanly (no side effects).

---

## Non-functional Requirements

- No changes to strategy internal logic.
- No UI changes (that is kkj.9).
- No DB schema changes.
- The registry must not trigger LLM connections at import time.
- Backward compatible: existing code that instantiates `TradingScheduler()` without `strategy=` must continue to work identically.

---

## Out of Scope

- UI selectors (kkj.9)
- New strategies
- DB model changes
- Changing internal strategy logic
