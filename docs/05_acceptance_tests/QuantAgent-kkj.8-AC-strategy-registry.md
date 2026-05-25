# QuantAgent-kkj.8 — Acceptance Criteria: Strategy Registry & Parametrized Scheduler

**Issue ID:** QuantAgent-kkj.8
**Title:** Crear strategy registry y parametrizar scheduler para selección de estrategia
**Type:** Feature

---

## AC-1: Registry exists and is importable without side effects

**Given** the repo is checked out and `.venv` is active  
**When** running:
```python
from quantagent.strategy import STRATEGY_REGISTRY, get_strategy_registry, get_strategy_names, build_strategy
```
**Then** no exception is raised, no LLM connections are made, no DB access occurs.

**Verification:**
```bash
python -c "from quantagent.strategy import STRATEGY_REGISTRY, get_strategy_registry, get_strategy_names, build_strategy; print('OK', list(STRATEGY_REGISTRY.keys()))"
```
Expected output includes all 4 strategy names.

---

## AC-2: Registry contains all 4 strategies

**Given** `get_strategy_registry()` is called  
**Then** the returned dict contains exactly these keys:
- `"RSIMeanReversionStrategy"`
- `"FiftyTwoWeekHighStrategy"`
- `"TripleScreenStrategy"`
- `"LLMAgentStrategy"`

And each entry has: `cls`, `type`, `display_name`, `description`, `params`, `min_bars`.

**Verification:**
```python
def test_registry_has_all_strategies():
    registry = get_strategy_registry()
    expected = {"RSIMeanReversionStrategy", "FiftyTwoWeekHighStrategy", "TripleScreenStrategy", "LLMAgentStrategy"}
    assert set(registry.keys()) == expected
    for name, entry in registry.items():
        assert "cls" in entry
        assert "type" in entry
        assert entry["type"] in ("deterministic", "llm")
        assert "display_name" in entry
        assert "params" in entry
        assert "min_bars" in entry
```

---

## AC-3: Deterministic strategies typed as "deterministic", LLM as "llm"

**Given** the registry  
**Then**:
- `registry["RSIMeanReversionStrategy"]["type"] == "deterministic"`
- `registry["FiftyTwoWeekHighStrategy"]["type"] == "deterministic"`
- `registry["TripleScreenStrategy"]["type"] == "deterministic"`
- `registry["LLMAgentStrategy"]["type"] == "llm"`

---

## AC-4: build_strategy instantiates with custom params

**Given** `build_strategy("RSIMeanReversionStrategy", rsi_period=20, oversold_threshold=25.0)`  
**Then** returns an `RSIMeanReversionStrategy` instance with `rsi_period=20` and `oversold_threshold=25.0`.

**Verification:**
```python
def test_build_strategy_custom_params():
    strategy = build_strategy("RSIMeanReversionStrategy", rsi_period=20, oversold_threshold=25.0)
    assert isinstance(strategy, RSIMeanReversionStrategy)
    assert strategy.rsi_period == 20
    assert strategy.oversold_threshold == 25.0
```

---

## AC-5: build_strategy with defaults

**Given** `build_strategy("RSIMeanReversionStrategy")` called with no extra kwargs  
**Then** returns `RSIMeanReversionStrategy` with default `rsi_period=14`.

---

## AC-6: TradingScheduler accepts any TradingStrategy

**Given** a `TradingScheduler` is instantiated with `strategy=RSIMeanReversionStrategy()`  
**Then** `scheduler.strategy` is an instance of `RSIMeanReversionStrategy`, not `LLMAgentStrategy`.

**Verification:**
```python
def test_scheduler_uses_provided_strategy(mock_trading_graph, mock_order_manager, mock_data_provider, db_session):
    rsi = RSIMeanReversionStrategy()
    scheduler = TradingScheduler(
        trading_graph=mock_trading_graph,
        order_manager=mock_order_manager,
        data_provider=mock_data_provider,
        db_session=db_session,
        strategy=rsi,
    )
    assert isinstance(scheduler.strategy, RSIMeanReversionStrategy)
    assert not isinstance(scheduler.strategy, LLMAgentStrategy)
```

---

## AC-7: TradingScheduler default is LLMAgentStrategy (backward compat)

**Given** a `TradingScheduler` is instantiated with no `strategy=` argument  
**Then** `scheduler.strategy` is an instance of `LLMAgentStrategy`.

---

## AC-8: _process_asset does not raise TypeError with deterministic strategy

**Given** `TradingScheduler` is instantiated with `RSIMeanReversionStrategy`  
**When** `_process_asset(symbol)` is called with mocked market data (sufficient bars)  
**Then** no `TypeError` is raised about unexpected `thread_id` kwarg.

**Verification:**
```python
def test_process_asset_rsi_no_type_error(scheduler_with_rsi, mock_market_data):
    # Should not raise TypeError: generate_signal() got an unexpected keyword argument 'thread_id'
    scheduler_with_rsi._process_asset("BTCUSDT")  # or with mock that returns None signal
```

---

## AC-9: describe() returns strategy metadata

**Given** any concrete strategy class  
**When** `StrategyClass.describe()` is called  
**Then** returns a dict with `name`, `display_name`, `type`, `description`.

**Verification:**
```python
def test_describe_rsi():
    d = RSIMeanReversionStrategy.describe()
    assert d["name"] == "RSIMeanReversionStrategy"
    assert d["type"] == "deterministic"
    assert "display_name" in d

def test_describe_llm():
    d = LLMAgentStrategy.describe()
    assert d["type"] == "llm"
```

---

## AC-10: Registry params match strategy __init__ signatures

**Given** the registry params for each deterministic strategy  
**Then** `build_strategy(name, **{p: v["default"] for p, v in entry["params"].items()})` succeeds for all deterministic strategies.

**Verification:**
```python
def test_all_deterministic_strategies_buildable():
    registry = get_strategy_registry()
    for name, entry in registry.items():
        if entry["type"] != "deterministic":
            continue
        kwargs = {p: v["default"] for p, v in entry["params"].items()}
        strategy = build_strategy(name, **kwargs)
        assert isinstance(strategy, entry["cls"])
```

---

## Manual Smoke Test

```bash
cd /home/azureuser/repos/projects/QuantAgent
source .venv/bin/activate

# 1. Import smoke
python -c "
from quantagent.strategy import STRATEGY_REGISTRY, build_strategy
print('Strategies:', list(STRATEGY_REGISTRY.keys()))
rsi = build_strategy('RSIMeanReversionStrategy', rsi_period=20)
print('RSI period:', rsi.rsi_period)
print('Type:', STRATEGY_REGISTRY['RSIMeanReversionStrategy']['type'])
print('LLM type:', STRATEGY_REGISTRY['LLMAgentStrategy']['type'])
print('OK')
"

# 2. Run tests
pytest tests/test_strategy_registry.py -v

# 3. Compile check
python -m compileall -q quantagent/strategy/ quantagent/trading/scheduler.py
```
