# QuantAgent-c69 — Planning: M1 Strategy 2 — LLMAgentStrategy as Reference Pipeline

**Issue**: QuantAgent-c69  
**Parent**: QuantAgent-l0h (M1 Milestone — Backtesting Estable para 3+ Estrategias)  
**Status**: planning  
**Labels**: m1, strategy, langgraph, backtesting

---

## Objective

Establish `LLMAgentStrategy` + `TradingGraph` as the explicit, stable, and testable M1 Strategy 2. The pipeline already exists; the goal is to stabilize the integration contract, fix a known reasoning-extraction gap, and provide a documented reference backtest profile.

---

## Current State Analysis

### What exists

| Component | File | Status |
|-----------|------|--------|
| `LLMAgentStrategy` | `quantagent/strategy/llm_agent_strategy.py` | Exists, has critical gap |
| `TradingGraph` | `quantagent/trading_graph.py` | Fully functional |
| `SetGraph` (LangGraph) | `quantagent/graph_setup.py` | Fully functional |
| `IndicatorAgentState` | `quantagent/agent_state.py` | Typed state schema |
| Decision agent | `quantagent/decision_agent.py` | Returns `TradingDecision` Pydantic object |
| Backtest engine | `quantagent/backtesting/backtest.py` | Already uses `LLMAgentStrategy` as default |
| Unit tests | `tests/test_llm_agent_strategy.py` | Exist but cover only string-return path |

### Graph topology

```
START
  ├─→ Indicator Agent   (RSI, MACD, ROC, Stochastic, Williams %R)
  ├─→ Pattern Agent     (vision LLM — candlestick chart image)
  └─→ Trend Agent       (support/resistance trendline image)
           ↓  ↓  ↓  (fan-in)
        Decision Maker  → TradingDecision (Pydantic)
              ↓
            END
```

### Critical gap: reasoning is not extracted

`decision_agent.py` returns `{"final_trade_decision": TradingDecision(...)}`.

`TradingDecision` is a Pydantic model with `.decision`, `.confidence`, `.reasoning`, `.risk_level` fields.

`LLMAgentStrategy.generate_signal()` currently does:
```python
trading_decision_raw = result.get("final_trade_decision", "HOLD")
decision, confidence = self._parse_decision(trading_decision_raw)
reasoning = result.get("reasoning", "")         # always empty
decision_report = result.get("decision_report", {})  # always empty
reasoning = decision_report.get("reasoning", "LLM Agent analysis")  # always fallback
```

**Effect**: Signals are persisted with `reasoning = "LLM Agent analysis"` regardless of what the agents produced. No actual reasoning evidence is preserved.

`_parse_decision()` does `str(TradingDecision).upper()`, which accidentally finds "LONG"/"SHORT" in the string representation, but confidence extraction via regex is fragile on Pydantic repr strings.

### Test gap

`tests/test_llm_agent_strategy.py` mocks the graph returning strings like `"LONG with 0.75 confidence"`. The real pipeline returns a `TradingDecision` Pydantic object. The Pydantic-object path is not tested, so the reasoning and confidence extraction failures are not caught.

---

## Scope of Change

### In scope

1. **Fix `LLMAgentStrategy.generate_signal()`**: detect when `final_trade_decision` is a `TradingDecision` instance and extract `.decision`, `.confidence`, `.reasoning` directly.
2. **Improve `_parse_decision()`**: handle both string and `TradingDecision` inputs; keep string fallback for backwards compatibility.
3. **Add deterministic tests**: cover the `TradingDecision` object path — including reasoning extraction and full `TradingSignal` contract validation.
4. **Reference backtest profile**: document minimum config to run a stable reference backtest with `LLMAgentStrategy`.
5. **Documentation**: per-issue PL, AC, and IM docs.

### Out of scope (per issue)

- Changing LLM provider or model routing
- Prompt or fine-tuning changes
- Rewriting `TradingGraph` or `SetGraph`
- Adding new agents or graph nodes

---

## Implementation Steps

### Step 1 — Fix `LLMAgentStrategy.generate_signal()`

**File**: `quantagent/strategy/llm_agent_strategy.py`

In `generate_signal()`, replace the reasoning extraction block:

```python
# Current (broken path):
reasoning = result.get("reasoning", "")
if not reasoning:
    decision_report = result.get("decision_report", {})
    ...
    reasoning = decision_report.get("reasoning", "LLM Agent analysis")

# Fix: check for TradingDecision object first
from quantagent.agent_models import TradingDecision as TradingDecisionModel

final_decision = result.get("final_trade_decision", "HOLD")
if isinstance(final_decision, TradingDecisionModel):
    decision = final_decision.decision.upper()
    confidence = final_decision.confidence
    reasoning = final_decision.reasoning
else:
    # String fallback (legacy path)
    decision, confidence = self._parse_decision(final_decision)
    reasoning = result.get("reasoning", "LLM Agent analysis")
```

This removes the dependency on `_parse_decision()` for the primary path while keeping the string fallback.

### Step 2 — Update `_parse_decision()` (optional guard)

Add a type guard so the method works safely for both input types:

```python
def _parse_decision(self, decision_input) -> tuple[str, float]:
    from quantagent.agent_models import TradingDecision as TradingDecisionModel
    if isinstance(decision_input, TradingDecisionModel):
        return decision_input.decision.upper(), decision_input.confidence
    # existing string parsing logic unchanged
    ...
```

### Step 3 — Add tests in `tests/test_llm_agent_strategy.py`

New test class `TestTradingDecisionObjectPath`:

- `test_generate_signal_with_trading_decision_object`: graph returns `TradingDecision(decision="LONG", confidence=0.8, reasoning="Strong bullish alignment", risk_level="medium")`. Assert `signal.decision == "LONG"`, `signal.confidence == 0.8`, `signal.reasoning == "Strong bullish alignment"`.
- `test_generate_signal_extracts_reasoning_from_pydantic`: verify that when `TradingDecision.reasoning` is non-empty, `signal.reasoning` matches it (not the "LLM Agent analysis" fallback).
- `test_generate_signal_hold_via_pydantic`: graph returns `TradingDecision(decision="HOLD", ...)`. Assert returns `None`.
- `test_parse_decision_accepts_trading_decision_object`: unit test for `_parse_decision` with Pydantic input.

### Step 4 — Reference backtest profile

Document in `docs/06_implementation/QuantAgent-c69-IM-llm-agent-strategy.md`:

Minimum reproducible config for a reference backtest with `LLMAgentStrategy`:

```python
from datetime import datetime
from quantagent.backtesting.backtest import Backtest

backtest = Backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    assets=["BTC-USD"],
    timeframe="4h",
    initial_capital=100_000.0,
    config={
        "agent_llm_provider": "openai",
        "agent_llm_model": "gpt-4o-mini",
        "agent_llm_temperature": 0.1,
    },
    use_checkpointing=False,  # strategy=None → defaults to LLMAgentStrategy
)
metrics = backtest.run(name="QuantAgent-c69-reference")
```

No `strategy=` argument needed — `Backtest` already defaults to `LLMAgentStrategy(trading_graph)`.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| `TradingDecision` type is not returned by graph in all code paths | Low | Medium | Tests with real graph invocation path confirm contract |
| String repr fallback still works if Pydantic changes repr | Low | Low | Explicit isinstance check removes reliance on repr |
| Reasoning field empty in `TradingDecision` (LLM returns empty string) | Medium | Low | Already handled by `or "LLM Agent analysis"` fallback |
| Integration test requires live LLM API | High | Medium | Keep integration test behind marker; unit tests use mock |

---

## Handoff to Implementer

The implementer should:
1. Read this plan and `docs/05_acceptance_tests/QuantAgent-c69-AC-llm-agent-strategy.md`
2. Read `docs/06_implementation/QuantAgent-c69-IM-llm-agent-strategy.md` for technical context
3. Make changes to `quantagent/strategy/llm_agent_strategy.py` and `tests/test_llm_agent_strategy.py` only
4. Run `pytest tests/test_llm_agent_strategy.py -v` to verify all tests pass
5. Run `python -m compileall -q quantagent/strategy/` to verify no syntax errors
