# QuantAgent-c69 — Implementation Guide: M1 Strategy 2 — LLMAgentStrategy

**Issue**: QuantAgent-c69  
**Phase**: Implementation guidance (for implementer)  

---

## Overview

`LLMAgentStrategy` wraps `TradingGraph` (the LangChain/LangGraph multi-agent pipeline) to implement the M1 Strategy 2 reference. This doc covers:
1. The contract gap to fix
2. Minimum code changes required
3. Reference backtest profile
4. How to validate

---

## Contract Gap: `TradingDecision` → `TradingSignal` mapping

### Problem

`decision_agent.py` returns:
```python
{"final_trade_decision": TradingDecision(...), "messages": [...]}
```

where `TradingDecision` is a Pydantic model with `.decision`, `.confidence`, `.reasoning`, `.risk_level`.

`LLMAgentStrategy.generate_signal()` currently reads:
```python
trading_decision_raw = result.get("final_trade_decision", "HOLD")
decision, confidence = self._parse_decision(trading_decision_raw)
reasoning = result.get("reasoning", "")  # always empty — key doesn't exist in graph state
```

**Consequence**: `signal.reasoning` is always `"LLM Agent analysis"` (the static fallback). No actual reasoning evidence is preserved.

### Fix (minimal)

In `quantagent/strategy/llm_agent_strategy.py`, replace the decision/confidence/reasoning extraction block:

```python
from quantagent.agent_models import TradingDecision as TradingDecisionModel

final_decision_raw = result.get("final_trade_decision", "HOLD")

if isinstance(final_decision_raw, TradingDecisionModel):
    decision = final_decision_raw.decision.upper()
    confidence = final_decision_raw.confidence
    reasoning = final_decision_raw.reasoning or "LLM Agent analysis"
else:
    # String fallback — kept for test compatibility and legacy paths
    decision, confidence = self._parse_decision(final_decision_raw)
    reasoning = result.get("reasoning", "LLM Agent analysis")
```

Remove the `_parse_decision(trading_decision_raw)` call from the primary path. The `_parse_decision` method can remain for the string fallback.

### Optional: guard `_parse_decision` for Pydantic input

```python
def _parse_decision(self, decision_input) -> tuple[str, float]:
    from quantagent.agent_models import TradingDecision as TradingDecisionModel
    if isinstance(decision_input, TradingDecisionModel):
        return decision_input.decision.upper(), decision_input.confidence
    # existing string-parsing logic unchanged below
    ...
```

---

## Tests to add in `tests/test_llm_agent_strategy.py`

Add a new test class after the existing ones:

```python
class TestTradingDecisionObjectPath:
    """Tests for when graph returns TradingDecision Pydantic object (real pipeline path)."""

    def test_generate_signal_with_trading_decision_object(
        self, llm_strategy, mock_trading_graph, sample_kline_data
    ):
        """Graph returns TradingDecision object — extracts decision, confidence, reasoning."""
        from quantagent.agent_models import TradingDecision

        mock_trading_graph.graph.invoke.return_value = {
            "final_trade_decision": TradingDecision(
                decision="LONG",
                confidence=0.8,
                reasoning="RSI oversold, MACD crossover confirmed, upward trend",
                risk_level="medium",
            ),
        }

        signal = llm_strategy.generate_signal(sample_kline_data, "BTC-USD", "4h", 100.0)

        assert signal is not None
        assert signal.decision == "LONG"
        assert signal.confidence == 0.8
        assert signal.reasoning == "RSI oversold, MACD crossover confirmed, upward trend"
        assert signal.entry_price == 100.0
        assert signal.stop_loss == 98.0
        assert signal.take_profit == 103.0

    def test_generate_short_via_pydantic(
        self, llm_strategy, mock_trading_graph, sample_kline_data
    ):
        """SHORT signal via TradingDecision object."""
        from quantagent.agent_models import TradingDecision

        mock_trading_graph.graph.invoke.return_value = {
            "final_trade_decision": TradingDecision(
                decision="SHORT",
                confidence=0.65,
                reasoning="Bearish breakdown with pattern confirmation",
                risk_level="high",
            ),
        }

        signal = llm_strategy.generate_signal(sample_kline_data, "BTC-USD", "4h", 200.0)

        assert signal is not None
        assert signal.decision == "SHORT"
        assert signal.confidence == 0.65
        assert signal.reasoning == "Bearish breakdown with pattern confirmation"
        assert signal.stop_loss == 204.0
        assert signal.take_profit == 194.0

    def test_hold_via_pydantic_returns_none(
        self, llm_strategy, mock_trading_graph, sample_kline_data
    ):
        """HOLD via TradingDecision object returns None."""
        from quantagent.agent_models import TradingDecision

        mock_trading_graph.graph.invoke.return_value = {
            "final_trade_decision": TradingDecision(
                decision="HOLD",
                confidence=0.3,
                reasoning="Conflicting signals, insufficient alignment",
                risk_level="high",
            ),
        }

        signal = llm_strategy.generate_signal(sample_kline_data, "BTC-USD", "4h", 100.0)

        assert signal is None

    def test_reasoning_not_fallback_when_pydantic(
        self, llm_strategy, mock_trading_graph, sample_kline_data
    ):
        """Reasoning must come from TradingDecision, not static fallback."""
        from quantagent.agent_models import TradingDecision

        specific_reasoning = "MACD histogram expanding positive, RSI at 45 with upward momentum"
        mock_trading_graph.graph.invoke.return_value = {
            "final_trade_decision": TradingDecision(
                decision="LONG",
                confidence=0.75,
                reasoning=specific_reasoning,
                risk_level="low",
            ),
        }

        signal = llm_strategy.generate_signal(sample_kline_data, "BTC-USD", "4h", 150.0)

        assert signal is not None
        assert signal.reasoning == specific_reasoning
        assert signal.reasoning != "LLM Agent analysis"
```

---

## Reference Backtest Profile

The minimum config to run `LLMAgentStrategy` as a backtest with no custom strategy argument:

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
    use_checkpointing=False,
    # strategy=None → defaults to LLMAgentStrategy(TradingGraph(...))
)
metrics = backtest.run(name="QuantAgent-c69-reference")

print(f"Trades: {metrics.total_trades}")
print(f"Win Rate: {metrics.win_rate:.2%}")
print(f"Total PnL: ${metrics.total_pnl:,.2f}")
print(f"Total Return: {metrics.total_return_pct:.2%}")
```

**Prerequisites**:
- `.env` with `OPENAI_API_KEY` set (or equivalent provider keys)
- Database configured (SQLite for local dev is sufficient)
- Internet access for `yfinance` data fetch

**Note**: `Backtest` line 158 already defaults to `LLMAgentStrategy(self.trading_graph)` when `strategy=None`. No code change required to use it.

---

## Validation Commands

```bash
# Unit tests (no live LLM needed)
pytest tests/test_llm_agent_strategy.py -v

# Syntax check
python -m compileall -q quantagent/strategy/llm_agent_strategy.py

# Full test suite (no regressions)
pytest tests/ -v --ignore=tests/test_backtest_integration.py -x
```

---

## Files to Modify

| File | Change |
|------|--------|
| `quantagent/strategy/llm_agent_strategy.py` | Fix `generate_signal()` to extract from `TradingDecision` Pydantic object |
| `tests/test_llm_agent_strategy.py` | Add `TestTradingDecisionObjectPath` class |

No other files need modification. `Backtest` already defaults to `LLMAgentStrategy`.

---

## Known Limitations (out of scope for this issue)

- `LLMAgentStrategy.should_reevaluate()` always returns `False` — this is by design (no re-evaluation once in position)
- SL/TP values (2%/3% fixed) are hardcoded — refinement is future work, not M1 scope
- The graph does not stream intermediate agent outputs; full pipeline completes before signal is returned
