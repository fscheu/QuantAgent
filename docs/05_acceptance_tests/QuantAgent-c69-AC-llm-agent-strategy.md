# QuantAgent-c69 — Acceptance Criteria: M1 Strategy 2 — LLMAgentStrategy

**Issue**: QuantAgent-c69  
**Parent**: QuantAgent-l0h (AC3)  
**Type**: Acceptance Tests  

---

## AC1 — Documented path to run the multi-agent strategy as M1 Strategy 2

**Given** a developer clones the repo and has valid API keys configured in `.env`  
**When** they follow `docs/06_implementation/QuantAgent-c69-IM-llm-agent-strategy.md`  
**Then** they can instantiate `Backtest` (with no `strategy=` argument) and run it end-to-end without code changes

**Testable**: Manual check against the reference profile in the IM doc.

---

## AC2 — `LLMAgentStrategy.generate_signal()` correctly maps `TradingDecision` to `TradingSignal`

### AC2.1 — LONG signal from `TradingDecision` object

**Given** the graph returns a `TradingDecision(decision="LONG", confidence=0.8, reasoning="Bullish alignment", risk_level="medium")`  
**When** `LLMAgentStrategy.generate_signal()` is called with `current_price=100.0`  
**Then**:
- `signal.decision == "LONG"`
- `signal.confidence == 0.8`
- `signal.reasoning == "Bullish alignment"`
- `signal.entry_price == 100.0`
- `signal.stop_loss == 98.0` (2% below)
- `signal.take_profit == 103.0` (3% above)
- `signal.trailing_stop_pct == 0.05`

### AC2.2 — SHORT signal from `TradingDecision` object

**Given** the graph returns a `TradingDecision(decision="SHORT", confidence=0.65, reasoning="Bearish breakdown", risk_level="high")`  
**When** `LLMAgentStrategy.generate_signal()` is called with `current_price=200.0`  
**Then**:
- `signal.decision == "SHORT"`
- `signal.confidence == 0.65`
- `signal.reasoning == "Bearish breakdown"`
- `signal.stop_loss == 204.0` (2% above)
- `signal.take_profit == 194.0` (3% below)

### AC2.3 — Reasoning is always populated from agent output (not fallback)

**Given** the graph returns a `TradingDecision` with non-empty `.reasoning`  
**When** `generate_signal()` extracts the signal  
**Then** `signal.reasoning` equals the `.reasoning` field from `TradingDecision`, not the static fallback `"LLM Agent analysis"`

### AC2.4 — HOLD returns `None`

**Given** the graph returns a `TradingDecision(decision="HOLD", confidence=0.3, reasoning="Conflicting signals", risk_level="high")`  
**When** `LLMAgentStrategy.generate_signal()` is called  
**Then** the return value is `None`

---

## AC3 — Deterministic tests for pipeline → `TradingSignal` mapping

**Given** `tests/test_llm_agent_strategy.py`  
**When** `pytest tests/test_llm_agent_strategy.py -v` is run  
**Then**:
- All existing tests pass (no regressions)
- New tests for the `TradingDecision` Pydantic-object path pass:
  - LONG path with Pydantic object
  - SHORT path with Pydantic object
  - HOLD path with Pydantic object
  - Reasoning correctly extracted (not fallback)
  - `_parse_decision()` handles `TradingDecision` object input

---

## AC4 — Reference backtest completes without crashes

**Given** a test environment with:
- PostgreSQL or SQLite database configured
- Valid LLM API key in `.env`
- Market data available for the test symbol/period

**When** the reference backtest is run with the profile from `QuantAgent-c69-IM`:
```python
Backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    assets=["BTC-USD"],
    timeframe="4h",
    initial_capital=100_000.0,
    config={"agent_llm_provider": "openai", "agent_llm_model": "gpt-4o-mini", ...},
).run(name="QuantAgent-c69-reference")
```

**Then**:
- `backtest.run()` completes without raising exceptions
- `metrics.total_trades >= 0` (may be 0 if all signals are HOLD)
- `metrics.total_pnl` is a finite float
- Signals with non-empty `analysis_summary` are persisted in the database

---

## AC5 — Evidence of reasoning and decision for inspection

**Given** a completed backtest run (from AC4)  
**When** the `Signal` table is queried for signals from that backtest run  
**Then**:
- `signal.analysis_summary` is populated with the actual reasoning from the agents (not "LLM Agent analysis")
- `signal.confidence` reflects the value from `TradingDecision.confidence`

---

## AC6 — Validates multi-agent and multimodal flow (AC3 of QuantAgent-l0h)

**Given** the LLMAgentStrategy pipeline runs with a vision-capable model for the pattern agent  
**When** `generate_signal()` is called  
**Then**:
- The LangGraph executes all three agents: indicator (tools), pattern (vision), trend (vision)
- The decision agent receives reports from all three and produces `TradingDecision`
- This satisfies AC3 of `QuantAgent-l0h`: "at least one strategy validates the multi-agent and multimodal complete flow"

**Testable**: Via integration test or manual observation of logs with `event_type: "agent_start"` / `"agent_end"` for Indicator, Pattern, Trend, and Decision nodes.

---

## Non-goals (explicitly excluded)

- Performance benchmarks (PnL targets, win rate thresholds)
- LLM response quality evaluation
- Provider-specific behavior tests
