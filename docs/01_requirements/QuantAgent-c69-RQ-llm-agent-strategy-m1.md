# QuantAgent-c69 — Requirements: M1 Strategy 2 — LLMAgentStrategy

**Issue:** QuantAgent-c69  
**Parent:** QuantAgent-l0h (AC3)  
**Type:** Requirements

---

## Objective

Convert the existing `LLMAgentStrategy` + `TradingGraph` pipeline into the explicit and testable M1 Strategy 2 reference, preserving the current architecture and fixing only the minimum contract gaps needed for stable backtesting evidence.

---

## Scope In

- Use the existing `LLMAgentStrategy` as the reference wrapper for `TradingGraph`
- Ensure `TradingDecision` output is mapped correctly into `TradingSignal`
- Preserve actual agent reasoning in persisted signal evidence
- Add deterministic tests for the real `TradingDecision` object path
- Document a minimum reproducible backtest profile for this strategy

## Scope Out

- Provider/model routing changes
- Prompt optimization or fine-tuning
- Rewriting the LangGraph topology or adding new agents
- Performance tuning or strategy parameter optimization

---

## Functional Requirements

### FR1 — Explicit M1 reference strategy
The repo must have a documented, explicit path to run the original multi-agent pipeline as M1 Strategy 2 without introducing a new strategy abstraction for this ticket.

### FR2 — Correct decision mapping
When `TradingGraph` returns `final_trade_decision` as a `TradingDecision` object, `LLMAgentStrategy.generate_signal()` must:
- map `decision` to `TradingSignal.decision`
- map `confidence` to `TradingSignal.confidence`
- map `reasoning` to persisted signal reasoning / analysis summary
- keep the existing HOLD behavior (`None`)

### FR3 — Reasoning evidence preserved
Signals produced by the strategy must preserve the actual reasoning emitted by the decision agent instead of collapsing to a static fallback when reasoning is available.

### FR4 — Stable legacy fallback
If the strategy receives the legacy string-shaped decision path, it may keep the current parsing fallback so existing tests and non-Pydantic paths do not break unnecessarily.

### FR5 — Deterministic validation
The issue must include deterministic tests for LONG, SHORT, HOLD, confidence, and reasoning extraction on the `TradingDecision` object path.

### FR6 — Reference backtest
There must be a documented reference backtest profile showing how to run Strategy 2 end-to-end with existing repo capabilities.

---

## Edge Cases

- `TradingDecision.reasoning` empty → allow fallback text
- `TradingDecision.decision == HOLD` → return `None`
- Legacy string decisions must still parse without expanding scope

---

## Definition of Done

- `LLMAgentStrategy` correctly extracts decision/confidence/reasoning from `TradingDecision`
- Deterministic tests cover the real object path
- A reference backtest can be run from documented instructions
- Acceptance remains aligned with `docs/05_acceptance_tests/QuantAgent-c69-AC-llm-agent-strategy.md`
