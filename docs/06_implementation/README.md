# Implementation Notes

This folder contains implementation notes, technical details, and post-implementation documentation for specific features and changes.

## Reading Order

Agents working in this repository should read implementation documents relevant to the feature being worked on:

### Current Implementation Notes
- [backtest_data_pipeline_alignment.md](./backtest_data_pipeline_alignment.md) - OHLCV data pipeline alignment and remediation
- [20251130_Implementation_AgentParallelization.md](./20251130_Implementation_AgentParallelization.md) - Agent parallelization implementation details
- [SHORT_POSITIONS_IMPLEMENTATION.md](./SHORT_POSITIONS_IMPLEMENTATION.md) - Short position support implementation
- [IMPLEMENTATION_WEEK5_TRADING_EXECUTION.md](./IMPLEMENTATION_WEEK5_TRADING_EXECUTION.md) - Trading execution layer implementation
- [RISK_MANAGER_DUPLICATION.md](./RISK_MANAGER_DUPLICATION.md) - Risk manager duplication analysis and resolution

## Current Truth

Implementation notes document the actual code changes, technical decisions made during implementation, and any deviations from the original design. When working on related features:

- Review relevant implementation notes to understand what was actually built
- Check for known issues or technical debt mentioned in the notes
- Use these documents to understand implementation patterns and conventions

## Implementation Document Content

Implementation documents should include:

1. **Overview** - What was implemented
2. **Approach** - How it was implemented (key code changes)
3. **Deviations** - Any changes from the original design/plan
4. **Known Issues** - Technical debt or limitations
5. **Testing** - How the implementation was validated

## Active Per-Change Implementation

- [QuantAgent-app-IM-qa-streamlit-cutover.md](./QuantAgent-app-IM-qa-streamlit-cutover.md) - QA Streamlit cutover, Cloudflare tunnel alignment, and deploy_finished -> qa_verified contract.
- [QuantAgent-uzq-IM-fix-scheduler-heartbeat.md](./QuantAgent-uzq-IM-fix-scheduler-heartbeat.md) - Fix TradingScheduler heartbeat and scheduler unit-test regressions exposed by CI gate (P1, bug)
- [QuantAgent-b8r-IM-52week-high-momentum.md](./QuantAgent-b8r-IM-52week-high-momentum.md) - M1 Strategy 3: FiftyTwoWeekHighStrategy (52-week high momentum/breakout for US equities)
- [QuantAgent-c69-IM-llm-agent-strategy.md](./QuantAgent-c69-IM-llm-agent-strategy.md) - M1 Strategy 2: LLMAgentStrategy reference pipeline (m1, strategy)
- [QuantAgent-94d-IM-backtest-isolation.md](./QuantAgent-94d-IM-backtest-isolation.md) - Backtest Run ID Isolation (P1, bug)

## Per-Change Implementation Documents

Per-change implementation documents should follow this naming convention:

```
QuantAgent-<issue-id>-IM-<short-slug>.md
```

Example: `QuantAgent-a3f2dd-IM-fee-calculation.md`

These files describe implementation deltas and technical notes for specific changes linked to Beads issues (when applicable for system implementations).
