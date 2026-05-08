# Requirements

This folder contains functional requirements and UI specifications for the QuantAgent system.

## Reading Order

Agents working in this repository should read requirements documents in this order:

1. **Current System Requirements** (foundational)
   - [trading_system_requirements.md](./trading_system_requirements.md) - Core trading system functional requirements
   - [ui_streamlit_mvp_requirements.md](./ui_streamlit_mvp_requirements.md) - Streamlit UI MVP specifications

2. **Future Requirements** (reference)
   - [ideas_post_mvp.md](./ideas_post_mvp.md) - Post-MVP feature ideas and enhancements

## Current Truth

The files listed above represent the current functional requirements for the system. When implementing features:

- Verify requirements against these documents
- Ensure UI changes align with `ui_streamlit_mvp_requirements.md`
- Consider post-MVP ideas only when explicitly requested

## Active Per-Change Requirements

- [QuantAgent-40j-RQ-parallel-test-fix.md](./QuantAgent-40j-RQ-parallel-test-fix.md) - Fix missing benchmark fixture for parallel execution test (P1, blocks 82t)
- [QuantAgent-4w4-RQ-lookback-windows.md](./QuantAgent-4w4-RQ-lookback-windows.md) - Backtest must honor strategy-specific lookback windows (P1, blocks b8r)
- [QuantAgent-b8r-RQ-52week-high-momentum.md](./QuantAgent-b8r-RQ-52week-high-momentum.md) - M1 Strategy 3: 52-week high momentum / breakout for US equities
- [QuantAgent-c69-RQ-llm-agent-strategy-m1.md](./QuantAgent-c69-RQ-llm-agent-strategy-m1.md) - M1 Strategy 2: LLMAgentStrategy as explicit reference pipeline
- [QuantAgent-94d-RQ-backtest-isolation.md](./QuantAgent-94d-RQ-backtest-isolation.md) - Backtest Run ID Isolation (P1, bug)
- [QuantAgent-r78-RQ-trade-pnl-calculation.md](./QuantAgent-r78-RQ-trade-pnl-calculation.md) - Trade P&L calculation bug fix (P1)
- [QuantAgent-les-RQ-commissions-pnl.md](./QuantAgent-les-RQ-commissions-pnl.md) - Support commissions in P&L calculation (P3, feature)
- [QuantAgent-nu7-RQ-active-position-monitoring.md](./QuantAgent-nu7-RQ-active-position-monitoring.md) - Active Position Monitoring System (Epic: QuantAgent-nu7)
- [QuantAgent-7bn-RQ-azure-openai-support.md](./QuantAgent-7bn-RQ-azure-openai-support.md) - Azure OpenAI LLM provider support

## Per-Change Requirements

Per-change requirement documents should follow this naming convention:

```
QuantAgent-<issue-id>-RQ-<short-slug>.md
```

Example: `QuantAgent-a3f2dd-RQ-fee-tracking.md`

These files describe requirement deltas introduced by specific changes and are linked to Beads issues (when applicable for system implementations).
