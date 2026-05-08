# Planning

This folder contains project roadmaps, task management, and planning artifacts for the QuantAgent system.

## Reading Order

Agents working in this repository should read planning documents in this order:

1. **Current Phase Roadmaps** (start here)
   - [phase1_roadmap.md](./phase1_roadmap.md) - Phase 1 development plan and milestones
   - [phase2_roadmap.md](./phase2_roadmap.md) - Phase 2 development plan and milestones

## Current Truth

The roadmap files represent the current development phases and priorities. When planning work:

- Consult the relevant phase roadmap for context
- Align new tasks with phase objectives
- Use Beads (`bd ready`) to view current backlog and ready tasks

## Active Per-Change Planning

- [QuantAgent-40j-PL-parallel-test-fix.md](./QuantAgent-40j-PL-parallel-test-fix.md) - Fix missing benchmark fixture for parallel execution test (P1, blocks 82t)
- [QuantAgent-c69-PL-llm-agent-strategy-m1.md](./QuantAgent-c69-PL-llm-agent-strategy-m1.md) - M1 Strategy 2: LLMAgentStrategy reference pipeline (m1, strategy)
- [QuantAgent-r78-PL-trade-pnl-calculation.md](./QuantAgent-r78-PL-trade-pnl-calculation.md) - Trade P&L calculation bug fix (P1)
- [QuantAgent-les-PL-commissions-pnl.md](./QuantAgent-les-PL-commissions-pnl.md) - Support commissions in P&L calculation (P3, feature)
- [QuantAgent-nu7-PL-active-position-monitoring.md](./QuantAgent-nu7-PL-active-position-monitoring.md) - Active Position Monitoring System (Epic: QuantAgent-nu7)
- [QuantAgent-8vb-backtest-analysis.md](./QuantAgent-8vb-backtest-analysis.md) - Backtest log analysis (P0 bugs identified)
- [QuantAgent-7bn-PL-azure-openai-support.md](./QuantAgent-7bn-PL-azure-openai-support.md) - Azure OpenAI LLM provider support
- [QuantAgent-ou3-spx-data-fetch.md](./QuantAgent-ou3-spx-data-fetch.md) - SPX/yfinance intraday data fix (P2)
- [QuantAgent-lmn-deprecated-parameter.md](./QuantAgent-lmn-deprecated-parameter.md) - Deprecated wait_sec cleanup (P3)

## Per-Change Planning Documents

Per-change planning documents should follow this naming convention:

```
QuantAgent-<issue-id>-PL-<short-slug>.md
```

Example: `QuantAgent-a3f2dd-PL-backtest-optimization.md`

These files describe planning deltas, task breakdowns, and execution strategies for specific changes linked to Beads issues (when applicable for system implementations).
