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

- [QuantAgent-kkj.9-PL-strategy-selector-ui.md](./QuantAgent-kkj.9-PL-strategy-selector-ui.md) - 5-step implementation plan: CLI args, backtesting selector, paper trading selector, configuration defaults, tests

- [QuantAgent-kkj.11-PL-multi-provider-routing.md](./QuantAgent-kkj.11-PL-multi-provider-routing.md) - Multi-provider routing by role: 5-phase implementation plan (configuration, cost, llm)
- [QuantAgent-s62-PL-operational-observability.md](./QuantAgent-s62-PL-operational-observability.md) - Wire existing heartbeat + telemetry primitives into Streamlit dashboard and logs for paper trading observability (M2)
- [QuantAgent-69d-PL-token-time-metrics.md](./QuantAgent-69d-PL-token-time-metrics.md) - Minimal implementation plan for LLM token/runtime telemetry using existing logs
- [QuantAgent-40j-PL-parallel-test-fix.md](./QuantAgent-40j-PL-parallel-test-fix.md) - Fix missing benchmark fixture for parallel execution test (P1, blocks 82t)
- [QuantAgent-c69-PL-llm-agent-strategy-m1.md](./QuantAgent-c69-PL-llm-agent-strategy-m1.md) - M1 Strategy 2: LLMAgentStrategy reference pipeline (m1, strategy)
- [QuantAgent-r78-PL-trade-pnl-calculation.md](./QuantAgent-r78-PL-trade-pnl-calculation.md) - Trade P&L calculation bug fix (P1)
- [QuantAgent-les-PL-commissions-pnl.md](./QuantAgent-les-PL-commissions-pnl.md) - Support commissions in P&L calculation (P3, feature)
- [QuantAgent-nu7-PL-active-position-monitoring.md](./QuantAgent-nu7-PL-active-position-monitoring.md) - Active Position Monitoring System (Epic: QuantAgent-nu7)
- [QuantAgent-8vb-backtest-analysis.md](./QuantAgent-8vb-backtest-analysis.md) - Backtest log analysis (P0 bugs identified)
- [QuantAgent-7bn-PL-azure-openai-support.md](./QuantAgent-7bn-PL-azure-openai-support.md) - Azure OpenAI LLM provider support
- [QuantAgent-ou3-spx-data-fetch.md](./QuantAgent-ou3-spx-data-fetch.md) - SPX/yfinance intraday data fix (P2)
- [QuantAgent-lmn-deprecated-parameter.md](./QuantAgent-lmn-deprecated-parameter.md) - Deprecated wait_sec cleanup (P3)
- [QuantAgent-6t4-PL-structured-output-vision-agents.md](./QuantAgent-6t4-PL-structured-output-vision-agents.md) - Structured output refactor for pattern/trend vision agents
- [QuantAgent-1p7-PL-stategraph-image-paths.md](./QuantAgent-1p7-PL-stategraph-image-paths.md) - Minimal implementation plan for disk-backed StateGraph visualization artifacts
- [QuantAgent-sft-PL-paper-runtime-hardening.md](./QuantAgent-sft-PL-paper-runtime-hardening.md) - Plan to harden paper trading runtime for stable, observable QA operation
- [QuantAgent-339-PL-qa-validator-real-runtime.md](./QuantAgent-339-PL-qa-validator-real-runtime.md) - Plan to turn the QA validator PoC into real-runtime post-deploy validation
- [QuantAgent-kkj.3-PL-dashboard-environment-aware.md](./QuantAgent-kkj.3-PL-dashboard-environment-aware.md) - Implementation plan: restructure dashboard.py into paper mode (scheduler indicator + run grid + selector) and backtest mode (run grid + selector + metrics)

## Per-Change Planning Documents

Per-change planning documents should follow this naming convention:

```
QuantAgent-<issue-id>-PL-<short-slug>.md
```

Example: `QuantAgent-a3f2dd-PL-backtest-optimization.md`

These files describe planning deltas, task breakdowns, and execution strategies for specific changes linked to Beads issues (when applicable for system implementations).
