# Acceptance Tests & Criteria

This folder contains acceptance criteria, test oracles, and manual test cases for validating system behavior.

## Reading Order

Agents working in this repository should read acceptance test documents in this order:

### Current Test Cases
- [MVP_MANUAL_TEST_CASES.md](./MVP_MANUAL_TEST_CASES.md) - Manual test cases for MVP validation

## Current Truth

Acceptance test documents define the criteria for validating that features work as intended. When implementing or testing features:

- Review relevant acceptance criteria before implementation
- Execute manual test cases to validate behavior
- Update test cases when requirements change
- Automate manual tests where feasible (see `tests/` directory)

## Active Per-Change Acceptance Criteria

- [QuantAgent-s62-AC-operational-observability.md](./QuantAgent-s62-AC-operational-observability.md) - Acceptance criteria for paper trading observability: scheduler status, positions/PnL, LLM telemetry, env filter in logs
- [QuantAgent-69d-AC-token-time-metrics.md](./QuantAgent-69d-AC-token-time-metrics.md) - Acceptance criteria for LLM token/runtime telemetry persistence and isolation
- [QuantAgent-uzq-AC-fix-scheduler-heartbeat.md](./QuantAgent-uzq-AC-fix-scheduler-heartbeat.md) - Fix TradingScheduler heartbeat and scheduler unit-test regressions exposed by CI gate (P1, blocks 82t)
- [QuantAgent-40j-AC-parallel-test-fix.md](./QuantAgent-40j-AC-parallel-test-fix.md) - Fix missing benchmark fixture for parallel execution test (P1, blocks 82t)
- [QuantAgent-b8r-AC-52week-high-momentum.md](./QuantAgent-b8r-AC-52week-high-momentum.md) - M1 Strategy 3: 52-week high momentum / breakout for US equities (m1, strategy)
- [QuantAgent-c69-AC-llm-agent-strategy.md](./QuantAgent-c69-AC-llm-agent-strategy.md) - M1 Strategy 2: LLMAgentStrategy reference pipeline (m1, strategy)
- [QuantAgent-94d-AC-backtest-isolation.md](./QuantAgent-94d-AC-backtest-isolation.md) - Backtest Run ID Isolation (P1, bug)
- [QuantAgent-r78-AC-trade-pnl-calculation.md](./QuantAgent-r78-AC-trade-pnl-calculation.md) - Trade P&L calculation bug fix (P1)
- [QuantAgent-les-AC-commissions-pnl.md](./QuantAgent-les-AC-commissions-pnl.md) - Support commissions in P&L calculation (P3, feature)
- [QuantAgent-nu7-AC-active-position-monitoring.md](./QuantAgent-nu7-AC-active-position-monitoring.md) - Active Position Monitoring System (Epic: QuantAgent-nu7)
- [QuantAgent-7bn-AC-azure-openai-support.md](./QuantAgent-7bn-AC-azure-openai-support.md) - Azure OpenAI LLM provider support
- [QuantAgent-6t4-AC-structured-output-vision-agents.md](./QuantAgent-6t4-AC-structured-output-vision-agents.md) - Structured output refactor for pattern/trend vision agents
- [QuantAgent-1p7-AC-stategraph-image-paths.md](./QuantAgent-1p7-AC-stategraph-image-paths.md) - Acceptance criteria for disk-backed StateGraph visualization paths
- [QuantAgent-sft-AC-paper-runtime-hardening.md](./QuantAgent-sft-AC-paper-runtime-hardening.md) - Acceptance criteria for stable, observable paper runtime in QA
- [QuantAgent-339-AC-qa-validator-real-runtime.md](./QuantAgent-339-AC-qa-validator-real-runtime.md) - Acceptance criteria for real-runtime post-deploy QA validation

## Acceptance Criteria Format

Acceptance criteria should follow Given/When/Then format:

```
Given [initial context]
When [action or event]
Then [expected outcome]
```

Example:
```
Given a user has an open long position
When the market price drops below the stop-loss level
Then the position should be automatically closed
```

## Per-Change Acceptance Documents

Per-change acceptance documents should follow this naming convention:

```
QuantAgent-<issue-id>-AC-<short-slug>.md
```

Example: `QuantAgent-a3f2dd-AC-risk-limits.md`

These files describe acceptance criteria and test oracles for specific changes linked to Beads issues (when applicable for system implementations).
