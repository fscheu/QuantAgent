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

- [QuantAgent-94d-AC-backtest-isolation.md](./QuantAgent-94d-AC-backtest-isolation.md) - Backtest Run ID Isolation (P1, bug)
- [QuantAgent-r78-AC-trade-pnl-calculation.md](./QuantAgent-r78-AC-trade-pnl-calculation.md) - Trade P&L calculation bug fix (P1)
- [QuantAgent-nu7-AC-active-position-monitoring.md](./QuantAgent-nu7-AC-active-position-monitoring.md) - Active Position Monitoring System (Epic: QuantAgent-nu7)
- [QuantAgent-7bn-AC-azure-openai-support.md](./QuantAgent-7bn-AC-azure-openai-support.md) - Azure OpenAI LLM provider support
- [QuantAgent-69d-AC-token-time-metrics.md](./QuantAgent-69d-AC-token-time-metrics.md) - Tracking de tokens y tiempo de ejecución para control de costos

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
