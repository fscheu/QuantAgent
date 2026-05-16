# QuantAgent-aki — Requirements: Paper Trading Pilot

**Issue:** QuantAgent-aki  
**Parent:** QuantAgent-kkj (M2 Milestone — Paper Trading Operativo)  
**Created:** 2026-05-14  
**Related:** [PL](../02_planning/QuantAgent-aki-PL-paper-pilot.md) | [AC](../05_acceptance_tests/QuantAgent-aki-AC-paper-pilot.md)

---

## Context

M2 (Paper Trading Operativo) cannot be closed with merged features alone — it requires operational evidence. The blocking dependencies (QuantAgent-sft, QuantAgent-s62, QuantAgent-339) are now closed. The next step is a controlled, bounded paper trading pilot that produces concrete evidence to support a go/no-go decision about advancing to a real broker.

## Functional Requirements

### FR1 — Versioned Runbook
A runbook document must be committed under `docs/` specifying:
- Pilot universe (tickers)
- Strategies used (M1 set only)
- Number of cycles / time window
- Success and failure exit criteria
- Preconditions that must hold before the pilot starts

### FR2 — Pilot Execution
A pilot runner script must be available that:
- Initializes the paper trading system with pilot-specific settings
- Drives `N = 3` scheduler cycles programmatically (not time-gated by APScheduler)
- Collects evidence from each cycle (heartbeat rows, signals, orders, trades, positions)
- Emits a machine-readable evidence file (`pilot_evidence.json`) and a human-readable operational summary (`readiness_report.md`)

### FR3 — Evidence Content
After pilot execution, the evidence must include:
- Per-cycle: heartbeat status, signals generated (per ticker), orders created, trades filled
- Aggregate: total signals, fills, open positions, errors/exceptions
- LLM cost/latency: if LLM strategy is selected, token and latency totals from `llm_telemetry`
- Data fetch: tickers resolved, bars retrieved, data gaps or fetch errors

### FR4 — Readiness Report
The readiness report must state:
- **Go**: if pilot completed N cycles without critical errors and the signal→order→trade→position chain is intact
- **No-go / Conditional go**: if blockers were detected, each blocker must be described with enough detail to become a separate backlog ticket
- **Recommendation**: explicit next milestone recommendation (M3 paper continuity or further M2 stabilization)

## Non-Functional Requirements

- The pilot must be runnable in the local development environment (not QA docker) using a populated `paper` environment DB and real yfinance data
- The pilot script must be idempotent: repeated runs should not corrupt existing paper state (use isolated pilot session or scoped cleanup)
- Execution must complete in under 15 minutes wall-clock time
- No real capital, no live broker, no production service mutations

## Out of Scope

- New trading strategies beyond the M1 set
- Continuous / unattended operation
- Real broker integration (Alpaca or similar)
- UI changes
- Replay functionality (QuantAgent-375)
