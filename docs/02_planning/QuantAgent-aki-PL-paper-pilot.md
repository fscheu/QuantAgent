# QuantAgent-aki — Planning: Paper Trading Pilot

**Issue:** QuantAgent-aki  
**Related:** [RQ](../01_requirements/QuantAgent-aki-RQ-paper-pilot.md) | [AC](../05_acceptance_tests/QuantAgent-aki-AC-paper-pilot.md)

---

## Objective

Execute a bounded paper trading pilot with 1–2 M1 strategies, produce structured operational evidence, and emit a readiness report with an explicit go/no-go recommendation for broker real.

---

## Pilot Scope

| Parameter | Value | Rationale |
|---|---|---|
| Strategies | `RSIMeanReversionStrategy`, `FiftyTwoWeekHighStrategy` | Deterministic, no LLM API cost; well-tested in M1 |
| Universe | SPY, AAPL, MSFT | Small, liquid, yfinance-reliable |
| Cycles | 3 | Minimum viable multi-cycle evidence; repeatable in < 15 min |
| Environment | `paper` | Isolates pilot from backtest history |
| Timeframe | `1h` | Matches scheduler default |
| Lookback | `168h` (7 days) | Enough context for RSI and 52-week calculations |

### Why not LLMAgentStrategy?
LLMAgentStrategy introduces API cost and latency variability that could mask structural runtime issues. It can be added in a follow-up pilot once the deterministic path is confirmed stable.

---

## Preconditions

All of the following must be verified before the pilot runs:

1. **DB accessible**: `DATABASE_URL` resolves and Alembic migrations are current (paper-compatible schema including `SchedulerHeartbeat`, `ActivePosition`, `Trade`, `Order`, `TradeSignal`).
2. **Data provider reachable**: yfinance can fetch at least 7 days of 1h bars for SPY, AAPL, MSFT.
3. **Blocker tickets closed**: QuantAgent-sft ✅, QuantAgent-s62 ✅, QuantAgent-339 ✅ (closed in Beads).

### Known Risks at Planning Time

| Risk | Source | Impact | Mitigation |
|---|---|---|---|
| QA validator for sft was PARTIAL | `result.json` from `20260514T000542Z` | Heartbeat may not persist correctly in paper env | Run pilot against local dev DB, not QA docker; verify heartbeat rows after each cycle |
| QA validator for s62 was PARTIAL | `result.json` from `20260514T000532Z` | Observability wiring may be incomplete | Capture evidence directly from DB queries in pilot script, not from UI |
| QuantAgent-339 tester was BLOCKED | `result.json` from `20260514T023928Z` | QA workflow contract untested | Pilot does not depend on CI/deploy path; runs locally |
| yfinance rate-limiting or stale data | External dependency | Empty bars → no signals | Detect and report; do not fail silently |
| RSI / 52w signals may not fire in 3 cycles | Market conditions | Thin evidence (no trades) | Document as valid pilot outcome: "runtime healthy, signal threshold not met" — still actionable for go/no-go |

---

## Task Breakdown

### Task 1 — Pilot Runbook Document

**File:** `docs/06_implementation/QuantAgent-aki-IM-pilot-runbook.md`

Create a versionable runbook that specifies:
- Exact pilot parameters (universe, strategies, cycles, environment)
- Pre-run checklist (preconditions above)
- How to run the pilot script
- Expected output files and their location
- Success/failure exit criteria

**Estimated effort:** small (30–50 lines)

### Task 2 — Pilot Runner Script

**File:** `scripts/run_paper_pilot.py`

A standalone script (uses shared venv) that:

```
1. Parse CLI args: --cycles N (default 3), --tickers A B C, --output-dir PATH
2. Verify preconditions (DB ping, yfinance test fetch, alembic check)
3. Initialize: TradingGraph, OrderManager, DataProvider, PositionMonitor, TradingScheduler
4. For each cycle 1..N:
   a. Call scheduler's internal cycle method directly (bypassing APScheduler timer)
   b. Capture heartbeat row from DB after cycle
   c. Capture new signals, orders, trades since cycle start
   d. Log cycle summary to stdout
5. Collect aggregate evidence across all cycles
6. Write pilot_evidence.json to --output-dir
7. Write readiness_report.md to --output-dir
8. Exit 0 if no critical errors, exit 1 otherwise
```

**Key implementation note:** Call `scheduler._run_cycle()` or equivalent internal method rather than `scheduler.start()` to avoid APScheduler's async timer. If `_run_cycle` is private, refactor to expose a `run_cycle_once()` public method.

**Estimated effort:** medium (150–200 lines)

### Task 3 — Evidence Capture Helpers

The pilot script should query DB directly after each cycle using SQLAlchemy sessions. Relevant models: `SchedulerHeartbeat`, `Signal`, `TradeSignal`, `Order`, `Trade`, `ActivePosition`.

No new models. No migrations. Read-only queries on existing tables.

### Task 4 — Pilot Execution and Evidence Collection

After Task 2 is implemented:
1. Run the pilot script in the local environment against the dev/paper DB
2. Capture the output: `pilot_evidence.json`, `readiness_report.md`
3. Place artifacts under `docs/envelopes/QuantAgent-aki/<run_id>/`

**Note for implementer:** If the pilot cannot complete 3 cycles due to blockers (DB not ready, data fetch failure, model import error), document the blocker with full traceback in `readiness_report.md` and mark the pilot as BLOCKED, not FAIL.

### Task 5 — Readiness Report

The `readiness_report.md` must follow this structure:

```markdown
# Paper Trading Pilot — Readiness Report

**Date:** ...
**Pilot ID:** QuantAgent-aki-pilot-<timestamp>

## Configuration
- Strategies: ...
- Universe: ...
- Cycles: ...
- Environment: ...

## Cycle Summary
| Cycle | Heartbeat Status | Signals | Orders | Fills | Errors |
|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ... |

## Aggregate Results
- Total signals generated: ...
- Total orders placed: ...
- Total trades filled: ...
- Open positions at end: ...
- Critical errors: ...
- Exceptions/tracebacks: ...

## Cost & Latency (if LLM strategy used)
- Total LLM calls: ...
- Total tokens: ...
- Avg latency (ms): ...
- Approx cost (USD): ...

## Signal → Order → Trade → Position Chain
[ Describe whether the chain is reconstructible from DB ]

## Blockers Detected
[ List each blocker with evidence, or "None detected" ]

## Recommendation
**[ GO | NO-GO | CONDITIONAL GO ]**

Reasoning: ...

Next milestone: ...

Suggested follow-up tickets (if any): ...
```

---

## Implementation Order

1. Task 1 (runbook doc) — no code dependencies
2. Task 2 (pilot script) — depends on runbook scope
3. Task 3 (evidence helpers) — part of Task 2
4. Task 4 (execution) — depends on Task 2
5. Task 5 (report) — depends on Task 4 output

---

## Files to Create or Modify

| File | Action | Notes |
|---|---|---|
| `docs/06_implementation/QuantAgent-aki-IM-pilot-runbook.md` | Create | Versioned runbook |
| `scripts/run_paper_pilot.py` | Create | Pilot runner |
| `quantagent/trading/scheduler.py` | Modify (minimal) | Expose `run_cycle_once()` if needed |
| `docs/envelopes/QuantAgent-aki/<run_id>/pilot_evidence.json` | Create | Machine-readable evidence |
| `docs/envelopes/QuantAgent-aki/<run_id>/readiness_report.md` | Create | Human-readable report |

**No new models, no migrations, no UI changes, no `.env` modifications.**

---

## Recommended Routing

1. `autodev-implementer` — Tasks 1, 2, 3, 4, 5
2. Tech Lead review — validates readiness report and closes QuantAgent-aki
3. If NO-GO: Tech Lead creates follow-up tickets from blocker list before closing M2
