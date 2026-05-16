# QuantAgent-375 — Requirements: Scope Replay Signal Lookup to Selected Source Run

**Beads issue:** QuantAgent-375  
**Type:** bug (blocker for QuantAgent-3o8 integration)  
**Priority:** P1  
**Dependency:** QuantAgent-3o8 (Replay execution mode)

---

## Context

`Backtest.run_replay()` (implemented in QuantAgent-3o8) loads stored signals using a
`(symbol, generated_at)` composite key. This key is non-unique across overlapping backtest
runs that share the same symbol, timeframe, and date range. When the `signal_map` is built,
later signals overwrite earlier ones silently, causing a replay to consume signals from a
different source run.

**Proven failure:** proof run on 2026-05-10 showed `source_run_id=1` consuming signal ID 2
from a different overlapping run, violating acceptance criteria TC5 and TC11 of QuantAgent-3o8.

Current signal query in `run_replay()`:

```python
Signal.symbol.in_(source_run.assets),
Signal.timeframe == source_run.timeframe,
Signal.generated_at >= source_run.start_date,
Signal.generated_at <= source_run.end_date,
Signal.environment == Environment.BACKTEST,
```

This does not filter by which run produced the signal. Any backtest run over the same
assets/timeframe/dates will have its signals included in the map, causing non-deterministic
cross-run contamination.

---

## Required Changes

### R1 — Add `backtest_run_id` FK to `Signal`

The `Signal` model must carry a direct reference to the `BacktestRun` that produced it.
This is nullable to preserve compatibility with live/paper signals that are not scoped to
any run.

```
Signal.backtest_run_id  →  BacktestRun.id  (nullable FK, indexed)
```

### R2 — Populate `backtest_run_id` at signal creation time

Both `_create_signal_from_strategy()` and `_create_signal()` in `Backtest` must set
`backtest_run_id = self.backtest_run_id` when `self.backtest_run_id is not None`.

### R3 — Scope replay signal query to source run

`run_replay()` must filter signals by `Signal.backtest_run_id == source_run_id` instead
of (or in addition to) the current date/symbol filters.

### R4 — Alembic migration

A new migration must add the `backtest_run_id` column (nullable, with FK and index) to the
`signals` table without breaking existing rows.

### R5 — Regression tests for cross-run contamination

Add tests that:
- Create two overlapping backtest runs with the same symbol/timeframe/date range
- Verify that replaying run A uses only run A's signals
- Verify that replaying run B uses only run B's signals
- Verify the signal counts are correct (no extra signals from the other run)

---

## Out of Scope

- New replay UI features
- Parallel replay execution
- Broader backtest refactors beyond the signal scoping fix
- Retroactive backfill of `backtest_run_id` for existing rows in production
- Changes to live/paper trading signal creation paths
