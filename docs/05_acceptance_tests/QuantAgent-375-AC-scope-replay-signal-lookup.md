# QuantAgent-375 — Acceptance Tests: Scope Replay Signal Lookup to Selected Source Run

**Beads issue:** QuantAgent-375  
**Format:** Given / When / Then

---

## TC-375-1 — Signal creation populates `backtest_run_id`

**Given** a `Backtest` instance is initialized and `run()` is called  
**When** the strategy generates a signal and `_create_signal_from_strategy()` is invoked  
**Then** the persisted `Signal` record has `backtest_run_id` equal to `self.backtest_run_id`

**Testable:** yes — query `db.query(Signal).filter_by(backtest_run_id=run_id).count()` after
a mock backtest run and verify it matches the number of signals created.

---

## TC-375-2 — Replay of run A loads only run A's signals

**Given** two backtest runs (A and B) have been executed with overlapping symbols, timeframe,
and date range (so both produce signals with identical `(symbol, generated_at)` values)  
**When** `run_replay(source_run_id=A.id)` is called  
**Then** the `signal_map` contains only signals where `Signal.backtest_run_id == A.id`  
**And** no signals with `backtest_run_id == B.id` are present in the map

**Testable:** yes — intercept the `signal_map` after loading or count signals by
`backtest_run_id` in the DB before calling `run_replay()`.

---

## TC-375-3 — Replay of run B loads only run B's signals

**Given** same setup as TC-375-2  
**When** `run_replay(source_run_id=B.id)` is called  
**Then** only signals with `backtest_run_id == B.id` are used  
**And** the signal count matches the number of signals originally produced by run B

**Testable:** yes — same approach as TC-375-2.

---

## TC-375-4 — No regression on non-backtest signals

**Given** signals created in `Environment.PAPER` or `Environment.PROD` (live signals)  
**When** those signals are persisted  
**Then** `Signal.backtest_run_id` is `NULL` (not set)  
**And** those signals are never included in any replay signal query

**Testable:** yes — create a paper signal directly and assert `backtest_run_id is None`.

---

## TC-375-5 — Replay with overlapping runs yields correct metrics isolation

**Given** run A produced 3 signals for BTC on dates T1, T2, T3 (all LONG, high confidence)  
**And** run B produced 3 signals for the same dates but with different signal types (SHORT)  
**When** `run_replay(source_run_id=A.id)` is executed  
**Then** all replay trades are based on LONG signals  
**And** the replay `BacktestRun` record has `replay_source_run_id == A.id`

**Testable:** yes — use mock `TradingStrategy` that returns deterministic signals, verify trade
directions in DB after replay.

---

## TC-375-6 — Migration is safe for existing data

**Given** an existing database with `Signal` rows that have no `backtest_run_id` (pre-migration)  
**When** the migration `add_backtest_run_id_to_signals` is applied  
**Then** all existing rows have `backtest_run_id = NULL` (no data loss)  
**And** the migration can be applied and rolled back without errors

**Testable:** yes — run `alembic upgrade head && alembic downgrade -1` against a DB seeded
with existing signals; verify row count unchanged and no FK violations.

---

## TC-375-7 — Replay against pre-migration run raises ValueError

**Given** a `BacktestRun` exists in the DB that was created before the migration  
**And** its signals all have `backtest_run_id = NULL`  
**When** `run_replay(source_run_id=<that_run_id>)` is called  
**Then** a `ValueError` is raised with a message indicating no signals were found for the run

**Testable:** yes — insert a `BacktestRun` and signals with `backtest_run_id = NULL`, call
`run_replay()`, assert `ValueError`.

---

## Automated Test Location

New test file: `tests/test_replay_signal_scoping.py`

The tests above (TC-375-1 through TC-375-5, TC-375-7) must be automated using SQLite
in-memory DB (no external dependencies). TC-375-6 requires alembic and should be run
manually or in a dedicated migration test.
