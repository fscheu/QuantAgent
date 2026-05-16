# QuantAgent-375 — Design: Scope Replay Signal Lookup to Selected Source Run

**Beads issue:** QuantAgent-375  
**References:** `docs/03_design/backtesting_engine.md`, QuantAgent-3o8 design

---

## Problem Statement

`Backtest.run_replay()` (QuantAgent-3o8) queries signals by asset+timeframe+date+environment.
This key is non-unique: two backtest runs over the same symbols and date range produce signals
with identical `(symbol, generated_at)` tuples. When both runs are present in the DB, the
signal map is overwritten, causing replay to consume signals from the wrong source run.

---

## Design Decision

**Add an explicit `backtest_run_id` FK to `Signal`.**

This is the minimal, direct approach:
- One new nullable column with FK + index
- One new lookup path in `run_replay()`
- No changes to non-backtest code paths

### Why not use `thread_id`?

`thread_id` encodes a LangGraph checkpoint identity, not a backtest run scope. Not all
signals have a `thread_id` (checkpointing is optional). Repurposing it would couple two
unrelated concerns.

### Why not use a composite unique constraint?

Adding `(symbol, generated_at, backtest_run_id)` as a unique key would require the
`signal_map` lookup to change anyway, and would add overhead to every signal insert. The
direct FK is simpler and sufficient.

---

## Schema Delta

### `signals` table

```
backtest_run_id  INTEGER  NULLABLE  FK → backtest_runs.id  INDEX
```

Existing rows: `NULL` (no backfill needed; NULL means "not scoped to a backtest run").

### `Signal` ORM model

```python
backtest_run_id = Column(
    Integer, ForeignKey("backtest_runs.id"), nullable=True, index=True
)
backtest_run = relationship(
    "BacktestRun",
    foreign_keys=[backtest_run_id],
    back_populates="signals",
)
```

### `BacktestRun` ORM model (reverse relationship)

```python
signals = relationship(
    "Signal",
    foreign_keys="Signal.backtest_run_id",
    back_populates="backtest_run",
)
```

---

## Updated Signal Creation

`_create_signal_from_strategy()` and `_create_signal()` in `backtest.py` both create
`Signal` objects. Both must receive `backtest_run_id=self.backtest_run_id`.

`self.backtest_run_id` is guaranteed to be set before any signal is created because
`_create_backtest_run()` is called at the start of `run()` and `run_replay()`, which is the
only entry point that ever calls these two methods.

---

## Updated Replay Query

Current (buggy):

```python
signals = (
    self.db.query(Signal)
    .filter(
        Signal.symbol.in_(source_run.assets),
        Signal.timeframe == source_run.timeframe,
        Signal.generated_at >= source_run.start_date,
        Signal.generated_at <= source_run.end_date,
        Signal.environment == Environment.BACKTEST,
    )
    .all()
)
```

Correct (after fix):

```python
signals = (
    self.db.query(Signal)
    .filter(Signal.backtest_run_id == source_run_id)
    .all()
)
```

This is also simpler. Signals stored with the new `backtest_run_id` field do not need any
additional date/asset/environment filters because the FK to the source run already uniquely
determines the scope.

**Backward compatibility note:** Signals created before this migration will have
`backtest_run_id = NULL`. A replay attempted against such a run will return zero signals and
raise `ValueError`. This is acceptable — those runs were created before replay was supported.

---

## Provenance Chain After Fix

```
BacktestRun (id=1, source)
  └── Signal (id=5, backtest_run_id=1, symbol=BTC, generated_at=T)
        └── Order (trigger_signal_id=5)
              └── Trade

BacktestRun (id=2, replay of run 1)
  replay_source_run_id = 1
  └── signal_map loaded: Signal WHERE backtest_run_id = 1
        → Signal.id=5 only; Signal.id=12 (from run 3) is excluded
```

---

## Migration

The migration file must:
1. Add nullable `backtest_run_id` column to `signals`
2. Add FK constraint `fk_signals_backtest_run_id`
3. Add index `idx_signals_backtest_run_id`

Downgrade reverses all three steps in reverse order.
