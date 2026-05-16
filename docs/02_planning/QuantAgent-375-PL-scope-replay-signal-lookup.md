# QuantAgent-375 — Planning: Scope Replay Signal Lookup to Selected Source Run

**Beads issue:** QuantAgent-375  
**Blocks:** QuantAgent-3o8 re-integration  
**Branch base:** `main` (fresh branch required — 3o8 branch is 207 commits behind)

---

## Objective

Add a direct `backtest_run_id` FK to `Signal` so that `run_replay()` can scope its signal
query to the exact source run, eliminating cross-run contamination.

---

## Implementation Steps

### Step 1 — Schema change: add `backtest_run_id` to `Signal`

**File:** `quantagent/models.py`

Add to `Signal`:

```python
backtest_run_id = Column(
    Integer, ForeignKey("backtest_runs.id"), nullable=True, index=True
)
backtest_run = relationship("BacktestRun", foreign_keys=[backtest_run_id])
```

Add to `Signal.__table_args__`:

```python
Index("idx_signals_backtest_run_id", "backtest_run_id"),
```

Add to `BacktestRun` relationships:

```python
signals = relationship("Signal", foreign_keys="Signal.backtest_run_id", back_populates="backtest_run")
```

**Why nullable:** Live/paper signals are not scoped to any backtest run. Nullable preserves
compatibility with existing rows and non-backtest flows.

---

### Step 2 — Alembic migration

**File:** `alembic/versions/<hash>_add_backtest_run_id_to_signals.py`

```python
def upgrade() -> None:
    op.add_column(
        "signals",
        sa.Column("backtest_run_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_signals_backtest_run_id",
        "signals",
        "backtest_runs",
        ["backtest_run_id"],
        ["id"],
    )
    op.create_index("idx_signals_backtest_run_id", "signals", ["backtest_run_id"])


def downgrade() -> None:
    op.drop_index("idx_signals_backtest_run_id", table_name="signals")
    op.drop_constraint("fk_signals_backtest_run_id", "signals", type_="foreignkey")
    op.drop_column("signals", "backtest_run_id")
```

---

### Step 3 — Populate `backtest_run_id` at signal creation

**File:** `quantagent/backtesting/backtest.py`

In `_create_signal_from_strategy()`, add:

```python
backtest_run_id=self.backtest_run_id,
```

In `_create_signal()`, add the same field.

Note: `self.backtest_run_id` is set by `_create_backtest_run()` before any signal is ever
created, so no ordering issue exists.

---

### Step 4 — Scope `run_replay()` signal query

**File:** `quantagent/backtesting/backtest.py` (on the 3o8 feature branch)

Replace the broad date+asset filter with a direct run-scoped query:

```python
signals = (
    self.db.query(Signal)
    .filter(Signal.backtest_run_id == source_run_id)
    .all()
)
```

This is simpler, faster (single indexed FK lookup), and unambiguous.

---

### Step 5 — Regression tests

**File:** `tests/test_replay_signal_scoping.py` (new file)

Tests to cover:
- TC-375-1: Two overlapping runs with identical symbol/timeframe/date; replay of run A loads
  only run A's signals (count matches, no run B signal IDs present).
- TC-375-2: Replay of run B similarly isolated.
- TC-375-3: `Signal.backtest_run_id` is set correctly after `_create_signal_from_strategy()`.
- TC-375-4: Non-backtest signals (live/paper) retain `backtest_run_id = None`.

---

## Branch Strategy

- New branch: `feature/QuantAgent-375-scope-replay-signal-lookup`
- Base: current `main`
- QuantAgent-3o8 changes must be cherry-picked or re-applied on top after 375 is merged

---

## Files Changed

| File | Change |
|------|--------|
| `quantagent/models.py` | Add `backtest_run_id` FK + index to `Signal`; add relationship to `BacktestRun` |
| `alembic/versions/<hash>_add_backtest_run_id_to_signals.py` | New migration |
| `quantagent/backtesting/backtest.py` | Populate field in `_create_signal_from_strategy()` and `_create_signal()` |
| `quantagent/backtesting/backtest.py` (3o8 branch) | Update `run_replay()` query |
| `tests/test_replay_signal_scoping.py` | New regression tests |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Existing tests that create `Signal` objects without `backtest_run_id` break | Column is nullable; no existing test will break |
| Migration fails on existing DB with live data | Nullable add-column is always safe; FK is not enforced for NULLs in SQLite/Postgres |
| 3o8 branch cherry-pick conflicts after 375 lands | 3o8 branch must be rebased onto main after 375 merges; the `run_replay()` query change is small and conflict-free |

---

## Handoff to Implementer

The implementer receives this document plus:
- `docs/03_design/QuantAgent-375-DS-scope-replay-signal-lookup.md`
- `docs/05_acceptance_tests/QuantAgent-375-AC-scope-replay-signal-lookup.md`

Steps 1–4 are independent of the 3o8 branch. Step 4 only applies when 3o8 is rebased;
the implementer should note this clearly in their run-report.
