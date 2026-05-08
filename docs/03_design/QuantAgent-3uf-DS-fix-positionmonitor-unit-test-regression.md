# QuantAgent-3uf — Design: Fix PositionMonitor Unit-Test Regressions

**Issue ID:** QuantAgent-3uf
**Type:** Bug / Test Infrastructure

---

## Context

See requirements: `docs/01_requirements/QuantAgent-3uf-RQ-fix-positionmonitor-unit-test-regression.md`

The 4 failing tests are caused by test infrastructure problems, not by bugs in
`PositionMonitor` business logic. The design changes are minimal and surgical.

---

## Change 1: Remove local `db_session` from both test files

### Problem

Both test files define a `db_session` fixture locally that wins over the centralized
one in `conftest.py` whenever `DATABASE_URL` is set:

```python
# In test_position_monitor.py AND test_position_monitor_constraints.py
@pytest.fixture
def db_session():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        engine = create_engine(database_url)  # ← uses PostgreSQL in CI
    else:
        engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    db = TestSession()
    yield db
    db.close()  # ← commits are NOT rolled back; data persists
```

`conftest.py` already has the correct fixture (added by `QuantAgent-4ch`):
- Uses SQLite in-memory → function-scoped isolation
- FK constraints enabled via `PRAGMA foreign_keys=ON`
- Drops all tables on teardown

### Fix

Delete the entire `db_session` fixture block (fixture + `position_monitor` fixture that
depends on it) from both test files. The `conftest.py` fixture will be discovered
automatically by pytest.

**Also remove** unused imports: `os`, `create_engine`, `sessionmaker` — these were
only used by the local fixture.

Note: The `position_monitor` fixture in both test files creates `PositionMonitor(db_session)`.
This does NOT need to change — pytest will inject the `conftest.py` session.

Actually the `position_monitor` fixture is local and wraps `db_session` — it must remain.
Only the `db_session` definition is removed; the `position_monitor` fixture stays.

### File changes

**`tests/test_position_monitor.py`**: remove lines:
```python
import os
# ...
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# ...
@pytest.fixture
def db_session():
    """Create database session for testing using DATABASE_URL if available."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        engine = create_engine(database_url)
    else:
        # Fallback to SQLite for local development
        engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    db = TestSession()
    yield db
    db.close()
```

Same removal in **`tests/test_position_monitor_constraints.py`**.

---

## Change 2: Fix FK violation in `test_position_with_all_optional_fields`

### Problem

The test uses `trade_id=123, signal_id=456` — hardcoded integers that are FKs to
`trades.id` and `signals.id`. No parent records exist → PostgreSQL raises
`ForeignKeyViolation`. With SQLite + FK enforcement, this also fails.

### Fix

Create minimal `Trade` and `Signal` records before opening the position, then use
their auto-assigned IDs. Minimum required fields (non-nullable, no default):

**Trade** (`trades` table):
- `symbol` (String)
- `entry_price` (Numeric)
- `quantity` (Numeric)
- `side` (Enum OrderSide)
- `environment` (Enum Environment, default=PAPER)
- `opened_at` (DateTime, default=utcnow)

**Signal** (`signals` table):
- `symbol` (String)
- `signal` (Enum TradeSignal)
- `confidence` (Float)
- `timeframe` (String)
- `environment` (Enum Environment, default=PAPER)
- `generated_at` (DateTime, default=utcnow)

```python
def test_position_with_all_optional_fields(position_monitor, db_session):
    """Validate position creation with all optional fields set."""
    from quantagent.models import Trade, Signal, TradeSignal, OrderSide as OS, Environment
    from decimal import Decimal as D

    trade = Trade(
        symbol="BTCUSDT",
        entry_price=100.0,
        quantity=D("1.0"),
        side=OS.BUY,
    )
    signal = Signal(
        symbol="BTCUSDT",
        signal=TradeSignal.LONG,
        confidence=0.8,
        timeframe="1h",
    )
    db_session.add_all([trade, signal])
    db_session.flush()  # populate IDs without committing

    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.TRAILING_STOP,
        trailing_stop_pct=0.05,
        max_hold_candles=10,
        prediction_horizon=5,
        trade_id=trade.id,
        signal_id=signal.id,
    )

    assert position.trailing_stop_pct == 0.05
    assert position.max_hold_candles == 10
    assert position.prediction_horizon == 5
    assert position.trade_id == trade.id
    assert position.signal_id == signal.id
```

Note: `db_session.flush()` sends the INSERT to the DB within the current transaction
(populating auto-increment IDs) without issuing a COMMIT. Since `open_position` then
does its own `db_session.commit()`, all records commit together.

---

## Change 3: Add `ORDER BY` to `get_active_position`

### Problem

`get_active_position` uses `.first()` with no ordering:
```python
return query.first()
```

In PostgreSQL and SQLite, without ORDER BY the result is engine-dependent and may vary
across runs. When a test directly inserts a second position with the same symbol, which
record `.first()` returns is unpredictable.

### Fix

Add `.order_by(ActivePosition.id)` (ascending) to return the oldest active position
consistently:

```python
return query.order_by(ActivePosition.id).first()
```

**Rationale for ASC (oldest first):** The test `test_get_active_position_returns_most_recent_if_multiple`
asserts `active.id == pos1.id` where pos1 is the first-created position. ASC ordering
matches this expectation and is the safest default (if duplicate active positions exist,
the system was already in violation; returning the oldest is conservative).

This is a one-line change in `quantagent/trading/position_monitor.py`.

---

## Non-Changes

- `tests/conftest.py` — no changes (already correct)
- `quantagent/models.py` — no changes
- CI workflow — no changes
- Other test files — no changes

---

## Dependency Chain

Changes are independent. Recommended execution order:

1. Remove local `db_session` from both test files (unblocks 3 of 4 failures)
2. Fix FK in `test_position_with_all_optional_fields` (fixes 1 remaining failure)
3. Add `ORDER BY` to `get_active_position` (stabilizes behavior for edge-case test)
