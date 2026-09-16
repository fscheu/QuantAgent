"""Tests for backtest run isolation logic."""

import math
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from quantagent.backtesting.backtest import Backtest
from quantagent.models import (
    ActivePosition,
    BacktestRun,
    Environment,
    ExitPolicy,
    MarketData,
    OrderSide,
    Trade,
)
from quantagent.strategy.rsi_strategy import RSIMeanReversionStrategy
from quantagent.trading.position_monitor import PositionMonitor


def _create_backtest_run(session, name: str) -> BacktestRun:
    now = datetime.utcnow()
    run = BacktestRun(
        name=name,
        timeframe="1h",
        assets=["BTC"],
        start_date=now - timedelta(days=1),
        end_date=now,
        config_snapshot={},
    )
    session.add(run)
    session.commit()
    return run

def test_position_monitor_isolates_by_backtest_run_id(db_session):
    run_a = _create_backtest_run(db_session, "run-a")
    run_b = _create_backtest_run(db_session, "run-b")

    monitor_a = PositionMonitor(db_session, backtest_run_id=run_a.id)
    monitor_b = PositionMonitor(db_session, backtest_run_id=run_b.id)

    monitor_a.open_position(
        symbol="BTC",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )
    monitor_b.open_position(
        symbol="BTC",
        side=OrderSide.SELL,
        entry_price=120.0,
        stop_loss=125.0,
        take_profit=110.0,
        quantity=Decimal("1"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    pos_a = monitor_a.get_active_position("BTC")
    pos_b = monitor_b.get_active_position("BTC")

    assert pos_a is not None
    assert pos_b is not None
    assert pos_a.backtest_run_id == run_a.id
    assert pos_b.backtest_run_id == run_b.id

    # Cross-check: monitor A should not see run B's position
    monitor_a.backtest_run_id = run_a.id
    isolated = monitor_a.get_active_position("BTC")
    assert isolated.id == pos_a.id

def test_backtest_metrics_scope_to_current_run(db_session):
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=1)

    run_a = _create_backtest_run(db_session, "metrics-run-a")
    run_b = _create_backtest_run(db_session, "metrics-run-b")

    pos_run_a = ActivePosition(
        symbol="BTC",
        side=OrderSide.BUY,
        entry_price=Decimal("100"),
        stop_loss=Decimal("95"),
        take_profit=Decimal("110"),
        quantity=Decimal("1"),
        decision_timestamp=start,
        candles_since_entry=0,
        exit_policy=ExitPolicy.SL_TP_ONLY,
        prediction_horizon=2,
        candles_direction=["up", "up"],
        is_active=False,
        close_reason="close_tp",
        environment=Environment.BACKTEST,
        backtest_run_id=run_a.id,
    )

    pos_run_b = ActivePosition(
        symbol="BTC",
        side=OrderSide.BUY,
        entry_price=Decimal("200"),
        stop_loss=Decimal("190"),
        take_profit=Decimal("210"),
        quantity=Decimal("1"),
        decision_timestamp=start,
        candles_since_entry=0,
        exit_policy=ExitPolicy.SL_TP_ONLY,
        prediction_horizon=2,
        candles_direction=["down", "down"],
        is_active=False,
        close_reason="close_sl",
        environment=Environment.BACKTEST,
        backtest_run_id=run_b.id,
    )

    db_session.add_all([pos_run_a, pos_run_b])
    db_session.commit()

    backtest = Backtest(
        start_date=start,
        end_date=end,
        assets=["BTC"],
        timeframe="1h",
        db_session=db_session,
    )
    backtest.backtest_run_id = run_a.id

    mda, accuracy_by_candle = backtest._calculate_directional_accuracy()
    close_reasons = backtest._calculate_close_reasons()

    assert mda == 1.0
    assert accuracy_by_candle == {1: 1.0, 2: 1.0}
    assert close_reasons == {"close_tp": 1}

def test_multiple_positions_in_same_run(db_session):
    """Test that multiple positions in the same run are correctly isolated."""
    run_a = _create_backtest_run(db_session, "multi-pos-run")
    monitor = PositionMonitor(db_session, backtest_run_id=run_a.id)

    # Open positions for multiple symbols
    symbols = ["BTC", "ETH", "XRP"]
    for i, symbol in enumerate(symbols):
        monitor.open_position(
            symbol=symbol,
            side=OrderSide.BUY,
            entry_price=100.0 + i * 10,
            stop_loss=95.0 + i * 10,
            take_profit=110.0 + i * 10,
            quantity=Decimal("1"),
            exit_policy=ExitPolicy.SL_TP_ONLY,
        )

    # Verify all positions are retrieved for this run
    for symbol in symbols:
        pos = monitor.get_active_position(symbol)
        assert pos is not None
        assert pos.symbol == symbol
        assert pos.backtest_run_id == run_a.id

def test_position_monitor_without_backtest_run_id(db_session):
    """Test that PositionMonitor works without backtest_run_id (for backward compatibility)."""
    _create_backtest_run(db_session, "run-no-context")

    # Create monitor without explicit backtest_run_id
    monitor = PositionMonitor(db_session, backtest_run_id=None)

    # Should still be able to open positions
    monitor.open_position(
        symbol="BTC",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    # Should retrieve the position when backtest_run_id is None (no filtering)
    pos = monitor.get_active_position("BTC")
    assert pos is not None
    # Position can have any backtest_run_id when queried without context
    assert pos.symbol == "BTC"

def test_three_way_run_isolation(db_session):
    """Test isolation with three concurrent runs."""
    runs = [
        _create_backtest_run(db_session, f"run-{i}")
        for i in range(3)
    ]

    monitors = [
        PositionMonitor(db_session, backtest_run_id=run.id)
        for run in runs
    ]

    # Each monitor opens a position for the same symbol but different run
    for i, monitor in enumerate(monitors):
        monitor.open_position(
            symbol="BTC",
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            entry_price=100.0 + i * 10,
            stop_loss=95.0 + i * 10,
            take_profit=110.0 + i * 10,
            quantity=Decimal("1"),
            exit_policy=ExitPolicy.SL_TP_ONLY,
        )

    # Each monitor should only see its own position
    for i, monitor in enumerate(monitors):
        pos = monitor.get_active_position("BTC")
        assert pos is not None
        assert pos.backtest_run_id == runs[i].id
        assert pos.entry_price == Decimal(str(100.0 + i * 10))

def test_close_reasons_multi_run(db_session):
    """Test that close_reasons are correctly scoped to run."""
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=1)

    run_a = _create_backtest_run(db_session, "close-run-a")
    run_b = _create_backtest_run(db_session, "close-run-b")

    # Create 3 closed positions in run_a with different close reasons
    close_reasons_a = ["close_tp", "close_sl", "close_tp"]
    for i, reason in enumerate(close_reasons_a):
        pos = ActivePosition(
            symbol=f"ASSET{i}",
            side=OrderSide.BUY,
            entry_price=Decimal("100"),
            stop_loss=Decimal("95"),
            take_profit=Decimal("110"),
            quantity=Decimal("1"),
            decision_timestamp=start,
            candles_since_entry=i,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=2,
            candles_direction=["up", "up"],
            is_active=False,
            close_reason=reason,
            environment=Environment.BACKTEST,
            backtest_run_id=run_a.id,
        )
        db_session.add(pos)

    # Create different close reasons in run_b
    close_reasons_b = ["close_sl", "close_sl", "close_timeout"]
    for i, reason in enumerate(close_reasons_b):
        pos = ActivePosition(
            symbol=f"ASSET{i}",
            side=OrderSide.SELL,
            entry_price=Decimal("200"),
            stop_loss=Decimal("190"),
            take_profit=Decimal("210"),
            quantity=Decimal("1"),
            decision_timestamp=start,
            candles_since_entry=i,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=2,
            candles_direction=["down", "down"],
            is_active=False,
            close_reason=reason,
            environment=Environment.BACKTEST,
            backtest_run_id=run_b.id,
        )
        db_session.add(pos)

    db_session.commit()

    # Check run_a's close reasons
    backtest_a = Backtest(
        start_date=start,
        end_date=end,
        assets=["ASSET0", "ASSET1", "ASSET2"],
        timeframe="1h",
        db_session=db_session,
    )
    backtest_a.backtest_run_id = run_a.id
    close_reasons_a_result = backtest_a._calculate_close_reasons()

    # Check run_b's close reasons
    backtest_b = Backtest(
        start_date=start,
        end_date=end,
        assets=["ASSET0", "ASSET1", "ASSET2"],
        timeframe="1h",
        db_session=db_session,
    )
    backtest_b.backtest_run_id = run_b.id
    close_reasons_b_result = backtest_b._calculate_close_reasons()

    # Verify isolation
    assert close_reasons_a_result == {"close_tp": 2, "close_sl": 1}
    assert close_reasons_b_result == {"close_sl": 2, "close_timeout": 1}

def test_position_monitor_run_id_change(db_session):
    """Test changing backtest_run_id context in PositionMonitor."""
    run_a = _create_backtest_run(db_session, "context-run-a")
    run_b = _create_backtest_run(db_session, "context-run-b")

    monitor = PositionMonitor(db_session, backtest_run_id=run_a.id)

    # Open position in run_a context
    monitor.open_position(
        symbol="BTC",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    # Switch context to run_b
    monitor.backtest_run_id = run_b.id

    # Open position in run_b context
    monitor.open_position(
        symbol="BTC",
        side=OrderSide.SELL,
        entry_price=120.0,
        stop_loss=125.0,
        take_profit=110.0,
        quantity=Decimal("1"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    # Switch back to run_a and verify we see the run_a position
    monitor.backtest_run_id = run_a.id
    pos_a = monitor.get_active_position("BTC")
    assert pos_a.backtest_run_id == run_a.id
    assert pos_a.side == OrderSide.BUY

    # Switch to run_b and verify we see the run_b position
    monitor.backtest_run_id = run_b.id
    pos_b = monitor.get_active_position("BTC")
    assert pos_b.backtest_run_id == run_b.id
    assert pos_b.side == OrderSide.SELL

def test_backtest_run_id_fk_constraint(db_session):
    """Test that backtest_run_id foreign key constraint is enforced."""
    # Try to create a position with non-existent backtest_run_id
    invalid_pos = ActivePosition(
        symbol="BTC",
        side=OrderSide.BUY,
        entry_price=Decimal("100"),
        stop_loss=Decimal("95"),
        take_profit=Decimal("110"),
        quantity=Decimal("1"),
        decision_timestamp=datetime.utcnow(),
        candles_since_entry=0,
        exit_policy=ExitPolicy.SL_TP_ONLY,
        prediction_horizon=2,
        candles_direction=["up", "up"],
        is_active=True,
        environment=Environment.BACKTEST,
        backtest_run_id=999999,  # Non-existent run ID
    )

    db_session.add(invalid_pos)

    # Should raise integrity error due to FK constraint
    with pytest.raises(Exception):  # psycopg2.IntegrityError or SQLAlchemy exception
        db_session.commit()

    # Important: rollback the session to clean up the bad state
    # so the fixture teardown can complete cleanly
    db_session.rollback()


def _seed_rsi_market_data(session, symbol: str, start: datetime, end: datetime) -> int:
    """Deterministic sine-wave OHLCV so RSI actually swings overbought/oversold.

    Same pattern as scripts/repro_iip_timeframe.py, kept local to this test so
    it stays self-contained.
    """
    rows = []
    current = start - timedelta(days=10)
    idx = 0
    while current <= end:
        close = 100.0 + 15.0 * math.sin(idx / 6.0) + (idx * 0.01)
        rows.append(
            MarketData(
                symbol=symbol,
                timeframe="1h",
                timestamp=current,
                open=Decimal(str(round(close - 0.5, 4))),
                high=Decimal(str(round(close + 1.0, 4))),
                low=Decimal(str(round(close - 1.0, 4))),
                close=Decimal(str(round(close, 4))),
                volume=Decimal("1000"),
            )
        )
        idx += 1
        current += timedelta(hours=1)
    session.bulk_save_objects(rows)
    session.commit()
    return len(rows)


def test_trades_isolated_across_consecutive_backtest_runs(db_session):
    """Two backtests run back-to-back over the same data must not mix trades.

    Regression test for D08 of PLAN-30-DIAS: PortfolioManager.execute_trade()
    must stamp the *current* backtest_run_id on every Trade it creates. Before
    that change, Trade.backtest_run_id was never set, so this test fails
    (trades come back with backtest_run_id=None and neither per-run query
    finds them).
    """
    symbol = "BTC"
    end = datetime(2026, 6, 30)
    start = end - timedelta(days=90)
    _seed_rsi_market_data(db_session, symbol, start, end)

    config = {
        "base_position_pct": 0.05,
        "max_daily_loss_pct": 0.05,
        "max_position_pct": 0.10,
        "slippage_pct": 0.01,
        "market_hours_filter": False,
    }

    def _run(name: str):
        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=[symbol],
            timeframe="1h",
            initial_capital=100000.0,
            config=config,
            db_session=db_session,
            strategy=RSIMeanReversionStrategy(),
        )
        metrics = backtest.run(name=name)
        assert metrics.total_trades > 0, "fixture must actually produce trades"
        return backtest, metrics

    backtest_a, metrics_a = _run("isolation-run-a")
    backtest_b, metrics_b = _run("isolation-run-b")

    assert backtest_a.backtest_run_id != backtest_b.backtest_run_id

    trades_a = (
        db_session.query(Trade)
        .filter(Trade.backtest_run_id == backtest_a.backtest_run_id)
        .all()
    )
    trades_b = (
        db_session.query(Trade)
        .filter(Trade.backtest_run_id == backtest_b.backtest_run_id)
        .all()
    )

    # Every row created by execute_trade() (open leg + close leg) must carry
    # its own run's backtest_run_id -- none should be left unstamped (None).
    assert len(trades_a) > 0
    assert len(trades_b) > 0

    ids_a = {t.id for t in trades_a}
    ids_b = {t.id for t in trades_b}
    assert ids_a.isdisjoint(ids_b)

    all_trades = db_session.query(Trade).filter(Trade.symbol == symbol).all()
    assert len(all_trades) == len(trades_a) + len(trades_b)
    assert all(t.backtest_run_id is not None for t in all_trades)
