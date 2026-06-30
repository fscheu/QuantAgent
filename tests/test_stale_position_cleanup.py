"""Tests for stale position cleanup in sequential backtests."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import pytest

from quantagent.backtesting.backtest import Backtest
from quantagent.models import (
    ActivePosition,
    BacktestRun,
    Environment,
    ExitPolicy,
    OrderSide,
    Trade,
)


def _session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from quantagent.models import Base
    engine = create_engine("sqlite:///:memory:")
    TestingSession = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    return TestingSession()


def test_cleanup_stale_positions_force_closes():
    """
    Verify _cleanup_stale_positions() closes ALL stale positions.
    This tests the force-close path (no yfinance data available in test).
    """
    session = _session()
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=90)

    run = BacktestRun(
        name="stale-cleanup-test",
        timeframe="1h",
        assets=["BTC"],
        start_date=start,
        end_date=end,
        config_snapshot={},
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    stale_positions = []
    for i in range(3):
        pos = ActivePosition(
            symbol="BTC",
            side=OrderSide.BUY,
            entry_price=Decimal("100"),
            stop_loss=Decimal("95"),
            take_profit=Decimal("110"),
            quantity=Decimal("1"),
            decision_timestamp=start + timedelta(hours=i),
            candles_since_entry=i,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=2,
            candles_direction=["up"],
            is_active=True,
            environment=Environment.BACKTEST,
            backtest_run_id=run.id,
        )
        session.add(pos)
        stale_positions.append(pos)

    session.commit()

    backtest = Backtest(
        start_date=start,
        end_date=end,
        assets=["BTC"],
        timeframe="1h",
        db_session=session,
        config={"market_hours_filter": False},
    )

    backtest._cleanup_stale_positions()

    for pos in stale_positions:
        session.refresh(pos)
        assert pos.is_active is False
        assert pos.close_reason is not None
        assert "stale_cleanup" in pos.close_reason

    session.close()


def test_cleanup_stale_positions_does_not_touch_other_environments():
    """
    Verify cleanup only closes BACKTEST positions, not PAPER/PROD.
    """
    session = _session()
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=90)

    run = BacktestRun(
        name="env-isolation-test",
        timeframe="1h",
        assets=["BTC"],
        start_date=start,
        end_date=end,
        config_snapshot={},
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    backtest_pos = ActivePosition(
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
        candles_direction=["up"],
        is_active=True,
        environment=Environment.BACKTEST,
        backtest_run_id=run.id,
    )
    paper_pos = ActivePosition(
        symbol="BTC",
        side=OrderSide.SELL,
        entry_price=Decimal("200"),
        stop_loss=Decimal("190"),
        take_profit=Decimal("210"),
        quantity=Decimal("1"),
        decision_timestamp=start,
        candles_since_entry=0,
        exit_policy=ExitPolicy.SL_TP_ONLY,
        prediction_horizon=2,
        candles_direction=["down"],
        is_active=True,
        environment=Environment.PAPER,
        backtest_run_id=run.id,
    )
    session.add_all([backtest_pos, paper_pos])
    session.commit()

    backtest = Backtest(
        start_date=start,
        end_date=end,
        assets=["BTC"],
        timeframe="1h",
        db_session=session,
        config={"market_hours_filter": False},
    )

    backtest._cleanup_stale_positions()

    session.refresh(backtest_pos)
    session.refresh(paper_pos)

    assert backtest_pos.is_active is False
    assert backtest_pos.close_reason is not None
    assert paper_pos.is_active is True
    assert paper_pos.close_reason is None

    session.close()


def test_cleanup_stale_positions_with_associated_trades():
    """
    Verify _cleanup_stale_positions handles positions with trade_id.
    Trades linked to stale positions should be force-closed via order_manager.
    """
    session = _session()
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=90)

    run = BacktestRun(
        name="stale-trade-cleanup",
        timeframe="1h",
        assets=["BTC"],
        start_date=start,
        end_date=end,
        config_snapshot={},
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    trade = Trade(
        symbol="BTC",
        entry_price=Decimal("100"),
        quantity=Decimal("1"),
        side=OrderSide.BUY,
        pnl=Decimal("10"),
        opened_at=start,
        environment=Environment.BACKTEST,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)

    pos = ActivePosition(
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
        candles_direction=["up"],
        is_active=True,
        environment=Environment.BACKTEST,
        backtest_run_id=run.id,
        trade_id=trade.id,
    )
    session.add(pos)
    session.commit()

    backtest = Backtest(
        start_date=start,
        end_date=end,
        assets=["BTC"],
        timeframe="1h",
        db_session=session,
        config={"market_hours_filter": False},
    )

    backtest._cleanup_stale_positions()

    session.refresh(pos)
    assert pos.is_active is False
    assert pos.close_reason is not None
    assert "stale_cleanup" in pos.close_reason

    session.close()


def test_cleanup_in_run_prevents_contamination():
    """
    Integration test: running backtest with stale positions in DB
    should clean them up via _cleanup_stale_positions() called at start of run().
    Uses mocked DataProvider to avoid yfinance dependency.
    """
    session = _session()
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=90)

    old_run = BacktestRun(
        name="old-run",
        timeframe="1h",
        assets=["BTC"],
        start_date=start,
        end_date=end,
        config_snapshot={},
    )
    session.add(old_run)
    session.commit()
    session.refresh(old_run)

    stale_pos = ActivePosition(
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
        candles_direction=["up"],
        is_active=True,
        environment=Environment.BACKTEST,
        backtest_run_id=old_run.id,
    )
    session.add(stale_pos)
    session.commit()

    backtest = Backtest(
        start_date=start,
        end_date=end,
        assets=["BTC"],
        timeframe="1h",
        db_session=session,
        config={"market_hours_filter": False},
    )

    empty_df = pd.DataFrame()

    with patch.object(backtest.data_provider, "get_ohlc", return_value=empty_df):
        backtest.run(name="clean-run")

    session.refresh(stale_pos)
    assert stale_pos.is_active is False
    assert stale_pos.close_reason is not None
    assert "stale_cleanup" in stale_pos.close_reason

    fresh_positions = (
        session.query(ActivePosition)
        .filter(
            ActivePosition.is_active == True,
            ActivePosition.environment == Environment.BACKTEST,
        )
        .all()
    )
    assert len(fresh_positions) == 0

    session.close()
