from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.backtesting.backtest import Backtest
from quantagent.models import (
    ActivePosition,
    BacktestRun,
    Base,
    Environment,
    ExitPolicy,
    OrderSide,
    Trade,
)


def _session():
    engine = create_engine("sqlite:///:memory:")
    TestingSession = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    return TestingSession()


def test_calculate_metrics_scopes_trades_to_current_backtest_run():
    session = _session()
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=1)

    run_a = BacktestRun(
        name="run-a",
        timeframe="1h",
        assets=["BTC"],
        start_date=start,
        end_date=end,
        config_snapshot={},
    )
    run_b = BacktestRun(
        name="run-b",
        timeframe="4h",
        assets=["BTC"],
        start_date=start,
        end_date=end,
        config_snapshot={},
    )
    session.add_all([run_a, run_b])
    session.commit()
    session.refresh(run_a)
    session.refresh(run_b)

    trade_a = Trade(
        symbol="BTC",
        entry_price=Decimal("100"),
        quantity=Decimal("1"),
        side=OrderSide.BUY,
        pnl=Decimal("123.45"),
        opened_at=start + timedelta(hours=1),
        environment=Environment.BACKTEST,
    )
    session.add(trade_a)
    session.commit()
    session.refresh(trade_a)

    pos_a = ActivePosition(
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
        is_active=False,
        close_reason="close_tp",
        environment=Environment.BACKTEST,
        backtest_run_id=run_a.id,
        trade_id=trade_a.id,
    )
    session.add(pos_a)
    session.commit()

    backtest = Backtest(
        start_date=start,
        end_date=end,
        assets=["BTC"],
        timeframe="4h",
        db_session=session,
        config={"market_hours_filter": False},
    )
    backtest.backtest_run_id = run_b.id

    metrics = backtest._calculate_metrics()

    assert metrics.total_trades == 0
    assert metrics.total_pnl == 0.0
