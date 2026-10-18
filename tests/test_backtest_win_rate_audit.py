from datetime import datetime, timedelta
from decimal import Decimal
import math
from quantagent.models import BacktestRun, Environment, OrderSide, Trade
from quantagent.backtesting.backtest import Backtest


def test_hand_calculated_win_rate_profit_factor_and_neutral_trades(db_session):
    """Audit win rate and profit factor with 3 winners, 2 losers, 1 neutral trade."""
    start = datetime(2026, 1, 1, 10, 0)
    end = datetime(2026, 1, 2, 18, 0)
    initial_capital = 10000.0

    # 3 Winners (+100, +50, +30 = +180.0)
    # 2 Losers (-40, -20 = -60.0)
    # 1 Neutral / Breakeven (0.0)
    pnls = [100.0, 50.0, 30.0, -40.0, -20.0, 0.0]
    for i, pnl_val in enumerate(pnls):
        trade = Trade(
            symbol="SPY",
            entry_price=Decimal("100.0"),
            exit_price=Decimal(str(100.0 + pnl_val)),
            quantity=Decimal("1.0"),
            side=OrderSide.BUY,
            pnl=Decimal(str(pnl_val)),
            opened_at=start + timedelta(hours=i),
            closed_at=start + timedelta(hours=i, minutes=30),
            environment=Environment.BACKTEST,
        )
        db_session.add(trade)
    db_session.commit()

    bt = Backtest(
        start_date=start,
        end_date=end,
        assets=["SPY"],
        initial_capital=initial_capital,
        config={},
        db_session=db_session,
    )
    metrics = bt._calculate_metrics()

    assert metrics.total_trades == 6
    assert metrics.winning_trades == 3
    assert metrics.losing_trades == 2

    # Policy A: neutral trade counts in total_trades denominator -> 3 / 6 = 50.0%
    # This assert fails if neutral trades are excluded from denominator (which would give 3 / 5 = 60.0%)
    assert metrics.win_rate == 0.50
    assert metrics.win_rate != 0.60

    # Profit factor: gross_profit (180.0) / gross_loss (60.0) = 3.0
    assert metrics.profit_factor == 3.0


def test_profit_factor_without_losing_trades_persists_none_in_db(db_session):
    """Audit that BacktestRun.profit_factor is None (NULL) in DB when there are no losses, never inf."""
    start = datetime(2026, 1, 1, 10, 0)
    end = datetime(2026, 1, 2, 18, 0)

    run = BacktestRun(
        name="test-run-inf",
        timeframe="1h",
        assets=["SPY"],
        start_date=start,
        end_date=end,
        config_snapshot={},
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Only winning trades -> 0 losses
    for i, pnl_val in enumerate([100.0, 50.0]):
        trade = Trade(
            symbol="SPY",
            entry_price=Decimal("100.0"),
            exit_price=Decimal(str(100.0 + pnl_val)),
            quantity=Decimal("1.0"),
            side=OrderSide.BUY,
            pnl=Decimal(str(pnl_val)),
            opened_at=start + timedelta(hours=i),
            closed_at=start + timedelta(hours=i, minutes=30),
            environment=Environment.BACKTEST,
        )
        db_session.add(trade)
    db_session.commit()

    bt = Backtest(
        start_date=start,
        end_date=end,
        assets=["SPY"],
        initial_capital=10000.0,
        config={},
        db_session=db_session,
    )
    metrics = bt._calculate_metrics()

    assert metrics.total_trades == 2
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 0
    assert metrics.profit_factor == float("inf")

    bt.backtest_run_id = run.id
    bt._update_backtest_run(metrics)
    db_session.refresh(run)

    assert run.profit_factor is None
    assert run.profit_factor != float("inf")


def test_backtest_run_model_validator_never_stores_inf():
    """Direct model validation: BacktestRun converts inf or nan profit_factor to None."""
    run = BacktestRun(
        timeframe="1h",
        assets=["SPY"],
        start_date=datetime(2026, 1, 1),
        end_date=datetime(2026, 1, 2),
        config_snapshot={},
        profit_factor=float("inf"),
    )
    assert run.profit_factor is None

    run.profit_factor = float("-inf")
    assert run.profit_factor is None

    run.profit_factor = float("nan")
    assert run.profit_factor is None

    run.profit_factor = 2.5
    assert run.profit_factor == 2.5
