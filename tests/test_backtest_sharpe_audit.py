"""Tests for Sharpe ratio annualization in the backtest engine (QuantAgent-hx0.4)."""

from datetime import datetime
import math
from quantagent.backtesting.backtest import Backtest


def _make_synthetic_curve(
    n_periods: int, target_sharpe: float = 2.0, target_vol: float = 0.20, rf: float = 0.02
):
    target_mean = (target_sharpe * target_vol + rf) / n_periods
    target_std = target_vol / math.sqrt(n_periods)
    d = target_std * math.sqrt((n_periods - 1) / n_periods)
    rets = [target_mean + d if i % 2 == 0 else target_mean - d for i in range(n_periods)]

    t0 = datetime(2026, 1, 1, 0, 0, 0)
    t1 = datetime(2027, 1, 1, 6, 0, 0)  # 365.25 days
    dt = (t1 - t0) / n_periods

    curve = [{"date": t0, "equity": 100000.0}]
    curr = t0
    eq = 100000.0
    for r in rets:
        curr += dt
        eq *= 1.0 + r
        curve.append({"date": curr, "equity": eq})
    return curve


def test_engine_synthetic_equity_same_sharpe_in_both_calendars(db_session):
    """Backtest engine derives periods per year from equity curve and gives expected Sharpe in both calendars."""
    start = datetime(2026, 1, 1, 0, 0, 0)
    end = datetime(2027, 1, 1, 6, 0, 0)

    # 24/7 calendar: 8766 hourly periods in 365.25 days
    bt_24_7 = Backtest(
        start_date=start,
        end_date=end,
        assets=["SPY"],
        timeframe="1h",
        db_session=db_session,
    )
    bt_24_7.equity_curve = _make_synthetic_curve(8766)
    sharpe_24_7 = bt_24_7._calculate_sharpe_ratio()
    assert round(sharpe_24_7, 2) == 2.00

    # Market hours calendar: 1638 hourly periods in 365.25 days (252 trading days * 6.5 hours)
    bt_mkt = Backtest(
        start_date=start,
        end_date=end,
        assets=["SPY"],
        timeframe="1h",
        db_session=db_session,
    )
    bt_mkt.equity_curve = _make_synthetic_curve(1638)
    sharpe_mkt = bt_mkt._calculate_sharpe_ratio()
    assert round(sharpe_mkt, 2) == 2.00
