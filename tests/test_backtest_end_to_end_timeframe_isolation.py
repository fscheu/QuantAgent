from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd

from quantagent.backtesting.backtest import Backtest
from quantagent.models import ActivePosition, Environment, MarketData, Trade
from quantagent.strategy.base import ExitPolicy, TradingSignal, TradingStrategy


class SingleEntryHoldStrategy(TradingStrategy):
    def __init__(self):
        self._entered = False

    @property
    def required_history_bars(self) -> int:
        return 2

    def generate_signal(self, kline_data, symbol, timeframe, current_price, **kwargs):
        if self._entered:
            return None

        self._entered = True
        return TradingSignal(
            decision="LONG",
            confidence=1.0,
            entry_price=current_price,
            stop_loss=current_price * 0.5,
            take_profit=current_price * 2.0,
            reasoning=f"enter once on {timeframe}",
            exit_policy=ExitPolicy.TIME_BASED,
            max_hold_candles=999,
        )

    def should_reevaluate(self, position, current_price):
        return False


def _seed_market_data(session, symbol: str, start: datetime) -> None:
    lookback_start = start - timedelta(days=8)

    hourly_points = []
    current = lookback_start
    end_hourly = start + timedelta(hours=8)
    while current <= end_hourly:
        hours_from_start = int((current - start).total_seconds() // 3600)
        close = 100 + hours_from_start
        hourly_points.append((current, close))
        current += timedelta(hours=1)

    four_hour_points = []
    current = lookback_start
    end_4h = start + timedelta(hours=8)
    step_index = 0
    while current <= end_4h:
        close = 100 + step_index * 10
        four_hour_points.append((current, close))
        current += timedelta(hours=4)
        step_index += 1

    for timestamp, close in hourly_points:
        session.add(
            MarketData(
                symbol=symbol,
                timeframe="1h",
                timestamp=timestamp,
                open=Decimal(str(close - 0.5)),
                high=Decimal(str(close + 0.5)),
                low=Decimal(str(close - 1.0)),
                close=Decimal(str(close)),
                volume=Decimal("1000"),
            )
        )

    for timestamp, close in four_hour_points:
        session.add(
            MarketData(
                symbol=symbol,
                timeframe="4h",
                timestamp=timestamp,
                open=Decimal(str(close - 0.5)),
                high=Decimal(str(close + 0.5)),
                low=Decimal(str(close - 1.0)),
                close=Decimal(str(close)),
                volume=Decimal("1000"),
            )
        )

    session.commit()


def _config():
    return {
        "base_position_pct": 0.05,
        "max_daily_loss_pct": 1.0,
        "max_position_pct": 1.0,
        "slippage_pct": 0.0,
        "market_hours_filter": False,
    }


def test_backtest_end_close_updates_linked_trade_metrics(db_session, monkeypatch):
    monkeypatch.setattr(
        "quantagent.data.provider.DataProvider._fetch_yfinance",
        lambda self, symbol, timeframe, start_date, end_date: pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        ),
    )

    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(hours=8)
    _seed_market_data(db_session, "BTC", start)

    backtest = Backtest(
        start_date=start,
        end_date=end,
        assets=["BTC"],
        timeframe="1h",
        db_session=db_session,
        config=_config(),
        strategy=SingleEntryHoldStrategy(),
    )

    metrics = backtest.run(name="end-close-realization")

    opening_trade = db_session.query(Trade).order_by(Trade.id.asc()).first()

    assert metrics.total_trades == 1
    assert metrics.total_pnl > 0
    assert opening_trade is not None
    assert opening_trade.closed_at is not None
    assert opening_trade.exit_price is not None
    assert opening_trade.pnl is not None
    assert (
        db_session.query(ActivePosition)
        .filter(
            ActivePosition.environment == Environment.BACKTEST,
            ActivePosition.is_active.is_(True),
        )
        .count()
        == 0
    )


def test_sequential_1h_and_4h_backtests_diverge_without_yahoo(db_session, monkeypatch):
    monkeypatch.setattr(
        "quantagent.data.provider.DataProvider._fetch_yfinance",
        lambda self, symbol, timeframe, start_date, end_date: pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        ),
    )

    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(hours=8)
    _seed_market_data(db_session, "BTC", start)

    backtest_1h = Backtest(
        start_date=start,
        end_date=end,
        assets=["BTC"],
        timeframe="1h",
        db_session=db_session,
        config=_config(),
        strategy=SingleEntryHoldStrategy(),
    )
    metrics_1h = backtest_1h.run(name="seq-1h")

    backtest_4h = Backtest(
        start_date=start,
        end_date=end,
        assets=["BTC"],
        timeframe="4h",
        db_session=db_session,
        config=_config(),
        strategy=SingleEntryHoldStrategy(),
    )
    metrics_4h = backtest_4h.run(name="seq-4h")

    assert metrics_1h.total_trades == 1
    assert metrics_4h.total_trades == 1
    assert backtest_1h.total_candles_processed > backtest_4h.total_candles_processed
    assert metrics_1h.total_pnl != metrics_4h.total_pnl
    assert (
        db_session.query(ActivePosition)
        .filter(
            ActivePosition.environment == Environment.BACKTEST,
            ActivePosition.is_active.is_(True),
        )
        .count()
        == 0
    )
