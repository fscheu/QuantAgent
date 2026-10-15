from datetime import datetime, timedelta

import pytest

from quantagent.backtesting.backtest import Backtest
from quantagent.models import ActivePosition, MarketData, Trade
from quantagent.strategy.base import TradingSignal, TradingStrategy


class HandStrategy(TradingStrategy):
    required_history_bars = 3

    def __init__(self):
        self.windows = []

    def generate_signal(self, kline_data, symbol, timeframe, current_price):
        self.windows.append([bar["close"] for bar in kline_data])
        assert len(kline_data) == 3
        assert kline_data[-1]["close"] == current_price
        return TradingSignal(decision="LONG", confidence=1, stop_loss=90, take_profit=200)

    def should_exit(self, position, current_price, ohlc_data):
        return True, "RULE"

    def should_reevaluate(self, position, current_price):
        return False


@pytest.mark.parametrize("stop,last,exit_price", [(False, False, 120), (True, False, 110 * 90 / 101), (False, True, None)])
def test_next_open_fill(db_session, stop, last, exit_price):
    start = datetime(2026, 1, 5)
    for i, opening in enumerate([100, 100, 100, 110, 120]):
        db_session.add(MarketData(symbol="FILL", timeframe="1d", timestamp=start + timedelta(days=i),
                                  open=opening, high=opening + 2, low=89 if stop and i == 3 else opening - 1,
                                  close=opening + 1, volume=1000))
    db_session.commit()
    bt = Backtest(start, start + timedelta(days=2 if last else 4), ["FILL"], "1d",
                  config={"offline_data": True, "market_hours_filter": False,
                          "commission_pct": 0.001, "slippage_pct": 0.002},
                  db_session=db_session, strategy=HandStrategy(), fill="apertura")
    bt.run()
    assert bt.strategy.windows[0] == [101, 101, 101]
    trades = db_session.query(Trade).filter(Trade.backtest_run_id == bt.backtest_run_id).all()
    assert bt._build_config_snapshot()["fill"] == "apertura"
    if last:
        assert trades == []
        return
    trade = trades[0]
    assert float(trade.entry_price) == pytest.approx(110 * 1.002)
    assert float(trade.exit_price) == pytest.approx(exit_price * 0.998)
    qty = float(trade.quantity)
    expected = qty * (exit_price * 0.998 - 110 * 1.002 - (110 * 1.002 + exit_price * 0.998) * 0.001)
    assert float(trade.pnl) == pytest.approx(expected, abs=1e-6)
    assert trade.opened_at == start + timedelta(days=3)
    assert trade.closed_at == start + timedelta(days=3 if stop else 4)


class SignalParamsStrategy(TradingStrategy):
    required_history_bars = 1

    def __init__(self, decision="LONG", stop_loss=98, take_profit=103):
        self.decision = decision
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signal(self, kline_data, symbol, timeframe, current_price):
        if len(kline_data) == 1:
            return TradingSignal(
                decision=self.decision,
                confidence=1,
                stop_loss=self.stop_loss,
                take_profit=self.take_profit,
            )
        return TradingSignal(decision="HOLD", confidence=0)

    def should_exit(self, position, current_price, ohlc_data):
        return False, ""

    def should_reevaluate(self, position, current_price):
        return False


def test_stop_take_profit_rescaled_open_long(db_session):
    start = datetime(2026, 1, 5)
    db_session.add(MarketData(symbol="RESCLONG", timeframe="1d", timestamp=start,
                              open=100, high=100, low=100, close=100, volume=1000))
    db_session.add(MarketData(symbol="RESCLONG", timeframe="1d", timestamp=start + timedelta(days=1),
                              open=104, high=105, low=102, close=104, volume=1000))
    db_session.commit()
    strat = SignalParamsStrategy(decision="LONG", stop_loss=98, take_profit=103)
    bt = Backtest(start, start + timedelta(days=1), ["RESCLONG"], "1d",
                  config={"offline_data": True, "market_hours_filter": False,
                          "intrabar_stops": True},
                  db_session=db_session, strategy=strat, fill="apertura")
    bt.run()
    pos = db_session.query(ActivePosition).filter(
        ActivePosition.symbol == "RESCLONG",
        ActivePosition.backtest_run_id == bt.backtest_run_id,
    ).first()
    assert pos is not None
    assert float(pos.stop_loss) == pytest.approx(101.92)
    assert float(pos.take_profit) == pytest.approx(107.12)
    trade = db_session.query(Trade).filter(Trade.backtest_run_id == bt.backtest_run_id).first()
    assert trade is not None
    assert trade.exit_signal == "backtest_end"


def test_stop_take_profit_rescaled_open_short(db_session):
    start = datetime(2026, 1, 5)
    db_session.add(MarketData(symbol="RESCSHORT", timeframe="1d", timestamp=start,
                              open=100, high=100, low=100, close=100, volume=1000))
    db_session.add(MarketData(symbol="RESCSHORT", timeframe="1d", timestamp=start + timedelta(days=1),
                              open=96, high=97, low=95, close=96, volume=1000))
    db_session.commit()
    strat = SignalParamsStrategy(decision="SHORT", stop_loss=102, take_profit=97)
    bt = Backtest(start, start + timedelta(days=1), ["RESCSHORT"], "1d",
                  config={"offline_data": True, "market_hours_filter": False,
                          "intrabar_stops": True},
                  db_session=db_session, strategy=strat, fill="apertura")
    bt.run()
    pos = db_session.query(ActivePosition).filter(
        ActivePosition.symbol == "RESCSHORT",
        ActivePosition.backtest_run_id == bt.backtest_run_id,
    ).first()
    assert pos is not None
    assert float(pos.stop_loss) == pytest.approx(97.92)
    assert float(pos.take_profit) == pytest.approx(93.12)
    trade = db_session.query(Trade).filter(Trade.backtest_run_id == bt.backtest_run_id).first()
    assert trade is not None
    assert trade.exit_signal == "backtest_end"


def test_stop_take_profit_close_fill_unchanged(db_session):
    start = datetime(2026, 1, 5)
    db_session.add(MarketData(symbol="CLOSEFILL", timeframe="1d", timestamp=start,
                              open=100, high=100, low=100, close=100, volume=1000))
    db_session.commit()
    strat = SignalParamsStrategy(decision="LONG", stop_loss=98, take_profit=103)
    bt = Backtest(start, start, ["CLOSEFILL"], "1d",
                  config={"offline_data": True, "market_hours_filter": False},
                  db_session=db_session, strategy=strat, fill="cierre")
    bt.run()
    pos = db_session.query(ActivePosition).filter(
        ActivePosition.symbol == "CLOSEFILL",
        ActivePosition.backtest_run_id == bt.backtest_run_id,
    ).first()
    assert pos is not None
    assert float(pos.stop_loss) == pytest.approx(98)
    assert float(pos.take_profit) == pytest.approx(103)


def test_stop_take_profit_default_levels_open_fill(db_session):
    start = datetime(2026, 1, 5)
    db_session.add(MarketData(symbol="DEFOPEN", timeframe="1d", timestamp=start,
                              open=100, high=100, low=100, close=100, volume=1000))
    db_session.add(MarketData(symbol="DEFOPEN", timeframe="1d", timestamp=start + timedelta(days=1),
                              open=104, high=105, low=102, close=104, volume=1000))
    db_session.commit()
    strat = SignalParamsStrategy(decision="LONG", stop_loss=None, take_profit=None)
    bt = Backtest(start, start + timedelta(days=1), ["DEFOPEN"], "1d",
                  config={"offline_data": True, "market_hours_filter": False},
                  db_session=db_session, strategy=strat, fill="apertura")
    bt.run()
    pos = db_session.query(ActivePosition).filter(
        ActivePosition.symbol == "DEFOPEN",
        ActivePosition.backtest_run_id == bt.backtest_run_id,
    ).first()
    assert pos is not None
    assert float(pos.stop_loss) == pytest.approx(104 * 0.98)
    assert float(pos.take_profit) == pytest.approx(104 * 1.03)


