"""Tests for stop entry orders and bar expiry in backtest engine (QuantAgent-fnpr.3.3)."""

from datetime import datetime, timedelta

import pytest

from quantagent.backtesting.backtest import Backtest
from quantagent.models import MarketData, Trade
from quantagent.strategy.base import TradingSignal, TradingStrategy


class StopOrderStrategy(TradingStrategy):
    required_history_bars = 1

    def __init__(self, entry_price=100.0, entry_expiry_bars=None, entry_order="stop"):
        self.entry_price = entry_price
        self.entry_expiry_bars = entry_expiry_bars
        self.entry_order = entry_order
        self.signal_emitted = False

    def generate_signal(self, kline_data, symbol, timeframe, current_price):
        if not self.signal_emitted:
            self.signal_emitted = True
            return TradingSignal(
                decision="LONG",
                confidence=1.0,
                entry_order=self.entry_order,
                entry_price=self.entry_price,
                entry_expiry_bars=self.entry_expiry_bars,
                stop_loss=80.0,
                take_profit=200.0,
            )
        return TradingSignal(decision="HOLD", confidence=0.0)

    def should_exit(self, position, current_price, ohlc_data):
        return False, ""

    def should_reevaluate(self, position, current_price):
        return False


def _seed_candles(db_session, symbol, start, candle_specs):
    for i, (op, hi, lo, cl) in enumerate(candle_specs):
        db_session.add(
            MarketData(
                symbol=symbol,
                timeframe="1d",
                timestamp=start + timedelta(days=i),
                open=op,
                high=hi,
                low=lo,
                close=cl,
                volume=1000,
            )
        )
    db_session.commit()


def test_apertura_por_encima_del_stop_llena_en_apertura(db_session):
    start = datetime(2026, 2, 1)
    candles = [
        (90.0, 92.0, 89.0, 91.0),    # día 0: emite señal stop en 100.0
        (105.0, 110.0, 103.0, 108.0), # día 1: apertura (105.0) > stop (100.0)
    ]
    _seed_candles(db_session, "STOPGAP", start, candles)

    strat = StopOrderStrategy(entry_price=100.0, entry_expiry_bars=3)
    bt = Backtest(
        start_date=start,
        end_date=start + timedelta(days=1),
        assets=["STOPGAP"],
        timeframe="1d",
        config={
            "offline_data": True,
            "market_hours_filter": False,
            "slippage_pct": 0.0,
            "commission_pct": 0.0,
        },
        db_session=db_session,
        strategy=strat,
    )
    bt.run()

    trades = db_session.query(Trade).filter(Trade.backtest_run_id == bt.backtest_run_id).all()
    assert len(trades) == 1
    assert float(trades[0].entry_price) == pytest.approx(105.0)
    assert trades[0].opened_at == start + timedelta(days=1)


def test_vela_que_toca_stop_dentro_del_rango_llena_en_stop(db_session):
    start = datetime(2026, 2, 1)
    candles = [
        (90.0, 92.0, 89.0, 91.0),   # día 0: emite señal stop en 100.0
        (95.0, 103.0, 94.0, 98.0),  # día 1: open 95 < 100, high 103 >= 100
    ]
    _seed_candles(db_session, "STOPHIT", start, candles)

    strat = StopOrderStrategy(entry_price=100.0, entry_expiry_bars=3)
    bt = Backtest(
        start_date=start,
        end_date=start + timedelta(days=1),
        assets=["STOPHIT"],
        timeframe="1d",
        config={
            "offline_data": True,
            "market_hours_filter": False,
            "slippage_pct": 0.0,
            "commission_pct": 0.0,
        },
        db_session=db_session,
        strategy=strat,
    )
    bt.run()

    trades = db_session.query(Trade).filter(Trade.backtest_run_id == bt.backtest_run_id).all()
    assert len(trades) == 1
    assert float(trades[0].entry_price) == pytest.approx(100.0)
    assert trades[0].opened_at == start + timedelta(days=1)


def test_sin_llenado_en_3_velas_orden_vence_sin_trade(db_session):
    start = datetime(2026, 2, 1)
    candles = [
        (90.0, 92.0, 89.0, 91.0),   # día 0: emite señal stop en 100.0 con vencimiento 3
        (91.0, 95.0, 90.0, 93.0),   # día 1: barra 1 sin llenar (high 95 < 100)
        (92.0, 96.0, 91.0, 94.0),   # día 2: barra 2 sin llenar (high 96 < 100)
        (93.0, 97.0, 92.0, 95.0),   # día 3: barra 3 sin llenar (high 97 < 100) -> vence
        (105.0, 110.0, 104.0, 108.0), # día 4: high supera 100 pero la orden ya venció
    ]
    _seed_candles(db_session, "STOPEXP", start, candles)

    strat = StopOrderStrategy(entry_price=100.0, entry_expiry_bars=3)
    bt = Backtest(
        start_date=start,
        end_date=start + timedelta(days=4),
        assets=["STOPEXP"],
        timeframe="1d",
        config={
            "offline_data": True,
            "market_hours_filter": False,
            "slippage_pct": 0.0,
            "commission_pct": 0.0,
        },
        db_session=db_session,
        strategy=strat,
    )
    bt.run()

    trades = db_session.query(Trade).filter(Trade.backtest_run_id == bt.backtest_run_id).all()
    assert trades == []


def test_llenado_en_tercera_vela_antes_de_vencer(db_session):
    start = datetime(2026, 2, 1)
    candles = [
        (90.0, 92.0, 89.0, 91.0),   # día 0: emite señal stop en 100.0 con vencimiento 3
        (91.0, 95.0, 90.0, 93.0),   # día 1: barra 1 sin llenar
        (92.0, 96.0, 91.0, 94.0),   # día 2: barra 2 sin llenar
        (95.0, 102.0, 94.0, 99.0),  # día 3: barra 3 toca el stop -> debe llenar en 100.0
    ]
    _seed_candles(db_session, "STOPBAR3", start, candles)

    strat = StopOrderStrategy(entry_price=100.0, entry_expiry_bars=3)
    bt = Backtest(
        start_date=start,
        end_date=start + timedelta(days=3),
        assets=["STOPBAR3"],
        timeframe="1d",
        config={
            "offline_data": True,
            "market_hours_filter": False,
            "slippage_pct": 0.0,
            "commission_pct": 0.0,
        },
        db_session=db_session,
        strategy=strat,
    )
    bt.run()

    trades = db_session.query(Trade).filter(Trade.backtest_run_id == bt.backtest_run_id).all()
    assert len(trades) == 1
    assert float(trades[0].entry_price) == pytest.approx(100.0)
    assert trades[0].opened_at == start + timedelta(days=3)
