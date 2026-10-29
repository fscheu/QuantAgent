"""Tests para Arquetipo A (variante de Triple Screen) — QuantAgent-fnpr.3.5 (M5-22)."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantagent.backtesting.backtest import Backtest
from quantagent.models import ActivePosition, ExitPolicy, MarketData, OrderSide, Trade
from quantagent.strategy.arquetipo_a_strategy import ArquetipoAStrategy
from quantagent.strategy.base import TradingSignal, TradingStrategy
from quantagent.strategy.indicadores import atr, safezone_long


def _make_series(n: int, base: float = 50.0) -> pd.DataFrame:
    t = np.linspace(0, 4 * np.pi, n)
    closes = base + np.arange(n) * 0.5 + 3.0 * np.sin(t)
    for i in range(max(0, n - 25), n):
        closes[i] += ((i - (n - 25)) * 0.3) ** 1.3
    return pd.DataFrame(
        [{"open": c - 0.5, "high": c + 1.0, "low": c - 1.0, "close": c, "volume": 1000.0} for c in closes]
    )


def test_arquetipo_a_metadata_and_defaults():
    strategy = ArquetipoAStrategy()
    assert issubclass(ArquetipoAStrategy, TradingStrategy)
    assert strategy.describe()["name"] == "ArquetipoAStrategy"
    assert strategy.get_default_exit_policy() == ExitPolicy.TRAILING_STOP
    assert strategy.required_history_bars >= 185
    assert strategy._min_candles == strategy.required_history_bars


def test_screen1_weekly_slope_and_macd_hist():
    strategy = ArquetipoAStrategy()
    weekly_up = _make_series(45, base=50.0)
    assert strategy._screen1_weekly(weekly_up) is True

    weekly_down = weekly_up.copy()
    weekly_down.loc[weekly_down.index[-3:], "close"] = [
        weekly_down.loc[weekly_down.index[-4], "close"] + 3.0,
        weekly_down.loc[weekly_down.index[-4], "close"] + 2.0,
        weekly_down.loc[weekly_down.index[-4], "close"] + 1.0,
    ]
    assert strategy._screen1_weekly(weekly_down) is False
    assert strategy._screen1_weekly(_make_series(10)) is False


def test_screen2_daily_pullback_conditions():
    strategy = ArquetipoAStrategy()
    df = _make_series(40, base=100.0)
    df.loc[df.index[-1], "close"] = df.loc[df.index[-2], "close"] - 1.0
    assert strategy._screen2_daily(df) is True

    df_below = df.copy()
    df_below.loc[df_below.index[-1], "close"] = 50.0
    assert strategy._screen2_daily(df_below) is False


def test_stops_and_entries_atr_multiples():
    strategy = ArquetipoAStrategy()
    df = _make_series(200, base=50.0)
    df.loc[df.index[-1], "close"] = df.loc[df.index[-2], "close"] - 1.5
    sig = strategy.generate_signal(df.to_dict(orient="records"), "TEST", "1d", float(df["close"].iloc[-1]))

    assert sig is not None and sig.decision == "LONG" and sig.entry_order == "stop"
    assert sig.entry_expiry_bars == 3 and sig.take_profit is None and sig.sin_take_profit is True and sig.exit_policy == ExitPolicy.TRAILING_STOP

    atr_val = float(atr(df, n=14).iloc[-1])
    expected_entry = float(df["high"].iloc[-1]) + 0.1 * atr_val
    assert sig.entry_price == pytest.approx(expected_entry, rel=1e-5)

    stop_swing = float(df["low"].iloc[-5:].min()) - 2.0 * atr_val
    expected_stop = max(stop_swing, float(safezone_long(df, n=10, coef=2.0).iloc[-1]))
    assert sig.stop_loss == pytest.approx(expected_stop, rel=1e-5)


def test_chandelier_exit_in_should_exit():
    strategy = ArquetipoAStrategy(chandelier_mult=3.0, atr_period=14)
    df = _make_series(30, base=100.0)
    atr_val = float(atr(df, n=14).iloc[-1])
    pos = ActivePosition(
        id=1, symbol="TEST", side=OrderSide.BUY, entry_price=100.0, stop_loss=90.0,
        take_profit=None, quantity=10.0, candles_since_entry=5, exit_policy=ExitPolicy.TRAILING_STOP,
    )
    highest = float(df.iloc[- (pos.candles_since_entry + 1):]["high"].max())
    chand_stop = highest - 3.0 * atr_val

    should_exit, reason = strategy.should_exit(pos, chand_stop + 1.0, df)
    assert should_exit is False
    assert float(pos.stop_loss) == pytest.approx(max(90.0, chand_stop))

    should_exit, reason = strategy.should_exit(pos, chand_stop - 0.1, df)
    assert should_exit is True and reason == "TRAILING_STOP"


class FixedSignalStrategy(TradingStrategy):
    required_history_bars = 1

    def __init__(self, entry_price: float, stop_loss: float):
        self.entry_price, self.stop_loss, self.emitted = entry_price, stop_loss, False

    def generate_signal(self, kline_data, symbol, timeframe, current_price):
        if not self.emitted:
            self.emitted = True
            return TradingSignal(
                decision="LONG", confidence=1.0, entry_order="stop", entry_price=self.entry_price,
                entry_expiry_bars=3, stop_loss=self.stop_loss, exit_policy=ExitPolicy.TRAILING_STOP,
            )
        return None

    def should_exit(self, position, current_price, ohlc_data):
        return False, None

    def should_reevaluate(self, position, current_price):
        return False


def _run_bt(db_session, fill: str, candles: list, symbol: str) -> Trade:
    start = datetime(2026, 3, 1)
    for i, (op, hi, lo, cl) in enumerate(candles):
        db_session.add(MarketData(symbol=symbol, timeframe="1d", timestamp=start + timedelta(days=i),
                                  open=op, high=hi, low=lo, close=cl, volume=1000))
    db_session.commit()
    strat = FixedSignalStrategy(entry_price=100.0, stop_loss=90.0)
    bt = Backtest(start_date=start, end_date=start + timedelta(days=len(candles) - 1),
                  assets=[symbol], timeframe="1d",
                  config={"offline_data": True, "market_hours_filter": False, "slippage_pct": 0.0, "commission_pct": 0.0},
                  db_session=db_session, strategy=strat, fill=fill, intrabar_stops=True)
    bt.run()
    return db_session.query(Trade).filter(Trade.backtest_run_id == bt.backtest_run_id).one()


def test_stop_de_salida_en_misma_vela_del_llenado_stop_apertura(db_session):
    trade = _run_bt(db_session, "apertura", [
        (90.0, 92.0, 89.0, 91.0), (95.0, 105.0, 85.0, 88.0), (88.0, 90.0, 87.0, 89.0)
    ], "STOPSAME")
    assert float(trade.entry_price) == pytest.approx(100.0)
    assert float(trade.exit_price) == pytest.approx(90.0)
    assert trade.exit_signal == "STOP_LOSS"
    assert trade.opened_at == trade.closed_at


def test_stop_de_salida_en_misma_vela_del_llenado_stop_cierre(db_session):
    trade = _run_bt(db_session, "cierre", [
        (90.0, 92.0, 89.0, 91.0), (95.0, 105.0, 85.0, 92.0), (88.0, 90.0, 87.0, 89.0)
    ], "STOPCIERRE")
    assert float(trade.entry_price) == pytest.approx(100.0)
    assert trade.opened_at < trade.closed_at
    assert trade.exit_signal == "STOP_LOSS"


@pytest.mark.snapshot
def test_arquetipo_a_xlk_motor_sin_take_profit():
    """QuantAgent-fnpr.3.10: Arquetipo A en XLK no tiene salidas TAKE_PROFIT."""
    import os
    from pathlib import Path
    from quantagent.cli.backtest import _engine_config, _resolve_source_and_meta, correr_par

    snapshot_dir = os.environ.get("QUANTAGENT_SNAPSHOT_DIR")
    if not snapshot_dir or not (Path(snapshot_dir) / "etf-1d-2026-10-amplio").exists():
        pytest.skip("Snapshot amplio no disponible")

    config = _engine_config(None, "etf", None, "confianza", None, None)
    meta, symbol, to_date = _resolve_source_and_meta(
        "run", "arquetipo-a", None, "etf-1d-2026-10-amplio", "XLK",
        None, None, False, False,
    )
    r = correr_par(
        "arquetipo-a", meta, config, "etf", None, "etf-1d-2026-10-amplio", "XLK",
        None, to_date, True, False, con_filas=True, fill="cierre",
    )
    assert r["trades"] > 0
    assert not any(f.exit_reason == "TAKE_PROFIT" for f in r["filas"])

