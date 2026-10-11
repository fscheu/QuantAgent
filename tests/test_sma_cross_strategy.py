"""Tests de SmaCrossStrategy (QuantAgent-cok): series armadas a mano, sin red."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List

import pandas as pd
import pytest

from quantagent.backtesting.backtest import Backtest, check_intrabar_stops
from quantagent.cli.backtest import STRATEGY_ALIASES
from quantagent.models import ActivePosition, Environment, ExitPolicy, MarketData, OrderSide
from quantagent.strategy import build_strategy
from quantagent.strategy.sma_cross_strategy import SmaCrossStrategy

_START = datetime(2024, 1, 1)

# Con fast=3 y slow=5 (medias calculadas a mano):
#   índice 7: SMA3=6.00 < SMA5=6.20
#   índice 8: SMA3=7.00 > SMA5=6.40  -> cruce hacia arriba (entrada, cierre 8)
#   índice 13: SMA3=10.00 > SMA5=9.80
#   índice 14: SMA3=9.00 < SMA5=9.60 -> cruce hacia abajo (salida, cierre 8)
_CLOSES = [10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6, 5, 4]
_ENTRY_IDX = 8
_EXIT_IDX = 14


def _candles(closes: List[float]) -> List[Dict]:
    return [
        {
            "timestamp": _START + timedelta(days=i),
            "open": float(c),
            "high": float(c) + 0.5,
            "low": float(c) - 0.5,
            "close": float(c),
            "volume": 1_000_000.0,
        }
        for i, c in enumerate(closes)
    ]


def _signal(strategy: SmaCrossStrategy, candles: List[Dict]):
    return strategy.generate_signal(candles, "TEST", "1d", candles[-1]["close"])


def _position() -> ActivePosition:
    return ActivePosition(
        id=1,
        symbol="TEST",
        side=OrderSide.BUY,
        entry_price=8.0,
        stop_loss=0.01,
        take_profit=1_000_000_000.0,
        quantity=10.0,
        decision_timestamp=_START + timedelta(days=_ENTRY_IDX),
        candles_since_entry=0,
        exit_policy=ExitPolicy.SL_TP_ONLY,
        prediction_horizon=3,
        candles_direction=[],
        is_active=True,
        environment=Environment.BACKTEST,
    )


def test_entry_signal_only_on_the_session_of_the_upward_cross():
    """Entrada: LONG en la vela del cruce; nada antes ni con la tendencia ya en curso."""
    strategy = SmaCrossStrategy(fast=3, slow=5)
    candles = _candles(_CLOSES)

    assert _signal(strategy, candles[:_ENTRY_IDX]) is None  # todavía por debajo
    signal = _signal(strategy, candles[: _ENTRY_IDX + 1])
    assert signal is not None and signal.decision == "LONG"
    assert signal.entry_price == 8.0
    # Tendencia en curso (rápida ya arriba el día anterior): no entra a mitad de camino.
    for end in range(_ENTRY_IDX + 2, _EXIT_IDX + 1):
        assert _signal(strategy, candles[:end]) is None
    # Solo largos: en toda la serie no aparece ninguna señal que no sea LONG.
    decisions = {
        s.decision
        for s in (_signal(strategy, candles[:end]) for end in range(6, len(candles) + 1))
        if s is not None
    }
    assert decisions == {"LONG"}


def test_cross_from_equality_counts_as_entry():
    """'El día anterior estaba por debajo o igual': medias iguales ayer y rápida arriba hoy."""
    strategy = SmaCrossStrategy(fast=3, slow=5)
    flat = _candles([5, 5, 5, 5, 5, 5])  # SMA3 == SMA5
    assert _signal(strategy, flat) is None
    assert _signal(strategy, _candles([5, 5, 5, 5, 5, 5, 6])).decision == "LONG"


def test_exit_only_when_fast_goes_below_slow():
    """Salida: should_exit es falso mientras la rápida está arriba y verdadero en el cruce."""
    strategy = SmaCrossStrategy(fast=3, slow=5)
    df = pd.DataFrame(_candles(_CLOSES))

    for end in range(_ENTRY_IDX + 1, _EXIT_IDX + 1):
        window = df.iloc[:end]
        assert strategy.should_exit(_position(), float(window["close"].iloc[-1]), window) == (False, None)
    window = df.iloc[: _EXIT_IDX + 1]
    assert strategy.should_exit(_position(), 8.0, window) == (True, "SMA_CROSS_DOWN")


def test_no_take_profit_and_protective_stop_cannot_be_touched():
    """Los niveles de la señal no cierran la posición ni con una vela extrema."""
    strategy = SmaCrossStrategy(fast=3, slow=5)
    signal = _signal(strategy, _candles(_CLOSES[: _ENTRY_IDX + 1]))
    position = {"side": "buy", "stop_loss": signal.stop_loss, "take_profit": signal.take_profit}
    crash_and_spike = {"open": 8.0, "high": 8_000_000.0, "low": 0.02, "close": 1.0}
    assert check_intrabar_stops(position, crash_and_spike) is None
    assert signal.trailing_stop_pct is None and signal.max_hold_candles is None


def test_insufficient_history_returns_none_and_declares_slow_plus_one():
    """Sin historia suficiente no hay señal; la historia declarada es slow + 1."""
    default = SmaCrossStrategy()
    assert (default.fast, default.slow) == (50, 200)
    assert default.required_history_bars == 201
    rising = _candles([100.0 + i for i in range(200)])  # una vela menos que la necesaria
    assert _signal(default, rising) is None
    assert default.generate_signal([], "TEST", "1d", 100.0) is None

    small = SmaCrossStrategy(fast=3, slow=5)
    assert small.required_history_bars == 6
    # 5 velas alcanzan para las medias de hoy pero no para las de ayer: no puede haber cruce.
    assert _signal(small, _candles([5, 5, 5, 5, 9])) is None
    assert small.should_exit(_position(), 5.0, pd.DataFrame(_candles([9, 8, 7, 6]))) == (False, None)

    with pytest.raises(ValueError):
        SmaCrossStrategy(fast=200, slow=50)


def test_registered_and_aliased():
    assert STRATEGY_ALIASES["sma-cross"] == "SmaCrossStrategy"
    assert isinstance(build_strategy("SmaCrossStrategy"), SmaCrossStrategy)


def test_short_backtest_end_to_end_produces_one_complete_trade(db_session):
    """Punta a punta con el motor: entra en el cruce hacia arriba y sale en el cruce hacia abajo."""
    for candle in _candles(_CLOSES):
        db_session.add(
            MarketData(
                symbol="TEST",
                timeframe="1d",
                timestamp=candle["timestamp"],
                open=Decimal(str(candle["open"])),
                high=Decimal(str(candle["high"])),
                low=Decimal(str(candle["low"])),
                close=Decimal(str(candle["close"])),
                volume=Decimal(str(candle["volume"])),
            )
        )
    db_session.commit()

    backtest = Backtest(
        start_date=_START,
        end_date=_START + timedelta(days=len(_CLOSES) - 1),
        assets=["TEST"],
        timeframe="1d",
        initial_capital=100_000.0,
        config={"market_hours_filter": False, "offline_data": True},
        db_session=db_session,
        strategy=SmaCrossStrategy(fast=3, slow=5),
    )
    metrics = backtest.run(name="QuantAgent-cok-e2e")

    positions = (
        db_session.query(ActivePosition)
        .filter(ActivePosition.backtest_run_id == backtest.backtest_run_id)
        .all()
    )
    assert metrics.total_trades == 1
    assert len(positions) == 1
    position = positions[0]
    assert position.side == OrderSide.BUY
    assert position.is_active is False
    assert position.decision_timestamp == _START + timedelta(days=_ENTRY_IDX)
    assert position.closed_at == _START + timedelta(days=_EXIT_IDX)
    assert position.close_reason == "SMA_CROSS_DOWN"
    assert metrics.close_reasons == {"SMA_CROSS_DOWN": 1}
