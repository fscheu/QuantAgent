"""Tests de Momentum12mStrategy (QuantAgent-jgn): series armadas a mano, sin red."""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List

import pandas as pd
import pytest

from quantagent.backtesting.backtest import Backtest, check_intrabar_stops
from quantagent.cli.backtest import STRATEGY_ALIASES
from quantagent.models import ActivePosition, Environment, ExitPolicy, MarketData, OrderSide
from quantagent.strategy import build_strategy
from quantagent.strategy.momentum_12m_strategy import (
    Momentum12mStrategy,
    is_last_session_of_month,
)

# Sesiones reales de NYSE del 2023-09-05 al 2023-11-17 (lunes a viernes; en ese tramo
# no hay feriados). Últimas sesiones de mes: viernes 2023-09-29 y martes 2023-10-31.
_SESSIONS = [d.to_pydatetime() for d in pd.bdate_range("2023-09-05", "2023-11-17")]
_SEP_END = datetime(2023, 9, 29)
_OCT_END = datetime(2023, 10, 31)


def _closes() -> List[float]:
    """Sube 1 por sesión hasta fin de septiembre, baja 1 en octubre, sube en noviembre."""
    closes, price = [], 100.0
    for day in _SESSIONS:
        if day <= _SEP_END:
            price += 1.0
        elif day <= _OCT_END:
            price -= 1.0
        else:
            price += 1.0
        closes.append(price)
    return closes


def _candles(closes: List[float] = None, sessions: List[datetime] = None) -> List[Dict]:
    closes = _closes() if closes is None else closes
    sessions = _SESSIONS if sessions is None else sessions
    return [
        {
            "timestamp": day,
            "open": c,
            "high": c + 0.5,
            "low": c - 0.5,
            "close": c,
            "volume": 1_000_000.0,
        }
        for day, c in zip(sessions, closes)
    ]


def _upto(candles: List[Dict], day: datetime) -> List[Dict]:
    return [c for c in candles if c["timestamp"] <= day]


def _signal(strategy: Momentum12mStrategy, candles: List[Dict]):
    return strategy.generate_signal(candles, "TEST", "1d", candles[-1]["close"])


def _position() -> ActivePosition:
    return ActivePosition(
        id=1,
        symbol="TEST",
        side=OrderSide.BUY,
        entry_price=119.0,
        stop_loss=0.01,
        take_profit=1_000_000_000.0,
        quantity=10.0,
        decision_timestamp=_SEP_END,
        candles_since_entry=0,
        exit_policy=ExitPolicy.SL_TP_ONLY,
        prediction_horizon=3,
        candles_direction=[],
        is_active=True,
        environment=Environment.BACKTEST,
    )


def test_last_session_of_month_follows_the_market_calendar():
    """Fin de mes de mercado, no de calendario: fines de semana y feriados lo adelantan."""
    assert is_last_session_of_month(pd.Timestamp("2023-09-29"))  # el 30 es sábado
    assert not is_last_session_of_month(pd.Timestamp("2023-09-28"))
    assert is_last_session_of_month(pd.Timestamp("2023-10-31"))
    assert not is_last_session_of_month(pd.Timestamp("2023-11-01"))
    assert is_last_session_of_month(pd.Timestamp("2024-03-28"))  # el 29 es Viernes Santo
    assert is_last_session_of_month(pd.Timestamp("2018-12-31"))


def test_entry_only_on_last_session_of_month_with_positive_return():
    """Entrada: LONG a fin de mes con retorno positivo; ningún otro día genera señal."""
    strategy = Momentum12mStrategy(lookback=5)
    candles = _candles()

    signal = _signal(strategy, _upto(candles, _SEP_END))
    assert signal is not None and signal.decision == "LONG"
    assert signal.entry_price == 119.0  # 100 + 19 sesiones de septiembre

    # El retorno de 5 sesiones es positivo todo septiembre, pero solo decide el 29.
    signal_days = [
        day
        for day in _SESSIONS[5:]
        if _signal(strategy, _upto(candles, day)) is not None
    ]
    assert signal_days == [_SEP_END]

    # Fin de mes con retorno exactamente cero: no entra (tiene que ser mayor).
    flat = _candles([100.0] * len(_SESSIONS))
    assert _signal(strategy, _upto(flat, _SEP_END)) is None


def test_exit_only_on_last_session_of_month_with_non_positive_return():
    """Salida: a fin de mes con retorno <= 0; a mitad de mes no sale aunque el retorno sea negativo."""
    strategy = Momentum12mStrategy(lookback=5)
    df = pd.DataFrame(_candles())

    exit_days = []
    for day in _SESSIONS[5:]:
        window = df[df["timestamp"] <= day]
        should_exit, reason = strategy.should_exit(
            _position(), float(window["close"].iloc[-1]), window
        )
        if should_exit:
            exit_days.append((day, reason))
    # Octubre cae todos los días (retorno negativo desde la primera semana): sale recién el 31.
    assert exit_days == [(_OCT_END, "MOMENTUM_NON_POSITIVE")]

    # Retorno exactamente cero a fin de mes: sale (menor o igual a cero).
    flat = pd.DataFrame(_upto(_candles([100.0] * len(_SESSIONS)), _OCT_END))
    assert strategy.should_exit(_position(), 100.0, flat) == (True, "MOMENTUM_NON_POSITIVE")
    # Retorno positivo a fin de mes: se queda.
    rising = pd.DataFrame(_upto(_candles(), _SEP_END))
    assert strategy.should_exit(_position(), 119.0, rising) == (False, None)


def test_decision_uses_no_data_after_the_decision_candle():
    """Sin mirada al futuro: la vela de decisión se reconoce por su fecha, no por lo que sigue."""
    strategy = Momentum12mStrategy(lookback=5)
    candles = _candles()

    # 1) La estrategia decide con la serie cortada en la vela de decisión: no existe
    #    ninguna vela de octubre en lo que recibe y aun así sabe que el 29 es fin de mes.
    cut = _upto(candles, _SEP_END)
    assert cut[-1]["timestamp"] == _SEP_END
    decided = _signal(strategy, cut)
    assert decided is not None and decided.decision == "LONG"

    # 2) Lo que pase después no cambia la decisión: mismo resultado con dos futuros opuestos.
    for future in (1_000.0, 1.0):
        altered = [dict(c) for c in candles]
        for c in altered:
            if c["timestamp"] > _SEP_END:
                c.update(open=future, high=future, low=future, close=future)
        again = _signal(strategy, _upto(altered, _SEP_END))
        assert again.model_dump() == decided.model_dump()

    # 3) Trampa típica: tomar "la última vela que tengo" como fin de mes. Con la serie
    #    cortada el 28 (penúltima sesión) no hay señal aunque sea la última vela recibida.
    assert _signal(strategy, _upto(candles, datetime(2023, 9, 28))) is None
    # 4) La entrada usa el cierre de la vela de decisión y el de `lookback` sesiones antes,
    #    nada más: cambiar cualquier otra vela anterior no altera la señal.
    noisy = [dict(c) for c in cut]
    for c in noisy[:-6] + noisy[-5:-1]:
        c["close"] = 1.0
    assert _signal(strategy, noisy) is not None
    noisy[-6]["close"] = 500.0  # el cierre de 5 sesiones antes pasa a ser mayor: sin señal
    assert _signal(strategy, noisy) is None


def test_no_take_profit_and_protective_stop_cannot_be_touched():
    """Los niveles de la señal no cierran la posición ni con una vela extrema."""
    signal = _signal(Momentum12mStrategy(lookback=5), _upto(_candles(), _SEP_END))
    position = {"side": "buy", "stop_loss": signal.stop_loss, "take_profit": signal.take_profit}
    extreme = {"open": 119.0, "high": 119_000_000.0, "low": 0.02, "close": 1.0}
    assert check_intrabar_stops(position, extreme) is None
    assert signal.trailing_stop_pct is None and signal.max_hold_candles is None


def test_insufficient_history_returns_none_and_declares_lookback_plus_one():
    """Sin historia suficiente no hay señal ni salida; la historia declarada es lookback + 1."""
    default = Momentum12mStrategy()
    assert default.lookback == 252
    assert default.required_history_bars == 253
    assert default.generate_signal([], "TEST", "1d", 100.0) is None

    # 252 sesiones en alza que terminan en un fin de mes: falta una vela para el retorno.
    sessions = [d.to_pydatetime() for d in pd.bdate_range(end="2023-10-31", periods=252)]
    short = _candles([100.0 + i for i in range(252)], sessions)
    assert _signal(default, short) is None
    assert default.should_exit(_position(), 351.0, pd.DataFrame(short)) == (False, None)

    small = Momentum12mStrategy(lookback=5)
    assert small.required_history_bars == 6
    five = _upto(_candles(), _SEP_END)[-5:]
    assert _signal(small, five) is None
    assert small.should_exit(_position(), 119.0, pd.DataFrame(five)) == (False, None)

    with pytest.raises(ValueError):
        Momentum12mStrategy(lookback=0)


def test_registered_and_aliased():
    assert STRATEGY_ALIASES["momentum-12m"] == "Momentum12mStrategy"
    assert isinstance(build_strategy("Momentum12mStrategy"), Momentum12mStrategy)


def test_short_backtest_end_to_end_produces_one_complete_trade(db_session):
    """Punta a punta con el motor: entra a fin de septiembre y sale a fin de octubre."""
    for candle in _candles():
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
        start_date=_SESSIONS[0],
        end_date=_SESSIONS[-1],
        assets=["TEST"],
        timeframe="1d",
        initial_capital=100_000.0,
        config={"market_hours_filter": False, "offline_data": True},
        db_session=db_session,
        strategy=Momentum12mStrategy(lookback=5),
    )
    metrics = backtest.run(name="QuantAgent-jgn-e2e")

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
    assert position.decision_timestamp == _SEP_END
    assert position.closed_at == _OCT_END
    assert position.close_reason == "MOMENTUM_NON_POSITIVE"
    assert metrics.close_reasons == {"MOMENTUM_NON_POSITIVE": 1}
    assert metrics.total_pnl < 0  # compra a ~119 y vende a ~97
