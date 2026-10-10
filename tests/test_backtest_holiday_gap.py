"""QuantAgent-8to: test que verifica que un hueco de feriados no saltea la evaluacion de una posicion abierta."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from quantagent.backtesting.backtest import Backtest
from quantagent.models import MarketData, OrderSide
from quantagent.strategy.rsi_strategy import RSIMeanReversionStrategy


def test_holiday_gap_evaluates_active_position(db_session):
    """
    Verifica que cuando un activo tiene un hueco de feriados dentro de los 44 dias
    de calendario, pero tiene historia suficiente (>= 30 velas en total):
    - El engine pide la historia por cantidad de velas (30) y no por dias de calendario (44).
    - La posicion abierta se evalua y se cierra en la vela actual si toca el stop loss.
    Con el codigo anterior, la ventana de 44 dias traia solo 25 velas (< 30) y hacia
    return antes de evaluar la posicion abierta, dejandola sin cerrar (is_active=True).
    """
    symbol = "TEST_GAP"
    timeframe = "1d"

    # 10 velas del 1 al 10 de enero
    candles = []
    base_date = datetime(2026, 1, 1)
    for i in range(10):
        ts = base_date + timedelta(days=i)
        candles.append(
            MarketData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=ts,
                open=Decimal("100.0"),
                high=Decimal("102.0"),
                low=Decimal("98.0"),
                close=Decimal("100.0"),
                volume=Decimal("1000"),
            )
        )

    # Hueco de 20 dias (feriados/vacaciones)
    # 25 velas del 31 de enero al 24 de febrero (total: 35 velas)
    gap_start = datetime(2026, 1, 31)
    for i in range(25):
        ts = gap_start + timedelta(days=i)
        close_price = "90.0" if i == 24 else "100.0"  # En la ultima vela (24 de feb) el precio cae a 90
        low_price = "89.0" if i == 24 else "98.0"
        candles.append(
            MarketData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=ts,
                open=Decimal("100.0"),
                high=Decimal("102.0"),
                low=Decimal(low_price),
                close=Decimal(close_price),
                volume=Decimal("1000"),
            )
        )

    db_session.bulk_save_objects(candles)
    db_session.commit()

    eval_date = gap_start + timedelta(days=24)  # 2026-02-24

    # En 2026-02-24, los ultimos 44 dias de calendario empiezan el 2026-01-11.
    # Entre 2026-01-11 y 2026-02-24 solo hay 25 velas en la base (< 30 requeridas).
    # Pero en el total de la base hay 35 velas (>= 30).

    strategy = RSIMeanReversionStrategy()
    bt = Backtest(
        start_date=datetime(2026, 1, 1),
        end_date=eval_date,
        assets=[symbol],
        timeframe=timeframe,
        config={"offline_data": True, "market_hours_filter": False},
        db_session=db_session,
        strategy=strategy,
    )

    # Creamos una posicion activa abierta comprada a 100 con stop loss en 95
    pos = bt.position_monitor.open_position(
        symbol=symbol,
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=105.0,
        quantity=Decimal("10.0"),
        exit_policy="sl_tp_only",
        timestamp=gap_start + timedelta(days=23),
    )
    assert pos.is_active is True

    # Ejecutamos el analisis de la vela en eval_date (donde el precio cayo a 90)
    bt._analyze_and_trade(symbol, eval_date)

    # Con el engine corregido:
    # 1. Se obtienen las ultimas 30 velas cargadas (len == 30).
    # 2. La posicion activa se evalua contra el close de 90.0 y se cierra por STOP_LOSS.
    db_session.refresh(pos)
    assert pos.is_active is False, "La posicion abierta deberia haberse cerrado por stop loss"
    assert pos.close_reason == "STOP_LOSS"
    assert pos.closed_at == eval_date
    assert bt.total_candles_processed == 1
