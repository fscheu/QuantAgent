"""Tests de la referencia comprar y mantener (QuantAgent-rnl). Los esperados se calculan en el test, no salen del engine."""

import math
from datetime import datetime, timedelta

import pytest

from quantagent.backtesting.buy_and_hold import buy_and_hold

DAYS = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
CLOSES = [100.0, 90.0, 120.0]


def test_three_candles_without_costs_returns_the_price_change_on_all_the_capital():
    """Sin costos, todo el capital sigue al precio: 10000 * (120 / 100 - 1) = 2000; el pozo 100 -> 90 es 10%."""
    r = buy_and_hold(DAYS, CLOSES, 10_000, 0.0, 0.0)
    assert r["pnl"] == pytest.approx(2000.0)
    assert r["max_drawdown"] == pytest.approx(0.10)


def test_three_candles_with_slippage_and_commission_on_both_sides():
    """Compra a 100 * 1.001 pagando 0.2% de comisión con todo el capital; vende a 120 * 0.999 y paga 0.2% otra vez."""
    r = buy_and_hold(DAYS, CLOSES, 10_000, 0.001, 0.002)

    qty = 10_000 / (100 * 1.001 * 1.002)  # 99.7006... acciones, sin efectivo sobrante
    proceeds = qty * 120 * 0.999 * 0.998
    assert r["pnl"] == pytest.approx(proceeds - 10_000, abs=1e-9)  # 1928.22...
    assert r["pnl"] < 2000 - 10_000 * (0.001 + 0.002)  # los costos de entrada y salida se notan

    # Equity por vela: qty * 100, qty * 90, proceeds. N = 2 retornos / (2 días / 365.25) = 365.25.
    r1, r2 = -0.1, proceeds / (qty * 90) - 1
    mean, std = (r1 + r2) / 2, math.sqrt(((r1 - r2) / 2) ** 2 * 2)
    assert r["sharpe"] == pytest.approx((mean - 0.02 / 365.25) / std * math.sqrt(365.25))
    assert r["max_drawdown"] == pytest.approx(0.10)
