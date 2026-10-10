"""Referencia comprar y mantener (QuantAgent-rnl): misma ventana, mismo activo, mismos costos que la corrida."""

import itertools
import math
import statistics
from datetime import datetime
from typing import Sequence


def buy_and_hold(
    timestamps: Sequence[datetime],
    closes: Sequence[float],
    initial_capital: float,
    slippage_pct: float,
    commission_pct: float,
    risk_free_rate: float = 0.02,
) -> dict:
    """Compra con todo el capital al cierre de la primera vela y vende al cierre de la última.

    Acciones fraccionarias, como el engine: la compra (precio + slippage, más comisión) gasta todo
    el capital y no queda efectivo. Un punto de equity por vela marcado al cierre; el último es lo
    que queda tras vender (precio - slippage, menos comisión). Sharpe con N de backtest_metrics.md §5.2.
    """
    qty = initial_capital / (closes[0] * (1 + slippage_pct) * (1 + commission_pct))
    proceeds = qty * closes[-1] * (1 - slippage_pct) * (1 - commission_pct)
    equity = [qty * c for c in closes[:-1]] + [proceeds]

    max_drawdown = max((p - e) / p for p, e in zip(itertools.accumulate(equity, max), equity))

    sharpe = 0.0
    returns = [(b - a) / a for a, b in zip(equity, equity[1:])]
    years = (timestamps[-1] - timestamps[0]).total_seconds() / (365.25 * 86400.0)
    if len(returns) > 1 and years > 0 and statistics.stdev(returns) > 0:
        n = len(returns) / years
        sharpe = (statistics.mean(returns) - risk_free_rate / n) / statistics.stdev(returns) * math.sqrt(n)
    return {"pnl": proceeds - initial_capital, "sharpe": sharpe, "max_drawdown": max_drawdown}
