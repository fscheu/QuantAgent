"""Referencia comprar y mantener (QuantAgent-rnl): misma ventana, mismo activo, mismos costos que la corrida."""

import itertools
import math
import statistics
from datetime import datetime
from typing import Sequence


def annual_return(first: float, last: float, t0: datetime, t1: datetime) -> float:
    """(last / first) ** (1 / años) - 1, con años = (t1 - t0) / 365.25 días; 0.0 si la ventana no dura nada."""
    years = (t1 - t0).total_seconds() / (365.25 * 86400.0)
    return (last / first) ** (1 / years) - 1 if years > 0 else 0.0


def exposure(curve: Sequence[dict]) -> tuple:
    """(fracción de velas con posición, media de |positions_value| / equity) sobre la curva de equity del engine."""
    n = len(curve)
    return (
        sum(p["positions_value"] != 0 for p in curve) / n,
        sum(abs(p["positions_value"]) / p["equity"] for p in curve) / n,
    )


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
    return {
        "pnl": proceeds - initial_capital, "sharpe": sharpe, "max_drawdown": max_drawdown,
        "annual_return": annual_return(initial_capital, proceeds, timestamps[0], timestamps[-1]),
    }


def buy_and_hold_cuenta_chica(
    timestamps: Sequence[datetime],
    closes: Sequence[float],
    capital: float,
    slippage_pct: float,
    commission_pct: float,
    *,
    costo_entrada_usd: float = 0.0,
    dividend_rates: dict | None = None,
    withholding: float = 0.0,
) -> dict:
    """Benchmark de cuenta chica con acciones enteras, costos y retenciones."""
    dividend_rates = dividend_rates or {}
    shares = math.floor((capital - costo_entrada_usd) /
                        (closes[0] * (1 + slippage_pct) * (1 + commission_pct)))
    cash_left = capital - costo_entrada_usd - shares * closes[0] * (1 + slippage_pct) * (1 + commission_pct)
    curve = []
    taxes = 0.0
    for timestamp, price in zip(timestamps, closes):
        taxes += shares * price * dividend_rates.get(timestamp.date().isoformat(), 0.0) * withholding
        curve.append(cash_left + shares * price - taxes)
    final = cash_left + shares * closes[-1] * (1 - slippage_pct) * (1 - commission_pct) - taxes
    curve[-1] = final
    returns = [b / a - 1 for a, b in zip(curve, curve[1:]) if a]
    years = (timestamps[-1] - timestamps[0]).total_seconds() / (365.25 * 86400)
    deviation = statistics.stdev(returns) if len(returns) > 1 else 0.0
    sharpe = statistics.mean(returns) / deviation * math.sqrt(len(returns) / years) if deviation and years > 0 else 0.0
    return {"pnl": final - capital, "return": final / capital - 1, "sharpe": sharpe,
            "shares": shares, "cash_left": cash_left}


def worst_day(curve: Sequence[dict]) -> tuple:
    """(menor retorno diario entre días de calendario, fecha AAAA-MM-DD); (0.0, None) si ningún día cae."""
    daily = {}
    for p in curve:
        ts = p.get("date") or p.get("timestamp")
        day = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
        daily[day] = float(p["equity"])
    days = list(daily.items())
    drops = [
        ((curr - prev) / prev, day)
        for (_, prev), (day, curr) in zip(days, days[1:])
        if prev > 0 and curr < prev
    ]
    if not drops:
        return 0.0, None
    return min(drops, key=lambda x: x[0])

