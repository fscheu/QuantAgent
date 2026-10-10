#!/usr/bin/env python
"""Recalculate backtest metrics independently of the engine (QuantAgent-hx0.5, QuantAgent-hx0.6).

Usage:
  python scripts/recalc_metrics.py run.csv [--commission-pct C]
  python scripts/recalc_metrics.py --equity eq.csv [--periods-per-year N]
  python scripts/recalc_metrics.py run.csv --equity eq.csv [--periods-per-year N]
  python scripts/recalc_metrics.py --comprar-y-mantener velas.csv|velas.parquet [--slippage-pct S] [--commission-pct C]
      [--from YYYY-MM-DD] [--to YYYY-MM-DD]

Reads CSVs written by `backtest run` (--out and/or --equity-out). Imports nothing from
`quantagent`, so matching numbers are a second opinion and not the engine checking itself.
"""

import csv
import datetime
import itertools
import math
import statistics
import sys


def recalc(path, commission_pct=0.0):
    with open(path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["pnl"]]  # rows without pnl are still open

    pnls = [float(row["pnl"]) for row in rows]
    gross_profit = math.fsum(p for p in pnls if p > 0)  # sum of the winning trades
    gross_loss = -math.fsum(p for p in pnls if p < 0)  # sum of the losing trades, as a positive number

    trades = len(pnls)  # closed trades, one per row
    total_pnl = math.fsum(pnls)  # wins minus losses
    win_rate = sum(p > 0 for p in pnls) / trades if trades else 0.0  # winners / all trades (a 0 pnl is not a win)
    profit_factor = gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else 0.0)  # $ won per $ lost

    matched = 0
    diffs = []  # pnl del CSV menos pnl recalculado, solo filas que no coinciden
    for row in rows:
        entry = float(row["entry_price"])
        exit_ = float(row["exit_price"])
        qty = float(row["qty"])
        trade_pnl = (exit_ - entry) * qty if row["side"].lower() == "buy" else (entry - exit_) * qty
        # Comisión por lado sobre el nocional ejecutado: una al entrar y otra al salir.
        trade_pnl -= commission_pct * qty * (entry + exit_)
        if abs(trade_pnl - float(row["pnl"])) <= 0.01:
            matched += 1
        else:
            diffs.append(float(row["pnl"]) - trade_pnl)

    return {
        "trades": trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "matched": matched,
        "diffs": diffs,
    }


def recalc_equity(path, periods_per_year=None, risk_free_rate=0.02):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("equity")]
    timestamps = [r.get("timestamp") or r.get("date") for r in rows]
    return equity_metrics([float(r["equity"]) for r in rows], timestamps, periods_per_year, risk_free_rate)


def equity_metrics(equities, timestamps, periods_per_year=None, risk_free_rate=0.02):
    if len(equities) < 2:
        return {"max_drawdown": 0.0, "sharpe": 0.0}

    # Drawdown formula: peak-to-trough decline from running maximum
    max_drawdown = max((p - e) / p for p, e in zip(itertools.accumulate(equities, max), equities))

    # Sharpe formula: annualized excess return over sample standard deviation
    returns = [(b - a) / a for a, b in zip(equities, equities[1:])]
    std = statistics.stdev(returns) if len(returns) > 1 else 0.0
    if not std:
        return {"max_drawdown": max_drawdown, "sharpe": 0.0}

    if periods_per_year is None:
        if len(timestamps) < 2 or not timestamps[0] or not timestamps[-1]:
            return {"max_drawdown": max_drawdown, "sharpe": 0.0}
        t0 = datetime.datetime.fromisoformat(str(timestamps[0]))
        t1 = datetime.datetime.fromisoformat(str(timestamps[-1]))
        elapsed_seconds = (t1 - t0).total_seconds()
        elapsed_years = elapsed_seconds / (365.25 * 86400.0)
        if elapsed_years <= 0:
            return {"max_drawdown": max_drawdown, "sharpe": 0.0}
        periods_per_year = len(returns) / elapsed_years

    if periods_per_year <= 0:
        return {"max_drawdown": max_drawdown, "sharpe": 0.0}

    sharpe = ((statistics.mean(returns) - risk_free_rate / periods_per_year) / std) * math.sqrt(periods_per_year)

    return {"max_drawdown": max_drawdown, "sharpe": sharpe, "periods_per_year": periods_per_year}


def buy_and_hold(path, slippage_pct=0.0, commission_pct=0.0, capital=100000.0, risk_free_rate=0.02,
                 from_date=None, to_date=None):
    """Compra todo al primer cierre (+slippage, +comisión) y vende al último (-slippage, -comisión).

    Acciones fraccionarias, sin efectivo sobrante; un punto de equity por vela al cierre y el último ya vendido.
    Velas: CSV con timestamp y close, o parquet del snapshot (índice de fechas; usa adj_close si está).
    from_date / to_date (YYYY-MM-DD, inclusivos) recortan las velas antes de calcular.
    """
    if str(path).endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(path)
        closes = df["adj_close" if "adj_close" in df else "close"].astype(float).tolist()
        timestamps = [ts.isoformat() for ts in df.index]
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        closes, timestamps = [float(r["close"]) for r in rows], [r["timestamp"] for r in rows]
    if from_date or to_date:
        keep = [(not from_date or t[:10] >= from_date) and (not to_date or t[:10] <= to_date) for t in timestamps]
        closes = [c for c, k in zip(closes, keep) if k]
        timestamps = [t for t, k in zip(timestamps, keep) if k]

    qty = capital / (closes[0] * (1 + slippage_pct) * (1 + commission_pct))
    proceeds = qty * closes[-1] * (1 - slippage_pct) * (1 - commission_pct)
    m = equity_metrics([qty * c for c in closes[:-1]] + [proceeds], timestamps, None, risk_free_rate)
    return {"pnl": proceeds - capital, **m}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("trades_csv", nargs="?")
    parser.add_argument("--equity")
    parser.add_argument("--periods-per-year", type=float, default=None)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--commission-pct", type=float, default=0.0, help="por lado; 0.001 = 0.10%%")
    parser.add_argument("--comprar-y-mantener", metavar="VELAS")
    parser.add_argument("--slippage-pct", type=float, default=0.0, help="por lado; solo con --comprar-y-mantener")
    parser.add_argument("--capital", type=float, default=100000.0, help="solo con --comprar-y-mantener")
    parser.add_argument("--from", dest="from_date", help="primera sesión YYYY-MM-DD; solo con --comprar-y-mantener")
    parser.add_argument("--to", dest="to_date", help="última sesión YYYY-MM-DD; solo con --comprar-y-mantener")
    args = parser.parse_args()

    if args.comprar_y_mantener:
        bh = buy_and_hold(
            args.comprar_y_mantener, args.slippage_pct, args.commission_pct, args.capital, args.risk_free_rate,
            args.from_date, args.to_date,
        )
        print(f"Comprar y mantener: PnL {bh['pnl']:.2f}, Sharpe {bh['sharpe']:.2f}, max drawdown {bh['max_drawdown']:.6f}")

    m = recalc(args.trades_csv, args.commission_pct) if args.trades_csv else None
    eq = recalc_equity(args.equity, args.periods_per_year, args.risk_free_rate) if args.equity else None

    if m:
        print(f"Trades: {m['trades']}")
        print(f"Win rate: {m['win_rate']:.2%}")
        pf = m["profit_factor"]
        if pf is None or math.isinf(pf):
            print("Profit factor: n/a")
        else:
            print(f"Profit factor: {pf:.2f}")
    if eq:
        print(f"Sharpe ratio: {eq['sharpe']:.2f}")
    if m:
        print(f"Total PnL: {m['total_pnl']:.2f}")
        print(f"PnL por trade: {m['matched']}/{m['trades']} filas coinciden")
        if m["diffs"]:
            d = m["diffs"]
            print(f"Diferencia CSV - recalculado: total {math.fsum(d):.2f}, min {min(d):.2f}, max {max(d):.2f}")
    if eq:
        print(f"Max drawdown: {eq['max_drawdown']:.6f}")
    if m and m["matched"] != m["trades"]:
        sys.exit(1)
