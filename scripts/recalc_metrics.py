#!/usr/bin/env python
"""Recalculate backtest metrics independently of the engine (QuantAgent-hx0.5, QuantAgent-hx0.6).

Usage:
  python scripts/recalc_metrics.py run.csv
  python scripts/recalc_metrics.py --equity eq.csv [--periods-per-year N]
  python scripts/recalc_metrics.py run.csv --equity eq.csv [--periods-per-year N]

Reads CSVs written by `backtest run` (--out and/or --equity-out). Imports nothing from
`quantagent`, so matching numbers are a second opinion and not the engine checking itself.
"""

import csv
import itertools
import math
import statistics
import sys


def recalc(path):
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
    for row in rows:
        entry = float(row["entry_price"])
        exit_ = float(row["exit_price"])
        qty = float(row["qty"])
        trade_pnl = (exit_ - entry) * qty if row["side"].lower() == "buy" else (entry - exit_) * qty
        if abs(trade_pnl - float(row["pnl"])) <= 0.01:
            matched += 1

    return {
        "trades": trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "matched": matched,
    }


def recalc_equity(path, periods_per_year=252 * 6.5, risk_free_rate=0.02):
    with open(path, newline="") as f:
        equities = [float(r["equity"]) for r in csv.DictReader(f) if r.get("equity")]

    if len(equities) < 2:
        return {"max_drawdown": 0.0, "sharpe": 0.0}

    # Drawdown formula: peak-to-trough decline from running maximum
    max_drawdown = max((p - e) / p for p, e in zip(itertools.accumulate(equities, max), equities))

    # Sharpe formula: annualized excess return over sample standard deviation
    returns = [(b - a) / a for a, b in zip(equities, equities[1:])]
    std = statistics.stdev(returns) if len(returns) > 1 else 0.0
    sharpe = ((statistics.mean(returns) - risk_free_rate / periods_per_year) / std) * math.sqrt(periods_per_year) if std else 0.0

    return {"max_drawdown": max_drawdown, "sharpe": sharpe}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("trades_csv", nargs="?")
    parser.add_argument("--equity")
    parser.add_argument("--periods-per-year", type=float, default=252 * 6.5)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    args = parser.parse_args()

    m = recalc(args.trades_csv) if args.trades_csv else None
    eq = recalc_equity(args.equity, args.periods_per_year, args.risk_free_rate) if args.equity else None

    if m:
        print(f"Trades: {m['trades']}")
        print(f"Win rate: {m['win_rate']:.2%}")
        print(f"Profit factor: {m['profit_factor']:.2f}")
    if eq:
        print(f"Sharpe ratio: {eq['sharpe']:.2f}")
    if m:
        print(f"Total PnL: {m['total_pnl']:.2f}")
        print(f"PnL por trade: {m['matched']}/{m['trades']} filas coinciden")
    if eq:
        print(f"Max drawdown: {eq['max_drawdown']:.6f}")
    if m and m["matched"] != m["trades"]:
        sys.exit(1)
