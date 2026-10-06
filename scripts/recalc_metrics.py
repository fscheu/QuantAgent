#!/usr/bin/env python
"""Recalculate backtest metrics from a trade log CSV, independently of the engine (QuantAgent-hx0.5).

Usage: python scripts/recalc_metrics.py run.csv
Reads the CSV written by `backtest run --out` and only uses its `pnl` column. Imports nothing
from `quantagent`, so a matching number is a second opinion and not the engine checking itself.
"""

import csv
import math
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


if __name__ == "__main__":
    m = recalc(sys.argv[1])
    # Same labels and rounding as `backtest run`, so both outputs can be compared line by line.
    print(f"Trades: {m['trades']}")
    print(f"Win rate: {m['win_rate']:.2%}")
    print(f"Profit factor: {m['profit_factor']:.2f}")
    print(f"Total PnL: {m['total_pnl']:.2f}")
    print(f"PnL por trade: {m['matched']}/{m['trades']} filas coinciden")
    if m["matched"] != m["trades"]:
        sys.exit(1)
