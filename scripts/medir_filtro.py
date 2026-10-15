"""Medición del filtro M3, exclusivamente en el ajuste 2007–2018."""

import argparse
from time import perf_counter

STARTED = perf_counter()  # incluye imports y primera compilación JIT

import pandas as pd

from quantagent.backtesting.fixtures import _snapshot_range
from quantagent.backtesting.signals import bar_signals
from quantagent.backtesting.vbt_filter import run_filter
from quantagent.settings import COSTOS_PERFILES
from quantagent.strategy.registry import build_strategy, cli_aliases

STRATEGIES = ("rsi", "fifty-two-week-high", "sma-cross", "momentum-12m")
SYMBOLS = ("SPY", "IWM", "EFA", "EEM", "TLT", "SHY", "GLD", "DBC", "XLE", "XLF", "XLK", "XLU")
SNAPSHOT = "etf-1d-2026-10"


def medir(out):
    rows = []
    for strategy in STRATEGIES:
        for symbol in SYMBOLS:
            candles = _snapshot_range(SNAPSHOT, symbol, pd.Timestamp("2007-01-01"), pd.Timestamp("2018-12-31"))
            candles = candles.rename_axis("timestamp").reset_index(drop=False).set_index("timestamp", drop=False)
            instance = build_strategy(cli_aliases()[strategy])
            start = perf_counter()
            signals = bar_signals(instance, candles, symbol, "1d")
            signals_s = perf_counter() - start
            start = perf_counter()
            metrics = run_filter(signals, candles, capital=100_000.0, **COSTOS_PERFILES["etf"])
            rows.append({"strategy": strategy, "symbol": symbol, **metrics,
                         "signals_s": signals_s, "simulation_s": perf_counter() - start})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Celdas: {len(rows)}; señales: {sum(r['signals_s'] for r in rows):.6f}s; "
          f"simulación: {sum(r['simulation_s'] for r in rows):.6f}s; pared total: {perf_counter() - STARTED:.6f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="CSV de las 48 celdas")
    medir(parser.parse_args().out)
