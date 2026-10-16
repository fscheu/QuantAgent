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
            rows.append({"strategy": strategy, "symbol": symbol, "desde": "2007-01-01", "hasta": "2018-12-31", **metrics,
                         "signals_s": signals_s, "simulation_s": perf_counter() - start})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Celdas: {len(rows)}; señales: {sum(r['signals_s'] for r in rows):.6f}s; "
          f"simulación: {sum(r['simulation_s'] for r in rows):.6f}s; pared total: {perf_counter() - STARTED:.6f}s")


def comparar(filtro, apertura, cierre):
    expected = {(s, a) for s in STRATEGIES for a in SYMBOLS}
    tables = []
    for path in (filtro, apertura, cierre):
        table = pd.read_csv(path).rename(columns={"estrategia": "strategy", "activo": "symbol"})
        keys = list(zip(table.strategy, table.symbol))
        if len(keys) != 48 or set(keys) != expected:
            raise ValueError(f"{path}: se requieren las mismas 48 claves únicas")
        if not ((table.desde >= "2007-01-01") & (table.hasta <= "2018-12-31")).all():
            raise ValueError(f"{path}: período requerido 2007-01-01 a 2018-12-31")
        tables.append(table.set_index(["strategy", "symbol"]).sort_index())
    for fill, motor in zip(("apertura", "cierre"), tables[1:]):
        ranks = [t.sharpe_tasa_0.rank(method="average") for t in (tables[0], motor)]
        print(f"Spearman {fill}: {ranks[0].corr(ranks[1]):.6f}")
        delta = (tables[0].trades - motor.trades).rename("delta_trades")
        print(f"Diferencias de trades {fill} (filtro - motor):")
        print(delta.to_string())
        print(f"Top cinco {fill} por diferencia absoluta:")
        print(delta.loc[delta.abs().sort_values(ascending=False, kind="stable").head(5).index].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="CSV de las 48 celdas; se lee al comparar")
    parser.add_argument("--apertura", help="CSV real de backtest compare con apertura")
    parser.add_argument("--cierre", help="CSV real de backtest compare con cierre")
    args = parser.parse_args()
    if bool(args.apertura) != bool(args.cierre):
        parser.error("--apertura y --cierre se requieren juntos")
    if args.apertura:
        comparar(args.out, args.apertura, args.cierre)
    else:
        medir(args.out)
