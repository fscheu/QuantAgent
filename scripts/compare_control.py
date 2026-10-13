#!/usr/bin/env python
"""Control de compare contra backtesting.py (V23b, V23c: sensibilidad apertura siguiente)."""

from __future__ import annotations

import argparse
import csv
import functools
import io
import os
import sys
from pathlib import Path

from tabulate import tabulate

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_old_slippage = os.environ.get("TRADING_SLIPPAGE_PCT")
import bt_adapter  # noqa: E402
import crossval_compare  # noqa: E402
import crossval_rsi as port  # noqa: E402

if _old_slippage is not None:
    os.environ["TRADING_SLIPPAGE_PCT"] = _old_slippage
else:
    os.environ.pop("TRADING_SLIPPAGE_PCT", None)

from quantagent import settings  # noqa: E402

settings.TRADING_SLIPPAGE_PCT = float(os.getenv("TRADING_SLIPPAGE_PCT", "0.0005"))
from quantagent.backtesting.export import trades_to_csv  # noqa: E402
from quantagent.cli.backtest import generar_markdown_compare  # noqa: E402
from quantagent.data.snapshot import load_manifest  # noqa: E402
from quantagent.strategy.registry import build_strategy, cli_aliases  # noqa: E402

QTY_RTOL = crossval_compare.QTY_RTOL
PRICE_TOL = crossval_compare.PRICE_TOL
NUMERIC = crossval_compare.NUMERIC
pnl_tol = crossval_compare.pnl_tol


@functools.lru_cache(maxsize=32)
def _load_snapshot_df(snapshot: str, symbol: str, from_date: str | None, to_date: str | None):
    return port.load_fixture(snapshot=snapshot, symbol=symbol, from_date=from_date, to_date=to_date)


def compare_trades_with_reasons(eng: list[dict], prt: list[dict]) -> str:
    if len(eng) != len(prt):
        return f"sin validar: distinta cantidad de trades (motor={len(eng)}, port={len(prt)})"
    for i, (e, p) in enumerate(zip(eng, prt)):
        bad = [c for c in ("entry_time", "exit_time", "symbol", "side", "exit_reason") if e.get(c) != p.get(c)]
        tols = {"qty": QTY_RTOL * abs(float(e["qty"])), "entry_price": PRICE_TOL, "exit_price": PRICE_TOL, "pnl": pnl_tol(e)}
        for c in NUMERIC:
            ve, vp = e.get(c, ""), p.get(c, "")
            if ve == "" or vp == "":
                if ve != vp:
                    bad.append(c)
            elif abs(float(ve) - float(vp)) > tols[c]:
                bad.append(c)
        if bad:
            return f"sin validar: fila {i + 1}, columna {bad[0]}"
    return f"coinciden {len(eng)} de {len(eng)}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tabla", required=True, help="CSV generado por backtest compare")
    ap.add_argument("--trades-dir", required=True, help="Directorio con trades del motor")
    ap.add_argument("--snapshot", required=True, help="Nombre del snapshot")
    ap.add_argument("--costos", choices=sorted(settings.COSTOS_PERFILES), help="Perfil de costos")
    ap.add_argument("--tamano-posicion", type=float, default=1.0, help="Tamaño de posición (default 1.0)")
    ap.add_argument("--regla-tamano", choices=["confianza", "fijo"], default="confianza", help="Regla de tamaño (default confianza)")
    ap.add_argument("--out", help="CSV de salida (default: sobrescribe --tabla)")
    ap.add_argument("--md", help="Documento Markdown de salida")
    args = ap.parse_args(argv)

    with open(args.tabla, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0

    if args.costos:
        costos_cfg = settings.COSTOS_PERFILES.get(args.costos, {})
        costos = {"commission_pct": costos_cfg.get("commission_pct", 0.0), "slippage_pct": costos_cfg.get("slippage_pct", 0.0)}
    else:
        costos = {"commission_pct": settings.TRADING_COMMISSION_PCT, "slippage_pct": settings.TRADING_SLIPPAGE_PCT}
    manifest = load_manifest(args.snapshot)
    timeframe = manifest["timeframe"]
    aliases = cli_aliases()

    for row in rows:
        strat, sym = row["estrategia"], row["activo"]
        motor_file = Path(args.trades_dir) / f"{strat}-{sym}.csv"
        if not motor_file.exists():
            row["control"] = f"sin validar: falta archivo {motor_file.name}"
            row["pnl_apertura_siguiente"] = ""
            row["trades_apertura_siguiente"] = ""
            continue
        with open(motor_file, newline="") as f:
            eng_rows = list(csv.DictReader(f))
        try:
            df = _load_snapshot_df(args.snapshot, sym, row.get("desde"), row.get("hasta")).copy()
            settings.TRADING_BASE_POSITION_PCT = args.tamano_posicion
            bt_adapter.settings.TRADING_BASE_POSITION_PCT = args.tamano_posicion
            stats = bt_adapter.run_adapter(build_strategy(aliases[strat]), df, symbol=sym, timeframe=timeframe, fill="close", **costos)
            prt_rows = list(csv.DictReader(io.StringIO(trades_to_csv(bt_adapter.trade_rows(stats, sym)))))
            row["control"] = compare_trades_with_reasons(eng_rows, prt_rows)
        except Exception as exc:
            row["control"] = f"sin validar: error del adaptador ({exc})"

        try:
            df_open = _load_snapshot_df(args.snapshot, sym, row.get("desde"), row.get("hasta")).copy()
            settings.TRADING_BASE_POSITION_PCT = args.tamano_posicion
            bt_adapter.settings.TRADING_BASE_POSITION_PCT = args.tamano_posicion
            stats_open = bt_adapter.run_adapter(build_strategy(aliases[strat]), df_open, symbol=sym, timeframe=timeframe, fill="next-open", **costos)
            open_rows = bt_adapter.trade_rows(stats_open, sym)
            row["pnl_apertura_siguiente"] = round(sum(r.pnl for r in open_rows), 2)
            row["trades_apertura_siguiente"] = len(open_rows)
        except Exception:
            row["pnl_apertura_siguiente"] = ""
            row["trades_apertura_siguiente"] = ""

    fieldnames = list(rows[0].keys())
    for col in ("control", "pnl_apertura_siguiente", "trades_apertura_siguiente"):
        if col not in fieldnames:
            fieldnames.append(col)
    out_path = args.out or args.tabla
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.md:
        md_text = generar_markdown_compare(
            rows,
            snapshot_name=args.snapshot,
            costos=args.costos,
            tamano_posicion=args.tamano_posicion,
            regla_tamano=args.regla_tamano,
            columns=fieldnames,
        )
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md_text)

    table_data = [[r.get(c, "") for c in fieldnames] for r in rows]
    print(tabulate(table_data, headers=fieldnames, tablefmt="github"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
