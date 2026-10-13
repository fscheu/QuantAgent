#!/usr/bin/env python
"""QuantAgent-ryaw (spike): pasa los closedTrades del JSON de resultados de LEAN al CSV de `--out`.

Uso:  python scripts/lean_spike/closed_trades_to_csv.py <resultado.json> --out trades.csv

Probado contra los resultados completos de SMA y RSI. Rechaza state.RuntimeError.
Los campos se buscan sin distinguir mayusculas.

Columnas: las de quantagent/backtesting/export.py::COLUMNS. entry_time y exit_time son la sesion de Nueva
York del fill a las 00:00, como en los CSV del motor y de bt_adapter.py. stop_loss y exit_reason salen de
los tags que pone lean_algos.py en las ordenes (entrada `stop=<nivel>`, salida el motivo), mediante
los orderIds de cada trade. LEAN exporta symbols como una lista y fechas de fills en UTC.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantagent.backtesting.export import TradeRow, trades_to_csv  # noqa: E402

NY = ZoneInfo("America/New_York")


def campo(obj: dict, nombre: str):
    for k, v in obj.items():
        if k.lower() == nombre.lower():
            return v
    return None


def instante(texto: str) -> datetime:
    return datetime.fromisoformat(texto.replace("Z", "+00:00"))


def sesion(texto: str) -> datetime:
    return datetime.combine(instante(texto).astimezone(NY).date(), datetime.min.time())


def filas(resultado: dict) -> list[TradeRow]:
    trades = campo(campo(resultado, "totalPerformance"), "closedTrades")
    ordenes = campo(resultado, "orders")
    out = []
    for t in trades:
        simbolo = campo(t, "symbols")[0]
        simbolo = campo(simbolo, "value") if isinstance(simbolo, dict) else simbolo
        largo = str(campo(t, "direction")).lower() in ("0", "long")
        ids = campo(t, "orderIds")
        tag_entrada = campo(ordenes[str(ids[0])], "tag") or ""
        stop = float(tag_entrada.split("=", 1)[1]) if tag_entrada.startswith("stop=") else None
        out.append(TradeRow(
            sesion(campo(t, "entryTime")), sesion(campo(t, "exitTime")), simbolo.split()[0],
            "buy" if largo else "sell", abs(float(campo(t, "quantity"))), float(campo(t, "entryPrice")),
            float(campo(t, "exitPrice")), stop, float(campo(t, "profitLoss")),
            campo(ordenes[str(ids[-1])], "tag") or "",
        ))
    return sorted(out, key=lambda r: r.entry_time)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("resultado", help="JSON de resultados de LEAN (<algoritmo>.json)")
    ap.add_argument("--out", required=True, help="CSV de trades")
    args = ap.parse_args(argv)
    resultado = json.loads(Path(args.resultado).read_text())
    estado = campo(resultado, "state") or {}
    if campo(estado, "runtimeError"):
        raise SystemExit(f"LEAN fallo: {campo(estado, 'runtimeError')}")
    rows = filas(resultado)
    Path(args.out).write_text(trades_to_csv(rows))
    print(f"Trades cerrados: {len(rows)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
