#!/usr/bin/env python
"""QuantAgent-ryaw (spike): compara dos CSV de trades (columnas de `--out`), control contra candidato.

Uso:  python scripts/lean_spike/compare_trades.py <control.csv> <candidato.csv> [--detalle]

scripts/crossval_compare.py no recibe CSV externos (corre el motor contra el port de rsi), asi que este
script importa sus funciones `read_rows` y `compare_trades` y agrega lo que pide el criterio del ticket:
- "Coinciden N de M": trades del control con un trade del candidato de igual lado, fecha de entrada y
  fecha de salida. Se cuenta por conjunto y no por posicion, para que un trade que falta no arrastre a los
  siguientes.
- cuantos coinciden solo en la entrada (misma fecha y lado, otra salida) y cuantos faltan o sobran;
- desvio relativo del PnL total del candidato contra el control.
Despues imprime la comparacion posicion por posicion de crossval_compare.py, con sus tolerancias.
Sale con 0 solo si N == M y no sobran trades. `--detalle` lista las fechas de los que no coinciden.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import crossval_compare  # noqa: E402


def clave(row: dict, con_salida: bool = True) -> tuple:
    return (row["side"], row["entry_time"][:10]) + ((row["exit_time"][:10],) if con_salida else ())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("control")
    ap.add_argument("candidato")
    ap.add_argument("--detalle", action="store_true", help="lista las fechas de los trades que no coinciden")
    args = ap.parse_args(argv)
    control = crossval_compare.read_rows(Path(args.control))
    candidato = crossval_compare.read_rows(Path(args.candidato))

    exactos = {clave(r) for r in control} & {clave(r) for r in candidato}
    entradas = {clave(r, False) for r in control} & {clave(r, False) for r in candidato}
    solo_entrada = [r for r in control if clave(r) not in exactos and clave(r, False) in entradas]
    faltan = [r for r in control if clave(r, False) not in entradas]
    sobran = [r for r in candidato if clave(r, False) not in {clave(c, False) for c in control}]
    print(f"Coinciden {len(exactos)} de {len(control)} (lado, fecha de entrada y fecha de salida)")
    print(f"Solo la entrada coincide: {len(solo_entrada)}; faltan en el candidato: {len(faltan)}; "
          f"sobran en el candidato: {len(sobran)} (candidato: {len(candidato)} trades)")
    if args.detalle:
        for titulo, rows in (("solo entrada", solo_entrada), ("falta", faltan), ("sobra", sobran)):
            for r in rows:
                print(f"  {titulo}: {r['side']} {r['entry_time'][:10]} -> {r['exit_time'][:10]} ({r['exit_reason']})")

    pnl_control = sum(float(r["pnl"]) for r in control)
    pnl_candidato = sum(float(r["pnl"]) for r in candidato)
    print(f"PnL total: control {pnl_control:.2f}, candidato {pnl_candidato:.2f}, "
          f"desvio relativo {(pnl_candidato - pnl_control) / abs(pnl_control):+.4%}")

    print("--- crossval_compare.compare_trades (motor = control, port = candidato) ---")
    lines, _ = crossval_compare.compare_trades(control, candidato)
    print("\n".join(lines))
    return 0 if len(exactos) == len(control) and not sobran else 1


if __name__ == "__main__":
    sys.exit(main())
