#!/usr/bin/env python
"""QuantAgent-ryaw (spike): escribe un simbolo diario del snapshot en el formato de datos de LEAN.

Uso:  python scripts/lean_spike/write_lean_data.py --out <carpeta-data> [--symbol SPY] [--enteros]

Escribe bajo <carpeta-data>/equity/usa/:
- daily/<simbolo>.zip con <simbolo>.csv, sin encabezado: `YYYYMMDD 00:00,open,high,low,close,volume`,
  precios multiplicados por 10.000 (deci-centavos, la escala de LEAN para equity).
- factor_files/<simbolo>.csv y map_files/<simbolo>.csv minimos: factor 1 y sin cambio de ticker. Los
  precios del snapshot ya vienen ajustados; el algoritmo los lee con normalizacion Raw.

Por defecto el precio escalado lleva 4 decimales (8 decimales del precio, los mismos que usan el motor
y bt_adapter.py, QuantAgent-bwqb). `--enteros` redondea a entero, que es lo que traen los datos de
QuantConnect. El Launcher local aceptó los decimales, factor 1 y map file sin cambios
en las corridas completas de SMA y RSI; no fue necesario usar --enteros.

El rango esta fijo en 2007-01-03..2018-12-31 (tramo de desarrollo): este script no lee mas alla.
La carpeta de salida va fuera del repo; los datos del snapshot no se commitean.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantagent.backtesting.fixtures import _snapshot_range  # noqa: E402

SNAPSHOT = "etf-1d-2026-10"
DESDE, HASTA = "2007-01-03", "2018-12-31"
ESCALA = 10_000


def lineas_diarias(df, enteros: bool) -> list[str]:
    """Una linea por vela, en el orden del snapshot. Los precios pasan primero a 8 decimales, como el control."""
    fmt = (lambda x: str(round(x))) if enteros else (lambda x: f"{x:.4f}")
    out = []
    for ts, row in df.iterrows():
        precios = [fmt(float(f"{row[c]:.8f}") * ESCALA) for c in ("open", "high", "low", "close")]
        out.append(",".join([ts.strftime("%Y%m%d 00:00"), *precios, str(int(row["volume"]))]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="carpeta de datos de LEAN (fuera del repo)")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--enteros", action="store_true", help="precios escalados redondeados a entero")
    args = ap.parse_args(argv)

    df = _snapshot_range(SNAPSHOT, args.symbol, DESDE, HASTA)
    if df.index.max().strftime("%Y-%m-%d") > HASTA:
        raise SystemExit(f"el rango leido pasa de {HASTA}")
    ticker = args.symbol.lower()
    base = Path(args.out) / "equity" / "usa"
    for sub in ("daily", "factor_files", "map_files"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base / "daily" / f"{ticker}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{ticker}.csv", "\n".join(lineas_diarias(df, args.enteros)) + "\n")
    # fecha, factor de precio, factor de split, precio de referencia
    (base / "factor_files" / f"{ticker}.csv").write_text("19980102,1,1,0\n20501231,1,1,0\n")
    # fecha, ticker, bolsa primaria (P = NYSE Arca)
    (base / "map_files" / f"{ticker}.csv").write_text(f"19980102,{ticker},P\n20501231,{ticker},P\n")
    print(f"{args.symbol}: {len(df)} velas, {df.index.min():%Y-%m-%d} a {df.index.max():%Y-%m-%d} -> {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
