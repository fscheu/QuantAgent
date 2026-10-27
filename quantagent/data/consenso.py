from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from quantagent.data.snapshot import snapshot_root


def guardar_foto(
    simbolos: Iterable[str],
    cliente: Any,
    fecha: Optional[date | str] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Guarda en Parquet la foto diaria de consenso de EPS para los símbolos indicados."""
    if fecha is None:
        fecha_obj = date.today()
    elif isinstance(fecha, str):
        fecha_obj = date.fromisoformat(fecha)
    else:
        fecha_obj = fecha

    fecha_str = fecha_obj.isoformat()
    carpeta = snapshot_root(base_dir) / "consenso-eps"
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo = carpeta / f"{fecha_str}.parquet"
    if archivo.exists():
        raise FileExistsError(f"el archivo de consenso para {fecha_str} ya existe: {archivo}")

    metodo = getattr(cliente, "estimaciones_analistas", None) or getattr(cliente, "analyst_estimates", None)
    if not callable(metodo) and callable(cliente):
        metodo = cliente

    filas = []
    for sym in simbolos:
        try:
            items = metodo(sym)
            if not items:
                print(f"{sym}: sin datos")
                continue
            for item in items:
                fila = dict(item)
                fila["fecha_foto"] = fecha_str
                if "symbol" not in fila:
                    fila["symbol"] = sym
                filas.append(fila)
        except Exception as exc:
            print(f"{sym}: error ({exc})")

    df = pd.DataFrame(filas)
    if df.empty:
        df = pd.DataFrame(columns=["symbol", "date", "fecha_foto", "epsAvg", "epsHigh", "epsLow", "numAnalystsEps", "revenueAvg"])
    df.to_parquet(archivo, index=False)
    return archivo
