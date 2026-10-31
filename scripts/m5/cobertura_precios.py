"""Cobertura sin red por miembro/año/sector. Uso: python scripts/m5/cobertura_precios.py."""

import pandas as pd
import pandas_market_calendars as mcal

from quantagent.data.snapshot import load_manifest, read_snapshot, snapshot_root
from quantagent.data.sp500 import RECLASIFICACIONES_CSV, miembros, sector

ACC, COMP = "acciones-sp500-2026-10c", "sp500-2026-10"


def cobertura(nombre=ACC):
    m = load_manifest(nombre)
    sesiones = mcal.get_calendar("NYSE").schedule(m["requested_start"], m["requested_end"]).index
    pertenencia = {}
    for d in sesiones:
        for t in miembros(COMP, d):
            pertenencia.setdefault(t, []).append(d)
    cortes = sorted(set(pd.read_csv(RECLASIFICACIONES_CSV)["desde"]))
    filas = []
    for t, dias in sorted(pertenencia.items()):
        precios = read_snapshot(nombre, t, adjusted=False) if t in m["symbols"] else pd.DataFrame()
        validos = precios.dropna(subset=["close", "adj_close"]).index if not precios.empty else []
        fechas = pd.DatetimeIndex(dias)
        for anio in sorted(set(fechas.year)):
            limites = [pd.Timestamp(anio, 1, 1), *[pd.Timestamp(c) for c in cortes if c.startswith(str(anio))],
                       pd.Timestamp(anio + 1, 1, 1)]
            for inicio, fin in zip(limites, limites[1:]):
                f = fechas[(fechas >= inicio) & (fechas < fin)]
                if len(f):
                    filas.append((anio, sector(COMP, t, f[0]) or "Desconocido", t, len(f), int(f.isin(validos).sum())))
    return pd.DataFrame(filas, columns=["anio", "sector", "ticker", "dias", "precios"]).groupby(
        ["anio", "sector", "ticker"], as_index=False
    )[["dias", "precios"]].sum()


if __name__ == "__main__":
    det = cobertura()
    det.to_csv(snapshot_root() / ACC / "cobertura.csv", index=False)
    det["con"] = det.precios.gt(0)
    det["sin"] = det.precios.eq(0)
    print(det.groupby(["sector", "anio"])[["con", "sin"]].sum().to_string())
