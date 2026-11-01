"""Lectura de artefactos semanales y retornos de sombra, sin ejecutar el pipeline."""

from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd

from quantagent.fabrica.screener import ETF_SECTOR
from quantagent.models import EquitySnapshot
from quantagent.trading.cartera_paper import ultima_sesion_cerrada_nyse
from quantagent.trading.pipeline_m5 import UNIVERSO_M5, fundamentales_vigente


def resumen_semanal(day, session, equity, *, carpeta, data_provider=None, base_dir=None):
    """Devuelve líneas para teléfono; proveedor y raíz de snapshots inyectables."""
    carpeta = Path(carpeta)
    tablas = {}
    for nombre in ("regimen", "candidatos", "sombra"):
        sufijo = "semanales" if nombre == "candidatos" else "semanal"
        path = carpeta / f"m5-{nombre}-{sufijo}.csv"
        tablas[nombre] = pd.read_csv(path).fillna("") if path.exists() else pd.DataFrame()
    descartes_path = carpeta / "m5-candidatos-descartes.csv"
    descartes = pd.read_csv(descartes_path).fillna("") if descartes_path.exists() else pd.DataFrame()
    corte = pd.Timestamp(day)
    regimen = tablas["regimen"]
    regimen = regimen[pd.to_datetime(regimen.fecha) <= corte] if not regimen.empty else regimen
    lines = ["Candidatos de la semana"]
    if regimen.empty:
        lines.append("- sin dato de régimen")
        descartes = descartes.iloc[:0]
    else:
        vigente = regimen.sort_values("fecha").iloc[-1]
        fecha = pd.Timestamp(vigente.fecha)
        candidatos = tablas["candidatos"]
        candidatos = candidatos[pd.to_datetime(candidatos.fecha) == fecha] if not candidatos.empty else candidatos
        descartes = descartes[pd.to_datetime(descartes.fecha) == fecha] if not descartes.empty else descartes
        globales = descartes[descartes.simbolo == ""].motivo.tolist() if not descartes.empty else []
        for etf in str(vigente.habilitados).split(","):
            if etf not in ETF_SECTOR:
                continue
            filas = candidatos[candidatos.etf == etf] if not candidatos.empty else candidatos
            seleccion = ", ".join(f"{r.simbolo} (rank {int(r.rank)})" for r in
                                  filas.sort_values("rank").itertuples()) if not filas.empty else ""
            motivo = "; ".join(globales or ([*dict.fromkeys(descartes.motivo)] if not descartes.empty else
                                           ["sin candidatos elegibles"]))
            lines.append(f"- {ETF_SECTOR[etf]} ({etf}): {seleccion or 'solo-ETF; ' + motivo}")
    for etiqueta, patron in (("precio", "precio"), ("fundamentales", "fundamentales")):
        filas = descartes[(descartes.simbolo != "") & descartes.motivo.str.contains(patron)] if not descartes.empty else descartes
        ejemplos = "; ".join(f"{r.simbolo}: {r.motivo}" for r in filas.head(3).itertuples())
        cantidad = filas.simbolo.nunique() if not filas.empty else 0
        lines.append(f"Descartadas por {etiqueta}: {cantidad}" + (f"; {ejemplos}" if ejemplos else ""))
    nombre = fundamentales_vigente(base_dir or carpeta)
    sello = nombre.removeprefix("fundamentales-sp500-")
    antiguedad = (corte - pd.Timestamp(sello)).days
    lines.append(f"Fundamentales vigentes: {sello} ({antiguedad} días; fecha mensual = mes)")
    lines.extend(comparacion_sombra(day, tablas["sombra"], data_provider, session, equity))
    return lines


def comparacion_sombra(day, sombra, proveedor, session, equity):
    """Encadena retornos cierre a cierre sin anticipar decisiones del cierre final."""
    lines = ["Comparación semanal / acumulada desde sombra", f"Equity pipeline: {equity:.2f} USD"]
    cierre = ultima_sesion_cerrada_nyse(datetime.combine(day, time.max))
    if sombra.empty:
        return lines + ["- Pipeline / H1 / H3 / 1/N: sin dato"]
    sombra = sombra.copy()
    sombra["fecha"] = pd.to_datetime(sombra.fecha)
    sombra = sombra[sombra.fecha < cierre]
    if sombra.empty:
        return lines + ["- Pipeline / H1 / H3 / 1/N: sin dato"]
    fechas = sorted(sombra.fecha.unique())
    limites = [*map(pd.Timestamp, fechas), cierre]
    lunes = day - timedelta(days=day.weekday())
    inicio_semana = ultima_sesion_cerrada_nyse(datetime.combine(lunes, time.min))
    panel = {}
    if proveedor is not None:
        activos = set(sombra.loc[(sombra.peso_h1 != 0) | (sombra.peso_h3 != 0), "activo"]) | set(UNIVERSO_M5)
        for activo in sorted(activos):
            try:
                df = proveedor.get_ohlc(activo, "1d", limites[0].to_pydatetime(),
                                        (cierre + pd.Timedelta(days=1)).to_pydatetime())
            except Exception:
                continue
            if not df.empty:
                fechas_precio = pd.to_datetime(df.timestamp).dt.tz_localize(None).dt.normalize()
                serie = pd.Series(df.close.to_numpy(), index=fechas_precio)
                panel[activo] = serie[~serie.index.duplicated(keep="last")]
    for etiqueta, columna in (("H1", "peso_h1"), ("H3", "peso_h3"), ("1/N", None)):
        retornos = []
        for inicio, fin in zip(limites, limites[1:]):
            filas = sombra[sombra.fecha == inicio]
            pesos = (dict.fromkeys(UNIVERSO_M5, 1 / len(UNIVERSO_M5)) if columna is None else
                     dict(zip(filas.activo, filas[columna])))
            retorno = 0.0
            for activo, peso in pesos.items():
                if not peso:
                    continue
                serie = panel.get(activo, pd.Series(dtype=float))
                a, b = serie.get(inicio), serie.get(fin)
                if a is None or b is None or pd.isna(a) or pd.isna(b) or a <= 0 or b <= 0:
                    retorno = float("nan")
                    break
                retorno += peso * (b / a - 1)
            retornos.append(retorno)
        acumulado = pd.Series([1 + r for r in retornos]).prod(skipna=False) - 1
        semanal = retornos[-1] if limites[-2] == inicio_semana else float("nan")
        lines.append(f"- {etiqueta}: {fmt(semanal)} / {fmt(acumulado)}")
    bases = {r.snapshot_date: r.equity for r in session.query(EquitySnapshot).all()}
    valores = [equity / bases[d.date()] - 1 if bases.get(d.date()) else float("nan")
               for d in (inicio_semana, limites[0])]
    lines.append(f"- Pipeline: {fmt(valores[0])} / {fmt(valores[1])}")
    return lines


def fmt(retorno):
    return "sin dato" if pd.isna(retorno) else f"{retorno:+.2%}"
