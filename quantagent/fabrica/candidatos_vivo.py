"""Entradas vivas del screener semanal, con reloj y proveedor inyectados (M5-55b)."""

import pandas as pd

from quantagent.data.fundamentales import metricas_fundamentales
from quantagent.data.snapshot import SnapshotError, load_manifest, snapshot_root
from quantagent.data.sp500 import miembros, sector
from quantagent.fabrica.screener import (
    COLS_CANDIDATOS,
    ETF_SECTOR,
    _retorno,
    candidatos_semanales,
)
from quantagent.trading.cartera_paper import ultima_sesion_cerrada_nyse


def candidatos_en_vivo(
    as_of, etfs_habilitados, data_provider, fundamentales,
    composicion="sp500-2026-10", base_dir=None,
):
    """Devuelve (candidatos, descartes) para la última sesión NYSE cerrada.

    `fundamentales` es el nombre del snapshot vigente elegido por el llamador.
    Un descarte con simbolo=None indica que falta el snapshot completo.
    """
    fecha = ultima_sesion_cerrada_nyse(as_of)
    vacios = pd.DataFrame(columns=COLS_CANDIDATOS)
    if fundamentales is None:
        return vacios, [{"simbolo": None, "motivo": "sin snapshot de fundamentales vigente"}]
    try:
        manifest = load_manifest(fundamentales, base_dir)
    except SnapshotError as exc:
        return vacios, [{"simbolo": None, "motivo": f"sin fundamentales: {exc}"}]
    if not manifest["symbols"]:
        return vacios, [{"simbolo": None, "motivo": f"snapshot de fundamentales vacío: {fundamentales}"}]

    composicion = str((snapshot_root(base_dir) / composicion).resolve())
    etfs = list(etfs_habilitados)
    sectores = {ETF_SECTOR[e] for e in etfs if e in ETF_SECTOR}
    acciones = [t for t in miembros(composicion, fecha)
                if sector(composicion, t, fecha) in sectores] if sectores else []
    descartes = [{"simbolo": t, "motivo": f"sin fundamentales en snapshot {fundamentales}"}
                 for t in acciones if t not in manifest["symbols"]]
    inicio = (fecha - pd.DateOffset(years=1)).to_pydatetime()
    fin = (fecha + pd.Timedelta(days=1)).to_pydatetime()
    precios = {}
    for simbolo in dict.fromkeys([*etfs, *acciones]):
        df = data_provider.get_ohlc(simbolo, "1d", inicio, fin)
        cierres = pd.Series(dtype=float)
        if not df.empty:
            fechas = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
            cierres = pd.Series(df["close"].to_numpy(), index=fechas)
            cierres = cierres[~cierres.index.duplicated(keep="last")].sort_index()
            cierres = cierres.loc[:fecha].dropna()
        precios[simbolo] = cierres
        if simbolo in acciones and all(pd.isna(_retorno(cierres, fecha, n)) for n in (63, 126)):
            motivo = "sin precio" if cierres.empty else "sin precio con historia suficiente o reciente"
            descartes.append({"simbolo": simbolo, "motivo": motivo})

    tickers = [t for t in acciones if t in manifest["symbols"]]
    paneles = metricas_fundamentales(fundamentales, tickers, pd.DatetimeIndex([fecha]), base_dir=base_dir)
    candidatos = candidatos_semanales(
        fecha, etfs, precios=precios, paneles_fundamentales=paneles, composicion=composicion,
    )
    return candidatos, descartes
