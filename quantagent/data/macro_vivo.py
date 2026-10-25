"""Lectura de macro reciente para decisiones paper, separada de snapshots históricos."""

import pandas as pd


class MacroVencida(RuntimeError):
    pass


def serie_vigente(serie: str, sesiones, cliente=None) -> pd.Series:
    """Devuelve el valor publicado antes de cada sesión; rechaza ausencias de más de 7 días."""
    if cliente is None:
        from quantagent.data.sources.fred import FredClient

        cliente = FredClient()
    sesiones = pd.DatetimeIndex(sesiones)
    df = cliente.observaciones(serie, sesiones[0] - pd.Timedelta(days=8), sesiones[-1], todas_las_versiones=True)
    valores = []
    for dia in sesiones:
        vigentes = df[(df.fecha < dia) & (df.realtime_start <= dia) & (dia <= df.realtime_end)].dropna(subset=["valor"])
        if vigentes.empty:
            raise MacroVencida(f"{serie}: sin observación vigente al {dia.date()}")
        obs = vigentes.iloc[-1]
        if dia - obs.fecha > pd.Timedelta(days=7):
            raise MacroVencida(f"{serie}: última observación {obs.fecha.date()} vencida al {dia.date()}")
        valores.append(float(obs.valor))
    return pd.Series(valores, index=sesiones, name=serie)
