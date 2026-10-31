"""Preparación H4 (M5-34.1): conjunto H3 y señales A, sin ejecución ni registros."""

from pathlib import Path

import h3
import yaml

from quantagent.backtesting.signals import bar_signals
from quantagent.strategy.arquetipo_a_strategy import ArquetipoAStrategy


def elegibilidad(w3, sesiones):
    """Activos habilitados por sesión según los pesos positivos de la semana vigente."""
    return w3.reindex(sesiones, method="ffill").fillna(0.0).gt(0).loc[h3.DESDE:h3.HASTA]


def senales_a(velas, simbolo):
    """Calienta A con historia previa y conserva todas las columnas de L24, aun fuera del conjunto."""
    spec = yaml.safe_load((Path(__file__).resolve().parents[2] / "hipotesis/arquetipo-a.yaml").read_text())
    estrategia = ArquetipoAStrategy(**{k: v[0] for k, v in spec["parametros"].items()})
    return bar_signals(estrategia, velas.loc[:h3.HASTA], simbolo, spec["timeframe"]).loc[h3.DESDE:]


def preparar_datos(cache="/tmp/ai-loop/h3-cache.pkl"):
    """Devuelve velas, pesos semanales, elegibilidad diaria y señales A para 2011–2018."""
    velas, w1, decisiones = h3.pesos_h1()
    etfs = {dec: [e for e in h3.scr.ETF_SECTOR if w1.loc[d, e] > 0] for d, dec in decisiones.items()}
    cand, _, _ = h3.candidatos(list(decisiones.values()), etfs, cache)
    w3 = h3.pesos_h3(w1, cand, decisiones)
    velas = {t: (velas[t] if t in velas else h3.read_snapshot(h3.ACC, t)).loc[:h3.HASTA] for t in w3.columns}
    habilitados = elegibilidad(w3, w1.index)
    senales = {t: senales_a(v, t) for t, v in velas.items()}
    return velas, w3, habilitados, senales
