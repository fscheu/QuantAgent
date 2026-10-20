"""Ficha de postmortem (QuantAgent-ui1p): por qué cortó una cartera, solo con causas estructurales.

La ficha lee las filas por celda de la corrida del motor, pero escribe únicamente el id y los nombres de
las causas: sin Sharpe, DSR, pnl ni conteos, para que el generador de ideas no pueda ajustar contra ellos.
"""

from pathlib import Path

import pandas as pd
import yaml

from quantagent.backtesting.verdict import MAX_DRAWDOWN
from quantagent.settings import COSTOS_PERFILES

# Umbrales fijos; no se ajustan por resultados.
MIN_TRADES = 30        # mismo mínimo de trades de cartera que `filtro.veredicto_etapa1` (DC cartera de primas, D3)
ROTACION_MAX = 12      # trades por activo y año: más de uno por mes ya no es una prima de señal lenta
COSTO_MAX = 0.5        # los costos son la mitad o más de la ganancia bruta
DOMINIO_MAX = 0.7      # una pierna aporta más del 70% del |pnl| de las piernas
CAPITAL = 100_000.0    # capital por celda del motor
CARPETA = Path("catalogo/postmortem")


def ficha_postmortem(h, filas, carpeta=CARPETA):
    """Clasifica la causa de corte de `h` según `filas` (trades, pnl, max_drawdown, desde, hasta por celda)."""
    trades = sum(int(f["trades"]) for f in filas)
    anios = (pd.Timestamp(filas[0]["hasta"]) - pd.Timestamp(filas[0]["desde"])).days / 365.25
    perfil = COSTOS_PERFILES["etf"]
    costo = trades * 2 * (perfil["slippage_pct"] + perfil["commission_pct"]) * CAPITAL
    pnl_pierna = {}
    for f in filas:
        pierna = f.get("pierna_clase") or f["estrategia"]
        pnl_pierna[pierna] = pnl_pierna.get(pierna, 0.0) + float(f["pnl"])
    bruto = sum(pnl_pierna.values()) + costo
    absolutos = [abs(v) for v in pnl_pierna.values()]
    causas = {
        "pocos_trades": trades < MIN_TRADES,
        "costos_comen_ganancia": bruto > 0 and costo >= COSTO_MAX * bruto,
        "rotacion_alta": trades / len(filas) / anios > ROTACION_MAX,
        "una_pierna_domina": len(absolutos) > 1 and max(absolutos) > DOMINIO_MAX * sum(absolutos),
        "drawdown": any(float(f["max_drawdown"] or 0) > MAX_DRAWDOWN for f in filas),
    }
    ficha = {"id": h.id, "causas": [c for c, activa in causas.items() if activa]}
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    (Path(carpeta) / f"{h.id}.yaml").write_text(yaml.safe_dump(ficha, allow_unicode=True), encoding="utf-8")
    return ficha
