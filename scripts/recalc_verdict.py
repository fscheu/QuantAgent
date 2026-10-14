#!/usr/bin/env python
"""Recalcula el veredicto de `backtest verdict` sin usar el motor (QuantAgent-0oqm, V36).

Uso: python scripts/recalc_verdict.py salida_verdict.csv

No importa `quantagent`: la regla del Juego 2 (V30) está reescrita acá con sus propias
constantes, así que si coincide es una segunda opinión y no el comando revisándose a sí mismo.
"""

import csv
import sys

# Juego 2, umbrales de V30.
MIN_TRADES = 2
MIN_SHARPE = 0.10
MAX_DD = 0.35

# Prefijo de motivo y sufijo de columna para cada llenado: al cierre y a la apertura siguiente.
LLENADOS = (("cierre", ""), ("apertura", "_apertura_siguiente"))


def motivos_llenado(fila, sufijo):
    motivos = []
    for campo, falla, texto in (
        ("trades", lambda v: v < MIN_TRADES, "trades < 2"),
        ("pnl", lambda v: v <= 0, "pnl <= 0"),
        ("sharpe_tasa_0", lambda v: v < MIN_SHARPE, "sharpe_tasa_0 < 0.10"),
        ("max_drawdown", lambda v: v > MAX_DD, "max_drawdown > 0.35"),
    ):
        valor = (fila.get(campo + sufijo) or "").strip()
        if not valor:
            motivos.append(f"sin {campo}")
        elif falla(float(valor)):
            motivos.append(texto)
    return motivos


def recalcular(fila):
    motivos = [f"{prefijo}: {m}" for prefijo, sufijo in LLENADOS for m in motivos_llenado(fila, sufijo)]
    return ("NO PASA" if motivos else "PASA"), "; ".join(motivos)


def main(argv):
    with open(argv[1], newline="") as f:
        filas = list(csv.DictReader(f))
    distintas = []
    for fila in filas:
        veredicto, motivos = recalcular(fila)
        if (veredicto, motivos) != (fila["veredicto"], fila["motivos"]):
            distintas.append((fila, veredicto, motivos))
    print(f"coinciden {len(filas) - len(distintas)} de {len(filas)}")
    for fila, veredicto, motivos in distintas:
        print(f"{fila['estrategia']} {fila['activo']}: salida={fila['veredicto']} ({fila['motivos']})"
              f" recalculado={veredicto} ({motivos})")
    return 1 if distintas else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
