"""Regla del Juego 2 para una celda estrategia × activo."""

# Umbrales fijados por Fede en el PR #140 (V30).
MIN_TRADES = 2
MIN_SHARPE_TASA_0 = 0.10
MAX_DRAWDOWN = 0.35


def evaluar(trades, pnl, sharpe_tasa_0, max_drawdown) -> tuple[bool, list[str]]:
    motivos = []
    criterios = (
        ("trades", trades, lambda v: v >= MIN_TRADES, "trades < 2"),
        ("pnl", pnl, lambda v: v > 0, "pnl <= 0"),
        ("sharpe_tasa_0", sharpe_tasa_0, lambda v: v >= MIN_SHARPE_TASA_0, "sharpe_tasa_0 < 0.10"),
        ("max_drawdown", max_drawdown, lambda v: v <= MAX_DRAWDOWN, "max_drawdown > 0.35"),
    )
    for campo, valor, cumple, motivo in criterios:
        if valor is None or valor == "":
            motivos.append(f"sin {campo}")
        elif not cumple(valor):
            motivos.append(motivo)
    return not motivos, motivos
