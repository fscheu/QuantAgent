"""Pure checks for paper trading state."""

from math import isclose

PAPER_CASH_RESERVE_PCT = 0.05


def reconciliar_cartera(posiciones, pendientes, precios, equity, pesos_sombra):
    """Compare effective holdings, including unfilled orders, with reserved shadow targets.

    Missing prices for effective holdings are reported; shadow-only symbols
    without a price retain comparar_pesos's default tolerance.
    """
    cantidades = {p["symbol"]: p["quantity"] for p in posiciones}
    for orden in pendientes:
        simbolo = orden["symbol"]
        restante = float(orden["quantity"]) - float(orden["filled_quantity"])
        cantidades[simbolo] = cantidades.get(simbolo, 0) + (
            restante if orden["side"] == "buy" else -restante)
    sin_precio = [f"{s}: sin precio para reconciliar posición y pendientes"
                  for s, q in sorted(cantidades.items()) if q and s not in precios]
    pesos = {s: float(q) * precios[s] / equity for s, q in cantidades.items()
             if s in precios} if equity else {}
    objetivos = {s: p * (1 - PAPER_CASH_RESERVE_PCT) for s, p in pesos_sombra.items()}
    tolerancias = {s: p / equity for s, p in precios.items()} if equity else {}
    return comparar_pesos(pesos, objetivos, tolerancias=tolerancias) + sin_precio


def comparar_pesos(pesos_paper, pesos_sombra, tolerancia=1e-4, tolerancias=None):
    """Report weight differences above each asset's tolerance, defaulting to tolerancia."""
    diferencias = []
    for activo in sorted(set(pesos_paper) | set(pesos_sombra)):
        peso_paper = pesos_paper.get(activo, 0.0)
        peso_sombra = pesos_sombra.get(activo, 0.0)
        limite = (tolerancias or {}).get(activo, tolerancia)
        diferencia = abs(peso_paper - peso_sombra)
        # Weight subtraction can put exactly one share slightly above its limit.
        en_limite = tolerancias and activo in tolerancias and isclose(
            diferencia, limite, rel_tol=1e-12, abs_tol=0.0)
        if diferencia > limite and not en_limite:
            diferencias.append(
                f"{activo}: peso paper {peso_paper} difiere de sombra {peso_sombra} "
                f"(tolerancia {limite})"
            )
    return diferencias


def reportar_rechazos(ordenes):
    """List rejected orders and flag any rejection without an explanation."""
    reportes = []
    for orden in ordenes:
        estado = getattr(getattr(orden, "status", None), "value", getattr(orden, "status", None))
        if str(estado).lower() != "rejected":
            continue
        motivo = getattr(orden, "comment", None)
        simbolo = getattr(orden, "symbol", "")
        reportes.append(f"{simbolo}: rechazo sin explicación" if not motivo else f"{simbolo}: rechazo, {motivo}")
    return reportes


def reportar_dividendos(dividendos):
    """List dividend records and flag any event without a recorded treatment."""
    reportes = []
    for dividendo in dividendos:
        simbolo, importe, moneda, tratamiento = dividendo
        if not tratamiento:
            reportes.append(f"{simbolo}: dividendo {importe} {moneda}, falta tratamiento")
        else:
            reportes.append(f"{simbolo}: dividendo {importe} {moneda}, tratamiento {tratamiento}")
    return reportes
