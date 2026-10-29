"""Pure checks for paper trading state."""

from math import isclose


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
