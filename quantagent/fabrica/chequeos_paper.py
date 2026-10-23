"""Pure checks for paper trading state."""


def comparar_pesos(pesos_paper, pesos_sombra, tolerancia=1e-4):
    """Return one explanation for each asset whose paper and shadow weights differ."""
    diferencias = []
    for activo in sorted(set(pesos_paper) | set(pesos_sombra)):
        peso_paper = pesos_paper.get(activo, 0.0)
        peso_sombra = pesos_sombra.get(activo, 0.0)
        if abs(peso_paper - peso_sombra) > tolerancia:
            diferencias.append(
                f"{activo}: peso paper {peso_paper} difiere de sombra {peso_sombra} "
                f"(tolerancia {tolerancia})"
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
