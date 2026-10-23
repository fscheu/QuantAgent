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
