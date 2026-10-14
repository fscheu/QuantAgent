"""Ventanas por año calendario y selección de parámetros para walk-forward."""

from datetime import datetime, timedelta
from typing import Any

import click


def construir_ventanas(
    desde: datetime, hasta: datetime, train_years: int, test_years: int
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Fechas inclusivas; el primer año puede empezar después del 1 de enero."""
    if train_years <= 0 or test_years <= 0:
        raise click.ClickException("Los años de ajuste y prueba deben ser positivos")
    ventanas = []
    inicio = desde
    while True:
        prueba_inicio = datetime(inicio.year + train_years, 1, 1)
        prueba_fin = datetime(prueba_inicio.year + test_years, 1, 1) - timedelta(days=1)
        if datetime(prueba_fin.year, 12, 24) > hasta:
            break
        ventanas.append((inicio, prueba_inicio - timedelta(days=1), prueba_inicio, prueba_fin))
        inicio = datetime(inicio.year + test_years, 1, 1)
    if not ventanas:
        raise click.ClickException("El rango no alcanza para una ventana completa de ajuste y prueba")
    return ventanas


def seleccionar_ganador(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    """Devuelve el mayor Sharpe de ajuste; un empate conserva el orden de grilla."""
    return max(resultados, key=lambda resultado: (
        resultado["sharpe_tasa_0"] is not None,
        resultado["sharpe_tasa_0"] if resultado["sharpe_tasa_0"] is not None else 0,
    ))
