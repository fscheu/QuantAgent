"""Ventanas por año calendario y selección de parámetros para walk-forward."""

from datetime import datetime, timedelta
from typing import Any

import click


def preparar_walkforward(
    strategy_name, symbol, snapshot_name, grid_path, from_date=None, to_date=None,
    train_years=4, test_years=2, costos=None, tamano_posicion=1.0,
    regla_tamano=None, orden_minima=None,
):
    """Valida el ajuste protegido y devuelve grilla, metadatos, config y ventanas."""
    from quantagent.backtesting.fixtures import snapshot_metadata
    from quantagent.cli.backtest import _engine_config, _prueba_desde, _reserva_desde
    from quantagent.cli.sweep import cargar_grilla
    from quantagent.data.snapshot import SnapshotError

    combos = cargar_grilla(grid_path, strategy_name)
    prueba, reserva = _prueba_desde(), _reserva_desde()
    if to_date is None:
        to_date = min(prueba, reserva) - timedelta(days=1)
        motivo = (
            f"el tramo de prueba empieza el {prueba:%Y-%m-%d}"
            if prueba <= reserva else f"la reserva empieza el {reserva:%Y-%m-%d}"
        )
        click.echo(f"Aviso: sin --to, se asume fin en {to_date:%Y-%m-%d} ({motivo}).", err=True)
    elif to_date >= reserva:
        raise click.ClickException("--to cae en la reserva; walkforward no abre la prueba ni la reserva")
    elif to_date >= prueba:
        raise click.ClickException("--to cae en el tramo de prueba; walkforward no abre la prueba ni la reserva")
    try:
        meta = snapshot_metadata(snapshot_name, symbol.upper(), from_date, to_date)
    except (SnapshotError, ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    ventanas = construir_ventanas(meta.start_date, meta.end_date, train_years, test_years)
    ventanas = [(a, b, c, min(d, to_date)) for a, b, c, d in ventanas]
    config = _engine_config(None, costos, tamano_posicion, regla_tamano, orden_minima)
    return combos, meta, config, ventanas


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
