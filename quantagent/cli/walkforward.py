"""Ventanas por año calendario y selección de parámetros para walk-forward."""

import csv
import json
from datetime import datetime, timedelta
from typing import Any

import click

from quantagent.settings import COSTOS_PERFILES
from quantagent.trading.position_sizer import PositionSizer


def preparar_walkforward(
    strategy_name, symbol, snapshot_name, grid_path, from_date=None, to_date=None,
    train_years=4, test_years=2, costos=None, tamano_posicion=1.0,
    regla_tamano=None, orden_minima=None, riesgo_por_trade=None,
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
        raise click.ClickException(f"--to {to_date:%Y-%m-%d} cae en la reserva, que empieza el {reserva:%Y-%m-%d}; walkforward no abre la prueba ni la reserva")
    elif to_date >= prueba:
        raise click.ClickException(f"--to {to_date:%Y-%m-%d} cae en el tramo de prueba, que empieza el {prueba:%Y-%m-%d}; walkforward no abre la prueba ni la reserva")
    try:
        meta = snapshot_metadata(snapshot_name, symbol.upper(), from_date, to_date)
    except (SnapshotError, ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    ventanas = construir_ventanas(meta.start_date, meta.end_date, train_years, test_years)
    ventanas = [(a, b, c, min(d, meta.end_date)) for a, b, c, d in ventanas]
    config = _engine_config(None, costos, tamano_posicion, regla_tamano, orden_minima, riesgo_por_trade)
    return combos, meta, config, ventanas


@click.command("walkforward", help="Elegir parámetros en ajuste y medir el ganador en la ventana siguiente.")
@click.option("--llenado", type=click.Choice(["apertura", "cierre"]), default="apertura", show_default=True)
@click.option("--strategy", "strategy_name", required=True)
@click.option("--symbol", required=True)
@click.option("--snapshot", "snapshot_name", required=True)
@click.option("--grid", "grid_path", required=True, type=click.Path(dir_okay=False))
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, writable=True))
@click.option("--from", "from_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--to", "to_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--train-years", default=4, type=click.IntRange(min=1))
@click.option("--test-years", default=2, type=click.IntRange(min=1))
@click.option("--costos", type=click.Choice(sorted(COSTOS_PERFILES)))
@click.option("--tamano-posicion", default=1.0, type=click.FloatRange(min=0.0, max=1.0, min_open=True))
@click.option("--regla-tamano", type=click.Choice(PositionSizer.REGLAS))
@click.option("--riesgo-por-trade", type=click.FloatRange(min=0.0, min_open=True))
@click.option("--orden-minima", type=click.FloatRange(min=0.0, min_open=True))
def walkforward_command(strategy_name, symbol, snapshot_name, grid_path, out_path, llenado, **options):
    from quantagent.backtesting.fixtures import snapshot_metadata
    from quantagent.cli.backtest import _isolated_sqlite_db, correr_par
    from quantagent.cli.sweep import registrar_intento

    combos, _, config, ventanas = preparar_walkforward(strategy_name, symbol, snapshot_name, grid_path, **options)
    symbol = symbol.upper()
    rows = []
    for a, b, c, d in ventanas:
        resultados = []
        for desde, hasta, parametros in ((a, b, combos), (c, d, None)):
            if parametros is None:
                ganador = seleccionar_ganador(resultados)
                parametros = [ganador["params"]]
            meta = snapshot_metadata(snapshot_name, symbol, desde, hasta)
            for combo in parametros:
                with _isolated_sqlite_db():
                    r = correr_par(strategy_name, meta, config, options["costos"], None,
                                   snapshot_name, symbol, desde, hasta, params=combo, fill=llenado)
                if desde == a:
                    registrar_intento(strategy_name, symbol, snapshot_name, options["costos"],
                                      options["tamano_posicion"], {**combo, "llenado": llenado},
                                      {**r, "pnl": round(r["pnl"], 2), "sharpe": round(r["sharpe"], 2),
                                       "desde": f"{meta.start_date:%Y-%m-%d}", "hasta": f"{meta.end_date:%Y-%m-%d}"},
                                      sharpe_base="tasa_libre")
                    resultados.append({**r, "params": combo})
        rows.append(dict(ajuste_desde=f"{a:%Y-%m-%d}", ajuste_hasta=f"{b:%Y-%m-%d}",
                         prueba_desde=f"{c:%Y-%m-%d}", prueba_hasta=f"{d:%Y-%m-%d}",
                         params=json.dumps(ganador["params"], sort_keys=True),
                         ajuste_sharpe_tasa_0="" if ganador["sharpe_tasa_0"] is None else round(ganador["sharpe_tasa_0"], 2),
                         trades=r["trades"], pnl=round(r["pnl"], 2),
                         sharpe_tasa_0="" if r["sharpe_tasa_0"] is None else round(r["sharpe_tasa_0"], 2)))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    click.echo(f"Ventanas: {len(rows)}; CSV: {out_path}")


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
