"""Paper trading reconciliation commands."""

import json
from types import SimpleNamespace

import click

from quantagent.models import Environment
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.trading.position_monitor import PositionMonitor
from quantagent.fabrica.chequeos_paper import comparar_pesos, reportar_dividendos, reportar_rechazos

from . import utils


@click.group("trading", help="Paper trading operations.")
def trading_group() -> None:
    pass


@trading_group.command("reconciliar", help="Compare local PAPER positions with Alpaca.")
def reconcile_positions() -> None:
    with utils.session_scope() as session:
        resolved = StrategyAssembler.from_profiles(
            overrides={"broker_type": "alpaca_paper"}, environment=Environment.PAPER
        )
        broker = StrategyAssembler.build_components(
            resolved, db_session=session, build_graph=False
        ).broker
        differences = PositionMonitor(session).reconcile_alpaca_positions(broker)

    if differences:
        for difference in differences:
            click.echo(difference)
    else:
        click.echo("Sin divergencias: posiciones locales y Alpaca coinciden.")


@trading_group.command("digest", help="Evaluar pesos, rechazos y dividendos desde un JSON.")
@click.argument("archivo", type=click.File("r", encoding="utf-8"))
def paper_digest(archivo) -> None:
    """El JSON contiene paper_weights, shadow_weights, orders y dividends."""
    data = json.load(archivo)
    secciones = (
        ("Pesos", comparar_pesos(data["paper_weights"], data["shadow_weights"])),
        ("Rechazos", reportar_rechazos([SimpleNamespace(**order) for order in data["orders"]])),
        ("Dividendos", reportar_dividendos(data["dividends"])),
    )
    fallo = False
    for titulo, reportes in secciones:
        click.echo(f"{titulo}:")
        for reporte in reportes:
            click.echo(f"- {reporte}")
            fallo |= titulo == "Pesos" or "sin explicación" in reporte or "falta tratamiento" in reporte
        if not reportes:
            click.echo("- Sin observaciones")
    if fallo:
        raise click.exceptions.Exit(1)
