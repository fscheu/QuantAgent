"""Paper trading reconciliation commands."""

import click

from quantagent.models import Environment
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.trading.position_monitor import PositionMonitor

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
