"""Paper trading reconciliation commands."""

import json
import os
from types import SimpleNamespace

import click
from sqlalchemy.exc import SQLAlchemyError

from quantagent.digest import build_digest
from quantagent.digest_datos import consultar_datos_paper
from quantagent.fabrica.chequeos_paper import comparar_pesos, reportar_dividendos, reportar_rechazos
from quantagent import settings
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


@trading_group.command("digest", help="Mostrar el digest diario de paper trading.")
@click.argument("archivo", required=False, type=click.File("r", encoding="utf-8"))
@click.option("--date", "digest_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--db", type=click.Path(dir_okay=False), help="Archivo SQLite; por defecto usa DATABASE_URL.")
@click.option("--portfolio", default="paper", show_default=True, help="Nombre de la cartera en el digest.")
def trading_digest(archivo, digest_date, db: str | None, portfolio: str) -> None:
    if archivo:
        data = json.load(archivo)
        differences = comparar_pesos(data["paper_weights"], data["shadow_weights"])
        sections = (("Pesos", differences),
                    ("Rechazos", reportar_rechazos([SimpleNamespace(**item) for item in data["orders"]])),
                    ("Dividendos", reportar_dividendos(data["dividends"])))
        for title, reports in sections:
            click.echo(f"{title}:")
            for report in reports:
                click.echo(f"- {report}")
            if not reports:
                click.echo("- Sin observaciones")
        if differences or any(str(item.get("status")).lower() == "rejected" and not item.get("comment")
                              for item in data["orders"]) or any(not item[3] for item in data["dividends"]):
            raise click.exceptions.Exit(1)
        return
    if digest_date is None:
        raise click.UsageError("Provide --date YYYY-MM-DD.")
    day = digest_date.date()
    previous_url = os.environ.get("DATABASE_URL")
    previous_settings_url = settings.DATABASE_URL
    if db:
        os.environ["DATABASE_URL"] = f"sqlite:///{os.path.abspath(db)}"
    try:
        with utils.session_scope() as session:
            portfolios, trades, heartbeats, cumulative_pnl, _orders = consultar_datos_paper(session, day, portfolio)
    except (SQLAlchemyError, ValueError):
        portfolios, trades, heartbeats, cumulative_pnl, _orders = [], [], [], 0.0, []
    finally:
        settings.DATABASE_URL = previous_settings_url
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url
    click.echo(build_digest(day, portfolios, trades, heartbeats, cumulative_pnl=cumulative_pnl))
