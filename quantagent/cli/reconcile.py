"""Paper trading reconciliation commands."""

import json
import os
from datetime import datetime, time, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import click
from sqlalchemy import create_engine, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from quantagent import settings
from quantagent.data.provider import DataProvider
from quantagent.data.sources import build_source
from quantagent.database import Base
from quantagent.digest import build_digest
from quantagent.digest_datos import (consultar_datos_cartera, consultar_datos_paper,
                                     consultar_valuacion_actual, crear_broker_alpaca_paper)
from quantagent.digest_m5 import build_digest_m5, consultar_datos_m5
from quantagent.fabrica.cartera import leer_sombra
from quantagent.fabrica.chequeos_paper import comparar_pesos, reportar_dividendos, reportar_rechazos
from quantagent.models import EquitySnapshot, Environment, Position
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.trading.cartera_paper import ultima_sesion_cerrada_nyse
from quantagent.trading.pipeline_m5 import correr_diario, correr_semanal
from quantagent.trading.position_monitor import PositionMonitor
from quantagent.trading.pre_apertura import correr_pre_apertura

from . import utils


@click.group("trading", help="Paper trading operations.")
def trading_group() -> None:
    pass


@trading_group.command("snapshot", help="Save today's Alpaca account equity and cash snapshot.",
                       epilog="Example: quantagent trading snapshot --db paper.db --portfolio paper-m4")
@click.option("--db", required=True, type=click.Path(dir_okay=False), help="SQLite database file to update.")
@click.option("--portfolio", required=True, help="Portfolio profile used to connect to Alpaca.")
def trading_snapshot(db: str, portfolio: str) -> None:
    previous_url = os.environ.get("DATABASE_URL")
    previous_settings_url = settings.DATABASE_URL
    os.environ["DATABASE_URL"] = f"sqlite:///{os.path.abspath(db)}"
    try:
        with utils.session_scope() as session:
            try:
                account = crear_broker_alpaca_paper().get_account()
            except Exception as exc:
                raise click.ClickException(f"Alpaca account query failed ({type(exc).__name__}).") from exc
            EquitySnapshot.__table__.create(session.get_bind(), checkfirst=True)
            snapshot_date = datetime.now(ZoneInfo("America/New_York")).date()
            row = session.scalar(select(EquitySnapshot).where(EquitySnapshot.snapshot_date == snapshot_date))
            values = {"taken_at": datetime.now(timezone.utc), "equity": float(account["equity"]),
                      "cash": float(account["cash"])}
            values["invested"] = values["equity"] - values["cash"]
            if row is None:
                session.add(EquitySnapshot(snapshot_date=snapshot_date, **values))
            else:
                for key, value in values.items():
                    setattr(row, key, value)
            click.echo(f"Equity snapshot saved for {portfolio} ({snapshot_date}).")
    finally:
        settings.DATABASE_URL = previous_settings_url
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url


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


@trading_group.command("pre-apertura", help="Prepare paper stops and expire stale entry orders before market open.",
                       epilog="Example: quantagent trading pre-apertura")
@click.option("--bucket", type=click.Choice(["m5"]),
              help="Trading bucket to prepare; defaults to the configured portfolio.")
@click.option("--db", type=click.Path(dir_okay=False), help="SQLite file; M5 defaults to local-data/m5.db.")
def pre_apertura(bucket: str | None, db: str | None) -> None:
    if bucket == "m5":
        from quantagent.trading.bucket_m5 import cargar_credenciales
        cargar_credenciales()
        engine = _crear_db_m5(db or "local-data/m5.db")
        session = sessionmaker(bind=engine, expire_on_commit=False)()
        try:
            resolved = StrategyAssembler.from_profiles(
                overrides={"broker_type": "alpaca_paper"}, environment=Environment.PAPER
            )
            broker = StrategyAssembler.build_components(
                _config_m5(resolved), db_session=session, build_graph=False
            ).broker
            informe = correr_pre_apertura(session, broker)
            session.commit()
        finally:
            session.close()
            engine.dispose()
    else:
        with utils.session_scope() as session:
            resolved = StrategyAssembler.from_profiles(
                overrides={"broker_type": "alpaca_paper"}, environment=Environment.PAPER
            )
            broker = StrategyAssembler.build_components(
                resolved, db_session=session, build_graph=False
            ).broker
            informe = correr_pre_apertura(session, broker)
    for symbol in informe["reenviados"]:
        click.echo(f"Stop reenviado: {symbol}")
    for order_id in informe["entradas_vencidas"]:
        click.echo(f"Entrada vencida cancelada: {order_id}")
    for aviso in informe["avisos"]:
        click.echo(aviso)
    if informe["sin_stop"]:
        for symbol in informe["sin_stop"]:
            click.echo(f"ALARMA: {symbol} sin stop de venta abierto")
        raise click.exceptions.Exit(1)
    click.echo("Preapertura verificada: todas las posiciones tienen stop de venta abierto.")


def _config_m5(resolved):
    from dataclasses import replace
    return replace(resolved, initial_cash=3000.0, max_position_pct=0.10,
                   regla_tamano="riesgo", riesgo_por_trade_pct=0.0025,
                   extras={**(resolved.extras or {}), "bucket_m5": True})


def _crear_db_m5(db_path):
    from quantagent import models  # noqa: F401
    engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


@trading_group.command("m5-correr", help="Run one M5 weekly or daily paper trading cycle.",
                       epilog="Example: quantagent trading m5-correr --modo semanal")
@click.option("--modo", type=click.Choice(["semanal", "diario"]), required=True, help="Cycle mode: semanal or diario.")
@click.option("--as-of", type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"]),
              default=None, show_default="current UTC time",
              help="Cycle timestamp in ISO format; defaults to the current UTC time.")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default="local-data/m5.db", show_default=True,
              help="SQLite file for M5 intents and orders.")
def m5_correr(modo: str, as_of: datetime | None, db_path: str) -> None:
    as_of = as_of or datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        from quantagent.trading.bucket_m5 import cargar_credenciales
        cargar_credenciales()
        engine = _crear_db_m5(db_path)
        session = sessionmaker(bind=engine, expire_on_commit=False)()
        try:
            resolved = StrategyAssembler.from_profiles(
                overrides={"broker_type": "alpaca_paper"}, environment=Environment.PAPER
            )
            components = StrategyAssembler.build_components(_config_m5(resolved), db_session=session, build_graph=False)
            provider = DataProvider(session, source=build_source("alpaca"))
            resultado = (correr_semanal(as_of, data_provider=provider, db=session)
                         if modo == "semanal" else
                         correr_diario(as_of, data_provider=provider, db=session, broker=components.broker))
        finally:
            session.close()
            engine.dispose()
        if modo == "semanal":
            estado = resultado.get("estado", "sin régimen") if resultado else "sin régimen"
            click.echo(f"M5 semanal: régimen {estado}")
        else:
            intents = len(resultado or [])
            ordenes = sum(1 for item in (resultado or []) if getattr(item, "order_id", None))
            click.echo(f"M5 diario: {intents} intents, {ordenes} órdenes")
    except Exception as exc:
        click.echo(f"M5: {type(exc).__name__}: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc


@trading_group.command("digest", help="Show the daily paper trading digest.",
                       epilog="Example: quantagent trading digest --bucket m5 --date 2026-10-29")
@click.argument("archivo", required=False, type=click.File("r", encoding="utf-8"))
@click.option("--date", "digest_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="Digest date in YYYY-MM-DD format.")
@click.option("--bucket", type=click.Choice(["m5"]), help="Trading bucket for the digest.")
@click.option("--db", type=click.Path(dir_okay=False), help="SQLite file; M5 defaults to local-data/m5.db, otherwise DATABASE_URL.")
@click.option("--shadow", type=click.Path(dir_okay=False), help="JSON file with shadow weights.")
@click.option("--portfolio", help="Dedicated portfolio for positions, executions, and current Alpaca valuation.")
@click.option("--dividendos-alpaca", is_flag=True, help="Read Alpaca dividends for the selected date.")
@click.option("--sin-dividendos", is_flag=True, help="Disable the Alpaca dividend query.")
def trading_digest(archivo, digest_date, bucket: str | None, db: str | None, shadow: str | None, portfolio: str,
                   dividendos_alpaca: bool, sin_dividendos: bool) -> None:
    if bucket:
        if digest_date is None:
            raise click.UsageError("Provide --date YYYY-MM-DD.")
        previous_url = os.environ.get("DATABASE_URL")
        previous_settings_url = settings.DATABASE_URL
        try:
            db_path = db or "local-data/m5.db"
            os.environ["DATABASE_URL"] = f"sqlite:///{os.path.abspath(db_path)}"
            datos = {}
            with utils.session_scope() as session:
                datos = consultar_datos_m5(session, digest_date.date())
            click.echo(build_digest_m5(digest_date.date(), datos))
        finally:
            settings.DATABASE_URL = previous_settings_url
            if previous_url is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = previous_url
        return
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
    dedicated = portfolio is not None
    portfolio = portfolio or "paper"
    valuation, executions, pending = None, None, None
    previous_url = os.environ.get("DATABASE_URL")
    previous_settings_url = settings.DATABASE_URL
    if db:
        os.environ["DATABASE_URL"] = f"sqlite:///{os.path.abspath(db)}"
    db_path = db or make_url(settings.DATABASE_URL).database
    shadow_path = Path(shadow) if shadow else Path(db_path).with_suffix(".sombra.json")
    warnings = [] if dedicated else ["Equity paper = valor invertido; la DB no guarda caja."]
    dividendos_alpaca = not sin_dividendos and (dividendos_alpaca or settings.TRADING_BROKER == "alpaca_paper")
    if not dividendos_alpaca:
        warnings.append("Dividendos: sin fuente de dividendos.")
    differences, dividendos_error = [], False
    dividendos = []
    if dividendos_alpaca:
        try:
            from quantagent.trading.alpaca_dividendos import leer_dividendos_alpaca
            client = crear_broker_alpaca_paper().client
            dividendos = leer_dividendos_alpaca(client, day)
        except Exception as exc:
            warnings.append(f"Dividendos: error al consultar Alpaca ({type(exc).__name__}).")
            if isinstance(exc, ValueError):
                warnings.append(f"Dividendos: {exc}")
            dividendos_error = True
        else:
            warnings.extend(reportar_dividendos(dividendos) or ["Dividendos: Sin observaciones"])
    orders = []
    try:
        with utils.session_scope() as session:
            portfolios, trades, heartbeats, cumulative_pnl, orders = consultar_datos_paper(session, day, portfolio)
            if dedicated:
                portfolios, executions, pending = consultar_datos_cartera(session, day, portfolio)
                trades, cumulative_pnl = [], None
                try:
                    broker = crear_broker_alpaca_paper()
                    valuation = consultar_valuacion_actual(broker, portfolios)
                except (RuntimeError, ValueError) as exc:
                    raise click.ClickException(str(exc)) from exc
                portfolios = valuation["portfolios"]
            values = {p["symbol"]: float(p["value"]) for p in portfolios[0]["positions"]} if portfolios else {}
            invested = sum(values.values())
            paper_weights = {symbol: value / invested for symbol, value in values.items()} if invested else {}
            session_day = ultima_sesion_cerrada_nyse(datetime.combine(day, time.max)).date()
            try:
                shadow_weights = leer_sombra(shadow_path, portfolio, session_day)["pesos"]
            except (KeyError, FileNotFoundError):
                warnings.append(f"Sombra sin datos para {portfolio} en {session_day}")
            else:
                tolerancias = {
                    symbol: float(price) / invested
                    for symbol, price in session.query(Position.symbol, Position.current_price)
                    .filter(Position.symbol.in_(set(paper_weights) | set(shadow_weights))).all()
                } if invested else {}
                differences = comparar_pesos(paper_weights, shadow_weights, tolerancias=tolerancias)
    except (SQLAlchemyError, ValueError) as exc:
        if dedicated:
            raise click.ClickException(f"Cartera: error al consultar datos ({type(exc).__name__}).") from exc
        portfolios, trades, heartbeats, cumulative_pnl, orders = [], [], [], 0.0, []
    finally:
        settings.DATABASE_URL = previous_settings_url
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url
    rejection_reports = reportar_rechazos([SimpleNamespace(**order) for order in orders])
    click.echo(build_digest(day, portfolios, trades, heartbeats, warnings=warnings,
                            reconciliations=[*differences, *rejection_reports], cumulative_pnl=cumulative_pnl,
                            valuation=valuation, executions=executions, pending=pending))
    if dividendos_error or differences or any(str(order.get("status")).lower() == "rejected" and not order.get("comment")
                          for order in orders):
        raise click.exceptions.Exit(1)
