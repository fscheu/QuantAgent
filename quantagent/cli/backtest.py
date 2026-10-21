"""CLI commands for running deterministic backtests (D13-D18 of PLAN-30-DIAS)."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import click
from sqlalchemy import select
from sqlalchemy.orm import Session
from tabulate import tabulate

from quantagent import settings
from quantagent.backtesting.backtest import Backtest
from quantagent.backtesting.buy_and_hold import annual_return, buy_and_hold, exposure, worst_day
from quantagent.backtesting.export import TradeRow, equity_to_csv, trades_to_csv
from quantagent.backtesting.fixtures import (
    FixtureMetadata,
    fixture_metadata,
    load_fixture,
    load_snapshot,
    snapshot_metadata,
)
from quantagent.data.snapshot import SnapshotError, load_manifest, snapshot_root
from quantagent.models import ActivePosition, MarketData, Trade
from quantagent.settings import COSTOS_PERFILES
from quantagent.strategy.registry import STRATEGY_REGISTRY, build_strategy, cli_aliases, effective_params
from quantagent.trading.position_sizer import PositionSizer

from . import utils
from .data import data_group
from .hipotesis import hipotesis_group
from .sweep import attempts_command, sweep_command
from .verdict import verdict_command
from .walkforward import walkforward_command


class _AliasChoice(click.Choice):
    """Choice over the registry's CLI aliases, read at parse time so late registrations count."""

    def __init__(self) -> None:
        super().__init__(())

    @property
    def choices(self):
        return sorted(cli_aliases())

    @choices.setter
    def choices(self, _value) -> None:
        pass


RESERVA_ENV = "QUANTAGENT_RESERVA_DESDE"
RESERVA_DEFAULT = "2023-01-01"
RESERVA_LOG = "reserva-accesos.log"

PRUEBA_ENV = "QUANTAGENT_PRUEBA_DESDE"
PRUEBA_DEFAULT = "2019-01-01"
PRUEBA_LOG = "prueba-accesos.log"


def _reserva_desde() -> datetime:
    value = os.environ.get(RESERVA_ENV, RESERVA_DEFAULT)
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise click.ClickException(f"{RESERVA_ENV}={value!r} no es una fecha YYYY-MM-DD")


def _prueba_desde() -> datetime:
    value = os.environ.get(PRUEBA_ENV, PRUEBA_DEFAULT)
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise click.ClickException(f"{PRUEBA_ENV}={value!r} no es una fecha YYYY-MM-DD")



def _registrar_apertura(comando: str) -> None:
    """Agrega fecha y comando a `$QUANTAGENT_SNAPSHOT_DIR/reserva-accesos.log`."""
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (snapshot_root() / RESERVA_LOG).open("a") as f:
        f.write(f"{ahora}\t{comando}\n")


def _registrar_apertura_prueba(comando: str) -> None:
    """Agrega fecha y comando a `$QUANTAGENT_SNAPSHOT_DIR/prueba-accesos.log`."""
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (snapshot_root() / PRUEBA_LOG).open("a") as f:
        f.write(f"{ahora}\t{comando}\n")


def _resolve_source_and_meta(
    subcommand: str,
    strategy_name: str,
    fixture_name: Optional[str],
    snapshot_name: Optional[str],
    symbol: Optional[str],
    from_date: Optional[datetime],
    to_date: Optional[datetime],
    abrir_reserva: bool,
    abrir_prueba: bool = False,
):
    if bool(fixture_name) == bool(snapshot_name):
        raise click.UsageError("Indicá exactamente uno: --fixture o --snapshot.")
    if snapshot_name and not symbol:
        raise click.UsageError("--snapshot requiere --symbol.")
    if fixture_name and (symbol or from_date or to_date or abrir_reserva or abrir_prueba):
        raise click.UsageError("--symbol, --from, --to, --abrir-reserva y --abrir-prueba solo se usan con --snapshot.")
    abriendo_reserva = False
    abriendo_prueba = False
    if snapshot_name:
        reserva = _reserva_desde()
        prueba = _prueba_desde()
        tope = min(prueba, reserva)
        if to_date is None:
            to_date = tope - timedelta(days=1)
            motivo = (
                f"el tramo de prueba empieza el {prueba:%Y-%m-%d}"
                if prueba <= reserva
                else f"la reserva empieza el {reserva:%Y-%m-%d}"
            )
            click.echo(
                f"Aviso: sin --to, se asume fin en {to_date:%Y-%m-%d} ({motivo}).",
                err=True,
            )
        elif to_date >= reserva:
            if not abrir_reserva:
                raise click.ClickException(
                    f"--to {to_date:%Y-%m-%d} cae en la reserva, que empieza el {reserva:%Y-%m-%d}: "
                    "ninguna corrida la lee. Para abrirla, --abrir-reserva (queda registrado)."
                )
            abriendo_reserva = True
        elif to_date >= prueba:
            if not abrir_prueba:
                raise click.ClickException(
                    f"--to {to_date:%Y-%m-%d} cae en el tramo de prueba, que empieza el {prueba:%Y-%m-%d}: "
                    "para abrirlo, --abrir-prueba (queda registrado)."
                )
            abriendo_prueba = True
    try:
        if snapshot_name:
            symbol = symbol.upper()
            meta = snapshot_metadata(snapshot_name, symbol, from_date, to_date)
        else:
            meta = fixture_metadata(fixture_name)
    except FileNotFoundError as exc:
        raise click.ClickException(f"Fixture not found: {fixture_name}") from exc
    except (SnapshotError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if abriendo_reserva:
        partes = [f"backtest {subcommand} --strategy {strategy_name} --snapshot {snapshot_name} --symbol {symbol}"]
        partes += [f"--from {from_date:%Y-%m-%d}"] if from_date else []
        partes += [f"--to {to_date:%Y-%m-%d} --abrir-reserva"]
        _registrar_apertura(" ".join(partes))
    elif abriendo_prueba:
        partes = [f"backtest {subcommand} --strategy {strategy_name} --snapshot {snapshot_name} --symbol {symbol}"]
        partes += [f"--from {from_date:%Y-%m-%d}"] if from_date else []
        partes += [f"--to {to_date:%Y-%m-%d} --abrir-prueba"]
        _registrar_apertura_prueba(" ".join(partes))

    return meta, symbol, to_date


def _engine_config(commission_pct: Optional[float], costos: Optional[str] = None,
                   tamano_posicion: Optional[float] = None, regla_tamano: Optional[str] = None,
                   orden_minima: Optional[float] = None) -> dict:
    config = {"market_hours_filter": False, "offline_data": True}
    if tamano_posicion is not None:
        config["base_position_pct"] = config["max_position_pct"] = tamano_posicion
    if regla_tamano is not None:
        config["regla_tamano"] = regla_tamano
    if orden_minima is not None:
        config["orden_minima"] = orden_minima
    if costos is not None:
        if commission_pct is not None:
            raise click.UsageError("--costos y --commission-pct son excluyentes: elegí uno.")
        config.update(COSTOS_PERFILES[costos])
    elif commission_pct is not None:
        config["commission_pct"] = commission_pct
    return config


def _tamano_options(f):
    f = click.option(
        "--orden-minima", "orden_minima", type=click.FloatRange(min=0.0, min_open=True), default=None,
        help="Monto mínimo en USD por orden de entrada.",
    )(f)
    f = click.option(
        "--regla-tamano", "regla_tamano", type=click.Choice(PositionSizer.REGLAS), default=None,
        help="Regla de tamaño: confianza (capital × fracción × confianza de la señal) o fijo. Default: confianza.",
    )(f)
    return click.option(
        "--tamano-posicion", "tamano_posicion", type=click.FloatRange(min=0.0, max=1.0, min_open=True), default=None,
        help="Fracción del capital por posición, en (0, 1] (1.0 = 100%). Fija base y máximo de la corrida.",
    )(f)


def _echo_tamano(config: dict) -> None:
    """Imprime con qué tamaño se corrió, solo si se pidió --tamano-posicion o --regla-tamano."""
    if "base_position_pct" in config or "regla_tamano" in config:
        pct = config.get("base_position_pct", settings.TRADING_BASE_POSITION_PCT)
        click.echo(f"Tamaño: {pct * 100:g}% del capital, regla {config.get('regla_tamano', 'confianza')}")


def cargar_params(path: Optional[str]) -> dict:
    """Lee {"<alias>": {"<parámetro>": valor}} y corta si nombra una estrategia o parámetro que no existe."""
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    aliases = cli_aliases()
    for strat, valores in data.items():
        if strat not in aliases:
            raise click.ClickException(f"--params: estrategia desconocida '{strat}'. Opciones: {', '.join(sorted(aliases))}")
        for nombre in valores:
            if nombre not in STRATEGY_REGISTRY[aliases[strat]]["params"]:
                raise click.ClickException(f"--params: la estrategia '{strat}' no tiene el parámetro '{nombre}'")
    return data


_params_option = click.option(
    "--params", "params", type=click.Path(exists=True, dir_okay=False), default=None, callback=lambda ctx, p, v: cargar_params(v),
    help='JSON {"<estrategia>": {"<parámetro>": valor}} con parámetros de estrategia (default: los del registro).',
)


def _hide_data_warnings(record: logging.LogRecord) -> bool:
    return getattr(record, "event_type", None) != "backtest_data_warning"


def _extract_trade_rows(session: Session, backtest_run_id: int) -> list[TradeRow]:
    trade_ids = select(ActivePosition.trade_id).where(
        ActivePosition.backtest_run_id == backtest_run_id,
        ActivePosition.trade_id.is_not(None),
    )
    trades = (
        session.query(Trade, ActivePosition.stop_loss)
        .join(ActivePosition, ActivePosition.trade_id == Trade.id)
        .filter(Trade.id.in_(trade_ids))
        .order_by(Trade.opened_at)
        .all()
    )
    return [
        TradeRow(
            entry_time=trade.opened_at,
            exit_time=trade.closed_at,
            symbol=trade.symbol,
            side=trade.side.value,
            qty=float(trade.quantity),
            entry_price=float(trade.entry_price),
            exit_price=float(trade.exit_price) if trade.exit_price is not None else None,
            stop_loss=float(stop_loss) if stop_loss is not None else None,
            pnl=float(trade.pnl) if trade.pnl is not None else None,
            exit_reason=trade.exit_signal,
        )
        for trade, stop_loss in trades
    ]


def correr_par(
    strategy_name: str,
    meta: FixtureMetadata,
    config: dict,
    costos: Optional[str],
    fixture_name: Optional[str],
    snapshot_name: Optional[str],
    symbol: Optional[str],
    from_date: Optional[datetime],
    to_date: Optional[datetime],
    intrabar_stops: bool = True,
    verbose: bool = False,
    con_filas: bool = False,
    params: Optional[dict] = None,
    fill: str = "cierre",
) -> dict:
    """Corre un par estrategia/activo y devuelve los números del resumen de `backtest run`.

    Claves: trades, win_rate, profit_factor (None o inf si no hay pérdidas), sharpe, pnl,
    slippage_pct, comision_pct, costos (nombre del perfil o None), tamano_pct (None si no se
    pidió tamaño), regla_tamano (None si no se pidió), bh_pnl, bh_sharpe, bh_max_drawdown,
    retorno_anualizado, exceso_sobre_referencia, tiempo_en_mercado, capital_invertido_medio,
    sharpe_tasa_0, peor_dia (estas seis son None si no hay curva de equity), max_drawdown,
    equity_curve, cierres (timestamp, cierre ajustado de cada vela), entradas_descartadas_orden_minima y filas (TradeRow; vacía salvo con con_filas=True).
    `params` son los parámetros del constructor de la estrategia (el resto, los defaults del registro).
    """
    reg_name = cli_aliases()[strategy_name]
    with utils.session_scope() as session:
        existing_count = (
            session.query(MarketData)
            .filter(
                MarketData.symbol == meta.symbol,
                MarketData.timeframe == meta.timeframe,
                MarketData.timestamp >= meta.start_date,
                MarketData.timestamp <= meta.end_date,
                MarketData.source == (f"snapshot:{snapshot_name}" if snapshot_name else "fixture"),
            )
            .count()
        )
        if existing_count == 0:
            if snapshot_name:
                load_snapshot(session, snapshot_name, symbol, from_date, to_date)
            else:
                load_fixture(session, fixture_name)
        elif existing_count != meta.row_count:
            raise click.ClickException(
                f"La base tiene otros datos para {meta.symbol} {meta.timeframe} en el rango "
                f"del {'snapshot' if snapshot_name else 'fixture'} ({existing_count} filas, se esperaban {meta.row_count}): "
                "usá una base limpia."
            )

        bt = Backtest(
            start_date=meta.start_date,
            end_date=meta.end_date,
            assets=[meta.symbol],
            timeframe=meta.timeframe,
            # Fixtures are versioned as continuous candles (every hour, no market-hours
            # gaps), so filtering to real trading hours would misalign against them.
            # offline_data avoids live yfinance calls for the lookback window before the
            # fixture's first row -- the whole point of a fixture-based run is to be
            # deterministic and network-free.
            config={**config, "source_name": f"snapshot:{snapshot_name}" if snapshot_name else "fixture",
                    "strategy_params": effective_params(reg_name, params)},
            db_session=session,
            strategy=build_strategy(reg_name, **(params or {})),
            intrabar_stops=intrabar_stops,
            fill=fill,
        )
        # The lookback warm-up logs one "Insufficient data" warning per candle; keep the
        # output to the metric lines unless --verbose asks for them.
        engine_logger = logging.getLogger("quantagent.backtesting.backtest")
        if not verbose:
            engine_logger.addFilter(_hide_data_warnings)
        try:
            metrics = bt.run(name=f"cli-{strategy_name}-{fixture_name or snapshot_name}")
        finally:
            engine_logger.removeFilter(_hide_data_warnings)

        slippage_pct = bt.order_manager.broker.slippage_pct
        commission = bt.order_manager.broker.commission_pct
        rows = _extract_trade_rows(session, bt.backtest_run_id) if con_filas else []
        candles = (
            session.query(MarketData.timestamp, MarketData.close)
            .filter(
                MarketData.symbol == meta.symbol,
                MarketData.timeframe == meta.timeframe,
                MarketData.timestamp >= meta.start_date,
                MarketData.timestamp <= meta.end_date,
            )
            .order_by(MarketData.timestamp)
            .all()
        )
        bh = buy_and_hold(
            [c.timestamp for c in candles], [float(c.close) for c in candles],
            bt.initial_capital, slippage_pct, commission,
        )

    curve = bt.equity_curve
    r = {
        "cierres": [(c.timestamp, float(c.close)) for c in candles],
        "trades": metrics.total_trades,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "sharpe": metrics.sharpe_ratio,
        "pnl": metrics.total_pnl,
        "slippage_pct": slippage_pct,
        "comision_pct": commission,
        "costos": costos,
        "tamano_pct": None,
        "regla_tamano": None,
        "bh_pnl": bh["pnl"],
        "bh_sharpe": bh["sharpe"],
        "bh_max_drawdown": bh["max_drawdown"],
        "retorno_anualizado": None,
        "exceso_sobre_referencia": None,
        "tiempo_en_mercado": None,
        "capital_invertido_medio": None,
        "sharpe_tasa_0": None,
        "peor_dia": None,
        "max_drawdown": metrics.max_drawdown,
        "equity_curve": curve,
        "filas": rows,
        "entradas_descartadas_orden_minima": bt.order_manager.entradas_descartadas_orden_minima,
    }
    if "base_position_pct" in config or "regla_tamano" in config:
        r["tamano_pct"] = config.get("base_position_pct", settings.TRADING_BASE_POSITION_PCT)
        r["regla_tamano"] = config.get("regla_tamano", "confianza")
    if curve:
        annual = annual_return(curve[0]["equity"], curve[-1]["equity"], curve[0]["date"], curve[-1]["date"])
        in_market, invested = exposure(curve)
        r["retorno_anualizado"] = annual
        r["exceso_sobre_referencia"] = annual - bh["annual_return"]
        r["tiempo_en_mercado"] = in_market
        r["capital_invertido_medio"] = invested
        r["sharpe_tasa_0"] = bt._calculate_sharpe_ratio(0.0)
        r["peor_dia"] = worst_day(curve)
    return r


@click.group(help="Run deterministic backtests against versioned fixtures.")
def backtest_group() -> None:
    """Root command group for backtest operations."""


@backtest_group.command("run", help="Run a backtest against a versioned fixture.")
@click.option(
    "--strategy",
    "strategy_name",
    required=True,
    type=_AliasChoice(),
    help="Strategy to run.",
)
@click.option(
    "--fixture",
    "fixture_name",
    help="Fixture name under tests/fixtures/ (without the .csv extension). Excluyente con --snapshot.",
)
@click.option("--snapshot", "snapshot_name", help="Snapshot en $QUANTAGENT_SNAPSHOT_DIR (velas ajustadas). Requiere --symbol.")
@click.option("--symbol", "symbol", help="Símbolo del snapshot (uno por corrida).")
@click.option("--from", "from_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="Primera sesión, YYYY-MM-DD (default: la primera del snapshot).")
@click.option("--to", "to_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="Última sesión, YYYY-MM-DD (default: la última del snapshot).")
@click.option(
    "--abrir-reserva",
    "abrir_reserva",
    is_flag=True,
    help="Permite que --to llegue a la reserva (desde $QUANTAGENT_RESERVA_DESDE, default 2023-01-01). Queda registrado.",
)
@click.option(
    "--abrir-prueba",
    "abrir_prueba",
    is_flag=True,
    help="Permite que --to llegue al tramo de prueba (desde $QUANTAGENT_PRUEBA_DESDE, default 2019-01-01). Queda registrado.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, writable=True),
    help="Write the trade log CSV to this path.",
)
@click.option(
    "--equity-out",
    "equity_out_path",
    type=click.Path(dir_okay=False, writable=True),
    help="Write the equity curve CSV (one row per candle) to this path.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Also show the engine's per-candle 'Insufficient data' messages on stderr.",
)
@click.option(
    "--commission-pct",
    "commission_pct",
    type=click.FloatRange(min=0.0),
    default=None,
    help="Comisión por lado sobre el nocional (0.001 = 0.10%). Default: TRADING_COMMISSION_PCT.",
)
@click.option(
    "--costos",
    type=click.Choice(sorted(COSTOS_PERFILES)),
    default=None,
    help="Perfil de costos (slippage y comisión por lado). Excluyente con --commission-pct.",
)
@click.option("--llenado", type=click.Choice(["apertura", "cierre"]), default="apertura", show_default=True)
@_tamano_options
@_params_option
@click.option(
    "--intrabar-stops/--no-intrabar-stops",
    default=True,
    help="Evaluate stop loss and take profit against intrabar high/low prices.",
)
def run_backtest(
    strategy_name: str,
    fixture_name: Optional[str],
    snapshot_name: Optional[str],
    symbol: Optional[str],
    from_date: Optional[datetime],
    to_date: Optional[datetime],
    abrir_reserva: bool,
    abrir_prueba: bool,
    out_path: Optional[str],
    equity_out_path: Optional[str],
    verbose: bool,
    intrabar_stops: bool,
    commission_pct: Optional[float],
    costos: Optional[str],
    tamano_posicion: Optional[float],
    regla_tamano: Optional[str],
    orden_minima: Optional[float],
    params: dict,
    llenado: str,
) -> None:
    config = _engine_config(commission_pct, costos, tamano_posicion, regla_tamano, orden_minima)
    meta, symbol, to_date = _resolve_source_and_meta(
        "run",
        strategy_name,
        fixture_name,
        snapshot_name,
        symbol,
        from_date,
        to_date,
        abrir_reserva,
        abrir_prueba,
    )

    r = correr_par(
        strategy_name, meta, config, costos, fixture_name, snapshot_name, symbol,
        from_date, to_date, intrabar_stops, verbose, con_filas=bool(out_path),
        params=params.get(strategy_name),
        fill=llenado,
    )
    rows = r["filas"]

    click.echo(f"Trades: {r['trades']}")
    click.echo(f"Win rate: {r['win_rate']:.2%}")
    if r["profit_factor"] is None or math.isinf(r["profit_factor"]):
        click.echo("Profit factor: n/a")
    else:
        click.echo(f"Profit factor: {r['profit_factor']:.2f}")
    click.echo(f"Sharpe ratio: {r['sharpe']:.2f}")
    click.echo(f"Total PnL: {r['pnl']:.2f}")
    click.echo(f"Slippage: {r['slippage_pct'] * 100:.2f}% por lado")
    click.echo(f"Comisión: {r['comision_pct'] * 100:.2f}% por lado")
    if costos:
        click.echo(f"Costos: {costos}")
    _echo_tamano(config)
    if "orden_minima" in config:
        click.echo(f"Entradas descartadas por orden mínima: {r['entradas_descartadas_orden_minima']}")
    click.echo(
        f"Comprar y mantener: PnL {r['bh_pnl']:.2f}, Sharpe {r['bh_sharpe']:.2f}, "
        f"max drawdown {r['bh_max_drawdown']:.6f}"
    )
    if r["retorno_anualizado"] is not None:
        click.echo(f"Retorno anualizado: {r['retorno_anualizado']:.2%}")
        click.echo(f"Exceso sobre la referencia: {r['exceso_sobre_referencia']:.2%}")
        click.echo(f"Tiempo en mercado: {r['tiempo_en_mercado']:.2%}")
        click.echo(f"Capital invertido medio: {r['capital_invertido_medio']:.2%}")
        click.echo(f"Sharpe (tasa 0): {r['sharpe_tasa_0']:.2f}")
        peor_pct, peor_fecha = r["peor_dia"]
        if peor_pct < 0 and peor_fecha:
            click.echo(f"Peor día: {peor_pct:.2%} ({peor_fecha})")
        else:
            click.echo("Peor día: 0.00%")

    if out_path:
        with open(out_path, "w", newline="") as f:
            f.write(trades_to_csv(rows))
        click.echo(f"Trade log written to {out_path}")

    if equity_out_path:
        with open(equity_out_path, "w", newline="") as f:
            f.write(equity_to_csv(r["equity_curve"]))
        click.echo(
            f"Equity curve written to {equity_out_path} "
            f"(max drawdown: {r['max_drawdown']:.6f})"
        )


@contextmanager
def _isolated_sqlite_db():
    import os
    import tempfile

    from sqlalchemy import event

    from quantagent import database, settings

    shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    with tempfile.TemporaryDirectory(dir=shm_dir) as tmp_dir:
        db_path = os.path.join(tmp_dir, "verify.db")
        db_url = f"sqlite:///{db_path}"
        old_env = os.environ.get("DATABASE_URL")
        old_settings_url = settings.DATABASE_URL
        os.environ["DATABASE_URL"] = db_url
        settings.DATABASE_URL = db_url
        if hasattr(database, "_engine"):
            database._engine = None
        if hasattr(database, "_SessionLocal"):
            database._SessionLocal = None
        try:
            eng = database._get_engine()

            def _set_sync_off(dbapi_con, _):
                cursor = dbapi_con.cursor()
                cursor.execute("PRAGMA synchronous=OFF")
                cursor.close()

            event.listen(eng, "connect", _set_sync_off)
            database.init_db()
            yield
        finally:
            if old_env is not None:
                os.environ["DATABASE_URL"] = old_env
            else:
                os.environ.pop("DATABASE_URL", None)
            settings.DATABASE_URL = old_settings_url
            if hasattr(database, "_engine"):
                if database._engine is not None:
                    database._engine.dispose()
                database._engine = None
            if hasattr(database, "_SessionLocal"):
                database._SessionLocal = None


def _run_verify_pass(
    strategy_name: str,
    fixture_name: Optional[str] = None,
    verbose: bool = False,
    *,
    snapshot_name: Optional[str] = None,
    symbol: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    meta: Optional[FixtureMetadata] = None,
    commission_pct: Optional[float] = None,
    costos: Optional[str] = None,
    config: Optional[dict] = None,
):
    if meta is None:
        if snapshot_name:
            meta = snapshot_metadata(snapshot_name, symbol, from_date, to_date)
        else:
            meta = fixture_metadata(fixture_name)
    with _isolated_sqlite_db(), utils.session_scope() as session:
        if snapshot_name:
            load_snapshot(session, snapshot_name, symbol, from_date, to_date)
        else:
            load_fixture(session, fixture_name)
        bt = Backtest(
            start_date=meta.start_date,
            end_date=meta.end_date,
            assets=[meta.symbol],
            timeframe=meta.timeframe,
            config={**(config or _engine_config(commission_pct, costos)),
                    "source_name": f"snapshot:{snapshot_name}" if snapshot_name else "fixture"},
            db_session=session,
            strategy=build_strategy(cli_aliases()[strategy_name]),
        )
        engine_logger = logging.getLogger("quantagent.backtesting.backtest")
        if not verbose:
            engine_logger.addFilter(_hide_data_warnings)
        try:
            metrics = bt.run(name=f"verify-{strategy_name}-{fixture_name or snapshot_name}")
        finally:
            engine_logger.removeFilter(_hide_data_warnings)

        slippage = bt.order_manager.broker.slippage_pct
        rows = _extract_trade_rows(session, bt.backtest_run_id)
        return metrics, slippage, trades_to_csv(rows)


@backtest_group.command("verify", help="Verify backtest reproducibility across two runs.")
@click.option(
    "--strategy",
    "strategy_name",
    required=True,
    type=_AliasChoice(),
    help="Strategy to run.",
)
@click.option(
    "--fixture",
    "fixture_name",
    help="Fixture name under tests/fixtures/ (without the .csv extension). Excluyente con --snapshot.",
)
@click.option("--snapshot", "snapshot_name", help="Snapshot en $QUANTAGENT_SNAPSHOT_DIR (velas ajustadas). Requiere --symbol.")
@click.option("--symbol", "symbol", help="Símbolo del snapshot (uno por corrida).")
@click.option("--from", "from_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="Primera sesión, YYYY-MM-DD (default: la primera del snapshot).")
@click.option("--to", "to_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="Última sesión, YYYY-MM-DD (default: la última del snapshot).")
@click.option(
    "--abrir-reserva",
    "abrir_reserva",
    is_flag=True,
    help="Permite que --to llegue a la reserva (desde $QUANTAGENT_RESERVA_DESDE, default 2023-01-01). Queda registrado.",
)
@click.option(
    "--abrir-prueba",
    "abrir_prueba",
    is_flag=True,
    help="Permite que --to llegue al tramo de prueba (desde $QUANTAGENT_PRUEBA_DESDE, default 2019-01-01). Queda registrado.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Also show engine warnings on stderr.",
)
@click.option(
    "--commission-pct",
    "commission_pct",
    type=click.FloatRange(min=0.0),
    default=None,
    help="Comisión por lado sobre el nocional (0.001 = 0.10%). Default: TRADING_COMMISSION_PCT.",
)
@click.option(
    "--costos",
    type=click.Choice(sorted(COSTOS_PERFILES)),
    default=None,
    help="Perfil de costos (slippage y comisión por lado). Excluyente con --commission-pct.",
)
@_tamano_options
def verify_backtest(
    strategy_name: str,
    fixture_name: Optional[str],
    snapshot_name: Optional[str],
    symbol: Optional[str],
    from_date: Optional[datetime],
    to_date: Optional[datetime],
    abrir_reserva: bool,
    abrir_prueba: bool,
    verbose: bool,
    commission_pct: Optional[float],
    costos: Optional[str],
    tamano_posicion: Optional[float],
    regla_tamano: Optional[str],
    orden_minima: Optional[float],
) -> None:
    import difflib

    config = _engine_config(commission_pct, costos, tamano_posicion, regla_tamano, orden_minima)
    meta, symbol, to_date = _resolve_source_and_meta(
        "verify",
        strategy_name,
        fixture_name,
        snapshot_name,
        symbol,
        from_date,
        to_date,
        abrir_reserva,
        abrir_prueba,
    )

    m1, s1, csv1 = _run_verify_pass(
        strategy_name,
        fixture_name,
        verbose,
        snapshot_name=snapshot_name,
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        meta=meta,
        commission_pct=commission_pct,
        costos=costos,
        config=config,
    )
    m2, s2, csv2 = _run_verify_pass(
        strategy_name,
        fixture_name,
        verbose,
        snapshot_name=snapshot_name,
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        meta=meta,
        commission_pct=commission_pct,
        costos=costos,
        config=config,
    )

    diff_lines = []
    if csv1 != csv2:
        diff_lines.extend(
            difflib.unified_diff(
                csv1.splitlines(keepends=True),
                csv2.splitlines(keepends=True),
                fromfile="run1_trades.csv",
                tofile="run2_trades.csv",
            )
        )

    for field in [
        "total_trades",
        "win_rate",
        "profit_factor",
        "sharpe_ratio",
        "total_pnl",
        "max_drawdown",
    ]:
        v1, v2 = getattr(m1, field), getattr(m2, field)
        if v1 != v2:
            diff_lines.append(f"Metric mismatch '{field}': run1={v1} != run2={v2}\n")

    if s1 != s2:
        diff_lines.append(f"Slippage mismatch: run1={s1} != run2={s2}\n")

    if diff_lines:
        click.echo("".join(diff_lines), nl=False)
        raise click.ClickException("Backtest reproducibility check failed")

    _echo_tamano(config)
    click.echo("OK reproducible")


COMPARE_COLUMNS = [
    "estrategia", "activo", "desde", "hasta", "trades", "win_rate", "profit_factor",
    "sharpe", "pnl", "max_drawdown", "bh_pnl", "bh_sharpe", "bh_max_drawdown",
    "retorno_anualizado", "exceso_sobre_referencia", "tiempo_en_mercado",
    "capital_invertido_medio", "sharpe_tasa_0",
]

CRISIS_ANIOS = (2008, 2020, 2022)


def _retorno_anio(puntos: list[tuple[datetime, float]], anio: int) -> float:
    """Último valor del año / último valor del año anterior, menos 1."""
    previo = [v for d, v in puntos if d.year == anio - 1][-1]
    return [v for d, v in puntos if d.year == anio][-1] / previo - 1


def crisis_columnas(curve: list[dict], cierres: list[tuple[datetime, float]], to_date: datetime) -> dict:
    """crisis_<año>: retorno de la estrategia en el año menos el de comprar y mantener; vacía si el año no está completo."""
    out = {}
    puntos = [(p["date"], p["equity"]) for p in curve]
    for anio in CRISIS_ANIOS:
        completo = (
            bool(curve) and curve[0]["date"].year < anio <= curve[-1]["date"].year
            and to_date >= datetime(anio, 12, 31)
        )
        out[f"crisis_{anio}"] = round(_retorno_anio(puntos, anio) - _retorno_anio(cierres, anio), 4) if completo else ""
    return out


def generar_markdown_compare(
    rows: list[dict],
    snapshot_name: str,
    costos: Optional[str] = None,
    tamano_posicion: Optional[float] = 1.0,
    regla_tamano: Optional[str] = None,
    columns: Optional[list[str]] = None,
    params: Optional[dict] = None,
) -> str:
    """Genera el documento Markdown de compare con encabezado de auditoría y tabla."""
    cols = columns or (list(rows[0].keys()) if rows else COMPARE_COLUMNS)
    desde = min((r["desde"] for r in rows), default="")
    hasta = max((r["hasta"] for r in rows), default="")

    manifest = load_manifest(snapshot_name)
    syms = sorted({r["activo"] for r in rows})
    manifest_syms = manifest.get("symbols", {})

    if costos and costos in COSTOS_PERFILES:
        c_cfg = COSTOS_PERFILES[costos]
        comm, slip = c_cfg.get("commission_pct", 0.0), c_cfg.get("slippage_pct", 0.0)
        costos_desc = f"{costos} (comisión: {comm * 100:g}%, slippage: {slip * 100:g}%)"
    else:
        comm, slip = settings.TRADING_COMMISSION_PCT, settings.TRADING_SLIPPAGE_PCT
        costos_desc = f"ninguno (comisión: {comm * 100:g}%, slippage: {slip * 100:g}%)"

    tamano = 1.0 if tamano_posicion is None else tamano_posicion
    regla = regla_tamano or "confianza"
    tamano_desc = f"{tamano * 100:g}% del capital, regla {regla}"

    aliases = cli_aliases()
    strats = sorted({r["estrategia"] for r in rows})
    strat_lines = []
    for s in strats:
        reg_name = aliases.get(s, s)
        params_info = effective_params(reg_name, (params or {}).get(s)) if reg_name in STRATEGY_REGISTRY else {}
        p_str = ", ".join(f"{k}={v}" for k, v in params_info.items()) if params_info else "sin parámetros"
        strat_lines.append(f"- **{s}**: {p_str}")

    hash_lines = [
        f"- **{sym}**: `{manifest_syms.get(sym, {}).get('sha256', 'n/a')}`"
        for sym in syms
    ]

    table_data = []
    for r in rows:
        row_vals = []
        trades_count = int(r.get("trades") or 0)
        for c in cols:
            val = r.get(c, "")
            if c == "profit_factor" and (val == "" or val is None) and trades_count > 0:
                val = "sin pérdidas"
            row_vals.append(val)
        table_data.append(row_vals)

    table_md = tabulate(table_data, headers=cols, tablefmt="github")

    lines = [
        "# Comparación de estrategias",
        "",
        "## Configuración",
        f"- **Rango:** {desde} a {hasta}",
        f"- **Snapshot:** {snapshot_name}",
        f"- **Costos:** {costos_desc}",
        f"- **Posición:** {tamano_desc}",
        "",
        "## Símbolos y hashes (SHA-256)",
        *hash_lines,
        "",
        "## Parámetros de estrategias",
        *strat_lines,
        "",
        "## Resultados",
        "",
        table_md,
        "",
    ]
    return "\n".join(lines)


@backtest_group.command("compare", help="Compare multiple strategies across symbols against a snapshot.")
@click.option("--strategies", required=True, help="Estrategias separadas por coma (ej: rsi,sma-cross).")
@click.option("--symbols", required=True, help="Símbolos separados por coma (ej: SPY,IWM).")
@click.option("--snapshot", "snapshot_name", required=True, help="Snapshot en $QUANTAGENT_SNAPSHOT_DIR (velas ajustadas).")
@click.option("--from", "from_date", type=click.DateTime(formats=["%Y-%m-%d"]), default=None, help="Primera sesión.")
@click.option("--to", "to_date", type=click.DateTime(formats=["%Y-%m-%d"]), default=None, help="Última sesión.")
@click.option("--costos", type=click.Choice(sorted(COSTOS_PERFILES)), default=None, help="Perfil de costos.")
@click.option("--tamano-posicion", "tamano_posicion", type=click.FloatRange(min=0.0, max=1.0, min_open=True), default=1.0)
@click.option("--regla-tamano", "regla_tamano", type=click.Choice(PositionSizer.REGLAS), default=None)
@click.option("--orden-minima", "orden_minima", type=click.FloatRange(min=0.0, min_open=True), default=None, help="Monto mínimo en USD por orden de entrada.")
@click.option("--abrir-reserva", "abrir_reserva", is_flag=True, help="Permite abrir la reserva.")
@click.option("--abrir-prueba", "abrir_prueba", is_flag=True, help="Permite abrir el tramo de prueba.")
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, writable=True), help="CSV de salida.")
@click.option("--trades-dir", "trades_dir", type=click.Path(file_okay=False, writable=True), default=None)
@click.option("--equity-dir", type=click.Path(file_okay=False, writable=True), default=None, help="CSV date,equity del motor por celda (equity con 2 decimales).")
@click.option("--md", "md_path", type=click.Path(dir_okay=False, writable=True), default=None, help="Documento Markdown de salida.")
@click.option("--crisis", is_flag=True, help="Suma crisis_2008/2020/2022: retorno del año menos comprar y mantener (informativo).")
@click.option("--llenado", type=click.Choice(["apertura", "cierre"]), default="apertura", show_default=True)
@_params_option
def compare_backtests(
    strategies: str, symbols: str, snapshot_name: str, from_date: Optional[datetime],
    to_date: Optional[datetime], costos: Optional[str], tamano_posicion: Optional[float],
    regla_tamano: Optional[str], abrir_reserva: bool, abrir_prueba: bool, out_path: str, trades_dir: Optional[str],
    params: dict, md_path: Optional[str] = None, orden_minima: Optional[float] = None, equity_dir: Optional[str] = None,
    crisis: bool = False,
    llenado: str = "apertura",
) -> None:
    columns = COMPARE_COLUMNS + [f"crisis_{a}" for a in CRISIS_ANIOS] if crisis else COMPARE_COLUMNS
    reserva = _reserva_desde()
    prueba = _prueba_desde()
    tope = min(prueba, reserva)
    abriendo_reserva = False
    abriendo_prueba = False
    if to_date is None:
        to_date = tope - timedelta(days=1)
        motivo = (
            f"el tramo de prueba empieza el {prueba:%Y-%m-%d}"
            if prueba <= reserva
            else f"la reserva empieza el {reserva:%Y-%m-%d}"
        )
        click.echo(
            f"Aviso: sin --to, se asume fin en {to_date:%Y-%m-%d} ({motivo}).",
            err=True,
        )
    elif to_date >= reserva:
        if not abrir_reserva:
            raise click.ClickException(
                f"--to {to_date:%Y-%m-%d} cae en la reserva, que empieza el {reserva:%Y-%m-%d}: "
                "ninguna corrida la lee. Para abrirla, --abrir-reserva (queda registrado)."
            )
        abriendo_reserva = True
    elif to_date >= prueba:
        if not abrir_prueba:
            raise click.ClickException(
                f"--to {to_date:%Y-%m-%d} cae en el tramo de prueba, que empieza el {prueba:%Y-%m-%d}: "
                "para abrirlo, --abrir-prueba (queda registrado)."
            )
        abriendo_prueba = True

    strats = [s.strip() for s in strategies.split(",") if s.strip()]
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not strats:
        raise click.UsageError("--strategies requiere al menos una estrategia.")
    if not syms:
        raise click.UsageError("--symbols requiere al menos un símbolo.")

    aliases = cli_aliases()
    for s in strats:
        if s not in aliases:
            raise click.UsageError(f"Estrategia desconocida: '{s}'. Opciones: {', '.join(sorted(aliases))}")

    if abriendo_reserva:
        partes = [f"backtest compare --strategies {strategies} --symbols {symbols} --snapshot {snapshot_name}"]
        if from_date:
            partes.append(f"--from {from_date:%Y-%m-%d}")
        partes.append(f"--to {to_date:%Y-%m-%d} --abrir-reserva")
        _registrar_apertura(" ".join(partes))
    elif abriendo_prueba:
        partes = [f"backtest compare --strategies {strategies} --symbols {symbols} --snapshot {snapshot_name}"]
        if from_date:
            partes.append(f"--from {from_date:%Y-%m-%d}")
        partes.append(f"--to {to_date:%Y-%m-%d} --abrir-prueba")
        _registrar_apertura_prueba(" ".join(partes))

    config = _engine_config(None, costos, tamano_posicion, regla_tamano, orden_minima)
    if trades_dir:
        os.makedirs(trades_dir, exist_ok=True)
    if equity_dir:
        os.makedirs(equity_dir, exist_ok=True)

    rows = []
    for strat in strats:
        for sym in syms:
            try:
                meta = snapshot_metadata(snapshot_name, sym, from_date, to_date)
            except (SnapshotError, ValueError, FileNotFoundError) as exc:
                raise click.ClickException(str(exc)) from exc

            with _isolated_sqlite_db():
                r = correr_par(
                    strat, meta, config, costos, None, snapshot_name, sym,
                    from_date, to_date, con_filas=bool(trades_dir), params=params.get(strat),
                    fill=llenado,
                )

            if trades_dir:
                with open(os.path.join(trades_dir, f"{strat}-{sym}.csv"), "w", newline="") as f:
                    f.write(trades_to_csv(r["filas"]))
            if equity_dir:
                with open(os.path.join(equity_dir, f"{strat}-{sym}.csv"), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["date", "equity"])
                    writer.writerows((p["date"].isoformat(), f'{p["equity"]:.2f}') for p in r["equity_curve"])

            def _fmt(v: Optional[float], digits: int) -> object:
                return "" if v is None else round(v, digits)

            pf = "" if r["profit_factor"] is None or math.isinf(r["profit_factor"]) else round(r["profit_factor"], 2)
            rows.append({
                "estrategia": strat, "activo": sym, "desde": f"{meta.start_date:%Y-%m-%d}",
                "hasta": f"{meta.end_date:%Y-%m-%d}", "trades": r["trades"],
                "win_rate": round(r["win_rate"], 4), "profit_factor": pf,
                "sharpe": round(r["sharpe"], 2), "pnl": round(r["pnl"], 2),
                "max_drawdown": round(r["max_drawdown"], 6), "bh_pnl": round(r["bh_pnl"], 2),
                "bh_sharpe": round(r["bh_sharpe"], 2), "bh_max_drawdown": round(r["bh_max_drawdown"], 6),
                "retorno_anualizado": _fmt(r["retorno_anualizado"], 4),
                "exceso_sobre_referencia": _fmt(r["exceso_sobre_referencia"], 4),
                "tiempo_en_mercado": _fmt(r["tiempo_en_mercado"], 4),
                "capital_invertido_medio": _fmt(r["capital_invertido_medio"], 4),
                "sharpe_tasa_0": _fmt(r["sharpe_tasa_0"], 2),
                **(crisis_columnas(r["equity_curve"], r["cierres"], to_date) if crisis else {}),
            })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    if md_path:
        md_text = generar_markdown_compare(
            rows,
            snapshot_name=snapshot_name,
            costos=costos,
            tamano_posicion=tamano_posicion,
            regla_tamano=regla_tamano,
            columns=columns,
            params=params,
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

    table_data = [[row[col] for col in columns] for row in rows]
    click.echo(tabulate(table_data, headers=columns, tablefmt="github"))


backtest_group.add_command(sweep_command, name="sweep")
backtest_group.add_command(attempts_command, name="attempts")
backtest_group.add_command(walkforward_command, name="walkforward")
backtest_group.add_command(verdict_command)
backtest_group.add_command(hipotesis_group)
backtest_group.add_command(data_group, name="data")

