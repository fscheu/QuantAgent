"""CLI commands for running deterministic backtests (D13-D18 of PLAN-30-DIAS)."""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import click
from sqlalchemy import select
from sqlalchemy.orm import Session

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
from quantagent.data.snapshot import SnapshotError, snapshot_root
from quantagent.models import ActivePosition, MarketData, Trade
from quantagent.settings import COSTOS_PERFILES
from quantagent.strategy.registry import build_strategy, cli_aliases
from quantagent.trading.position_sizer import PositionSizer

from . import utils


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


def _reserva_desde() -> datetime:
    value = os.environ.get(RESERVA_ENV, RESERVA_DEFAULT)
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise click.ClickException(f"{RESERVA_ENV}={value!r} no es una fecha YYYY-MM-DD")


def _registrar_apertura(comando: str) -> None:
    """Agrega fecha y comando a `$QUANTAGENT_SNAPSHOT_DIR/reserva-accesos.log`."""
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (snapshot_root() / RESERVA_LOG).open("a") as f:
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
):
    if bool(fixture_name) == bool(snapshot_name):
        raise click.UsageError("Indicá exactamente uno: --fixture o --snapshot.")
    if snapshot_name and not symbol:
        raise click.UsageError("--snapshot requiere --symbol.")
    if fixture_name and (symbol or from_date or to_date or abrir_reserva):
        raise click.UsageError("--symbol, --from, --to y --abrir-reserva solo se usan con --snapshot.")
    abriendo = False
    if snapshot_name:
        reserva = _reserva_desde()
        if to_date is None:
            to_date = reserva - timedelta(days=1)
        elif to_date >= reserva:
            if not abrir_reserva:
                raise click.ClickException(
                    f"--to {to_date:%Y-%m-%d} cae en la reserva, que empieza el {reserva:%Y-%m-%d}: "
                    "ninguna corrida la lee. Para abrirla, --abrir-reserva (queda registrado)."
                )
            abriendo = True
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

    if abriendo:
        partes = [f"backtest {subcommand} --strategy {strategy_name} --snapshot {snapshot_name} --symbol {symbol}"]
        partes += [f"--from {from_date:%Y-%m-%d}"] if from_date else []
        partes += [f"--to {to_date:%Y-%m-%d} --abrir-reserva"]
        _registrar_apertura(" ".join(partes))

    return meta, symbol, to_date


def _engine_config(commission_pct: Optional[float], costos: Optional[str] = None,
                   tamano_posicion: Optional[float] = None, regla_tamano: Optional[str] = None) -> dict:
    config = {"market_hours_filter": False, "offline_data": True}
    if tamano_posicion is not None:
        config["base_position_pct"] = config["max_position_pct"] = tamano_posicion
    if regla_tamano is not None:
        config["regla_tamano"] = regla_tamano
    if costos is not None:
        if commission_pct is not None:
            raise click.UsageError("--costos y --commission-pct son excluyentes: elegí uno.")
        config.update(COSTOS_PERFILES[costos])
    elif commission_pct is not None:
        config["commission_pct"] = commission_pct
    return config


def _tamano_options(f):
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
) -> dict:
    """Corre un par estrategia/activo y devuelve los números del resumen de `backtest run`.

    Claves: trades, win_rate, profit_factor (None o inf si no hay pérdidas), sharpe, pnl,
    slippage_pct, comision_pct, costos (nombre del perfil o None), tamano_pct (None si no se
    pidió tamaño), regla_tamano (None si no se pidió), bh_pnl, bh_sharpe, bh_max_drawdown,
    retorno_anualizado, exceso_sobre_referencia, tiempo_en_mercado, capital_invertido_medio,
    sharpe_tasa_0, peor_dia (estas seis son None si no hay curva de equity), max_drawdown,
    equity_curve y filas (TradeRow; vacía salvo con con_filas=True).
    """
    with utils.session_scope() as session:
        existing_count = (
            session.query(MarketData)
            .filter(
                MarketData.symbol == meta.symbol,
                MarketData.timeframe == meta.timeframe,
                MarketData.timestamp >= meta.start_date,
                MarketData.timestamp <= meta.end_date,
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
            config=config,
            db_session=session,
            strategy=build_strategy(cli_aliases()[strategy_name]),
            intrabar_stops=intrabar_stops,
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
@_tamano_options
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
    out_path: Optional[str],
    equity_out_path: Optional[str],
    verbose: bool,
    intrabar_stops: bool,
    commission_pct: Optional[float],
    costos: Optional[str],
    tamano_posicion: Optional[float],
    regla_tamano: Optional[str],
) -> None:
    config = _engine_config(commission_pct, costos, tamano_posicion, regla_tamano)
    meta, symbol, to_date = _resolve_source_and_meta(
        "run",
        strategy_name,
        fixture_name,
        snapshot_name,
        symbol,
        from_date,
        to_date,
        abrir_reserva,
    )

    r = correr_par(
        strategy_name, meta, config, costos, fixture_name, snapshot_name, symbol,
        from_date, to_date, intrabar_stops, verbose, con_filas=bool(out_path),
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

    from quantagent import database, settings

    with tempfile.TemporaryDirectory() as tmp_dir:
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
            database.init_db()
            yield
        finally:
            if old_env is not None:
                os.environ["DATABASE_URL"] = old_env
            else:
                os.environ.pop("DATABASE_URL", None)
            settings.DATABASE_URL = old_settings_url
            if hasattr(database, "_engine"):
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
            config=config or _engine_config(commission_pct, costos),
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
    verbose: bool,
    commission_pct: Optional[float],
    costos: Optional[str],
    tamano_posicion: Optional[float],
    regla_tamano: Optional[str],
) -> None:
    import difflib

    config = _engine_config(commission_pct, costos, tamano_posicion, regla_tamano)
    meta, symbol, to_date = _resolve_source_and_meta(
        "verify",
        strategy_name,
        fixture_name,
        snapshot_name,
        symbol,
        from_date,
        to_date,
        abrir_reserva,
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
