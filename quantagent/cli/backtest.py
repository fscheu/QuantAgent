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
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from tabulate import tabulate

from quantagent import settings
from quantagent.backtesting.backtest import Backtest
from quantagent.backtesting.buy_and_hold import annual_return, buy_and_hold, buy_and_hold_cuenta_chica, exposure, worst_day
from quantagent.backtesting.export import TradeRow, equity_to_csv, trades_to_csv
from quantagent.backtesting.fixtures import (
    FixtureMetadata,
    fixture_metadata,
    load_fixture,
    load_snapshot,
    snapshot_metadata,
)
from quantagent.data.snapshot import SnapshotError, load_manifest, read_snapshot, snapshot_root
from quantagent.data.asset_types import get_asset_type
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
                   orden_minima: Optional[float] = None, riesgo_por_trade: Optional[float] = None,
                   cuenta_chica: Optional[float] = None, retencion_dividendos: Optional[float] = None,
                   costo_fiscal_pct: float = 0.0, spread_bps: Optional[float] = None,
                   costo_entrada_usd: Optional[float] = None) -> dict:
    costo_entrada_pedido = costo_entrada_usd is not None
    costo_entrada_usd = costo_entrada_usd or 0.0
    config = {"market_hours_filter": False, "offline_data": True}
    if cuenta_chica is not None:
        config.update(initial_capital=cuenta_chica - costo_entrada_usd, round_shares=True,
                      costo_entrada_usd=costo_entrada_usd)
    elif spread_bps is not None or costo_entrada_pedido:
        raise click.UsageError("--spread-bps y --costo-entrada-usd requieren --cuenta-chica.")
    if spread_bps is not None:
        config["spread_bps"] = spread_bps
    if cuenta_chica is not None or retencion_dividendos is not None or costo_fiscal_pct:
        config["retencion_dividendos"] = (0.30 if cuenta_chica is not None else 0.0) if retencion_dividendos is None else retencion_dividendos
        config["costo_fiscal_pct"] = costo_fiscal_pct
    if tamano_posicion is not None:
        config["base_position_pct"] = config["max_position_pct"] = tamano_posicion
    if regla_tamano is not None:
        config["regla_tamano"] = regla_tamano
    if riesgo_por_trade is not None:
        config["riesgo_por_trade_pct"] = riesgo_por_trade
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
        help="Minimum USD amount per entry order.",
    )(f)
    f = click.option(
        "--regla-tamano", "regla_tamano", type=click.Choice(PositionSizer.REGLAS), default=None,
        help="Position sizing rule: confidence (capital × fraction × signal confidence) or fixed. Default: confidence.",
    )(f)
    f = click.option("--riesgo-por-trade", type=click.FloatRange(min=0.0, min_open=True), default=None,
                     help="Capital fraction at risk per trade.")(f)
    return click.option(
        "--tamano-posicion", "tamano_posicion", type=click.FloatRange(min=0.0, max=1.0, min_open=True), default=None,
        help="Capital fraction per position, in (0, 1] (1.0 = 100%). Sets the run's base and maximum size.",
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
    help='JSON {"<strategy>": {"<parameter>": value}} with strategy parameters (default: registry values).',
)


def _hide_data_warnings(record: logging.LogRecord) -> bool:
    return getattr(record, "event_type", None) != "backtest_data_warning"


def _extract_trade_rows(session: Session, backtest_run_id: int, spread_bps: Optional[float] = None) -> list[TradeRow]:
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
            spread_cost_entry_usd=None if spread_bps is None else float(trade.quantity) * float(trade.entry_price) * spread_bps / 10000,
            spread_cost_exit_usd=None if spread_bps is None or trade.exit_price is None
            else float(trade.quantity) * float(trade.exit_price) * spread_bps / 10000,
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
    sharpe_tasa_0, peor_dia (estas seis son None si no hay curva de equity), posiciones_en_cero
    (solo con --cuenta-chica),
    max_drawdown,
    equity_curve, backtest_run_id (id de BacktestRun o None si no se guardó la corrida), cierres (timestamp, cierre ajustado de cada vela), entradas_descartadas_orden_minima y filas (TradeRow; vacía salvo con con_filas=True).
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

        if snapshot_name and config.get("retencion_dividendos", 0.0) + config.get("costo_fiscal_pct", 0.0):
            frame = read_snapshot(snapshot_name, symbol)
            rates = frame["close"].pct_change() - frame["close_sin_dividendos"].pct_change()
            config = {
                **config,
                "dividend_rates": {
                    ts.date().isoformat(): float(rate)
                    for ts, rate in rates.loc[meta.start_date:meta.end_date].fillna(0).items()
                    if abs(rate) >= 1e-9
                },
            }
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
        spread_bps = config.get("spread_bps")
        if spread_bps is None:
            spread_bps = settings.SPREAD_BPS_POR_CLASE[get_asset_type(meta.symbol).value] if config.get("round_shares") else None
        rows = _extract_trade_rows(session, bt.backtest_run_id, spread_bps) if con_filas else []
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
        "backtest_run_id": bt.backtest_run_id,
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
        **({"posiciones_en_cero": bt.positions_en_cero} if config.get("round_shares") else {}),
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


@backtest_group.command(
    "run", help="Run a backtest against a versioned fixture.",
    epilog="Example: quantagent backtest run --strategy rsi --fixture spy-90d --cuenta-chica 2500",
)
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
    help="Fixture name under tests/fixtures/ (without the .csv extension). Mutually exclusive with --snapshot.",
)
@click.option("--snapshot", "snapshot_name", help="Snapshot in $QUANTAGENT_SNAPSHOT_DIR (adjusted bars). Requires --symbol.")
@click.option("--symbol", "symbol", help="Snapshot symbol (one per run).")
@click.option("--from", "from_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="First session, YYYY-MM-DD (default: first in the snapshot).")
@click.option("--to", "to_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="Last session, YYYY-MM-DD (default: last in the snapshot).")
@click.option(
    "--abrir-reserva",
    "abrir_reserva",
    is_flag=True,
    help="Allow --to to reach the reserve period (from $QUANTAGENT_RESERVA_DESDE, default 2023-01-01). Logged.",
)
@click.option(
    "--abrir-prueba",
    "abrir_prueba",
    is_flag=True,
    help="Allow --to to reach the test period (from $QUANTAGENT_PRUEBA_DESDE, default 2019-01-01). Logged.",
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
    help="Commission per side on notional (0.001 = 0.10%). Default: TRADING_COMMISSION_PCT.",
)
@click.option(
    "--costos",
    type=click.Choice(sorted(COSTOS_PERFILES)),
    default=None,
    help="Cost profile (slippage and commission per side). Mutually exclusive with --commission-pct.",
)
@click.option("--llenado", type=click.Choice(["apertura", "cierre"]), default="apertura", show_default=True,
              help="Fill price: opening or closing price of the candle.")
@click.option("--cuenta-chica", type=click.FloatRange(min=0.0, min_open=True), default=None,
              help="Initial backtest capital in USD; enables whole-share entries. Default: disabled ($100,000 capital).")
@click.option("--spread-bps", type=click.FloatRange(min=0.0), default=None,
              help="Spread per side in basis points (defaults to the asset-class estimate); requires --cuenta-chica.")
@click.option("--costo-entrada-usd", type=click.FloatRange(min=0.0), default=None,
              help="Fixed USD cost charged once at account start (default: USD 0); requires --cuenta-chica.")
@click.option("--retencion-dividendos", type=click.FloatRange(min=0.0, max=1.0), default=None,
              help="Dividend withholding fraction (0–1); defaults to 0.30 with --cuenta-chica, otherwise 0.")
@click.option("--costo-fiscal-pct", type=click.FloatRange(min=0.0, max=1.0), default=0.0, show_default=True,
              help="Additional fraction on implicit dividends; placeholder for future capital gains/personal assets taxes.")
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
    riesgo_por_trade: Optional[float],
    cuenta_chica: Optional[float],
    spread_bps: Optional[float],
    costo_entrada_usd: Optional[float],
    retencion_dividendos: Optional[float],
    costo_fiscal_pct: float,
    params: dict,
    llenado: str,
) -> None:
    config = _engine_config(commission_pct, costos, tamano_posicion, regla_tamano, orden_minima, riesgo_por_trade, cuenta_chica, retencion_dividendos, costo_fiscal_pct, spread_bps, costo_entrada_usd)
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

    if r["backtest_run_id"] is not None:
        equity_dir = os.environ.get("QUANTAGENT_BACKTEST_EQUITY_DIR", "local-data/backtest-equity")
        os.makedirs(equity_dir, exist_ok=True)
        with open(os.path.join(equity_dir, f"{r['backtest_run_id']}.csv"), "w", newline="") as f:
            f.write(equity_to_csv(r["equity_curve"]))

    click.echo(f"Trades: {r['trades']}")
    click.echo(f"Win rate: {r['win_rate']:.2%}")
    if r["profit_factor"] is None or math.isinf(r["profit_factor"]):
        click.echo("Profit factor: n/a")
    else:
        click.echo(f"Profit factor: {r['profit_factor']:.2f}")
    click.echo(f"Sharpe ratio: {r['sharpe']:.2f}")
    click.echo(f"Total PnL: {r['pnl']:.2f}")
    if cuenta_chica is not None:
        click.echo(f"Costo fijo de entrada: {(costo_entrada_usd or 0.0):.2f} USD")
        click.echo(f"Capital inicial efectivo: {config['initial_capital']:.2f} USD")
    click.echo(f"Slippage: {r['slippage_pct'] * 100:.2f}% por lado")
    click.echo(f"Comisión: {r['comision_pct'] * 100:.2f}% por lado")
    if costos:
        click.echo(f"Costos: {costos}")
    _echo_tamano(config)
    if "orden_minima" in config:
        click.echo(f"Entradas descartadas por orden mínima: {r['entradas_descartadas_orden_minima']}")
    if cuenta_chica is not None:
        dates_by_symbol = {}
        for position in r["posiciones_en_cero"]:
            dates = dates_by_symbol.setdefault(position["symbol"], {})
            dates[position["date"].strftime("%Y-%m-%d")] = None
        summary = "; ".join(
            f"{symbol}: {', '.join(list(dates)[:10])}{', …' if len(dates) > 10 else ''}"
            for symbol, dates in dates_by_symbol.items()
        )
        click.echo(f"posicion_en_cero: {len(r['posiciones_en_cero'])}" + (f" ({summary})" if summary else ""))
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


@backtest_group.command("cuenta-chica", help="Compare small-account backtests with a cost-matched SPY buy-and-hold benchmark.",
                        epilog="Example: quantagent backtest cuenta-chica --hipotesis hipotesis.yaml --snapshot etf-1d")
@click.option("--strategy", "strategy_name", required=False, type=_AliasChoice(), help="Strategy to run when using a fixture or one-symbol snapshot.")
@click.option("--hipotesis", type=click.Path(exists=True, dir_okay=False), default=None, help="Portfolio hypothesis YAML to simulate.")
@click.option("--fixture", "fixture_name", help="Fixture name under tests/fixtures/ (without .csv).")
@click.option("--snapshot", "snapshot_name", help="Snapshot in $QUANTAGENT_SNAPSHOT_DIR.")
@click.option("--macro-snapshot", help="Macro snapshot in $QUANTAGENT_SNAPSHOT_DIR (required for curva-pendiente).")
@click.option("--symbol", help="Snapshot symbol (one per run).")
@click.option("--from", "from_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="First session, YYYY-MM-DD (default: first in the snapshot).")
@click.option("--to", "to_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="Last session, YYYY-MM-DD (default: last in the snapshot).")
@click.option("--capitales", default="1000,2500,5000", show_default=True, help="Comma-separated initial capital amounts in USD.")
@click.option("--commission-pct", type=click.FloatRange(min=0.0), default=None, help="Commission per side on notional (0.001 = 0.10%). Default: TRADING_COMMISSION_PCT.")
@click.option("--costos", type=click.Choice(sorted(COSTOS_PERFILES)), default=None, help="Cost profile (slippage and commission per side; default: none).")
@click.option("--spread-bps", type=click.FloatRange(min=0.0), default=None, help="Spread per side in basis points (defaults to the asset-class estimate).")
@click.option("--costo-entrada-usd", type=click.FloatRange(min=0.0), default=None, help="Fixed USD cost charged once at account start (default: USD 0).")
@click.option("--retencion-dividendos", type=click.FloatRange(min=0.0, max=1.0), default=None, help="Dividend withholding fraction (0–1); defaults to 0.30.")
@click.option("--costo-fiscal-pct", type=click.FloatRange(min=0.0, max=1.0), default=0.0, show_default=True, help="Additional fraction on implicit dividends.")
@_tamano_options
@_params_option
@click.option("--intrabar-stops/--no-intrabar-stops", default=True, help="Evaluate stop loss and take profit against intrabar high/low prices (default: enabled).")
@click.option("--llenado", type=click.Choice(["apertura", "cierre"]), default="cierre", show_default=True, help="Fill price: opening or closing price of the candle.")
def cuenta_chica_command(strategy_name, hipotesis, fixture_name, snapshot_name, macro_snapshot, symbol, from_date, to_date, capitales,
                         commission_pct, costos, spread_bps, costo_entrada_usd, retencion_dividendos,
                         costo_fiscal_pct, tamano_posicion, regla_tamano, orden_minima, riesgo_por_trade,
                         params, intrabar_stops, llenado):
    """Run one card per initial capital and compare it with cost-matched SPY."""
    try:
        amounts = [float(value) for value in capitales.split(",")]
        if not amounts or any(value <= 0 for value in amounts):
            raise ValueError
    except ValueError as exc:
        raise click.BadParameter("Use comma-separated positive USD amounts.", param_hint="--capitales") from exc
    # Portfolio hypotheses use the complete panel and the paper portfolio's weight/rebalance rules.
    hipotesis_path = hipotesis
    if macro_snapshot and not hipotesis_path:
        raise click.UsageError("--macro-snapshot requiere --hipotesis.")
    if hipotesis_path:
        if strategy_name or fixture_name or symbol:
            raise click.UsageError("--hipotesis es mutuamente excluyente con --strategy, --fixture y --symbol.")
        if not snapshot_name:
            raise click.UsageError("--hipotesis requiere --snapshot.")
        if to_date and to_date.date() >= datetime(2019, 1, 1).date():
            raise click.ClickException("cuenta-chica con hipótesis no admite --to >= 2019-01-01.")
        from quantagent.fabrica.hipotesis import cargar_hipotesis
        from quantagent.backtesting.cartera_cuenta_chica import pesos_operables_por_fecha, simular_cartera_cuenta_chica
        try:
            h = cargar_hipotesis(hipotesis_path)
            requiere_macro = any(p.clase == "curva-pendiente" for p in h.piernas)
            if macro_snapshot and not requiere_macro:
                raise click.UsageError("--macro-snapshot solo se admite con una hipótesis que incluya curva-pendiente.")
            if requiere_macro and not macro_snapshot:
                raise click.ClickException("La hipótesis incluye curva-pendiente; indicá --macro-snapshot <nombre>.")
            symbols = sorted({a for p in h.piernas for a in p.activos})
            frames = {s: read_snapshot(snapshot_name, s) for s in symbols}
            panel = pd.DataFrame({s: f["close"] for s, f in frames.items()})
            timezone = panel.index.tz
            to_bound = pd.Timestamp(to_date) if to_date else pd.Timestamp("2018-12-31")
            from_bound = pd.Timestamp(from_date) if from_date else None
            if timezone is not None:
                to_bound = to_bound.tz_localize(timezone)
                if from_bound is not None:
                    from_bound = from_bound.tz_localize(timezone)
            panel = panel.loc[:to_bound]
            if from_date:
                panel = panel.loc[from_bound:]
            if panel.empty:
                raise click.ClickException("El snapshot no tiene cierres en el rango solicitado.")
            macro = None
            if macro_snapshot:
                from quantagent.data.macro import serie_punto_en_el_tiempo
                macro = serie_punto_en_el_tiempo(macro_snapshot, "T10Y2Y", panel.index)
        except (ValueError, SnapshotError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc)) from exc
        base_slippage = settings.TRADING_SLIPPAGE_PCT
        commission = settings.TRADING_COMMISSION_PCT if commission_pct is None else commission_pct
        if costos:
            base_slippage, commission = COSTOS_PERFILES[costos]["slippage_pct"], COSTOS_PERFILES[costos]["commission_pct"]
        elif commission_pct is not None:
            commission = commission_pct
        slippage = {
            s: base_slippage + (spread_bps if spread_bps is not None else settings.SPREAD_BPS_POR_CLASE[get_asset_type(s).value]) / 10000
            for s in symbols
        }
        if costos and commission_pct is not None:
            raise click.UsageError("--costos y --commission-pct son excluyentes: elegí uno.")
        pesos_fecha = pesos_operables_por_fecha(h, panel, macro)
        withholding = (0.30 if retencion_dividendos is None else retencion_dividendos) + costo_fiscal_pct
        div = {}
        for s, f in frames.items():
            if "close_sin_dividendos" not in f:
                continue
            rates = f["close"].pct_change() - f["close_sin_dividendos"].pct_change()
            div[s] = {ts.date().isoformat(): float(v) for ts, v in rates.items() if pd.notna(v)}
        timestamps = list(panel.index.to_pydatetime())
        if "SPY" not in panel.columns:
            raise click.ClickException("La hipótesis debe incluir SPY para calcular el benchmark.")
        spy = "SPY"
        benchmark_prices = panel[spy].tolist()
        for capital in map(float, capitales.split(",")):
            result = simular_cartera_cuenta_chica(h, panel, capital, slippage, commission,
                                                   costo_entrada_usd or 0, div, withholding, macro=macro,
                                                   pesos_fecha=pesos_fecha)
            bench = buy_and_hold_cuenta_chica(timestamps, benchmark_prices, capital, slippage[spy], commission,
                                               costo_entrada_usd=costo_entrada_usd or 0,
                                               dividend_rates=div.get(spy, {}), withholding=withholding)
            click.echo(f"Capital: USD {capital:,.0f}")
            click.echo(f"  Sharpe: {result['sharpe']:.2f} | Retorno: {result['return']:.2%}")
            click.echo(f"  Costos pagados: USD {result['costs']:.2f} | Posiciones en cero: {', '.join(result['zero_positions']) or 'ninguna'}")
            click.echo(f"  Posición más concentrada: {result['concentration']:.2%}")
            click.echo(f"  SPY comprar y mantener: retorno {bench['return']:.2%}, Sharpe {bench['sharpe']:.2f}; diferencia: {result['return']-bench['return']:+.2%}")
        return
    if not strategy_name:
        raise click.UsageError("Indicá --strategy o --hipotesis.")
    meta, symbol, to_date = _resolve_source_and_meta("cuenta-chica", strategy_name, fixture_name, snapshot_name,
                                                      symbol, from_date, to_date, False)
    for capital in amounts:
        config = _engine_config(commission_pct, costos, tamano_posicion, regla_tamano, orden_minima,
                                riesgo_por_trade, capital, retencion_dividendos, costo_fiscal_pct,
                                spread_bps, costo_entrada_usd)
        r = correr_par(strategy_name, meta, config, costos, fixture_name, snapshot_name, symbol,
                       from_date, to_date, intrabar_stops, con_filas=True,
                       params=params.get(strategy_name), fill=llenado)
        prices = [close for _, close in r["cierres"]]
        timestamps = [ts for ts, _ in r["cierres"]]
        slippage = r["slippage_pct"]
        commission = r["comision_pct"]
        withholding = (0.30 if retencion_dividendos is None else retencion_dividendos) + costo_fiscal_pct
        dividend_rates = {}
        if snapshot_name and withholding:
            frame = read_snapshot(snapshot_name, symbol)
            rates = frame["close"].pct_change() - frame["close_sin_dividendos"].pct_change()
            dividend_rates = {ts.date().isoformat(): float(rate) for ts, rate in rates.items() if abs(rate) >= 1e-9}
        benchmark = buy_and_hold_cuenta_chica(
            timestamps, prices, capital, slippage, commission,
            costo_entrada_usd=costo_entrada_usd or 0.0,
            dividend_rates=dividend_rates, withholding=withholding,
        )
        strategy_taxes = sum(
            abs(point["positions_value"]) * dividend_rates.get(point["date"].date().isoformat(), 0.0) * withholding
            for point in r["equity_curve"]
        )
        paid = (costo_entrada_usd or 0.0) + strategy_taxes + sum(
            (row.qty * row.entry_price + (row.qty * row.exit_price if row.exit_price else 0)) * (slippage + commission)
            for row in r["filas"]
        )
        curve = r["equity_curve"]
        strategy_return = curve[-1]["equity"] / capital - 1 if curve else 0
        concentration = max((abs(point["positions_value"]) / point["equity"] for point in curve if point["equity"]), default=0)
        zeros = sorted({position["symbol"] for position in r["posiciones_en_cero"]})
        click.echo(f"Capital: USD {capital:,.0f}")
        click.echo(f"  Sharpe: {r['sharpe']:.2f} | Retorno: {strategy_return:.2%}")
        click.echo(f"  Costos pagados: USD {paid:.2f} | Posiciones en cero: {', '.join(zeros) if zeros else 'ninguna'}")
        click.echo(f"  Posición más concentrada: {concentration:.2%}")
        click.echo(f"  SPY comprar y mantener: retorno {benchmark['return']:.2%}, Sharpe {benchmark['sharpe']:.2f}; diferencia: {strategy_return-benchmark['return']:+.2%}")


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
    help="Fixture name under tests/fixtures/ (without the .csv extension). Mutually exclusive with --snapshot.",
)
@click.option("--snapshot", "snapshot_name", help="Snapshot in $QUANTAGENT_SNAPSHOT_DIR (adjusted bars). Requires --symbol.")
@click.option("--symbol", "symbol", help="Snapshot symbol (one per run).")
@click.option("--from", "from_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="First session, YYYY-MM-DD (default: first in the snapshot).")
@click.option("--to", "to_date", type=click.DateTime(formats=["%Y-%m-%d"]), help="Last session, YYYY-MM-DD (default: last in the snapshot).")
@click.option(
    "--abrir-reserva",
    "abrir_reserva",
    is_flag=True,
    help="Allow --to to reach the reserve period (from $QUANTAGENT_RESERVA_DESDE, default 2023-01-01). Logged.",
)
@click.option(
    "--abrir-prueba",
    "abrir_prueba",
    is_flag=True,
    help="Allow --to to reach the test period (from $QUANTAGENT_PRUEBA_DESDE, default 2019-01-01). Logged.",
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
    help="Commission per side on notional (0.001 = 0.10%). Default: TRADING_COMMISSION_PCT.",
)
@click.option(
    "--costos",
    type=click.Choice(sorted(COSTOS_PERFILES)),
    default=None,
    help="Cost profile (slippage and commission per side). Mutually exclusive with --commission-pct.",
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
    riesgo_por_trade: Optional[float],
) -> None:
    import difflib

    config = _engine_config(commission_pct, costos, tamano_posicion, regla_tamano, orden_minima, riesgo_por_trade)
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
@click.option("--strategies", required=True, help="Comma-separated strategies (e.g. rsi,sma-cross).")
@click.option("--symbols", required=True, help="Comma-separated symbols (e.g. SPY,IWM).")
@click.option("--snapshot", "snapshot_name", required=True, help="Snapshot in $QUANTAGENT_SNAPSHOT_DIR (adjusted bars).")
@click.option("--from", "from_date", type=click.DateTime(formats=["%Y-%m-%d"]), default=None, help="First session.")
@click.option("--to", "to_date", type=click.DateTime(formats=["%Y-%m-%d"]), default=None, help="Last session.")
@click.option("--costos", type=click.Choice(sorted(COSTOS_PERFILES)), default=None, help="Cost profile.")
@click.option("--tamano-posicion", "tamano_posicion", type=click.FloatRange(min=0.0, max=1.0, min_open=True), default=1.0,
              help="Capital fraction per position, in (0, 1]. Sets the run's base and maximum size.")
@click.option("--regla-tamano", "regla_tamano", type=click.Choice(PositionSizer.REGLAS), default=None,
              help="Position sizing rule: confidence or fixed.")
@click.option("--orden-minima", "orden_minima", type=click.FloatRange(min=0.0, min_open=True), default=None, help="Minimum USD amount per entry order.")
@click.option("--abrir-reserva", "abrir_reserva", is_flag=True, help="Allow access to the reserve period.")
@click.option("--abrir-prueba", "abrir_prueba", is_flag=True, help="Allow access to the test period.")
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, writable=True), help="Output CSV file.")
@click.option("--trades-dir", "trades_dir", type=click.Path(file_okay=False, writable=True), default=None,
              help="Directory for per-strategy trade CSV files.")
@click.option("--equity-dir", type=click.Path(file_okay=False, writable=True), default=None, help="Engine date,equity CSV per bar (equity rounded to 2 decimals).")
@click.option("--md", "md_path", type=click.Path(dir_okay=False, writable=True), default=None, help="Output Markdown document.")
@click.option("--crisis", is_flag=True, help="Add crisis_2008/2020/2022: annual return minus buy and hold (informational).")
@click.option("--llenado", type=click.Choice(["apertura", "cierre"]), default="apertura", show_default=True,
              help="Fill price: opening or closing price of the candle.")
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

