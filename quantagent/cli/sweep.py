"""Carga y validación de grillas para backtest sweep (V31a)."""

import csv
import itertools
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import click
from tabulate import tabulate

from quantagent.settings import COSTOS_PERFILES
from quantagent.strategy.registry import STRATEGY_REGISTRY, cli_aliases
from quantagent.trading.position_sizer import PositionSizer

MAX_SWEEP_COMBINATIONS = 50


def validar_grilla(data: Any, strategy: str) -> list[dict[str, Any]]:
    """Valida un objeto de grilla y devuelve combinaciones con claves alfabéticas."""
    if not isinstance(data, dict) or not data:
        raise click.ClickException("La grilla debe ser un objeto JSON no vacío (parámetro: lista de valores)")
    for param, values in data.items():
        if not isinstance(values, list) or len(values) == 0:
            raise click.ClickException(f"El parámetro '{param}' debe ser una lista no vacía")
        if len(values) != len(set(values)):
            raise click.ClickException(f"El parámetro '{param}' contiene valores duplicados")

    aliases = cli_aliases()
    reg_name = aliases.get(strategy) or (strategy if strategy in STRATEGY_REGISTRY else None)
    if not reg_name:
        raise click.ClickException(f"Estrategia desconocida '{strategy}'. Opciones: {', '.join(sorted(aliases))}")

    allowed = STRATEGY_REGISTRY[reg_name]["params"]
    for param in data:
        if param not in allowed:
            raise click.ClickException(
                f"Parámetro desconocido '{param}' para la estrategia '{strategy}'. Opciones válidas: {', '.join(sorted(allowed))}"
            )

    total = 1
    for v in data.values():
        total *= len(v)
    if total > MAX_SWEEP_COMBINATIONS:
        raise click.ClickException(
            f"La grilla genera {total} combinaciones, superando el límite máximo de {MAX_SWEEP_COMBINATIONS}"
        )

    sorted_params = sorted(data.keys())
    return [dict(zip(sorted_params, p)) for p in itertools.product(*(data[k] for k in sorted_params))]


def cargar_grilla(source: str | Path | dict[str, Any], strategy: str) -> list[dict[str, Any]]:
    """Carga una grilla desde archivo, JSON string o dict, y valida sus combinaciones."""
    if isinstance(source, dict):
        return validar_grilla(source, strategy)
    if isinstance(source, (str, Path)) and Path(source).exists():
        try:
            with open(source, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"JSON inválido en grilla '{source}': {e}")
        return validar_grilla(data, strategy)
    if isinstance(source, str) and source.strip().startswith("{"):
        try:
            data = json.loads(source)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"JSON inválido en grilla: {e}")
        return validar_grilla(data, strategy)
    raise click.ClickException(f"No se encontró el archivo de grilla: {source}")


load_grid = cargar_grilla
validate_grid = validar_grilla


def grilla_vecinos(strategy: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Default y vecinos de ±20%, variando un parámetro por vez."""
    name = cli_aliases().get(strategy, strategy)
    if name not in STRATEGY_REGISTRY:
        raise click.ClickException(f"Estrategia desconocida '{strategy}'")
    defaults = {k: v["default"] for k, v in STRATEGY_REGISTRY[name]["params"].items()}
    combos, labels = [defaults], ["default"]
    for param, value in sorted(defaults.items()):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value == 0:
            continue
        seen = {value}
        for factor in (0.8, 1.2):
            vecino = round(value * factor) if isinstance(value, int) else value * factor
            if vecino not in seen:
                combos.append({**defaults, param: vecino})
                labels.append(f"{param}x{factor}")
                seen.add(vecino)
    if len(combos) > MAX_SWEEP_COMBINATIONS:
        raise click.ClickException(f"La grilla supera el límite máximo de {MAX_SWEEP_COMBINATIONS}")
    return combos, labels


CAMPOS_INTENTOS = [
    "fecha_utc", "estrategia", "activo", "snapshot", "desde", "hasta", "costos",
    "tamano", "params", "trades", "pnl", "sharpe", "hipotesis", "etapa",
]


def contar_intentos(hipotesis: Optional[str] = None) -> int:
    """Cuenta filas de intentos.csv; con `hipotesis`, solo las de esa hipótesis."""
    from quantagent.data.snapshot import snapshot_root

    path = snapshot_root() / "intentos.csv"
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for r in csv.DictReader(f) if hipotesis is None or r.get("hipotesis") == hipotesis)


def registrar_intento(
    strategy_name: str,
    symbol: str,
    snapshot_name: str,
    costos: Optional[str],
    tamano_posicion: Optional[float],
    combo: dict[str, Any],
    row: dict[str, Any],
    hipotesis: Optional[str] = None,
    etapa: Optional[str] = None,
) -> None:
    """Agrega una corrida al registro, conservando sus valores.

    Un archivo con encabezado viejo se reescribe con el nuevo (columnas faltantes vacías).
    """
    from quantagent.data.snapshot import snapshot_root

    intentos_path = snapshot_root() / "intentos.csv"
    intento = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "estrategia": strategy_name,
        "activo": symbol,
        "snapshot": snapshot_name,
        "desde": row["desde"],
        "hasta": row["hasta"],
        "costos": costos or "",
        "tamano": tamano_posicion,
        "params": json.dumps(combo, sort_keys=True),
        "trades": row["trades"],
        "pnl": row["pnl"],
        "sharpe": row["sharpe"],
        "hipotesis": hipotesis or "",
        "etapa": etapa or "",
    }
    previas = []
    if intentos_path.exists():
        with intentos_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == CAMPOS_INTENTOS:
                previas = None
            else:
                previas = list(reader)
    nuevo = not intentos_path.exists() or previas is not None
    with intentos_path.open("w" if nuevo else "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS_INTENTOS, restval="", extrasaction="ignore")
        if nuevo:
            writer.writeheader()
            writer.writerows(previas or [])
        writer.writerow(intento)


@click.command("attempts", help="Contar corridas y combinaciones registradas por sweep.")
@click.option("--strategy", "strategy_name", required=True)
@click.option("--symbol", default=None)
def attempts_command(strategy_name: str, symbol: Optional[str]) -> None:
    from quantagent.data.snapshot import snapshot_root

    aliases = cli_aliases()
    strategy = aliases.get(strategy_name, strategy_name)
    if strategy not in STRATEGY_REGISTRY:
        raise click.ClickException(f"Estrategia desconocida '{strategy_name}'")
    path = snapshot_root() / "intentos.csv"
    corridas = 0
    combinaciones = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not {"estrategia", "activo", "params"}.issubset(reader.fieldnames or []):
                raise click.ClickException("Encabezado inválido en intentos.csv")
            for row in reader:
                if aliases.get(row["estrategia"], row["estrategia"]) != strategy:
                    continue
                if symbol is not None and row["activo"] != symbol.upper():
                    continue
                try:
                    params = json.dumps(json.loads(row["params"]), sort_keys=True)
                except (ValueError, TypeError) as exc:
                    raise click.ClickException("params JSON inválido en intentos.csv") from exc
                corridas += 1
                combinaciones.add(params)
    click.echo(f"Corridas: {corridas}")
    click.echo(f"Combinaciones distintas: {len(combinaciones)}")


@click.command("sweep", help="Correr una grilla de parámetros de una estrategia sobre un activo.")
@click.option("--strategy", "strategy_name", required=True, help="Estrategia a optimizar.")
@click.option("--symbol", "symbol", required=True, help="Símbolo del activo (ej: SPY).")
@click.option("--snapshot", "snapshot_name", required=True, help="Snapshot en $QUANTAGENT_SNAPSHOT_DIR (velas ajustadas).")
@click.option("--grid", "grid_path", type=click.Path(dir_okay=False), help="Archivo JSON con la grilla de parámetros.")
@click.option("--vecinos-20", is_flag=True, help="Default y ±20% por parámetro, uno por vez; excluye --grid.")
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, writable=True), help="CSV de salida.")
@click.option("--from", "from_date", type=click.DateTime(formats=["%Y-%m-%d"]), default=None, help="Primera sesión.")
@click.option("--to", "to_date", type=click.DateTime(formats=["%Y-%m-%d"]), default=None, help="Última sesión.")
@click.option("--costos", type=click.Choice(sorted(COSTOS_PERFILES)), default=None, help="Perfil de costos.")
@click.option("--tamano-posicion", "tamano_posicion", type=click.FloatRange(min=0.0, max=1.0, min_open=True), default=1.0)
@click.option("--regla-tamano", "regla_tamano", type=click.Choice(PositionSizer.REGLAS), default=None)
@click.option("--orden-minima", "orden_minima", type=click.FloatRange(min=0.0, min_open=True), default=None, help="Monto mínimo en USD por orden de entrada.")
def sweep_command(
    strategy_name: str,
    symbol: str,
    snapshot_name: str,
    grid_path: Optional[str],
    vecinos_20: bool,
    out_path: str,
    from_date: Optional[datetime],
    to_date: Optional[datetime],
    costos: Optional[str],
    tamano_posicion: Optional[float],
    regla_tamano: Optional[str],
    orden_minima: Optional[float],
) -> None:
    from quantagent.backtesting.fixtures import snapshot_metadata
    from quantagent.cli.backtest import (
        COMPARE_COLUMNS,
        _engine_config,
        _isolated_sqlite_db,
        _prueba_desde,
        _reserva_desde,
        correr_par,
    )
    from quantagent.data.snapshot import SnapshotError

    if vecinos_20 == (grid_path is not None):
        raise click.ClickException("Elegí exactamente una opción: --grid o --vecinos-20")
    combos, vecinos = grilla_vecinos(strategy_name) if vecinos_20 else (cargar_grilla(grid_path, strategy_name), [])

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
        raise click.ClickException(
            f"--to {to_date:%Y-%m-%d} cae en la reserva, que empieza el {reserva:%Y-%m-%d}: "
            "sweep no abre la prueba ni la reserva (usar compare con la opción)."
        )
    elif to_date >= prueba:
        raise click.ClickException(
            f"--to {to_date:%Y-%m-%d} cae en el tramo de prueba, que empieza el {prueba:%Y-%m-%d}: "
            "sweep no abre la prueba ni la reserva (usar compare con la opción)."
        )

    sym = symbol.upper()
    try:
        meta = snapshot_metadata(snapshot_name, sym, from_date, to_date)
    except (SnapshotError, ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    config = _engine_config(None, costos, tamano_posicion, regla_tamano, orden_minima)

    param_cols = sorted(combos[0].keys())
    metric_cols = [c for c in COMPARE_COLUMNS if c not in ("estrategia", "activo")]
    fieldnames = (["vecino"] if vecinos_20 else []) + param_cols + metric_cols

    rows = []
    for combo in combos:
        with _isolated_sqlite_db():
            r = correr_par(
                strategy_name,
                meta,
                config,
                costos,
                None,
                snapshot_name,
                sym,
                from_date,
                to_date,
                con_filas=False,
                params=combo,
            )

        def _fmt(v: Optional[float], digits: int) -> object:
            return "" if v is None else round(v, digits)

        pf = "" if r["profit_factor"] is None or math.isinf(r["profit_factor"]) else round(r["profit_factor"], 2)
        row = {
            **combo,
            "desde": f"{meta.start_date:%Y-%m-%d}",
            "hasta": f"{meta.end_date:%Y-%m-%d}",
            "trades": r["trades"],
            "win_rate": round(r["win_rate"], 4),
            "profit_factor": pf,
            "sharpe": round(r["sharpe"], 2),
            "pnl": round(r["pnl"], 2),
            "max_drawdown": round(r["max_drawdown"], 6),
            "bh_pnl": round(r["bh_pnl"], 2),
            "bh_sharpe": round(r["bh_sharpe"], 2),
            "bh_max_drawdown": round(r["bh_max_drawdown"], 6),
            "retorno_anualizado": _fmt(r["retorno_anualizado"], 4),
            "exceso_sobre_referencia": _fmt(r["exceso_sobre_referencia"], 4),
            "tiempo_en_mercado": _fmt(r["tiempo_en_mercado"], 4),
            "capital_invertido_medio": _fmt(r["capital_invertido_medio"], 4),
            "sharpe_tasa_0": _fmt(r["sharpe_tasa_0"], 2),
        }
        rows.append(row)
        if vecinos_20:
            row["vecino"] = vecinos[len(rows) - 1]
        registrar_intento(strategy_name, sym, snapshot_name, costos, tamano_posicion, combo, row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table_data = []
    for r in rows:
        row_vals = []
        trades_count = int(r.get("trades") or 0)
        for c in fieldnames:
            val = r.get(c, "")
            if c == "profit_factor" and (val == "" or val is None) and trades_count > 0:
                val = "sin pérdidas"
            row_vals.append(val)
        table_data.append(row_vals)
    click.echo(tabulate(table_data, headers=fieldnames, tablefmt="github"))
