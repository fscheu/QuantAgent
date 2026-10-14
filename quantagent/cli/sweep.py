"""Carga y validación de grillas para backtest sweep (V31a)."""

import itertools
import json
from pathlib import Path
from typing import Any

import click

from quantagent.strategy.registry import STRATEGY_REGISTRY, cli_aliases

MAX_SWEEP_COMBINATIONS = 50


def validar_grilla(data: Any, strategy: str) -> list[dict[str, Any]]:
    """Valida un objeto de grilla y devuelve combinaciones con claves alfabéticas."""
    if not isinstance(data, dict) or not data:
        raise click.ClickException("La grilla debe ser un objeto JSON no vacío (parámetro: lista de valores)")
    for param, values in data.items():
        if not isinstance(values, list) or len(values) == 0:
            raise click.ClickException(f"El parámetro '{param}' debe ser una lista no vacía")

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
