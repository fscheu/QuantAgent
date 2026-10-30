"""Funciones puras para proponer y ajustar stops sin aflojar el riesgo."""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Sequence, TypeVar

from ..models import OrderSide

logger = logging.getLogger(__name__)
Precio = TypeVar("Precio", int, float, Decimal)


@dataclass(frozen=True)
class StopDecision:
    stop: Precio
    rechazado: bool


def nuevo_stop(actual: Precio, propuesto: Precio, lado: OrderSide) -> StopDecision:
    """Devuelve el stop que conserva o reduce el riesgo de la posición."""
    propuesta = Decimal(str(propuesto)) if isinstance(actual, Decimal) else propuesto
    if lado == OrderSide.BUY:
        rechazado = propuesta < actual
        stop = max(actual, propuesta)
    else:
        rechazado = propuesta > actual
        stop = min(actual, propuesta)
    if rechazado:
        logger.warning("Propuesta de stop rechazada por aflojar el stop actual")
    return StopDecision(stop, rechazado)


def propuesta_breakeven(entry: Precio, stop_inicial: Precio, precio: Precio,
                        lado: OrderSide, umbral_r: float = 1.5) -> Optional[Precio]:
    """Propone breakeven cuando el avance alcanza el umbral expresado en R."""
    riesgo = abs(entry - stop_inicial)
    avance = precio - entry if lado == OrderSide.BUY else entry - precio
    return entry if riesgo and avance >= riesgo * umbral_r else None


def propuesta_chandelier(maximo_desde_entrada: Precio, atr: Precio,
                          lado: OrderSide, mult: float = 3.0) -> Precio:
    """Propone Chandelier desde el extremo favorable y el ATR."""
    distancia = atr * mult
    return maximo_desde_entrada - distancia if lado == OrderSide.BUY else maximo_desde_entrada + distancia


def propuesta_invalidacion(ultimas_2: Sequence[Precio], lado: OrderSide) -> Precio:
    """Propone el mínimo (largo) o máximo (corto) de las dos últimas velas."""
    return min(ultimas_2) if lado == OrderSide.BUY else max(ultimas_2)
