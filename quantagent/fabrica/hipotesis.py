"""Hipótesis de la fábrica de estrategias: un archivo YAML por hipótesis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_TEXTOS = ("id", "autor", "justificacion", "timeframe")


@dataclass(frozen=True)
class Pierna:
    clase: str
    parametros: dict[str, list[Any]]
    activos: tuple[str, ...]
    universo: str
    elegibilidad: str | None = None

    def combinaciones(self) -> list[dict[str, Any]]:
        # Import diferido: hipotesis -> cli.sweep -> ... -> trading.cartera_paper -> hipotesis es un ciclo;
        # a nivel de módulo, importar hipotesis (p. ej. desde cartera_paper) falla con ImportError parcial.
        from quantagent.cli.sweep import validar_grilla

        from quantagent.fabrica.transversal import es_transversal

        if not self.parametros and es_transversal(self.clase):
            return [{}]  # métrica transversal: sin parámetros, una única combinación vacía
        return validar_grilla(self.parametros, self.clase)


@dataclass(frozen=True, init=False)
class Hipotesis:
    id: str
    autor: str
    justificacion: str
    timeframe: str
    piernas: tuple[Pierna, ...]
    fuente: str | None = None

    def __init__(
        self,
        id: str,
        autor: str,
        justificacion: str,
        timeframe: str = "1d",
        piernas: tuple[Pierna, ...] | None = None,
        fuente: str | None = None,
        clase: str | None = None,
        parametros: dict[str, list[Any]] | None = None,
        activos: tuple[str, ...] | list[str] | None = None,
        universo: str = "",
        **kwargs: Any,
    ):
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "autor", autor)
        object.__setattr__(self, "justificacion", justificacion)
        object.__setattr__(self, "timeframe", kwargs.get("timeframe", timeframe))
        object.__setattr__(self, "fuente", kwargs.get("fuente", fuente))
        if piernas is not None:
            object.__setattr__(self, "piernas", tuple(piernas))
        elif clase is not None:
            p = Pierna(
                clase=clase,
                parametros=parametros or {},
                activos=tuple(activos or ()),
                universo=universo,
            )
            object.__setattr__(self, "piernas", (p,))
        else:
            object.__setattr__(self, "piernas", ())

    @property
    def clase(self) -> str:
        return self.piernas[0].clase

    @property
    def parametros(self) -> dict[str, list[Any]]:
        return self.piernas[0].parametros

    @property
    def activos(self) -> tuple[str, ...]:
        return self.piernas[0].activos

    @property
    def universo(self) -> str:
        return self.piernas[0].universo

    @property
    def es_cartera(self) -> bool:
        return len(self.piernas) > 1 or self.id.startswith("cartera-")

    def combinaciones(self) -> list[dict[str, Any]]:
        return self.piernas[0].combinaciones()


def _validar_pierna(d: Any) -> Pierna:
    if not isinstance(d, dict):
        raise ValueError("Cada pierna debe ser un mapa")
    for campo in ("clase", "parametros", "activos", "universo"):
        if d.get(campo) is None:
            raise ValueError(f"Falta el campo '{campo}'")
    for campo in ("clase", "universo"):
        if not isinstance(d[campo], str) or not d[campo].strip():
            raise ValueError(f"Campo '{campo}' debe ser un texto no vacío")
    activos = d["activos"]
    if not isinstance(activos, list) or not activos or not all(isinstance(a, str) for a in activos):
        raise ValueError("Campo 'activos' debe ser una lista no vacía de símbolos")
    if d.get("elegibilidad") is not None and not isinstance(d["elegibilidad"], str):
        raise ValueError("Campo 'elegibilidad' debe ser un texto")
    p = Pierna(clase=d["clase"], parametros=d["parametros"], activos=tuple(activos), universo=d["universo"],
               elegibilidad=d.get("elegibilidad"))
    p.combinaciones()
    return p


def cargar_hipotesis(path: str | Path) -> Hipotesis:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("La hipótesis debe ser un mapa YAML")
    for campo in _TEXTOS:
        if data.get(campo) is None:
            raise ValueError(f"Falta el campo '{campo}'")
        if not isinstance(data[campo], str) or not data[campo].strip():
            raise ValueError(f"Campo '{campo}' debe ser un texto no vacío")
    if not re.fullmatch(r"[a-z0-9-]+", data["id"]):
        raise ValueError(f"Campo 'id' inválido '{data['id']}': debe cumplir [a-z0-9-]+")
    if not isinstance(data.get("fuente", ""), str):
        raise ValueError("Campo 'fuente' debe ser un texto")
    if "piernas" in data:
        raw_piernas = data["piernas"]
        if not isinstance(raw_piernas, list) or not raw_piernas:
            raise ValueError("Campo 'piernas' debe ser una lista no vacía")
        piernas = [_validar_pierna(p) for p in raw_piernas]
    else:
        piernas = [_validar_pierna(data)]
    return Hipotesis(
        id=data["id"],
        autor=data["autor"],
        justificacion=data["justificacion"],
        timeframe=data["timeframe"],
        piernas=tuple(piernas),
        fuente=data.get("fuente"),
    )
