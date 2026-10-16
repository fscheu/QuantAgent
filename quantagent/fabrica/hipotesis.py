"""Hipótesis de la fábrica de estrategias: un archivo YAML por hipótesis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from quantagent.cli.sweep import validar_grilla

_TEXTOS = ("id", "autor", "justificacion", "clase", "timeframe")


@dataclass(frozen=True)
class Hipotesis:
    id: str
    autor: str
    justificacion: str
    clase: str
    parametros: dict[str, list[Any]]
    activos: tuple[str, ...]
    timeframe: str
    fuente: str | None = None

    def combinaciones(self) -> list[dict[str, Any]]:
        return validar_grilla(self.parametros, self.clase)


def cargar_hipotesis(path: str | Path) -> Hipotesis:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("La hipótesis debe ser un mapa YAML")
    for campo in (*_TEXTOS, "parametros", "activos"):
        if data.get(campo) is None:
            raise ValueError(f"Falta el campo '{campo}'")
    for campo in _TEXTOS:
        if not isinstance(data[campo], str) or not data[campo].strip():
            raise ValueError(f"Campo '{campo}' debe ser un texto no vacío")
    if not re.fullmatch(r"[a-z0-9-]+", data["id"]):
        raise ValueError(f"Campo 'id' inválido '{data['id']}': debe cumplir [a-z0-9-]+")
    activos = data["activos"]
    if not isinstance(activos, list) or not activos or not all(isinstance(a, str) for a in activos):
        raise ValueError("Campo 'activos' debe ser una lista no vacía de símbolos")
    if not isinstance(data.get("fuente", ""), str):
        raise ValueError("Campo 'fuente' debe ser un texto")
    h = Hipotesis(**{c: data[c] for c in (*_TEXTOS, "parametros")}, activos=tuple(activos), fuente=data.get("fuente"))
    h.combinaciones()  # valida la grilla; propaga el error de validar_grilla
    return h
