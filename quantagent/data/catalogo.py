"""Carga del catálogo y cobertura de las carpetas locales, sin leer los datos."""

from fnmatch import fnmatchcase
from pathlib import Path

import yaml


def cargar_catalogo(directorio: Path) -> tuple[list[dict], list[dict]]:
    """Devuelve datasets y exclusiones desde los YAML del directorio."""
    datasets = []
    for archivo in sorted(Path(directorio).glob("*.yaml")):
        if archivo.name != "_no-son-datos.yaml":
            datasets.append(yaml.safe_load(archivo.read_text(encoding="utf-8")))
    exclusiones = yaml.safe_load(
        (Path(directorio) / "_no-son-datos.yaml").read_text(encoding="utf-8")
    )
    return datasets, exclusiones


def verificar_cobertura(datasets: list[dict], exclusiones: list[dict], raiz: Path) -> list[str]:
    """Lista carpetas huérfanas, patrones sin carpeta y fuentes que impiden borrado aislado.

    Los patrones son nombres de carpetas directamente bajo la raíz. El llamador
    decide si omite el chequeo cuando no hay datos locales disponibles.
    """
    carpetas = sorted(p.name for p in Path(raiz).iterdir() if p.is_dir())
    errores = []
    for nombre in carpetas:
        propietarios = [d for d in datasets if any(fnmatchcase(nombre, p) for p in d["carpeta"])]
        if not propietarios and not any(fnmatchcase(nombre, e["entrada"]) for e in exclusiones):
            errores.append(f"carpeta sin catálogo: {nombre}")
        if any(d["fuente"]["regla_de_borrado"] for d in propietarios):
            if len({d["fuente"]["nombre"] for d in propietarios}) > 1:
                errores.append(f"carpeta compartida entre fuentes con regla de borrado: {nombre}")
    for dataset in datasets:
        if dataset["datos_esperados"]:
            for patron in dataset["carpeta"]:
                if not any(fnmatchcase(nombre, patron) for nombre in carpetas):
                    errores.append(f"{dataset['id']}: patrón sin carpeta: {patron}")
    return errores
