"""Snapshots macro congelados: un Parquet por serie de ALFRED (todas las versiones) y un manifiesto con hashes.

Mismo patrón que `snapshot.py`: `<snapshot_root>/<nombre>/<serie>.parquet` y `manifest.json`.
Un snapshot macro se congela: no se vuelve a bajar con el mismo nombre (vj37).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

from quantagent.data.snapshot import MANIFEST_NAME, VERSIONED_DIR, SnapshotError, _sha256, load_manifest, snapshot_root

PRIMERA_FECHA = "1776-07-04"
ULTIMA_FECHA = "9999-12-31"


def crear_snapshot_macro(nombre: str, series: Iterable[str], cliente, base_dir: Optional[Path] = None) -> dict:
    """Baja todas las versiones de cada serie con `cliente.observaciones` y escribe parquet + manifiesto."""
    series = sorted(set(series))
    if not series:
        raise SnapshotError("no hay series para guardar")
    carpeta = snapshot_root(base_dir) / nombre
    if carpeta.exists():
        raise SnapshotError(f"el snapshot '{nombre}' ya existe: {carpeta}")
    frames = {s: cliente.observaciones(s, PRIMERA_FECHA, ULTIMA_FECHA, todas_las_versiones=True) for s in series}
    vacias = [s for s, df in frames.items() if df.empty]
    if vacias:
        raise SnapshotError(f"sin filas: {', '.join(vacias)}")

    return _escribir(carpeta, nombre, frames)


def _escribir(carpeta: Path, nombre: str, frames: dict) -> dict:
    carpeta.mkdir(parents=True, exist_ok=True)
    info = {}
    for serie, df in frames.items():
        ruta = carpeta / f"{serie}.parquet"
        df.to_parquet(ruta, index=False)
        info[serie] = {
            "file": ruta.name,
            "rows": len(df),
            "start": df["fecha"].min().date().isoformat() if len(df) else None,
            "end": df["fecha"].max().date().isoformat() if len(df) else None,
            "sha256": _sha256(ruta),
        }
    manifiesto = {
        "name": nombre,
        "source": "alfred",
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "series": info,
    }
    (carpeta / MANIFEST_NAME).write_text(json.dumps(manifiesto, indent=2, sort_keys=True) + "\n")
    return manifiesto


def recortar_snapshot_macro(nombre: str, al, destino: Path, base_dir: Optional[Path] = None) -> dict:
    """Escribe en `destino/nombre` el snapshot como estaba publicado en `al` (QuantAgent-d4cq.71).

    Descarta las versiones con `realtime_start > al` y acota `realtime_end` a `al`: nada posterior a `al`
    queda visible, ni por `valor_conocido` ni leyendo la última versión a mano.
    """
    al = pd.Timestamp(al)
    manifiesto = load_manifest(nombre, base_dir)
    carpeta = snapshot_root(base_dir) / nombre
    frames = {}
    for serie, info in manifiesto["series"].items():
        df = pd.read_parquet(carpeta / info["file"])
        df = df[df["realtime_start"] <= al].copy()
        df["realtime_end"] = df["realtime_end"].clip(upper=al)
        frames[serie] = df
    return _escribir(Path(destino) / nombre, nombre, frames)


def verificar_snapshot_macro(nombre: str, base_dir: Optional[Path] = None, versioned_dir: Optional[Path] = None) -> dict:
    """Recalcula el sha256 de cada serie contra `manifest.json` y, si existe, contra el manifiesto versionado.

    Si algo no coincide, `SnapshotError` nombra cada serie.
    """
    manifiesto = load_manifest(nombre, base_dir)
    carpeta = snapshot_root(base_dir) / nombre
    problemas = []
    for serie, info in manifiesto["series"].items():
        ruta = carpeta / info["file"]
        if not ruta.exists():
            problemas.append(f"{serie}: falta el archivo {info['file']}")
        elif _sha256(ruta) != info["sha256"]:
            problemas.append(f"{serie}: sha256 distinto del manifiesto")

    versionado = (VERSIONED_DIR if versioned_dir is None else Path(versioned_dir)) / f"{nombre}.json"
    if versionado.exists():
        esperado = json.loads(versionado.read_text())["series"]
        for serie in sorted(set(esperado) | set(manifiesto["series"])):
            if esperado.get(serie) != manifiesto["series"].get(serie):
                problemas.append(f"{serie}: no coincide con el manifiesto versionado {versionado.name}")
    if problemas:
        raise SnapshotError("; ".join(problemas))
    return manifiesto


def _versiones(nombre: str, serie: str, base_dir: Optional[Path]) -> pd.DataFrame:
    manifiesto = load_manifest(nombre, base_dir)
    if serie not in manifiesto["series"]:
        raise SnapshotError(f"{serie} no está en el snapshot macro '{nombre}'")
    df = pd.read_parquet(snapshot_root(base_dir) / nombre / manifiesto["series"][serie]["file"])
    return df.dropna(subset=["valor"]).sort_values("fecha")


def _conocido(df: pd.DataFrame, al: pd.Timestamp) -> Optional[Tuple[pd.Timestamp, float]]:
    vigentes = df[(df["realtime_start"] <= al) & (al <= df["realtime_end"])]
    if vigentes.empty:
        return None
    ultima = vigentes.iloc[-1]
    return ultima["fecha"], float(ultima["valor"])


def valor_conocido(nombre: str, serie: str, al, base_dir: Optional[Path] = None) -> Optional[Tuple[pd.Timestamp, float]]:
    """Observación más reciente cuya versión estaba vigente en `al` (`realtime_start <= al <= realtime_end`, ambos extremos inclusivos como en FRED), o None.

    Una versión `.` (NaN) no cuenta como valor conocido.
    """
    return _conocido(_versiones(nombre, serie, base_dir), pd.Timestamp(al))


def serie_punto_en_el_tiempo(nombre: str, serie: str, sesiones, base_dir: Optional[Path] = None) -> pd.Series:
    """Para cada sesión, el último valor conocido ese día; NaN si todavía no se conocía ninguno."""
    df = _versiones(nombre, serie, base_dir)
    sesiones = pd.DatetimeIndex(sesiones)
    valores = []
    for dia in sesiones:
        conocido = _conocido(df, dia)
        valores.append(float("nan") if conocido is None else conocido[1])
    return pd.Series(valores, index=sesiones, name=serie)
