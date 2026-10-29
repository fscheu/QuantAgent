"""Snapshots de composición histórica del S&P 500 y sector GICS punto en el tiempo (M5-04)."""

from __future__ import annotations

import functools
import io
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from quantagent.data.snapshot import MANIFEST_NAME, VERSIONED_DIR, SnapshotError, _sha256, load_manifest, snapshot_root

SOURCE_REPO = "fja05680/sp500"
SOURCE_SHA = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
HISTORICAL_CSV = "S&P 500 Historical Components & Changes (Updated).csv"
CURRENT_CSV = "sp500.csv"
TABLAS_DIR = Path(__file__).resolve().parent / "tablas"
SALIDOS_CSV = TABLAS_DIR / "gics_salidos.csv"
RECLASIFICACIONES_CSV = TABLAS_DIR / "gics_reclasificaciones.csv"


@functools.lru_cache(maxsize=16)
def _parquet_cacheado(ruta_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_parquet(ruta_str)


@functools.lru_cache(maxsize=8)
def _csv_cacheado(ruta_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(ruta_str, keep_default_na=False)


def _leer_csv_fuente(nombre_archivo: str, csv_dir: Optional[Path] = None) -> pd.DataFrame:
    if csv_dir is not None:
        return pd.read_csv(Path(csv_dir) / nombre_archivo)
    url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_SHA}/{urllib.parse.quote(nombre_archivo)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return pd.read_csv(io.StringIO(resp.read().decode("utf-8")))


def crear_snapshot_sp500(nombre: str, base_dir: Optional[Path] = None, csv_dir: Optional[Path] = None) -> dict:
    """Baja los dos CSV del commit fijado de fja05680/sp500 y los guarda como Parquet con manifest.json."""
    carpeta = snapshot_root(base_dir) / nombre
    if carpeta.exists():
        raise SnapshotError(f"el snapshot '{nombre}' ya existe: {carpeta}")

    df_hist = _leer_csv_fuente(HISTORICAL_CSV, csv_dir=csv_dir)
    df_actual = _leer_csv_fuente(CURRENT_CSV, csv_dir=csv_dir)
    if df_hist.empty or df_actual.empty:
        raise SnapshotError("los archivos de composición del S&P 500 no pueden estar vacíos")

    df_hist = df_hist.assign(date=df_hist["date"].astype(str).str.strip()).sort_values("date").reset_index(drop=True)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_hist, ruta_actual = carpeta / "composicion.parquet", carpeta / "sp500.parquet"
    df_hist.to_parquet(ruta_hist, index=False)
    df_actual.to_parquet(ruta_actual, index=False)

    primera, ultima = str(df_hist["date"].iloc[0]), str(df_hist["date"].iloc[-1])
    manifiesto = {
        "name": nombre,
        "source": SOURCE_REPO,
        "sha": SOURCE_SHA,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(df_hist),
        "start": primera,
        "end": ultima,
        "files": {
            "composicion": {
                "file": ruta_hist.name,
                "rows": len(df_hist),
                "start": primera,
                "end": ultima,
                "sha256": _sha256(ruta_hist),
            },
            "sp500": {"file": ruta_actual.name, "rows": len(df_actual), "sha256": _sha256(ruta_actual)},
        },
    }
    (carpeta / MANIFEST_NAME).write_text(json.dumps(manifiesto, indent=2, sort_keys=True) + "\n")
    return manifiesto


def verificar_snapshot_sp500(
    nombre: str, base_dir: Optional[Path] = None, versioned_dir: Optional[Path] = None
) -> dict:
    """Recalcula el sha256 de cada Parquet contra manifest.json y, si existe, contra el manifiesto versionado."""
    manifiesto = load_manifest(nombre, base_dir)
    carpeta = snapshot_root(base_dir) / nombre
    problemas = []
    for clave, info in manifiesto["files"].items():
        ruta = carpeta / info["file"]
        if not ruta.exists():
            problemas.append(f"{clave}: falta el archivo {info['file']}")
        elif _sha256(ruta) != info["sha256"]:
            problemas.append(f"{clave}: sha256 distinto del manifiesto")

    versionado = (VERSIONED_DIR if versioned_dir is None else Path(versioned_dir)) / f"{nombre}.json"
    if versionado.exists():
        esperado = json.loads(versionado.read_text())
        if esperado.get("sha") != manifiesto.get("sha"):
            problemas.append(f"sha del repo no coincide con {versionado.name}")
        archivos_esp = esperado.get("files", {})
        for clave in sorted(set(archivos_esp) | set(manifiesto["files"])):
            if archivos_esp.get(clave) != manifiesto["files"].get(clave):
                problemas.append(f"{clave}: no coincide con el manifiesto versionado {versionado.name}")
    if problemas:
        raise SnapshotError("; ".join(problemas))
    return manifiesto


def miembros(nombre: str, fecha, base_dir: Optional[Path] = None) -> list[str]:
    """Miembros del S&P 500 en la última fila con date <= fecha. Antes de la primera fecha lanza SnapshotError."""
    manifiesto = load_manifest(nombre, base_dir)
    ruta = snapshot_root(base_dir) / nombre / manifiesto["files"]["composicion"]["file"]
    df = _parquet_cacheado(str(ruta), ruta.stat().st_mtime_ns)
    fecha_str = pd.Timestamp(fecha).date().isoformat()
    previas = df[df["date"].astype(str) <= fecha_str]
    if previas.empty:
        raise SnapshotError(f"fecha {fecha_str} anterior a la primera fecha ({df['date'].iloc[0]})")
    return [t.strip() for t in str(previas.iloc[-1]["tickers"]).split(",") if t.strip()]


def sector(nombre: str, ticker: str, fecha, base_dir: Optional[Path] = None) -> Optional[str]:
    """Sector GICS vigente para ticker en fecha, partiendo del sector base y revirtiendo reclasificaciones posteriores."""
    manifiesto = load_manifest(nombre, base_dir)
    ruta = snapshot_root(base_dir) / nombre / manifiesto["files"]["sp500"]["file"]
    df_actual = _parquet_cacheado(str(ruta), ruta.stat().st_mtime_ns)
    t, fecha_str = ticker.strip().upper(), pd.Timestamp(fecha).date().isoformat()

    sec: Optional[str] = None
    actual_match = df_actual[df_actual["Symbol"].astype(str).str.strip().str.upper() == t]
    if not actual_match.empty:
        val = actual_match.iloc[0]["GICS Sector"]
        if pd.notna(val) and str(val).strip():
            sec = str(val).strip()
    elif SALIDOS_CSV.exists():
        df_salidos = _csv_cacheado(str(SALIDOS_CSV), SALIDOS_CSV.stat().st_mtime_ns)
        salido_match = df_salidos[df_salidos["ticker"].astype(str).str.strip().str.upper() == t]
        if not salido_match.empty:
            fila = salido_match.iloc[0]
            val = str(fila["sector"]).strip()
            if val and str(fila["fuente"]).strip() != "desconocido":
                sec = val

    if sec is None:
        return None

    if RECLASIFICACIONES_CSV.exists():
        df_recl = _csv_cacheado(str(RECLASIFICACIONES_CSV), RECLASIFICACIONES_CSV.stat().st_mtime_ns)
        recl_t = df_recl[df_recl["ticker"].astype(str).str.strip().str.upper() == t].sort_values(
            "desde", ascending=False
        )
        for _, fila in recl_t.iterrows():
            if fecha_str < str(fila["desde"]).strip():
                sec = str(fila["sector_anterior"]).strip()
    return sec
