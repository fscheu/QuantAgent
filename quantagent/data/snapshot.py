"""Snapshots de datos de mercado: un Parquet por símbolo y un manifiesto con hashes.

Los datos viven fuera del repo, en `$QUANTAGENT_SNAPSHOT_DIR/<nombre>/` (iuf D5, opción C).
Adentro del repo solo va el manifiesto, en `tests/fixtures/snapshots/<nombre>.json`.
Un snapshot se congela: `write_snapshot` no pisa un nombre que ya existe (iuf D6).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

SNAPSHOT_DIR_ENV = "QUANTAGENT_SNAPSHOT_DIR"
MANIFEST_NAME = "manifest.json"
VERSIONED_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "snapshots"


class SnapshotError(Exception):
    """El snapshot no existe, ya existe o no coincide con su manifiesto."""


def snapshot_root(base_dir: Optional[Path] = None) -> Path:
    """Carpeta que contiene los snapshots: `base_dir` o `$QUANTAGENT_SNAPSHOT_DIR`."""
    if base_dir is not None:
        return Path(base_dir)
    value = os.environ.get(SNAPSHOT_DIR_ENV)
    if not value:
        raise SnapshotError(f"{SNAPSHOT_DIR_ENV} no está definida")
    return Path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_snapshot(
    name: str,
    frames: Dict[str, pd.DataFrame],
    *,
    source: str = "yahoo",
    timeframe: str = "1d",
    base_dir: Optional[Path] = None,
) -> dict:
    """Escribe un Parquet por símbolo y `manifest.json`. Falla si `name` ya existe."""
    if not frames:
        raise SnapshotError("no hay símbolos para guardar")
    empty = [symbol for symbol, df in frames.items() if df.empty]
    if empty:
        raise SnapshotError(f"sin filas: {', '.join(sorted(empty))}")
    folder = snapshot_root(base_dir) / name
    if folder.exists():
        raise SnapshotError(f"el snapshot '{name}' ya existe: {folder}")
    folder.mkdir(parents=True)

    try:
        yf_version = metadata.version("yfinance")
    except metadata.PackageNotFoundError:  # pragma: no cover
        yf_version = None

    symbols = {}
    for symbol, df in sorted(frames.items()):
        path = folder / f"{symbol}.parquet"
        df.to_parquet(path)
        symbols[symbol] = {
            "file": path.name,
            "rows": len(df),
            "start": df.index.min().date().isoformat(),
            "end": df.index.max().date().isoformat(),
            "sha256": _sha256(path),
        }
    manifest = {
        "name": name,
        "source": source,
        "timeframe": timeframe,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "yfinance_version": yf_version,
        "symbols": symbols,
    }
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_manifest(name: str, base_dir: Optional[Path] = None) -> dict:
    path = snapshot_root(base_dir) / name / MANIFEST_NAME
    if not path.exists():
        raise SnapshotError(f"el snapshot '{name}' no existe: {path}")
    return json.loads(path.read_text())


def read_snapshot(name: str, symbol: str, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Lee el Parquet de `symbol` tal como se guardó."""
    manifest = load_manifest(name, base_dir)
    if symbol not in manifest["symbols"]:
        raise SnapshotError(f"{symbol} no está en el snapshot '{name}'")
    return pd.read_parquet(snapshot_root(base_dir) / name / manifest["symbols"][symbol]["file"])


def verify_snapshot(name: str, base_dir: Optional[Path] = None, versioned_dir: Optional[Path] = None) -> dict:
    """Recalcula el sha256 de cada Parquet contra `manifest.json` y, si existe, contra el manifiesto versionado.

    Devuelve el manifiesto si todo coincide; si no, `SnapshotError` con cada archivo que no coincide.
    """
    manifest = load_manifest(name, base_dir)
    folder = snapshot_root(base_dir) / name
    problems: List[str] = []
    for symbol, info in manifest["symbols"].items():
        path = folder / info["file"]
        if not path.exists():
            problems.append(f"{info['file']}: falta el archivo")
        elif _sha256(path) != info["sha256"]:
            problems.append(f"{info['file']}: sha256 distinto del manifiesto")

    versioned = (VERSIONED_DIR if versioned_dir is None else Path(versioned_dir)) / f"{name}.json"
    if versioned.exists():
        expected = json.loads(versioned.read_text())["symbols"]
        for symbol in sorted(set(expected) | set(manifest["symbols"])):
            if expected.get(symbol) != manifest["symbols"].get(symbol):
                problems.append(f"{symbol}.parquet: no coincide con el manifiesto versionado {versioned.name}")
    if problems:
        raise SnapshotError("; ".join(problems))
    return manifest
