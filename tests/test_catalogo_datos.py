"""Contrato del catálogo: esquema, manifiestos y cobertura local opcional."""

import os
from collections import Counter
from pathlib import Path

import pytest

from quantagent.data.catalogo import cargar_catalogo, verificar_cobertura

ROOT = Path(__file__).resolve().parents[1]
CATALOGO = ROOT / "catalogo/datos"


def texto(valor):
    assert isinstance(valor, str) and valor.strip()


def test_esquema():
    datasets, exclusiones = cargar_catalogo(CATALOGO)
    archivos = sorted(p for p in CATALOGO.glob("*.yaml") if p.name != "_no-son-datos.yaml")
    assert datasets and len(datasets) == len(archivos)
    for dataset, archivo in zip(datasets, archivos, strict=True):
        assert isinstance(dataset, dict)
        assert set(dataset) == {
            "id", "descripcion", "fuente", "campos", "columna_cuando_se_supo",
            "frecuencia", "carpeta", "snapshots", "lectores", "datos_esperados",
        }
        texto(dataset["id"])
        assert dataset["id"] == archivo.stem
        texto(dataset["descripcion"])
        assert isinstance(dataset["datos_esperados"], bool)
        fuente = dataset["fuente"]
        assert isinstance(fuente, dict)
        assert set(fuente) == {"nombre", "url", "licencia", "regla_de_borrado"}
        for campo in ("nombre", "licencia"):
            texto(fuente[campo])
        for campo in ("url", "regla_de_borrado"):
            if fuente[campo] is not None:
                texto(fuente[campo])
        assert isinstance(dataset["campos"], list) and dataset["campos"]
        for campo in dataset["campos"]:
            assert isinstance(campo, dict) and set(campo) == {"nombre", "unidad", "descripcion"}
            for valor in campo.values():
                texto(valor)
        for clave, nombres in (
            ("columna_cuando_se_supo", {"columna", "descripcion"}),
            ("frecuencia", {"capa_viva", "congelado"}),
        ):
            assert isinstance(dataset[clave], dict) and set(dataset[clave]) == nombres
            for valor in dataset[clave].values():
                texto(valor)
        for clave in ("carpeta", "snapshots", "lectores"):
            assert isinstance(dataset[clave], list)
        assert dataset["carpeta"]
        for patron in dataset["carpeta"]:
            texto(patron)
            assert "/" not in patron and patron not in {".", ".."}
        for snapshot in dataset["snapshots"]:
            texto(snapshot)
            assert Path(snapshot).name == snapshot and snapshot.endswith(".json")
        for lector in dataset["lectores"]:
            assert isinstance(lector, dict) and set(lector) == {"archivo", "tipo", "descripcion"}
            for valor in lector.values():
                texto(valor)
            assert (ROOT / lector["archivo"]).is_file()
            assert lector["tipo"] in {"estrategias", "embudo", "paper", "web", "datos", "investigacion"}
    assert Counter(d["id"] for d in datasets) == Counter(p.stem for p in archivos)
    assert isinstance(exclusiones, list) and exclusiones
    for entrada in exclusiones:
        assert isinstance(entrada, dict) and set(entrada) == {"entrada", "motivo"}
        for valor in entrada.values():
            texto(valor)


def test_manifiestos_versionados():
    datasets, _ = cargar_catalogo(CATALOGO)
    referencias = Counter(s for d in datasets for s in d["snapshots"])
    manifiestos = {p.name for p in (ROOT / "tests/fixtures/snapshots").glob("*.json")}
    assert referencias == Counter(manifiestos)


@pytest.mark.snapshot
def test_cobertura_local():
    raiz = Path(os.environ.get("QUANTAGENT_SNAPSHOT_DIR", ROOT / "local-data"))
    if not raiz.exists():
        pytest.skip(f"sin directorio de datos locales: {raiz}")
    if not any(p.is_dir() for p in raiz.iterdir()):
        pytest.skip(f"sin carpetas de datos locales (checkout de CI): {raiz}")
    datasets, exclusiones = cargar_catalogo(CATALOGO)
    assert verificar_cobertura(datasets, exclusiones, raiz) == []


def test_cobertura_detecta_ambos_fallos(tmp_path):
    datasets = [{"id": "precios", "carpeta": ["precios-*"], "datos_esperados": True,
                 "fuente": {"nombre": "prueba", "regla_de_borrado": None}}]
    (tmp_path / "huerfana").mkdir()
    assert verificar_cobertura(datasets, [], tmp_path) == [
        "carpeta sin catálogo: huerfana", "precios: patrón sin carpeta: precios-*",
    ]
    (tmp_path / "precios-2026-10").mkdir()
    assert verificar_cobertura(datasets, [{"entrada": "huerfana", "motivo": "resultado"}], tmp_path) == []
    datasets[0]["carpeta"] = ["precios-*", "ausente-*"]
    assert verificar_cobertura(datasets, [{"entrada": "huerfana"}], tmp_path) == [
        "precios: patrón sin carpeta: ausente-*",
    ]
    datasets[0]["datos_esperados"] = False
    assert verificar_cobertura(datasets, [{"entrada": "huerfana"}], tmp_path) == []


def test_regla_de_borrado_exige_carpeta_propia(tmp_path):
    # Fuentes ficticias: no declara un dataset Sharadar que todavía no existe.
    datasets = [
        {"id": "a", "carpeta": ["fuente-*"], "datos_esperados": True,
         "fuente": {"nombre": "fuente-a", "regla_de_borrado": "borrar al vencer licencia"}},
        {"id": "b", "carpeta": ["fuente-2026"], "datos_esperados": True,
         "fuente": {"nombre": "fuente-b", "regla_de_borrado": None}},
    ]
    (tmp_path / "fuente-2026").mkdir()
    assert verificar_cobertura(datasets, [], tmp_path) == [
        "carpeta compartida entre fuentes con regla de borrado: fuente-2026",
    ]
    datasets[1]["carpeta"] = ["otra-fuente"]
    (tmp_path / "otra-fuente").mkdir()
    assert verificar_cobertura(datasets, [], tmp_path) == []
