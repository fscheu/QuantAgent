import json
from pathlib import Path

import click
import pytest

from quantagent.cli.sweep import MAX_SWEEP_COMBINATIONS, cargar_grilla, validar_grilla


def test_grilla_rsi_2x2_devuelve_cuatro_combinaciones_deterministas(tmp_path: Path):
    grid = {"rsi_period": [7, 14], "oversold_threshold": [25.0, 30.0]}
    f = tmp_path / "grid.json"
    f.write_text(json.dumps(grid), encoding="utf-8")

    combos = cargar_grilla(f, "rsi")
    assert len(combos) == 4
    for c in combos:
        assert list(c.keys()) == ["oversold_threshold", "rsi_period"]
    expected = [
        {"oversold_threshold": 25.0, "rsi_period": 7},
        {"oversold_threshold": 25.0, "rsi_period": 14},
        {"oversold_threshold": 30.0, "rsi_period": 7},
        {"oversold_threshold": 30.0, "rsi_period": 14},
    ]
    assert combos == expected
    assert validar_grilla({"oversold_threshold": [25.0, 30.0], "rsi_period": [7, 14]}, "rsi") == expected


def test_parametro_desconocido_falla(tmp_path: Path):
    f = tmp_path / "grid.json"
    f.write_text(json.dumps({"inexistente": [10, 20]}), encoding="utf-8")
    with pytest.raises(click.ClickException, match="Parámetro desconocido 'inexistente'"):
        cargar_grilla(f, "rsi")


def test_limite_50_combinaciones(tmp_path: Path):
    assert len(validar_grilla({"rsi_period": list(range(MAX_SWEEP_COMBINATIONS))}, "rsi")) == 50
    f = tmp_path / "grid_51.json"
    f.write_text(json.dumps({"rsi_period": list(range(51))}), encoding="utf-8")
    with pytest.raises(click.ClickException, match="51 combinaciones"):
        cargar_grilla(f, "rsi")


def test_json_invalido_se_rechaza(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ json roto", encoding="utf-8")
    with pytest.raises(click.ClickException, match="JSON inválido"):
        cargar_grilla(bad, "rsi")
    with pytest.raises(click.ClickException, match="No se encontró"):
        cargar_grilla(tmp_path / "missing.json", "rsi")
    with pytest.raises(click.ClickException, match="objeto JSON"):
        validar_grilla([1, 2, 3], "rsi")


def test_listas_vacias_y_grilla_vacia_se_rechazan():
    with pytest.raises(click.ClickException, match="no vacío"):
        validar_grilla({}, "rsi")
    with pytest.raises(click.ClickException, match="lista no vacía"):
        validar_grilla({"rsi_period": []}, "rsi")
    with pytest.raises(click.ClickException, match="lista no vacía"):
        validar_grilla({"rsi_period": 14}, "rsi")
