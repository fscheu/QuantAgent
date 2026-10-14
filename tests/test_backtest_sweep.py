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


def test_grilla_con_valores_repetidos_se_rechaza():
    with pytest.raises(click.ClickException, match="valores duplicados"):
        validar_grilla({"rsi_period": [14, 14]}, "rsi")


def test_cli_help_expone_sweep_sin_abrir_prueba_ni_reserva():
    from click.testing import CliRunner
    from quantagent.cli.backtest import backtest_group

    runner = CliRunner()
    res_root = runner.invoke(backtest_group, ["--help"])
    assert res_root.exit_code == 0
    assert "sweep" in res_root.output

    res_sweep = runner.invoke(backtest_group, ["sweep", "--help"])
    assert res_sweep.exit_code == 0
    assert "--abrir-prueba" not in res_sweep.output
    assert "--abrir-reserva" not in res_sweep.output


def test_cli_parametro_desconocido_sale_1_sin_csv(tmp_path: Path):
    from click.testing import CliRunner
    from quantagent.cli.backtest import backtest_group

    grid_file = tmp_path / "grid_bad.json"
    grid_file.write_text(json.dumps({"inexistente": [10, 20]}), encoding="utf-8")
    out_csv = tmp_path / "out.csv"
    runner = CliRunner()
    res = runner.invoke(backtest_group, [
        "sweep",
        "--strategy", "rsi",
        "--symbol", "SPY",
        "--snapshot", "etf-1d-2026-10",
        "--grid", str(grid_file),
        "--out", str(out_csv),
    ])
    assert res.exit_code == 1
    assert "Parámetro desconocido" in res.output
    assert not out_csv.exists()


def test_cli_mas_de_50_combinaciones_sale_1_sin_csv(tmp_path: Path):
    from click.testing import CliRunner
    from quantagent.cli.backtest import backtest_group

    grid_file = tmp_path / "grid_51.json"
    grid_file.write_text(json.dumps({"rsi_period": list(range(51))}), encoding="utf-8")
    out_csv = tmp_path / "out.csv"
    runner = CliRunner()
    res = runner.invoke(backtest_group, [
        "sweep",
        "--strategy", "rsi",
        "--symbol", "SPY",
        "--snapshot", "etf-1d-2026-10",
        "--grid", str(grid_file),
        "--out", str(out_csv),
    ])
    assert res.exit_code == 1
    assert "límite máximo de 50" in res.output or "superando el límite" in res.output
    assert not out_csv.exists()


def test_cli_to_en_tramo_prueba_sale_1_sin_csv(tmp_path: Path):
    from click.testing import CliRunner
    from quantagent.cli.backtest import backtest_group

    grid_file = tmp_path / "grid.json"
    grid_file.write_text(json.dumps({"rsi_period": [14]}), encoding="utf-8")
    out_csv = tmp_path / "out.csv"
    runner = CliRunner()
    res = runner.invoke(backtest_group, [
        "sweep",
        "--strategy", "rsi",
        "--symbol", "SPY",
        "--snapshot", "etf-1d-2026-10",
        "--grid", str(grid_file),
        "--out", str(out_csv),
        "--to", "2019-01-01",
    ])
    assert res.exit_code == 1
    assert "cae en el tramo de prueba" in res.output
    assert not out_csv.exists()


def test_cli_valores_duplicados_sale_1_sin_csv(tmp_path: Path):
    from click.testing import CliRunner
    from quantagent.cli.backtest import backtest_group

    grid_file = tmp_path / "grid_dup.json"
    grid_file.write_text(json.dumps({"rsi_period": [14, 14]}), encoding="utf-8")
    out_csv = tmp_path / "out.csv"
    runner = CliRunner()
    res = runner.invoke(backtest_group, [
        "sweep",
        "--strategy", "rsi",
        "--symbol", "SPY",
        "--snapshot", "etf-1d-2026-10",
        "--grid", str(grid_file),
        "--out", str(out_csv),
    ])
    assert res.exit_code == 1
    assert "valores duplicados" in res.output
    assert not out_csv.exists()

