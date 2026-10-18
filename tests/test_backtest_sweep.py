import json
from pathlib import Path

import click
import pytest

from quantagent.cli.sweep import CAMPOS_INTENTOS, MAX_SWEEP_COMBINATIONS, cargar_grilla, validar_grilla


def test_vecinos_rsi_un_parametro_por_vez():
    from quantagent.cli.sweep import grilla_vecinos
    combos, labels = grilla_vecinos("rsi")
    default = combos[0]
    assert len(combos) == len(labels) == 13
    assert labels[0] == "default" and combos.count(default) == 1
    assert [c["rsi_period"] for c, label in zip(combos, labels) if label.startswith("rsi_period")] == [11, 17]
    for combo, label in zip(combos[1:], labels[1:]):
        param, factor = label.split("x")
        assert [p for p in default if combo[p] != default[p]] == [param]
        expected = default[param] * float(factor)
        assert combo[param] == (round(expected) if isinstance(default[param], int) else expected)


def test_vecinos_fijos_redondeo_y_tope(monkeypatch):
    from quantagent.cli.sweep import grilla_vecinos, STRATEGY_REGISTRY
    params = {k: {"default": v} for k, v in {"zero": 0, "bool": True, "text": "x", "one": 1}.items()}
    monkeypatch.setitem(STRATEGY_REGISTRY, "Test", {"params": params})
    assert grilla_vecinos("Test") == ([{k: v["default"] for k, v in params.items()}], ["default"])
    params.update({f"p{i}": {"default": 10} for i in range(25)})
    with pytest.raises(click.ClickException, match="límite máximo de 50"):
        grilla_vecinos("Test")


def test_cli_vecinos_excluye_grid(tmp_path):
    from click.testing import CliRunner
    from quantagent.cli.backtest import backtest_group
    out = tmp_path / "out.csv"
    args = ["sweep", "--strategy", "rsi", "--symbol", "SPY", "--snapshot", "missing", "--out", str(out)]
    for options in ([], ["--vecinos-20", "--grid", "missing.json"]):
        result = CliRunner().invoke(backtest_group, args + options)
        assert result.exit_code == 1 and "exactamente una" in result.output
        assert not out.exists()


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
    assert "attempts" in res_root.output

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
    assert "--abrir-" not in res.output
    assert "sweep no abre la prueba ni la reserva" in res.output
    assert not out_csv.exists()

    res_reserva = runner.invoke(backtest_group, [
        "sweep",
        "--strategy", "rsi",
        "--symbol", "SPY",
        "--snapshot", "etf-1d-2026-10",
        "--grid", str(grid_file),
        "--out", str(out_csv),
        "--to", "2024-01-01",
    ])
    assert res_reserva.exit_code == 1
    assert "cae en la reserva" in res_reserva.output
    assert "--abrir-" not in res_reserva.output
    assert "sweep no abre la prueba ni la reserva" in res_reserva.output


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


def test_sweep_integracion_snapshot_sintetico_rsi_2x2(monkeypatch, tmp_path: Path):
    import csv
    import pandas as pd
    from click.testing import CliRunner
    from quantagent.backtesting.fixtures import FIXTURES_DIR, snapshot_metadata
    from quantagent.cli.backtest import (
        COMPARE_COLUMNS,
        _engine_config,
        _isolated_sqlite_db,
        backtest_group,
        correr_par,
    )
    from quantagent.data.snapshot import write_snapshot

    snap_dir = tmp_path / "snaps"
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(snap_dir))

    with (FIXTURES_DIR / "spy-2y-1d.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(
        {
            "open": [float(r["open"]) for r in rows],
            "high": [float(r["high"]) for r in rows],
            "low": [float(r["low"]) for r in rows],
            "close": [float(r["close"]) for r in rows],
            "volume": [int(r["volume"]) for r in rows],
        },
        index=pd.bdate_range("2017-01-03", periods=len(rows), name="timestamp"),
    )
    df.insert(4, "adj_close", df["close"])
    write_snapshot("snap_rsi", {"SPY": df})

    grid_file = tmp_path / "grid_rsi.json"
    grid_file.write_text(json.dumps({"rsi_period": [7, 14], "oversold_threshold": [25.0, 30.0]}), encoding="utf-8")
    out_csv = tmp_path / "out_rsi.csv"

    runner = CliRunner()
    res = runner.invoke(backtest_group, [
        "sweep",
        "--strategy", "rsi",
        "--symbol", "SPY",
        "--snapshot", "snap_rsi",
        "--grid", str(grid_file),
        "--out", str(out_csv),
    ])
    assert res.exit_code == 0, res.output

    with out_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        out_rows = list(reader)

    expected_cols = ["oversold_threshold", "rsi_period"] + [
        c for c in COMPARE_COLUMNS if c not in ("estrategia", "activo")
    ]
    assert header == expected_cols
    assert len(out_rows) == 4

    intentos = snap_dir / "intentos.csv"
    prefijo = intentos.read_bytes()
    res = runner.invoke(backtest_group, [
        "sweep", "--strategy", "rsi", "--symbol", "spy",
        "--snapshot", "snap_rsi", "--grid", str(grid_file), "--out", str(out_csv),
    ])
    assert res.exit_code == 0, res.output
    assert intentos.read_bytes().startswith(prefijo)
    with intentos.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CAMPOS_INTENTOS
        intentos_rows = list(reader)
    assert len(intentos_rows) == 8
    antes = intentos.read_bytes()
    consulta = runner.invoke(backtest_group, ["attempts", "--strategy", "rsi"])
    assert consulta.exit_code == 0, consulta.output
    assert consulta.output == "Corridas: 8\nCombinaciones distintas: 4\n"
    assert intentos.read_bytes() == antes
    from datetime import datetime, timezone
    for intento, salida in zip(intentos_rows, out_rows * 2):
        assert datetime.fromisoformat(intento["fecha_utc"]).utcoffset() == timezone.utc.utcoffset(None)
        assert (intento["estrategia"], intento["activo"], intento["snapshot"]) == ("rsi", "SPY", "snap_rsi")
        assert (intento["costos"], intento["tamano"]) == ("", "1.0")
        assert intento["params"] == json.dumps({
            "llenado": "apertura",
            "oversold_threshold": float(salida["oversold_threshold"]),
            "rsi_period": int(salida["rsi_period"]),
        }, sort_keys=True)
        for campo in ("desde", "hasta", "trades", "pnl", "sharpe"):
            assert intento[campo] == salida[campo]
    for existente in (True, False):
        if not existente:
            intentos.unlink()
        anterior = intentos.read_bytes() if existente else None
        grid_file.write_text(json.dumps({"inexistente": [1]}), encoding="utf-8")
        res = runner.invoke(backtest_group, [
            "sweep", "--strategy", "rsi", "--symbol", "SPY",
            "--snapshot", "snap_rsi", "--grid", str(grid_file), "--out", str(out_csv),
        ])
        assert res.exit_code == 1
        assert (intentos.read_bytes() if intentos.exists() else None) == anterior

    meta = snapshot_metadata("snap_rsi", "SPY", None, None)
    config = _engine_config(None, None, 1.0, None, None)
    with _isolated_sqlite_db():
        baseline = correr_par("rsi", meta, config, None, None, "snap_rsi", "SPY", None, None, fill="apertura")

    filas_por_combo = {}
    for r in out_rows:
        combo = {"oversold_threshold": float(r["oversold_threshold"]), "rsi_period": int(r["rsi_period"])}
        with _isolated_sqlite_db():
            esperado = correr_par("rsi", meta, config, None, None, "snap_rsi", "SPY", None, None, params=combo, fill="apertura")
        assert int(r["trades"]) == esperado["trades"]
        assert float(r["pnl"]) == round(esperado["pnl"], 2)
        filas_por_combo[(combo["oversold_threshold"], combo["rsi_period"])] = r

    default_row = filas_por_combo[(30.0, 14)]
    assert int(default_row["trades"]) == baseline["trades"]
    assert float(default_row["pnl"]) == round(baseline["pnl"], 2)

    for ot in [25.0, 30.0]:
        f7 = filas_por_combo[(ot, 7)]
        f14 = filas_por_combo[(ot, 14)]
        assert (int(f7["trades"]) != int(f14["trades"])) or (float(f7["pnl"]) != float(f14["pnl"]))

    res = runner.invoke(backtest_group, [
        "sweep", "--strategy", "rsi", "--symbol", "SPY", "--snapshot", "snap_rsi",
        "--vecinos-20", "--out", str(out_csv),
    ])
    assert res.exit_code == 0, res.output
    with out_csv.open() as f:
        vecinos = list(csv.DictReader(f))
    with intentos.open() as f:
        assert len(list(csv.DictReader(f))) == len(vecinos) == 13
    assert vecinos[0]["vecino"] == "default"
    assert (int(vecinos[0]["trades"]), float(vecinos[0]["pnl"])) == (baseline["trades"], round(baseline["pnl"], 2))

    grid_file.write_text(json.dumps({"rsi_period": [7, 14]}))
    resultados = []
    for flags, fill in [([], "apertura"), (["--llenado", "apertura"], "apertura"),
                        (["--llenado", "cierre"], "cierre")]:
        res = runner.invoke(backtest_group, [
            "sweep", "--strategy", "rsi", "--symbol", "SPY", "--snapshot", "snap_rsi",
            "--grid", str(grid_file), "--out", str(out_csv), *flags])
        assert res.exit_code == 0, res.output
        with out_csv.open() as f:
            resultados.append(list(csv.DictReader(f)))
        with intentos.open() as f:
            ultimos = list(csv.DictReader(f))[-2:]
        assert [json.loads(r["params"]) for r in ultimos] == [
            {"rsi_period": p, "llenado": fill} for p in [7, 14]]
        for fila, p in zip(resultados[-1], [7, 14]):
            with _isolated_sqlite_db():
                esperado = correr_par("rsi", meta, config, None, None, "snap_rsi", "SPY",
                                      None, None, params={"rsi_period": p}, fill=fill)
            assert float(fila["pnl"]) == round(esperado["pnl"], 2)
    assert resultados[0] == resultados[1]
    assert [r["pnl"] for r in resultados[0]] != [r["pnl"] for r in resultados[2]]


def test_attempts_filtra_alias_y_activo_sin_modificar(monkeypatch, tmp_path):
    import csv
    from click.testing import CliRunner
    from quantagent.cli.backtest import backtest_group

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    path = tmp_path / "intentos.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["estrategia", "activo", "params"])
        writer.writerows([
            ["rsi", "SPY", '{"rsi_period": 14, "oversold_threshold": 30}'],
            ["RSIMeanReversionStrategy", "SPY", '{"oversold_threshold":30,"rsi_period":14}'],
            ["rsi", "TLT", '{"rsi_period": 7}'],
            ["sma-cross", "SPY", '{"fast": 50}'],
        ])
    antes = path.read_bytes()
    runner = CliRunner()
    for strategy, symbol, expected in [
        ("rsi", None, (3, 2)), ("RSIMeanReversionStrategy", "spy", (2, 1)),
        ("rsi", "TLT", (1, 1)), ("sma-cross", "SPY", (1, 1)), ("rsi", "QQQ", (0, 0)),
    ]:
        args = ["attempts", "--strategy", strategy] + (["--symbol", symbol] if symbol else [])
        res = runner.invoke(backtest_group, args)
        assert res.exit_code == 0, res.output
        assert res.output == f"Corridas: {expected[0]}\nCombinaciones distintas: {expected[1]}\n"
        assert path.read_bytes() == antes


def test_attempts_registro_ausente_o_encabezado_invalido(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from quantagent.cli.backtest import backtest_group

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    runner = CliRunner()
    args = ["attempts", "--strategy", "rsi"]
    assert runner.invoke(backtest_group, args).output == "Corridas: 0\nCombinaciones distintas: 0\n"
    path = tmp_path / "intentos.csv"
    path.write_text("otra,columna\n", encoding="utf-8")
    res = runner.invoke(backtest_group, args)
    assert res.exit_code == 1
    assert "Encabezado inválido" in res.output
    assert path.read_text() == "otra,columna\n"



def test_registrar_intento_migra_encabezado_viejo_y_cuenta_por_hipotesis(monkeypatch, tmp_path):
    import csv
    from quantagent.cli.sweep import CAMPOS_INTENTOS, contar_intentos, registrar_intento

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    path = tmp_path / "intentos.csv"
    viejo = CAMPOS_INTENTOS[:-3]
    filas = [{c: f"{c}{i}" for c in viejo} for i in range(3)]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=viejo)
        w.writeheader()
        w.writerows(filas)
    assert contar_intentos() == 3 and contar_intentos("h1") == 0

    row = {"desde": "2010-01-01", "hasta": "2012-01-01", "trades": 4, "pnl": 1.5, "sharpe": 0.3}
    registrar_intento("rsi", "SPY", "snap", None, 1.0, {"a": 1}, row, hipotesis="h1", etapa="E1")
    registrar_intento("rsi", "SPY", "snap", None, 1.0, {"a": 2}, row)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CAMPOS_INTENTOS
        rows = list(reader)
    assert len(rows) == 5
    for orig, nueva in zip(filas, rows[:3]):
        assert {c: nueva[c] for c in viejo} == orig
        assert (nueva["hipotesis"], nueva["etapa"], nueva["sharpe_base"]) == ("", "", "")
    assert (rows[3]["hipotesis"], rows[3]["etapa"]) == ("h1", "E1")
    assert contar_intentos() == 5 and contar_intentos("h1") == 1


def test_registrar_intento_migra_encabezado_sin_sharpe_base(monkeypatch, tmp_path):
    import csv
    from quantagent.cli.sweep import CAMPOS_INTENTOS, registrar_intento

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    path = tmp_path / "intentos.csv"
    sin_sharpe_base = CAMPOS_INTENTOS[:-1]
    filas = [{c: f"v_{c}_{i}" for c in sin_sharpe_base} for i in range(2)]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sin_sharpe_base)
        w.writeheader()
        w.writerows(filas)

    row = {"desde": "2010-01-01", "hasta": "2012-01-01", "trades": 4, "pnl": 1.5, "sharpe": 0.3}
    registrar_intento("rsi", "SPY", "snap", None, 1.0, {"a": 1}, row, hipotesis="h1", etapa="1", sharpe_base="tasa_0")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CAMPOS_INTENTOS
        rows = list(reader)
    assert len(rows) == 3
    for orig, guardada in zip(filas, rows[:2]):
        assert {c: guardada[c] for c in sin_sharpe_base} == orig
        assert guardada["sharpe_base"] == ""
    assert rows[2]["sharpe_base"] == "tasa_0"
    assert rows[2]["hipotesis"] == "h1"
