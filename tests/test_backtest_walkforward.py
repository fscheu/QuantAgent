import csv
import json
from datetime import datetime

import click
import pytest

from quantagent.cli.walkforward import construir_ventanas, preparar_walkforward, seleccionar_ganador


@pytest.mark.parametrize("hasta", ["2012-12-23", "2019-01-01", "2023-01-01", "2016-12-30", "2016-12-31"])
def test_comando_csv_y_rechazos(snapshot_walkforward, hasta):
    from click.testing import CliRunner

    from quantagent.cli.backtest import backtest_group

    grid, out = snapshot_walkforward / "grid.json", snapshot_walkforward / "out.csv"
    grid.write_text(json.dumps({"rsi_period": [7, 14]}))
    runner = CliRunner()
    help_result = runner.invoke(backtest_group, ["walkforward", "--help"])
    assert help_result.exit_code == 0 and "--train-years" in help_result.output
    assert "--abrir-" not in help_result.output
    antes = {p: p.read_bytes() for p in snapshot_walkforward.rglob("*") if p.is_file()}
    res = runner.invoke(backtest_group, ["walkforward", "--strategy", "rsi", "--symbol", "spy",
                        "--snapshot", "wf", "--grid", str(grid), "--out", str(out),
                        "--from", "2011-01-03", "--to", hasta, "--llenado", "cierre"])
    if hasta[:4] != "2016":
        assert res.exit_code == 1, res.output
        assert ("ventana completa" if hasta[:4] == "2012" else hasta) in res.output
        assert {p: p.read_bytes() for p in snapshot_walkforward.rglob("*") if p.is_file()} == antes
        return
    assert res.exit_code == 0, res.output
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0] == dict(ajuste_desde="2011-01-03", ajuste_hasta="2014-12-31",
                           prueba_desde="2015-01-01", prueba_hasta="2016-12-30",
                           params='{"rsi_period": 7}', ajuste_sharpe_tasa_0="-0.71",
                           trades="1", pnl="-99.95", sharpe_tasa_0="-1.0")
    with (snapshot_walkforward / "intentos.csv").open() as f:
        intentos = list(csv.DictReader(f))
    assert [json.loads(r["params"]) for r in intentos] == [
        {"rsi_period": p, "llenado": "cierre"} for p in [7, 14]]
    assert all(r["hasta"] == "2014-12-31" and r["activo"] == "SPY" for r in intentos)


@pytest.mark.parametrize("fill", ["apertura", "cierre"])
def test_seleccion_motor_real_y_registro_solo_ajustes(snapshot_walkforward, monkeypatch, fill):
    import numpy as np
    import pandas as pd
    from click.testing import CliRunner

    from quantagent.backtesting.fixtures import snapshot_metadata
    from quantagent.cli import backtest
    from quantagent.cli.sweep import registrar_intento
    from quantagent.data.snapshot import write_snapshot

    fechas = pd.date_range("2007-01-05", "2010-12-31", freq="W-FRI")
    precios = 100 + 15 * np.sin(np.arange(len(fechas)) / 4)
    df = pd.DataFrame(dict(open=precios, high=precios + 1, low=precios - 1,
                           close=precios, adj_close=precios, volume=1000), index=fechas)
    write_snapshot("real", {"SPY": df})
    grid, out = snapshot_walkforward / "grid.json", snapshot_walkforward / "out.csv"
    combos = [{"rsi_period": p} for p in [7, 14, 21]]
    grid.write_text(json.dumps({"rsi_period": [7, 14, 21]}))
    args = ["walkforward", "--strategy", "rsi", "--symbol", "SPY", "--snapshot", "real",
            "--grid", str(grid), "--out", str(out), "--train-years", "2", "--test-years", "1",
            "--costos", "etf", "--orden-minima", "50"]
    if fill == "cierre":
        args += ["--llenado", fill]
    _, _, config, ventanas = preparar_walkforward(
        "rsi", "SPY", "real", grid, train_years=2, test_years=1, costos="etf", orden_minima=50)
    motor = backtest.correr_par
    esperados, llamadas = [], []
    for a, b, c, d in ventanas:
        ajustes = []
        for inicio, fin, params in [(a, b, p) for p in combos]:
            with backtest._isolated_sqlite_db():
                ajustes.append(motor("rsi", snapshot_metadata("real", "SPY", inicio, fin), config,
                                     "etf", None, "real", "SPY", inicio, fin, params=params, fill=fill))
        indice = max(range(len(combos)), key=lambda i: ajustes[i]["sharpe_tasa_0"])
        with backtest._isolated_sqlite_db():
            prueba = motor("rsi", snapshot_metadata("real", "SPY", c, d), config,
                           "etf", None, "real", "SPY", c, d, params=combos[indice], fill=fill)
        esperados.append((combos[indice], ajustes, prueba))

    def observar(*args, **kwargs):
        assert kwargs["fill"] == fill
        llamadas.append((args[7], args[8], kwargs["params"]))
        return motor(*args, **kwargs)

    monkeypatch.setattr(backtest, "correr_par", observar)
    registro = snapshot_walkforward / "intentos.csv"
    previo = esperados[0][1][0]
    registrar_intento("rsi", "SPY", "real", "etf", 1.0, combos[0],
                      {**previo, "desde": "2007-01-05", "hasta": "2008-12-26"})
    prefijo = registro.read_bytes()
    result = CliRunner().invoke(backtest.backtest_group, args)
    assert result.exit_code == 0, result.output
    assert registro.read_bytes().startswith(prefijo)
    with out.open() as f:
        filas = list(csv.DictReader(f))
    with registro.open() as f:
        intentos = list(csv.DictReader(f))[1:]
    assert len(filas) == len(ventanas) == 2 and len(intentos) == 6
    assert all(a < b < c <= d for a, b, c, d in ventanas) and ventanas[0][3] < ventanas[1][2]
    assert any(r["trades"] > 0 for _, _, r in esperados)
    assert any(len({r["sharpe_tasa_0"] for r in ajustes}) > 1 for _, ajustes, _ in esperados)
    for i, ((a, b, c, d), fila, (ganador, ajustes, prueba)) in enumerate(zip(ventanas, filas, esperados)):
        assert llamadas[i * 4:i * 4 + 4] == [(a, b, p) for p in combos] + [(c, d, ganador)]
        assert json.loads(fila["params"]) == ganador
        assert float(fila["ajuste_sharpe_tasa_0"]) == round(max(r["sharpe_tasa_0"] for r in ajustes), 2)
        for campo in ("trades", "pnl", "sharpe_tasa_0"):
            assert float(fila[campo]) == round(prueba[campo], 2)
        for intento, combo, ajuste in zip(intentos[i * 3:i * 3 + 3], combos, ajustes):
            assert json.loads(intento["params"]) == {**combo, "llenado": fill}
            assert intento["desde"] == f"{snapshot_metadata('real', 'SPY', a, b).start_date:%Y-%m-%d}"
            assert intento["hasta"] == f"{snapshot_metadata('real', 'SPY', a, b).end_date:%Y-%m-%d}"
            assert int(intento["trades"]) == ajuste["trades"]
            assert float(intento["pnl"]) == round(ajuste["pnl"], 2)
            assert float(intento["sharpe"]) == round(ajuste["sharpe"], 2)


@pytest.fixture
def snapshot_walkforward(monkeypatch, tmp_path):
    import pandas as pd

    from quantagent.data.snapshot import write_snapshot

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("QUANTAGENT_PRUEBA_DESDE", "2019-01-01")
    monkeypatch.setenv("QUANTAGENT_RESERVA_DESDE", "2023-01-01")
    df = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "adj_close": 100., "volume": 1000},
                      index=pd.bdate_range("2007-01-03", "2024-01-01"))
    write_snapshot("wf", {"SPY": df})
    return tmp_path


def test_preparacion_default_y_recorte(snapshot_walkforward, capsys):
    combos, meta, config, ventanas = preparar_walkforward("rsi", "spy", "wf", {"rsi_period": [7, 14]})
    assert combos == [{"rsi_period": 7}, {"rsi_period": 14}]
    assert meta.end_date == ventanas[-1][3] == datetime(2018, 12, 31)
    assert config["offline_data"] and config["base_position_pct"] == 1.0
    assert "sin --to, se asume fin en 2018-12-31" in capsys.readouterr().err
    _, meta, _, ventanas = preparar_walkforward(
        "rsi", "SPY", "wf", {"rsi_period": [14]}, to_date=datetime(2016, 12, 30))
    assert len(ventanas) == 3
    assert meta.end_date == ventanas[-1][3] == datetime(2016, 12, 30)
    assert not (snapshot_walkforward / "intentos.csv").exists()


@pytest.mark.parametrize("fecha,mensaje", [("2012-12-23", "ventana completa"),
                                           ("2019-01-01", "tramo de prueba"), ("2023-01-01", "reserva")])
def test_preparacion_rechaza_rangos_sin_escribir(snapshot_walkforward, fecha, mensaje):
    archivos = {p: p.read_bytes() for p in snapshot_walkforward.rglob("*") if p.is_file()}
    with pytest.raises(click.ClickException, match=mensaje):
        preparar_walkforward("rsi", "SPY", "wf", {"rsi_period": [14]}, to_date=datetime.fromisoformat(fecha))
    assert {p: p.read_bytes() for p in snapshot_walkforward.rglob("*") if p.is_file()} == archivos


def test_cuatro_ventanas_calendario_sin_solapamiento():
    ventanas = construir_ventanas(datetime(2007, 1, 3), datetime(2018, 12, 31), 4, 2)
    assert ventanas == [
        (datetime(2007, 1, 3), datetime(2010, 12, 31), datetime(2011, 1, 1), datetime(2012, 12, 31)),
        (datetime(2009, 1, 1), datetime(2012, 12, 31), datetime(2013, 1, 1), datetime(2014, 12, 31)),
        (datetime(2011, 1, 1), datetime(2014, 12, 31), datetime(2015, 1, 1), datetime(2016, 12, 31)),
        (datetime(2013, 1, 1), datetime(2016, 12, 31), datetime(2017, 1, 1), datetime(2018, 12, 31)),
    ]
    assert all(ajuste_fin < prueba_inicio <= prueba_fin for _, ajuste_fin, prueba_inicio, prueba_fin in ventanas)
    assert all(a[3] < b[2] for a, b in zip(ventanas, ventanas[1:]))


def test_fecha_bisiesta_y_descarta_ultima_ventana_incompleta():
    ventanas = construir_ventanas(datetime(2008, 2, 29), datetime(2015, 12, 23), 4, 2)
    assert ventanas == [
        (datetime(2008, 2, 29), datetime(2011, 12, 31), datetime(2012, 1, 1), datetime(2013, 12, 31)),
    ]


@pytest.mark.parametrize("train,test", [(0, 2), (-1, 2), (4, 0), (4, -1)])
def test_rechaza_anios_no_positivos(train, test):
    with pytest.raises(click.ClickException, match="deben ser positivos"):
        construir_ventanas(datetime(2007, 1, 3), datetime(2018, 12, 31), train, test)


def test_rango_insuficiente():
    with pytest.raises(click.ClickException, match="ventana completa"):
        construir_ventanas(datetime(2007, 1, 3), datetime(2012, 12, 23), 4, 2)


@pytest.mark.parametrize("dia,cantidad", [(30, 3), (24, 3), (23, 2)])
def test_ultima_semana_del_anio_completa_la_ventana(dia, cantidad):
    assert len(construir_ventanas(datetime(2007, 1, 3), datetime(2016, 12, dia), 4, 2)) == cantidad


@pytest.mark.parametrize("sharpes,ganador", [([None, -0.5, None], 1), ([-0.5, None], 0), ([None, None], 0)])
def test_sharpe_ausente_pierde_y_empate_conserva_el_primero(sharpes, ganador):
    resultados = [{"sharpe_tasa_0": sharpe} for sharpe in sharpes]
    assert seleccionar_ganador(resultados) is resultados[ganador]


def test_maximo_y_empate_conservan_orden_de_grilla():
    resultados = [
        {"params": {"rsi_period": 7}, "sharpe_tasa_0": -0.5},
        {"params": {"rsi_period": 14}, "sharpe_tasa_0": 0.8},
        {"params": {"rsi_period": 21}, "sharpe_tasa_0": 0.8},
    ]
    assert seleccionar_ganador(resultados) is resultados[1]
    assert seleccionar_ganador(list(reversed(resultados))) is resultados[2]
    assert seleccionar_ganador(resultados[:1]) is resultados[0]
