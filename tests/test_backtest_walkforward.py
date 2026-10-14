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
                        "--from", "2011-01-03", "--to", hasta])
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
                           params='{"rsi_period": 7}', ajuste_sharpe_tasa_0="-0.7081740842923033",
                           trades="1", pnl="-99.95002499", sharpe_tasa_0="-1.0019921641686198")
    with (snapshot_walkforward / "intentos.csv").open() as f:
        intentos = list(csv.DictReader(f))
    assert [json.loads(r["params"]) for r in intentos] == [{"rsi_period": 7}, {"rsi_period": 14}]
    assert all(r["hasta"] == "2014-12-31" and r["activo"] == "SPY" for r in intentos)


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
