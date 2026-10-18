import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from quantagent.cli import hipotesis
from quantagent.cli.backtest import backtest_group
from quantagent.data.snapshot import write_snapshot


@pytest.mark.parametrize("rechazo", [True, False])
def test_correr_corte_real_y_estado(tmp_path, monkeypatch, rechazo):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2007-01-01", "2018-12-31", freq="7D")
    velas = pd.DataFrame(
        {c: 100.0 for c in ("open", "high", "low", "close", "adj_close", "volume")}, index=fechas,
    )
    write_snapshot("sintetico", {"SPY": velas}, base_dir=tmp_path)
    data = yaml.safe_load((Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml").read_text())
    data["parametros"] = {"rsi_period": [7], "oversold_threshold": [30.0]}
    if rechazo:
        data["justificacion"] = "corta"
    archivo = tmp_path / "h.yaml"
    archivo.write_text(yaml.safe_dump(data))

    def prohibido(*args, **kwargs):
        pytest.fail("No debe ejecutar compare ni etapa 2")

    monkeypatch.setattr("subprocess.run", prohibido)
    result = CliRunner().invoke(backtest_group, [
        "hipotesis", "correr", str(archivo), "--snapshot", "sintetico", "--carpeta", str(tmp_path / "salida"),
    ])
    assert result.exit_code == 1, result.output
    with (tmp_path / "hipotesis.csv").open() as f:
        filas = list(csv.DictReader(f))
    assert [(r["etapa"], r["veredicto"]) for r in filas] == (
        [("0", "CORTA")] if rechazo else [("0", "PASA"), ("1", "CORTA")]
    )
    assert all(r["hipotesis"] == data["id"] for r in filas)
    assert result.output.splitlines() == [
        *[f"Etapa {r['etapa']}: {r['veredicto']} ({r['motivo']})" for r in filas],
        f"Llegó a la etapa {filas[-1]['etapa']}: CORTA ({filas[-1]['motivo']})",
    ]
    assert ("justificación" if rechazo else "menos de 30 trades") in filas[-1]["motivo"]
    with (tmp_path / "intentos.csv").open() as f:
        intentos = list(csv.DictReader(f))
    assert [r["etapa"] for r in intentos] == ["0" if rechazo else "1"]
    assert not (tmp_path / "hipotesis-candado.csv").exists()


@pytest.mark.parametrize("cortar", [False, True])
def test_correr_control_real(tmp_path, monkeypatch, cortar):
    pytest.importorskip("backtesting")
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2007-01-01", "2018-12-31", freq="2D")
    snapshots = {}
    for activo, periodo in [("SPY", 5), ("TLT", 9)]:
        close = 100 + 10 * np.sin(np.arange(len(fechas)) / periodo)
        snapshots[activo] = pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1,
             "close": close, "adj_close": close, "volume": 1000.0}, index=fechas,
        )
    write_snapshot("sintetico", snapshots, base_dir=tmp_path)
    data = yaml.safe_load((Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml").read_text())
    data.update(activos=["SPY", "TLT"], parametros={"rsi_period": [7, 21], "oversold_threshold": [30.0]})
    archivo = tmp_path / "h.yaml"
    archivo.write_text(yaml.safe_dump(data))
    real, controles = hipotesis.validar_etapa2, []
    filtro, elegidos = hipotesis.filtrar_hipotesis, []

    def observar_filtro(*args):
        r = filtro(*args)
        elegidos.append(r[1])
        return r

    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", observar_filtro)

    def observar(h, snapshot, params, carpeta):
        controles.append((params, real(h, snapshot, params, carpeta)))
        return controles[-1][1]

    monkeypatch.setattr(hipotesis, "validar_etapa2", observar)
    monkeypatch.setattr(hipotesis, "validar_etapa3", lambda *args: (
        "PASA", [{"activo": "SPY", "v30": "PASA", "dsr": 0.95, "motivos": []}],
    ))
    if cortar:
        import subprocess
        run = subprocess.run

        def sin_validar(cmd, **kwargs):
            resultado = run(cmd, **kwargs)
            if "--tabla" in cmd:
                tabla = Path(cmd[cmd.index("--tabla") + 1])
                tabla.write_text(tabla.read_text().replace("coinciden", "sin validar"))
            return resultado

        monkeypatch.setattr(subprocess, "run", sin_validar)
    result = CliRunner().invoke(backtest_group, [
        "hipotesis", "correr", str(archivo), "--snapshot", "sintetico", "--carpeta", str(tmp_path / "motor"),
    ])
    assert result.exit_code == int(cortar), result.output
    params, (veredicto, resultados) = controles[0]
    assert params == elegidos[0] and set(params) == {"SPY", "TLT"}
    assert veredicto == ("CORTA" if cortar else "PASA")
    with (tmp_path / "intentos.csv").open() as f:
        intentos = list(csv.DictReader(f))
    assert [r["etapa"] for r in intentos] == ["1", "2", "2"]
    assert {r["activo"]: json.loads(r["params"]) for r in intentos[1:]} == params
    assert all((tmp_path / "motor" / f"{a}.csv").exists() for a in params)
    with (tmp_path / "hipotesis.csv").open() as f:
        filas = list(csv.DictReader(f))
    assert [(r["etapa"], r["veredicto"]) for r in filas] == [
        ("0", "PASA"), ("1", "PASA"), ("2", veredicto),
    ] + ([] if cortar else [("3", "PASA")])
    assert (tmp_path / "hipotesis-candado.csv").exists() == (not cortar)
    if cortar:
        assert all(m in filas[-1]["motivo"] for r in resultados for m in r["motivos"])
    else:
        activos = ", ".join(r["activo"] for r in resultados if r["v30"] == "PASA")
        assert filas[2]["motivo"] == f"pasan V30: {activos}; control coincide en 2 de 2"
    assert [line for line in result.output.splitlines() if not line.startswith("|")] == [
        *[f"Etapa {r['etapa']}: {r['veredicto']} ({r['motivo']})" for r in filas],
        f"Llegó a la etapa {filas[-1]['etapa']}: {veredicto} ({filas[-1]['motivo']})",
    ]


@pytest.mark.parametrize("rechazo", [False, True])
def test_correr_prueba_real_y_reapertura(tmp_path, monkeypatch, rechazo):
    pytest.importorskip("backtesting")
    import subprocess

    from quantagent.fabrica.candado import abierto

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2007-01-01", "2023-02-01", freq="2D")
    close = 100 + 10 * np.sin(np.arange(len(fechas)) / 5)
    close[fechas >= "2023-01-01"] = 1_000_000  # Centinelas fuera de la prueba.
    velas = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "adj_close": close, "volume": 1000.0}, index=fechas,
    )
    write_snapshot("sintetico", {"SPY": velas}, base_dir=tmp_path)
    data = yaml.safe_load((Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml").read_text())
    data.update(activos=["SPY"], parametros={"rsi_period": [7, 21], "oversold_threshold": [30.0]})
    archivo = tmp_path / "h.yaml"
    archivo.write_text(yaml.safe_dump(data))
    run, comandos = subprocess.run, []

    def observar(cmd, **kwargs):
        if "compare" in cmd:
            desde, hasta = (cmd[cmd.index(flag) + 1] for flag in ("--from", "--to"))
            prueba = desde == "2019-01-01"
            assert (desde, hasta) == (("2019-01-01", "2022-12-31") if prueba else
                                      ("2007-01-01", "2018-12-31"))
            assert ("--abrir-prueba" in cmd) == prueba
            assert not prueba or abierto(data["id"], "prueba")
            assert cmd[cmd.index("--symbols") + 1] == "SPY"
            assert cmd[cmd.index("--llenado") + 1] == "apertura"
            assert json.loads(Path(cmd[cmd.index("--params") + 1]).read_text()) == {
                "rsi": {"rsi_period": 21, "oversold_threshold": 30.0},
            }
            comandos.append((desde, hasta))
        return run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", observar)
    args = ["hipotesis", "correr", str(archivo), "--snapshot", "sintetico",
            "--carpeta", str(tmp_path / "motor")]
    runner = CliRunner()
    primera = runner.invoke(backtest_group, args)
    assert primera.exception is None or isinstance(primera.exception, SystemExit), primera.output
    with (tmp_path / "hipotesis.csv").open() as f:
        estados = list(csv.DictReader(f))
    assert [(r["etapa"], r["veredicto"]) for r in estados[:3]] == [
        ("0", "PASA"), ("1", "PASA"), ("2", "PASA"),
    ]
    assert len(estados) == 4 and estados[-1]["etapa"] == "3"
    assert primera.exit_code == int(estados[-1]["veredicto"] == "CORTA"), primera.output
    assert "V30" in primera.output and "DSR" in primera.output
    candado = (tmp_path / "hipotesis-candado.csv").read_text()
    assert [r[:2] for r in csv.reader(candado.splitlines())] == [[data["id"], "prueba"]]
    with (tmp_path / "intentos.csv").open() as f:
        intentos = list(csv.DictReader(f))
    assert [r["etapa"] for r in intentos] == ["1", "2", "3"]
    for carpeta, desde, hasta in [("motor", "2007-01-01", "2018-12-31"),
                                 ("motor/etapa3", "2019-01-01", "2022-12-31")]:
        curva = pd.read_csv(tmp_path / carpeta / "equity/rsi-SPY.csv")
        assert list(pd.to_datetime(curva["date"])) == list(velas.loc[desde:hasta].index)
    antes = {p: (tmp_path / p).read_bytes() for p in ("intentos.csv", "hipotesis.csv")}

    def filtro_prohibido(*args):
        pytest.fail("La reapertura no debe ejecutar el filtro")

    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", filtro_prohibido)
    if rechazo:
        data["justificacion"] = "corta"
        archivo.write_text(yaml.safe_dump(data))
    segunda = runner.invoke(backtest_group, args)
    fecha = next(csv.reader(candado.splitlines()))[2]
    assert segunda.exit_code == 1
    assert segunda.output == f"prueba de {data['id']} ya abierta el {fecha}; no se vuelve a correr\n"
    assert comandos == [("2007-01-01", "2018-12-31"), ("2019-01-01", "2022-12-31")]
    assert (tmp_path / "hipotesis-candado.csv").read_text() == candado
    assert {p: (tmp_path / p).read_bytes() for p in antes} == antes


@pytest.mark.parametrize("final", ["PASA", "CORTA"])
def test_correr_etapa3(tmp_path, monkeypatch, final):
    from quantagent.fabrica.candado import abierto
    from quantagent.fabrica.hipotesis import cargar_hipotesis

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    archivo = Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml"
    h = cargar_hipotesis(archivo)
    params = {"SPY": {"rsi_period": 7}}
    monkeypatch.setattr(hipotesis, "admitir_hipotesis", lambda *a: (h, True, []))
    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", lambda *a: ([["SPY", 1, 30]], params, "PASA", 0))
    monkeypatch.setattr(hipotesis, "validar_etapa2", lambda *a: (
        "PASA", [{"activo": "SPY", "v30": "PASA", "motivos": [], "sharpe_tasa_0": 1}],
    ))
    vistos = []

    def validar(h_recibida, snapshot, elegidos, carpeta):
        assert abierto(h.id, "prueba")
        assert (h_recibida, snapshot, elegidos, carpeta) == (h, "sintetico", params, tmp_path / "motor/etapa3")
        vistos.append(True)
        return final, [{"activo": "SPY", "v30": "PASA", "dsr": 0.95 if final == "PASA" else 0.94,
                        "motivos": [] if final == "PASA" else ["SPY: DSR < 0.95"]}]

    monkeypatch.setattr(hipotesis, "validar_etapa3", validar)
    result = CliRunner().invoke(backtest_group, ["hipotesis", "correr", str(archivo),
        "--snapshot", "sintetico", "--carpeta", str(tmp_path / "motor")])
    assert result.exit_code == int(final != "PASA"), result.output
    with (tmp_path / "hipotesis.csv").open() as f:
        filas = list(csv.DictReader(f))
    assert [f["etapa"] for f in filas] == ["0", "1", "2", "3"]
    assert filas[-1]["veredicto"] == ("PASA" if final == "PASA" else "CORTA")
    assert f"Llegó a la etapa 3: {filas[-1]['veredicto']} ({filas[-1]['motivo']})" in result.output
    assert vistos == [True]
    assert "V30" in result.output and "DSR" in result.output
    assert ("0.95" if final == "PASA" else "0.94") in result.output


def test_admision_rango_y_registro(tmp_path, monkeypatch):
    rechazo = True
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.to_datetime(["2006-12-31", "2007-01-01", "2010-06-01", "2018-12-31", "2019-01-01"])
    velas = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "adj_close", "volume")}, index=fechas)
    write_snapshot("sintetico", {"SPY": velas}, base_dir=tmp_path)
    data = yaml.safe_load((Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml").read_text())
    if rechazo:
        data.update(justificacion="corta", activos=["SPY", "AAPL"])
    archivo = tmp_path / "h.yaml"
    archivo.write_text(yaml.safe_dump(data))
    real = hipotesis.admitir
    vistos = []

    def observar(h, recibidas):
        vistos.extend(recibidas.index)
        return real(h, recibidas)

    monkeypatch.setattr(hipotesis, "admitir", observar)
    admision, admisiones = hipotesis.admitir_hipotesis, []

    def observar_admision(*args):
        r = admision(*args)
        admisiones.append(r)
        return r

    monkeypatch.setattr(hipotesis, "admitir_hipotesis", observar_admision)
    runner = CliRunner()
    out = tmp_path / "rechazada.csv"
    result = runner.invoke(backtest_group, ["hipotesis", "filtrar", str(archivo), "--snapshot", "sintetico", "--out", str(out)])
    assert not out.exists()
    assert result.exit_code == int(rechazo), result.output
    assert vistos == list(fechas[1:4])
    assert "PASA" not in result.output
    h, ok, motivos = admisiones[0]
    assert h.id == data["id"] and not ok
    assert motivos and all(motivo in result.output for motivo in motivos)
    if rechazo:
        assert "justificación" in result.output and "AAPL" in result.output
        with (tmp_path / "intentos.csv").open() as f:
            filas = list(csv.DictReader(f))
        assert len(filas) == 1 and filas[0]["hipotesis"] == data["id"] and filas[0]["etapa"] == "0"
    ayuda = runner.invoke(backtest_group, ["hipotesis", "filtrar", "--help"])
    assert ayuda.exit_code == 0 and "--snapshot" in ayuda.output and "ARCHIVO" in ayuda.output


@pytest.mark.parametrize("exportar", [False, True])
def test_filtro_real_dos_activos(tmp_path, monkeypatch, exportar):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2006-12-01", "2019-02-01", freq="7D")
    snapshots = {}
    for activo, periodo in [("SPY", 5), ("TLT", 9)]:
        close = 100 + 10 * np.sin(np.arange(len(fechas)) / periodo)
        snapshots[activo] = pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1,
             "close": close, "adj_close": close, "volume": 1000.0}, index=fechas,
        )
    write_snapshot("sintetico", snapshots, base_dir=tmp_path)
    data = yaml.safe_load((Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml").read_text())
    data.update(activos=["SPY", "TLT"], parametros={"rsi_period": [7], "oversold_threshold": [30.0]})
    archivo = tmp_path / "h.yaml"
    archivo.write_text(yaml.safe_dump(data))
    real, resultados = hipotesis.filtrar_activo, []

    def observar(h, velas, **kwargs):
        esperado = snapshots[h.activos[len(resultados)]].loc["2007":"2018"]
        assert list(velas.timestamp) == list(esperado.index)
        assert kwargs == {"costos": hipotesis.COSTOS_PERFILES["etf"], "capital": 100_000.0}
        r = real(h, velas, **kwargs)
        resultados.append(r)
        return r

    monkeypatch.setattr(hipotesis, "filtrar_activo", observar)
    filtro, filtros = hipotesis.filtrar_hipotesis, []

    def observar_filtro(*args):
        r = filtro(*args)
        filtros.append(r)
        return r

    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", observar_filtro)
    out = tmp_path / "filtro.csv"
    args = ["--out", str(out)] if exportar else []
    result = CliRunner().invoke(backtest_group, ["hipotesis", "filtrar", str(archivo), "--snapshot", "sintetico", *args])
    assert result.exit_code == 0, result.output
    assert "CORTA" in result.output
    with (tmp_path / "intentos.csv").open() as f:
        filas = list(csv.DictReader(f))
    assert len(filas) == 1 and filas[0]["activo"] == "cartera"
    for activo, r in zip(data["activos"], resultados):
        celdas = next(line for line in result.output.splitlines() if f" {activo} " in line).split("|")
        assert float(celdas[2]) == pytest.approx(r["sharpe_tasa_0"], rel=1e-4)
        assert int(celdas[3]) == r["trades"]
    assert out.exists() == exportar
    metricas, parametros, veredicto, motivo = filtros[0][:4]
    assert metricas == [[a, r["sharpe_tasa_0"], r["trades"]] for a, r in zip(data["activos"], resultados)]
    assert veredicto == "CORTA"
    assert parametros == {a: hipotesis.parametros_elegidos(r["ganadores"]) for a, r in zip(data["activos"], resultados)}
    if exportar:
        with out.open() as f:
            filas = list(csv.DictReader(f))
        assert len(filas) == 2
        for fila, activo, r in zip(filas, data["activos"], resultados):
            assert list(fila) == ["activo", "sharpe_fuera_muestra", "trades", "params", "veredicto"]
            assert fila["activo"] == activo and fila["veredicto"] == "CORTA"
            assert float(fila["sharpe_fuera_muestra"]) == r["sharpe_tasa_0"]
            assert int(fila["trades"]) == r["trades"]
            assert fila["params"] == json.dumps(hipotesis.parametros_elegidos(r["ganadores"]), sort_keys=True)
            assert json.loads(fila["params"]) == parametros[activo]


def test_etapa1_cartera_12_activos_5_trades_pasa(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    activos, fechas = ["SPY", "IWM", "EFA", "EEM", "TLT", "SHY", "GLD", "DBC", "XLE", "XLF", "XLK", "XLU"], pd.date_range("2007-01-01", periods=100)
    velas = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "adj_close", "volume")}, index=fechas)
    write_snapshot("sintetico", dict.fromkeys(activos, velas), base_dir=tmp_path)
    data = yaml.safe_load((Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml").read_text())
    data.update(activos=activos, parametros={"rsi_period": [7], "oversold_threshold": [30.0]})
    archivo = tmp_path / "h.yaml"
    archivo.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(hipotesis, "filtrar_activo", lambda *a, **kw: {
        "sharpe_tasa_0": 0.5, "trades": 5, "ganadores": [{"params": {}, "prueba_desde": fechas[0]}],
        "retornos": pd.Series(0.001, index=fechas[60:]),
    })
    res = CliRunner().invoke(backtest_group, ["hipotesis", "filtrar", str(archivo), "--snapshot", "sintetico"])
    assert res.exit_code == 0 and "PASA" in res.output
    with (tmp_path / "intentos.csv").open() as f:
        filas = list(csv.DictReader(f))
    assert len(filas) == 1 and filas[0]["activo"] == "cartera" and int(filas[0]["trades"]) == 60


def test_etapa1_cartera_dos_piernas_tres_activos_un_intento(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    activos, fechas = ["SPY", "TLT", "GLD"], pd.date_range("2007-01-01", periods=100)
    velas = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "adj_close", "volume")}, index=fechas)
    write_snapshot("sintetico", dict.fromkeys(activos, velas), base_dir=tmp_path)
    p1 = {"clase": "rsi", "parametros": {"rsi_period": [7], "oversold_threshold": [30.0]}, "activos": activos, "universo": "M"}
    p2 = {"clase": "sma-cross", "parametros": {"fast": [20], "slow": [200]}, "activos": activos, "universo": "M"}
    archivo = tmp_path / "c.yaml"
    archivo.write_text(yaml.safe_dump({
        "id": "c-2x3", "autor": "fede", "timeframe": "1d",
        "justificacion": "Combinar tendencia y momentum diversifica el riesgo en ETFs.",
        "piernas": [p1, p2],
    }))
    monkeypatch.setattr(hipotesis, "filtrar_activo", lambda *a, **kw: {
        "sharpe_tasa_0": 0.5, "trades": 10, "ganadores": [{"params": {}, "prueba_desde": fechas[0]}],
        "retornos": pd.Series(0.001, index=fechas[60:]),
    })
    res = CliRunner().invoke(backtest_group, ["hipotesis", "filtrar", str(archivo), "--snapshot", "sintetico"])
    assert res.exit_code == 0
    with (tmp_path / "intentos.csv").open() as f:
        filas = list(csv.DictReader(f))
    assert len(filas) == 1 and filas[0]["activo"] == "cartera" and filas[0]["etapa"] == "1" and int(filas[0]["trades"]) == 60


def test_formato_viejo_una_pierna_coincide_con_cartera(tmp_path, monkeypatch):
    from quantagent.fabrica.hipotesis import cargar_hipotesis

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2007-01-01", periods=100)
    velas = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "adj_close", "volume")}, index=fechas)
    write_snapshot("sintetico", {"SPY": velas}, base_dir=tmp_path)
    monkeypatch.setattr(hipotesis, "filtrar_activo", lambda *a, **kw: {
        "sharpe_tasa_0": 0.55, "trades": 35, "ganadores": [{"params": {}, "prueba_desde": fechas[0]}],
        "retornos": pd.Series(0.001, index=fechas[60:]),
    })
    p = {"clase": "rsi", "activos": ["SPY"], "universo": "U", "parametros": {"rsi_period": [7], "oversold_threshold": [30.0]}}
    f_v, f_n = tmp_path / "v.yaml", tmp_path / "n.yaml"
    f_v.write_text(yaml.safe_dump({"id": "v", "autor": "f", "justificacion": "Formato viejo de hipótesis de prueba.", "timeframe": "1d", **p}))
    f_n.write_text(yaml.safe_dump({"id": "n", "autor": "f", "justificacion": "Formato nuevo de hipótesis de prueba.", "timeframe": "1d", "piernas": [p]}))
    r_v = hipotesis.filtrar_hipotesis(cargar_hipotesis(f_v), "sintetico")
    r_n = hipotesis.filtrar_hipotesis(cargar_hipotesis(f_n), "sintetico")
    assert r_v[2] == r_n[2] == "PASA" and r_v[4] == pytest.approx(r_n[4]) and r_v[5] == r_n[5] == 35
