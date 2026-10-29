import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from quantagent.backtesting.vbt_filter import equity_metrics
from quantagent.cli import hipotesis
from quantagent.cli.backtest import backtest_group
from quantagent.data.macro import crear_snapshot_macro
from quantagent.data.snapshot import write_snapshot
from quantagent.fabrica.cartera import combinar, pesos
from quantagent.fabrica.hipotesis import Hipotesis, Pierna
from quantagent.strategy.rebalanceo_fin_de_mes_strategy import RebalanceoFinDeMesStrategy
from tests.test_filtrar_pierna import resultado_ganadores_distintos as resultado_ganadores_distintos


def test_reserva_simulacro_valida_y_no_lee_datos(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    archivo = Path(__file__).parents[1] / "hipotesis/rebalanceo-fin-de-mes.yaml"
    fechas = pd.bdate_range("2022-12-30", "2024-01-02")
    velas = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "adj_close", "volume")}, index=fechas)
    write_snapshot("reserva-sintetica", {"SPY": velas, "IEF": velas}, base_dir=tmp_path)
    monkeypatch.setattr(hipotesis, "ultimo_estado", lambda _: {"etapa": "2", "veredicto": "PASA"})
    monkeypatch.setattr(hipotesis, "fecha_apertura", lambda *_: None)
    monkeypatch.setattr(hipotesis, "read_snapshot", lambda *a, **k: pytest.fail("simulacro leyó velas"))
    result = CliRunner().invoke(hipotesis.hipotesis_group, ["reserva", str(archivo), "--snapshot", "reserva-sintetica"])
    assert result.exit_code == 0, result.output
    assert "reserva 2023-01-01 a 2024-01-02" in result.output
    assert "no se abrió ni leyó" in result.output
    assert not (tmp_path / "hipotesis-candado.csv").exists()


def test_reserva_rechaza_sin_etapa_dos_pasa(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    archivo = Path(__file__).parents[1] / "hipotesis/rebalanceo-fin-de-mes.yaml"
    monkeypatch.setattr(hipotesis, "ultimo_estado", lambda _: {"etapa": "1", "veredicto": "PASA"})
    result = CliRunner().invoke(hipotesis.hipotesis_group, ["reserva", str(archivo), "--snapshot", "ausente"])
    assert result.exit_code == 1
    assert "no tiene etapa 2 PASA" in result.output


def test_reserva_rechaza_fuera_de_lista_y_apertura_previa(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fuente = yaml.safe_load((Path(__file__).parents[1] / "hipotesis/rebalanceo-fin-de-mes.yaml").read_text())
    fuente["id"] = "cambio-de-mes"
    archivo = tmp_path / "fuera.yaml"
    archivo.write_text(yaml.safe_dump(fuente))
    monkeypatch.setattr(hipotesis, "ultimo_estado", lambda _: {"etapa": "2", "veredicto": "PASA"})
    runner = CliRunner()
    fuera = runner.invoke(hipotesis.hipotesis_group, ["reserva", str(archivo), "--snapshot", "ausente"])
    assert "fuera de la lista" in fuera.output
    archivo = Path(__file__).parents[1] / "hipotesis/rebalanceo-fin-de-mes.yaml"
    candado = tmp_path / "hipotesis-candado.csv"
    original = "rebalanceo-fin-de-mes,reserva,2026-01-01\n"
    candado.write_text(original)
    abierta = runner.invoke(hipotesis.hipotesis_group, ["reserva", str(archivo), "--snapshot", "ausente"])
    assert "ya fue abierta" in abierta.output
    assert candado.read_text() == original


def test_reserva_abrir_pre_registra_y_rechaza_segunda_apertura(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("QUANTAGENT_RESERVA_DESDE", "2023-01-01")
    archivo = Path(__file__).parents[1] / "hipotesis/rebalanceo-fin-de-mes.yaml"
    fechas = pd.bdate_range("2022-12-30", "2024-01-02")
    velas = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "adj_close", "volume")}, index=fechas)
    write_snapshot("reserva-sintetica", {"SPY": velas, "IEF": velas}, base_dir=tmp_path)
    monkeypatch.setattr(hipotesis, "ultimo_estado", lambda _: {"etapa": "2", "veredicto": "PASA"})
    abiertas, orden = set(), []
    monkeypatch.setattr(hipotesis, "fecha_apertura", lambda h, p: "sintética" if (h, p) in abiertas else None)

    def observar_apertura(hipotesis_id, periodo):
        preregistro = (tmp_path / "reserva-preregistro.log").read_text()
        orden.append((hipotesis_id in preregistro, "elegible" in preregistro,
                      not (tmp_path / "reserva-accesos.log").exists()))
        abiertas.add((hipotesis_id, periodo))

    monkeypatch.setattr(hipotesis, "abrir", observar_apertura)
    monkeypatch.setattr(hipotesis, "validar_etapa3", lambda *a, **k: ("PASA", []))
    monkeypatch.setattr("quantagent.fabrica.informe.diagnostico_exceso", lambda *a, **k: {
        "disponible": True, "sharpe_cartera": 0.0, "sharpe_benchmark": 0.0})
    runner = CliRunner()
    args = ["reserva", str(archivo), "--snapshot", "reserva-sintetica", "--abrir"]
    primera = runner.invoke(hipotesis.hipotesis_group, args)
    assert primera.exit_code == 0, primera.output
    assert orden == [(True, True, True)]
    log = (tmp_path / "reserva-accesos.log").read_text()
    assert "hipotesis reserva" in log and "--abrir" in log
    segunda = runner.invoke(hipotesis.hipotesis_group, args)
    assert segunda.exit_code == 1
    assert "ya fue abierta" in segunda.output
    assert (tmp_path / "reserva-accesos.log").read_text() == log


def test_reserva_abierta_imprime_c0_m1_y_registra_intento(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("QUANTAGENT_RESERVA_DESDE", "2023-01-01")
    archivo = Path(__file__).parents[1] / "hipotesis/rebalanceo-fin-de-mes.yaml"
    fechas = pd.bdate_range("2022-12-30", "2024-01-02")
    velas = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "adj_close", "volume")}, index=fechas)
    write_snapshot("reserva-sintetica", {"SPY": velas, "IEF": velas}, base_dir=tmp_path)
    monkeypatch.setattr(hipotesis, "ultimo_estado", lambda _: {"etapa": "2", "veredicto": "PASA"})
    monkeypatch.setattr(hipotesis, "fecha_apertura", lambda *_: None)
    monkeypatch.setattr(hipotesis, "abrir", lambda *_: None)

    from quantagent.data.macro import crear_snapshot_macro
    filas_macro = [(pd.Timestamp(f"{year}-01-01"), 2.0, pd.Timestamp(f"{year}-01-01"),
                    pd.Timestamp("9999-12-31")) for year in (2023, 2024)]
    macro = pd.DataFrame(filas_macro, columns=["fecha", "valor", "realtime_start", "realtime_end"])

    class ClienteSintetico:
        def observaciones(self, *args, **kwargs):
            return macro

    crear_snapshot_macro("macro-2026-10", ["DGS3MO"], ClienteSintetico(), base_dir=tmp_path)

    def correr_motor(h, snapshot, parametros, desde, hasta, carpeta, etapa, **kwargs):
        eq_dir = carpeta / "equity"
        eq_dir.mkdir(parents=True)
        fechas_eq = pd.bdate_range(desde, periods=4)
        from quantagent.cli.sweep import registrar_intento
        for activo in ("SPY", "IEF"):
            pd.DataFrame({"date": fechas_eq, "equity": [100000, 101000, 100500, 102000]}).to_csv(
                eq_dir / f"{h.clase}-{activo}.csv", index=False)
            registrar_intento(h.clase, activo, snapshot, "etf", 1.0, {},
                              {"desde": desde, "hasta": hasta, "trades": 3, "pnl": 100, "sharpe": 0.3},
                              hipotesis=h.id, etapa=etapa)
        return [{"activo": a, "trades": "3", "pnl": "100", "sharpe_tasa_0": "0.3",
                 "max_drawdown": "0.01"} for a in ("SPY", "IEF")]

    from quantagent.fabrica import validacion
    monkeypatch.setattr(validacion, "correr_motor", correr_motor)
    result = CliRunner().invoke(hipotesis.hipotesis_group,
                                ["reserva", str(archivo), "--snapshot", "reserva-sintetica", "--abrir"])
    assert result.exit_code == 0, result.output
    assert "C0: CORTA" in result.output and "Sharpe en exceso de DGS3MO" in result.output
    with (tmp_path / "intentos.csv").open() as f:
        intentos = list(csv.DictReader(f))
    intento = next(f for f in intentos if f["sharpe_base"] == "DGS3MO")
    assert intento["activo"] == "cartera" and intento["etapa"] == "3"
    assert intento["desde"] == "2023-01-01" and intento["hasta"] == "2024-01-02"
    assert "CORTA" in (tmp_path / "hipotesis.csv").read_text()


def test_reserva_veredicto_recupera_m1_una_vez_desde_curvas(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2023-01-02", periods=4)
    carpeta = tmp_path / "reserva" / "rebalanceo-fin-de-mes" / "equity"
    carpeta.mkdir(parents=True)
    for activo, valores in (("SPY", [100, 102, 103, 106]), ("IEF", [100, 101, 103, 104])):
        pd.DataFrame({"date": fechas, "equity": valores}).to_csv(carpeta / f"{activo}.csv", index=False)
    trades_dir = tmp_path / "reserva" / "rebalanceo-fin-de-mes" / "trades"
    trades_dir.mkdir()
    (trades_dir / "rebalanceo-fin-de-mes-SPY.csv").write_text("date,symbol\n2023-01-02,SPY\n2023-01-03,SPY\n")
    frame = pd.DataFrame({a: [100, 101, 102, 103] for a in ("SPY", "IEF")}, index=fechas)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda *a, **k: frame.assign(close=frame[a[1]]))
    monkeypatch.setattr(hipotesis, "fecha_apertura", lambda *_: "ya abierta")
    monkeypatch.setattr(hipotesis, "abrir", lambda *_: pytest.fail("abrir llamado"))
    from quantagent.fabrica import validacion
    monkeypatch.setattr(validacion, "correr_motor", lambda *a, **k: pytest.fail("motor llamado"))
    from quantagent.data.macro import _escribir
    macro = pd.DataFrame({"fecha": [fechas[0]], "realtime_start": [fechas[0]],
                          "realtime_end": [pd.Timestamp("2099-01-01")], "valor": [0.0]})
    _escribir(tmp_path / "macro-2026-10", "macro-2026-10", {"DGS3MO": macro})
    runner = CliRunner()
    args = ["reserva-veredicto", str(Path(__file__).parents[1] / "hipotesis/rebalanceo-fin-de-mes.yaml"),
            "--snapshot", "reserva-sintetica"]
    primera = runner.invoke(hipotesis.hipotesis_group, args)
    segunda = runner.invoke(hipotesis.hipotesis_group, args)
    assert primera.exit_code == 0, primera.output
    assert segunda.exit_code == 0 and "ya recuperada" in segunda.output
    from quantagent.fabrica.criterio import evaluar_etapa3_cartera, sharpe_benchmark
    h = hipotesis.cargar_hipotesis(args[1])
    retornos = pd.DataFrame({"SPY": pd.Series([100, 102, 103, 106], index=fechas).pct_change(),
                             "IEF": pd.Series([100, 101, 103, 104], index=fechas).pct_change()}).dropna().sum(axis=1)
    sharpe_m1 = retornos.iloc[1:]
    esperado = sharpe_m1.mean() / sharpe_m1.std() * (365.25 ** 0.5)
    cierres = pd.DataFrame({a: frame[a] for a in ("SPY", "IEF")})
    benchmark = (cierres / cierres.iloc[0]).mean(axis=1)
    ret_benchmark = benchmark.pct_change().dropna().iloc[1:]
    esperado_benchmark = ret_benchmark.mean() / ret_benchmark.std() * (365.25 ** 0.5)
    cierres_etapa2 = pd.DataFrame({a: frame[a].loc["2007-01-01":"2018-12-31"] for a in ("SPY", "IEF")})
    eq_cartera = hipotesis.curva_cartera_guardada(h, tmp_path / "reserva" / h.id, None)
    pasa, _ = evaluar_etapa3_cartera(eq_cartera, sharpe_benchmark(cierres), None,
                                     sharpe_benchmark(cierres_etapa2) if not cierres_etapa2.empty else None,
                                     trades=2)
    intentos = list(csv.DictReader((tmp_path / "intentos.csv").open()))
    assert len(intentos) == 1 and float(intentos[0]["sharpe"]) == pytest.approx(esperado, abs=0.01)
    assert len([r for r in csv.DictReader((tmp_path / "hipotesis.csv").open()) if r["etapa"] == "3"]) == 1
    assert "M1 cartera" in primera.output
    assert f"C0: {'PASA' if pasa else 'CORTA'}" in primera.output
    assert f"M1 benchmark {esperado_benchmark:.2f}" in primera.output and "trades 2" in primera.output


def test_curva_guardada_posiciones_objetivo_suma_retornos_y_benchmark_50_50(tmp_path):
    from types import SimpleNamespace
    from quantagent.fabrica.validacion import curva_cartera_guardada
    from quantagent.fabrica.criterio import curva_benchmark

    fechas = pd.date_range("2023-01-02", periods=3)
    carpeta = tmp_path / "equity"
    carpeta.mkdir()
    for activo, valores in (("SPY", [100_000, 110_000, 121_000]), ("IEF", [100_000, 90_000, 81_000])):
        pd.DataFrame({"date": fechas, "equity": valores}).to_csv(carpeta / f"{activo}.csv", index=False)
    pierna = SimpleNamespace(clase="rebalanceo-fin-de-mes", activos=["SPY", "IEF"])
    h = SimpleNamespace(piernas=[pierna])
    curva = curva_cartera_guardada(h, tmp_path, None)
    retornos = pd.Series([0.1 + (-0.1), 0.1 + (-0.1)], index=fechas[1:])
    pd.testing.assert_series_equal(curva, 100_000 * (1 + retornos).cumprod(), check_names=False, check_freq=False)
    cierres = pd.DataFrame({"SPY": [100, 110, 121], "IEF": [100, 90, 81]}, index=fechas)
    benchmark = curva_benchmark(cierres, pesos=pd.DataFrame(0.5, index=fechas, columns=cierres.columns))
    esperado_benchmark = 100_000 * (cierres / cierres.iloc[0]).mul(0.5).sum(axis=1)
    pd.testing.assert_series_equal(benchmark, esperado_benchmark, check_names=False)


def test_tendencia_larga_corta_aprobada_pasa_admision_con_snapshots_sinteticos(tmp_path, monkeypatch):
    datos = tmp_path / "datos"
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(datos))
    propuesta = Path(__file__).parents[1] / "hipotesis/tendencia-larga-corta.yaml"
    hipotesis_dir = tmp_path / "hipotesis"
    hipotesis_dir.mkdir()
    (hipotesis_dir / propuesta.name).write_text(propuesta.read_text(encoding="utf-8"), encoding="utf-8")
    fechas = pd.bdate_range("2007-01-01", "2018-12-31")
    precios = np.linspace(80, 120, len(fechas))
    velas = pd.DataFrame({"open": precios, "high": precios + 1, "low": precios - 1,
                          "close": precios, "adj_close": precios, "volume": 1000.0}, index=fechas)
    activos = yaml.safe_load((hipotesis_dir / propuesta.name).read_text())['activos']
    write_snapshot("etf-1d-2026-10-amplio", dict.fromkeys(activos, velas), base_dir=datos)

    class ClienteSintetico:
        def observaciones(self, serie, desde, hasta, todas_las_versiones=True):
            filas = [(pd.Timestamp(f"{anio}-01-01"), 2.0, pd.Timestamp(f"{anio}-01-01"),
                      pd.Timestamp("9999-12-31")) for anio in range(2007, 2020)]
            return pd.DataFrame(filas, columns=["fecha", "valor", "realtime_start", "realtime_end"])

    crear_snapshot_macro("macro-2026-10", ["DGS3MO"], ClienteSintetico(), base_dir=datos)
    h, ok, motivos = hipotesis.admitir_hipotesis(
        hipotesis_dir / propuesta.name, "etf-1d-2026-10-amplio")
    assert ok, motivos
    assert h.clase == "tsmom-largo-corto"
    assert h.parametros == {"snapshot": ["macro-2026-10"]}


def test_rebalanceo_aprobado_pasa_admision_sin_parametros(tmp_path, monkeypatch):
    datos = tmp_path / "datos"
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(datos))
    propuestas_dir, hipotesis_dir = tmp_path / "propuestas", tmp_path / "hipotesis"
    propuestas_dir.mkdir()
    propuesta = Path(__file__).parents[1] / "hipotesis/rebalanceo-fin-de-mes.yaml"
    (propuestas_dir / propuesta.name).write_text(propuesta.read_text(encoding="utf-8"), encoding="utf-8")
    fechas = pd.bdate_range("2007-01-01", "2018-12-31")
    precios = np.linspace(80, 120, len(fechas))
    velas = pd.DataFrame({"open": precios, "high": precios + 1, "low": precios - 1,
                          "close": precios, "adj_close": precios, "volume": 1000.0}, index=fechas)
    write_snapshot("etf-1d-2026-10-amplio", {"SPY": velas, "IEF": velas}, base_dir=datos)
    aprobado = CliRunner().invoke(backtest_group, ["hipotesis", "propuestas", "aprobar", "rebalanceo-fin-de-mes",
        "--quien", "fede", "--propuestas-dir", str(propuestas_dir), "--hipotesis-dir", str(hipotesis_dir)])
    assert aprobado.exit_code == 0, aprobado.output
    h, ok, motivos = hipotesis.admitir_hipotesis(hipotesis_dir / propuesta.name, "etf-1d-2026-10-amplio")
    assert ok, motivos
    assert h.parametros == {}


@pytest.mark.parametrize("calendario_desigual", [False, True])
def test_filtrar_hipotesis_transporta_seleccion_oos_al_calculo_real(monkeypatch, resultado_ganadores_distintos, calendario_desigual):
    h, p, paneles, r = resultado_ganadores_distintos
    snapshots = {a: v.drop(columns="timestamp") for a, v in paneles.items()}
    if calendario_desigual:
        snapshots["TLT"] = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "volume")},
                                        index=pd.bdate_range("2009-01-02", "2018-12-31"))
        h = Hipotesis("t", "a", "x" * 50, piernas=(p, Pierna("halloween", {}, ("TLT",), "sintético")))
    monkeypatch.setattr(hipotesis, "read_snapshot", lambda snap, activo: snapshots[activo])
    evaluaciones, evaluar = [], hipotesis.evaluar_cartera

    def observar(cierres, piernas, retornos, *, seleccion, semanales=()):
        assert set(seleccion) == {0}
        pd.testing.assert_frame_equal(seleccion[0], r["seleccion"].reindex(cierres.index, fill_value=0))
        w = pesos(cierres, piernas, seleccion=seleccion)
        meses = w.index.to_period("M")
        inicio_mes = ~meses.duplicated()
        mascara = seleccion[0].reindex(w.index).eq(0).loc[inicio_mes]
        assert w[0].loc[inicio_mes].where(mascara).fillna(0).eq(0).all().all()
        df_ret = pd.DataFrame(retornos).fillna(0.0)
        referencia = 100_000 * (1 + combinar(df_ret, w.reindex(df_ret.index))).cumprod()
        evaluaciones.append(evaluar(cierres, piernas, retornos, seleccion=seleccion))
        assert evaluaciones[-1] == pytest.approx(equity_metrics(referencia, 0.0)["sharpe"])
        return evaluaciones[-1]

    monkeypatch.setattr(hipotesis, "evaluar_cartera", observar)
    resultado = hipotesis.filtrar_hipotesis(h, "sintetico")
    assert resultado[4] == evaluaciones[0]
    esperados = {(p.clase, a) if h.es_cartera else a: hipotesis.parametros_elegidos(r["ganadores"]) for a in p.activos}
    if calendario_desigual:
        esperados["halloween", "TLT"] = {}
    assert resultado[1] == esperados
    assert resultado[5] >= r["trades"]


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
    data.update(fidelidad="desvío: solo largo", muestra_de_la_cita="1985-2012")
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
        "Advertencia etapa 0 (no bloquea): desvio_de_la_cita",
        "Advertencia etapa 0 (no bloquea): muestra_solapada",
        "Advertencia etapa 0 (no bloquea): overlay_solo_largo",
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
        "Advertencia etapa 0 (no bloquea): overlay_solo_largo",
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


def test_etapa1_pesos_objetivo_pasa_panel_y_suma_ordenes(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.bdate_range("2007-01-01", "2018-12-31")
    x = np.arange(len(fechas))
    cierres = pd.DataFrame({"SPY": 100 * np.exp(x * 0.0002 + 0.03 * np.sin(x / 18)),
                            "IEF": 100 * np.exp(x * 0.0001 - 0.02 * np.sin(x / 18))}, index=fechas)
    snapshots = {a: pd.DataFrame({c: cierres[a] for c in ("open", "high", "low", "close", "adj_close")}
                                 | {"volume": 1.0}, index=fechas) for a in ("SPY", "IEF")}
    write_snapshot("sintetico", snapshots, base_dir=tmp_path)
    h = Hipotesis(id="f1", autor="test", justificacion="test", clase="rebalanceo-fin-de-mes", activos=("SPY", "IEF"))
    objetivos = pd.DataFrame(0.5, index=fechas, columns=["SPY", "IEF"])
    cambios_esperados = []
    for mes in pd.period_range("2011-01", "2018-12", freq="M"):
        ultimos = fechas[fechas.to_period("M") == mes][-5:]
        for fecha in ultimos[[0, 2, 4]]:
            cambios_esperados.append(fecha)
            objetivos.loc[fecha:] = (1.0, 0.0) if len(cambios_esperados) % 2 else (0.0, 1.0)
    monkeypatch.setattr(RebalanceoFinDeMesStrategy, "posiciones_objetivo", staticmethod(lambda _: objetivos))
    resultado = hipotesis.filtrar_hipotesis(h, "sintetico")
    assert [r[0] for r in resultado[0]] == ["SPY", "IEF"]
    assert len(cambios_esperados) == 288
    assert [r[2] for r in resultado[0]] == [288, 288]
    assert resultado[5] == sum(r[2] for r in resultado[0]) == 576


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


@pytest.mark.parametrize("final,dsr,motivo_3", [
    ("PASA", 0.96, "cartera pasa V30 y DSR"),
    ("CORTA", 0.94, "DSR de cartera < 0.95"),
])
def test_correr_cartera_etapa3_y_reapertura(tmp_path, monkeypatch, final, dsr, motivo_3):
    from quantagent.fabrica.candado import abierto
    from quantagent.fabrica.hipotesis import cargar_hipotesis

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    velas = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "adj_close": [100.0], "volume": [1000.0]},
        index=pd.date_range("2007-01-01", periods=1),
    )
    write_snapshot("sintetico", {"SPY": velas, "TLT": velas}, base_dir=tmp_path)
    data = {
        "id": f"c-c07-{final.lower()}", "autor": "fede", "timeframe": "1d",
        "justificacion": "Combinar tendencia y momentum diversifica el riesgo en ETFs.",
        "piernas": [
            {"clase": "rsi", "parametros": {"rsi_period": [7]}, "activos": ["SPY"], "universo": "U"},
            {"clase": "sma-cross", "parametros": {"fast": [20]}, "activos": ["TLT"], "universo": "U"},
        ],
    }
    archivo = tmp_path / "c.yaml"
    archivo.write_text(yaml.safe_dump(data))
    h = cargar_hipotesis(archivo)
    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", lambda *a, **kw: (
        [["SPY", 1.0, 30], ["TLT", 1.0, 30]], {("rsi", "SPY"): {}, ("sma-cross", "TLT"): {}}, "PASA", "sharpe 1.0 >= 0.10 y 60 trades >= 30", 1.0, 60, [],
    ))
    monkeypatch.setattr(hipotesis, "validar_etapa2", lambda *a, **kw: ("PASA", [
        {"activo": "SPY", "v30": "PASA", "motivos": [], "sharpe_cartera": 0.5},
        {"activo": "TLT", "v30": "PASA", "motivos": [], "sharpe_cartera": 0.5},
    ]))
    vistos_etapa3 = []

    def fake_etapa3(h_obj, snap, params, carp):
        vistos_etapa3.append(carp)
        motivos = [] if final == "PASA" else ["DSR de cartera < 0.95"]
        return final, [
            {"activo": "SPY", "estrategia": "rsi", "v30": "PASA", "dsr": dsr, "motivos": motivos},
            {"activo": "TLT", "estrategia": "sma-cross", "v30": "PASA", "dsr": dsr, "motivos": motivos},
        ]

    monkeypatch.setattr(hipotesis, "validar_etapa3", fake_etapa3)

    runner = CliRunner()
    res_correr = runner.invoke(backtest_group, [
        "hipotesis", "correr", str(archivo), "--snapshot", "sintetico", "--carpeta", str(tmp_path / "salida_correr"),
    ])
    assert res_correr.exit_code == int(final != "PASA"), res_correr.output
    assert f"Etapa 3: {final} ({motivo_3}" in res_correr.output
    assert f"dsr_efectivo={dsr:.2f}" in res_correr.output
    assert f"Llegó a la etapa 3: {final}" in res_correr.output
    assert abierto(h.id, "prueba")
    assert len(vistos_etapa3) == 1

    if final == "PASA":
        res_segunda = runner.invoke(backtest_group, [
            "hipotesis", "correr", str(archivo), "--snapshot", "sintetico", "--carpeta", str(tmp_path / "salida_segunda"),
        ])
        assert res_segunda.exit_code == 1, res_segunda.output
        assert f"prueba de {h.id} ya abierta el" in res_segunda.output
        assert "no se vuelve a correr" in res_segunda.output


def test_correr_cartera_alarma(tmp_path, monkeypatch):
    from quantagent.fabrica.hipotesis import cargar_hipotesis

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    data = {
        "id": "c-alarma", "autor": "fede", "timeframe": "1d",
        "justificacion": "Combinar tendencia y momentum diversifica el riesgo en ETFs.",
        "piernas": [
            {"clase": "rsi", "parametros": {"rsi_period": [7]}, "activos": ["SPY"], "universo": "U"},
            {"clase": "sma-cross", "parametros": {"fast": [20]}, "activos": ["TLT"], "universo": "U"},
        ],
    }
    archivo = tmp_path / "c_al.yaml"
    archivo.write_text(yaml.safe_dump(data))
    h = cargar_hipotesis(archivo)
    monkeypatch.setattr(hipotesis, "admitir_hipotesis", lambda *a, **kw: (h, True, []))
    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", lambda *a, **kw: (
        [["SPY", 1.0, 30], ["TLT", 1.0, 30]], {("rsi", "SPY"): {}, ("sma-cross", "TLT"): {}}, "PASA", "sharpe 1.0 >= 0.10 y 60 trades >= 30", 0.5, 60, [],
    ))
    monkeypatch.setattr(hipotesis, "validar_etapa2", lambda *a, **kw: ("PASA", [
        {"activo": "SPY", "v30": "PASA", "motivos": [], "sharpe_cartera": -0.1},
        {"activo": "TLT", "v30": "PASA", "motivos": [], "sharpe_cartera": -0.1},
    ]))
    monkeypatch.setattr(hipotesis, "validar_etapa3", lambda *a, **kw: pytest.fail("etapa 3 no debe llamarse"))
    monkeypatch.setattr(hipotesis, "abrir", lambda *a, **kw: pytest.fail("candado no debe abrirse"))

    runner = CliRunner()
    res = runner.invoke(backtest_group, [
        "hipotesis", "correr", str(archivo), "--snapshot", "sintetico", "--carpeta", str(tmp_path / "salida"),
    ])
    assert res.exit_code == 1, res.output
    assert "Etapa 2: ALARMA (filtro optimista en la cartera)" in res.output
    assert "Llegó a la etapa 2: ALARMA (filtro optimista en la cartera)" in res.output
    with (tmp_path / "hipotesis.csv").open() as f:
        estados = list(csv.DictReader(f))
    assert estados[-1]["veredicto"] == "ALARMA"
    assert estados[-1]["motivo"] == "filtro optimista en la cartera"


def test_filtrar_hipotesis_cartera_dos_piernas_mismo_activo_parametros_por_clase_y_activo(monkeypatch):
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    fechas = pd.date_range("2007-01-01", "2018-12-31", freq="7D")
    p1 = Pierna("rsi", {"rsi_period": [7]}, ("SPY",), "U")
    p2 = Pierna("sma-cross", {"fast": [20]}, ("SPY",), "U")
    h = Hipotesis("h-2p", "autor", "just", "1d", piernas=(p1, p2))

    def fake_filtrar(p, velas, **kw):
        if p.clase == "rsi":
            return {"sharpe_tasa_0": 0.5, "trades": 35, "ganadores": [{"params": {"rsi_period": 7}, "prueba_desde": "2018-01-01"}], "retornos": pd.Series(0.001, index=velas.timestamp)}
        return {"sharpe_tasa_0": 0.4, "trades": 30, "ganadores": [{"params": {"fast": 20}, "prueba_desde": "2018-01-01"}], "retornos": pd.Series(0.001, index=velas.timestamp)}

    monkeypatch.setattr(hipotesis, "filtrar_activo", fake_filtrar)
    monkeypatch.setattr(hipotesis, "read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas)}, index=fechas))
    monkeypatch.setattr(hipotesis, "evaluar_cartera", lambda *a, **kw: 0.6)
    monkeypatch.setattr(hipotesis, "registrar_intento", lambda *a, **k: None)

    ret = hipotesis.filtrar_hipotesis(h, "sintetico")
    parametros = ret[1]
    assert parametros == {
        ("rsi", "SPY"): {"rsi_period": 7},
        ("sma-cross", "SPY"): {"fast": 20},
    }


def test_hipotesis_correr_sintetico_no_escribe_en_local_data():
    local_data = Path(__file__).resolve().parents[1] / "local-data"
    for cand in (Path("/home/azureuser/repos/projects/QuantAgent/local-data"), local_data):
        if (cand / "intentos.csv").exists():
            local_data = cand
            break

    intentos_real = local_data / "intentos.csv"
    creado = False
    if not intentos_real.exists():
        intentos_real.write_text("fecha_utc,estrategia,activo,snapshot,desde,hasta,costos,tamano,params,trades,pnl,sharpe,hipotesis,etapa,sharpe_base\n")
        creado = True
    try:
        h_antes = hashlib.sha256(intentos_real.read_bytes()).hexdigest()

        snapshot_dir = Path(os.environ["QUANTAGENT_SNAPSHOT_DIR"])
        intentos_aislado = snapshot_dir / "intentos.csv"
        if not intentos_aislado.exists():
            intentos_aislado.write_text("fecha_utc,estrategia,activo,snapshot,desde,hasta,costos,tamano,params,trades,pnl,sharpe,hipotesis,etapa,sharpe_base\n")

        fechas = pd.date_range("2007-01-01", "2018-12-31", freq="7D")
        velas = pd.DataFrame(
            {c: 100.0 for c in ("open", "high", "low", "close", "adj_close", "volume")}, index=fechas,
        )
        write_snapshot("sintetico", {"SPY": velas}, base_dir=snapshot_dir)
        data = yaml.safe_load((Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml").read_text())
        data["parametros"] = {"rsi_period": [7], "oversold_threshold": [30.0]}
        archivo = snapshot_dir / "h.yaml"
        archivo.write_text(yaml.safe_dump(data))

        CliRunner().invoke(backtest_group, [
            "hipotesis", "correr", str(archivo), "--snapshot", "sintetico", "--carpeta", str(snapshot_dir / "salida"),
        ])

        assert (snapshot_dir / "intentos.csv").exists()
        h_despues = hashlib.sha256(intentos_real.read_bytes()).hexdigest()
        assert h_despues == h_antes
    finally:
        if creado and intentos_real.exists():
            intentos_real.unlink()


@pytest.mark.parametrize("hasta_etapa", [0, 1, 2, 3, None])
def test_correr_hasta_etapa(tmp_path, monkeypatch, hasta_etapa):
    from quantagent.fabrica.candado import abierto
    from quantagent.fabrica.hipotesis import cargar_hipotesis

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    archivo = Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml"
    h = cargar_hipotesis(archivo)
    params = {"SPY": {"rsi_period": 7}}
    llamadas = []
    real_abrir = hipotesis.abrir
    monkeypatch.setattr(hipotesis, "admitir_hipotesis", lambda *a: (h, True, []))
    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", lambda *a: (
        llamadas.append("1") or [["SPY", 1.0, 30]], params, "PASA", "sharpe >= 0.10", 1.0, 30, [],
    ))
    monkeypatch.setattr(hipotesis, "validar_etapa2", lambda *a: (
        llamadas.append("2") or "PASA", [{"activo": "SPY", "v30": "PASA", "motivos": [], "sharpe_tasa_0": 1.0}],
    ))
    monkeypatch.setattr(hipotesis, "abrir", lambda *a: llamadas.append(("abrir", a)) or real_abrir(*a))
    monkeypatch.setattr(hipotesis, "validar_etapa3", lambda *a: (
        llamadas.append("3") or "PASA", [{"activo": "SPY", "v30": "PASA", "dsr": 0.95, "motivos": []}],
    ))
    tope = 3 if hasta_etapa is None else hasta_etapa
    extra = [] if hasta_etapa is None else ["--hasta-etapa", str(hasta_etapa)]
    res = CliRunner().invoke(backtest_group, [
        "hipotesis", "correr", str(archivo), "--snapshot", "sintetico", "--carpeta", str(tmp_path / "motor"), *extra,
    ])
    assert res.exit_code == 0, res.output
    with (tmp_path / "hipotesis.csv").open() as f:
        filas = list(csv.DictReader(f))
    assert [f["etapa"] for f in filas] == [str(i) for i in range(tope + 1)]
    assert res.output.splitlines()[-1] == f"Llegó a la etapa {tope}: PASA ({filas[-1]['motivo']})"
    assert abierto(h.id, "prueba") == (tope == 3)
    assert (tmp_path / "hipotesis-candado.csv").exists() == (tope == 3)
    assert ("abrir", (h.id, "prueba")) in llamadas if tope == 3 else ("abrir", (h.id, "prueba")) not in llamadas
    assert [x for x in llamadas if isinstance(x, str)] == [str(i) for i in range(1, tope + 1)]


@pytest.mark.parametrize("valor", ["-1", "4"])
def test_correr_hasta_etapa_fuera_de_rango(tmp_path, monkeypatch, valor):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    archivo = Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml"
    res = CliRunner().invoke(backtest_group, [
        "hipotesis", "correr", str(archivo), "--snapshot", "sintetico",
        "--carpeta", str(tmp_path / "motor"), "--hasta-etapa", valor,
    ])
    assert res.exit_code != 0
    assert "--hasta-etapa" in res.output and "0<=x<=3" in res.output
    assert not (tmp_path / "hipotesis.csv").exists()
    assert not (tmp_path / "hipotesis-candado.csv").exists()



