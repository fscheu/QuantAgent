import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from quantagent.cli.backtest import backtest_group
from quantagent.cli.sweep import CAMPOS_INTENTOS
from quantagent.fabrica.estado import registrar_estado


@pytest.fixture
def raiz(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    registrar_estado("h-corta1", "1", "CORTA", "2 de 2 bajo 0,10")
    for e, v, m in (("1", "PASA", "ok"), ("2", "PASA", "ok"), ("3", "CORTA", "DSR bajo en QQQ")):
        registrar_estado("h-corta3", e, v, m)
    filas = [("h-corta1", "1", "SPY", "0.02", "12"), ("h-corta1", "1", "IWM", "0.05", "20"),
             ("h-corta3", "2", "SPY", "0.9", "40"), ("h-corta3", "3", "QQQ", "0.31", "55"),
             ("h-corta3", "3", "EFA", "0.44", "61")]
    with (tmp_path / "intentos.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_INTENTOS, restval="")
        w.writeheader()
        for h, e, a, s, t in filas:
            w.writerow({"hipotesis": h, "etapa": e, "activo": a, "sharpe": s, "trades": t, "pnl": "123.4"})
    return tmp_path


def informe(raiz, hid, archivo=None):
    out = raiz / "i.html"
    args = ["hipotesis", "informe", hid, "--out", str(out)]
    if archivo:
        args.extend(["--archivo", str(archivo)])
    return CliRunner().invoke(backtest_group, args), out


def test_corte_en_etapa_1_nombra_etapa_motivo_y_activos(raiz):
    r, out = informe(raiz, "h-corta1")
    html = out.read_text()
    assert r.exit_code == 0, r.output
    assert "CORTA en la etapa 1" in html and "2 de 2 bajo 0,10" in html
    assert "SPY" in html and "IWM" in html and "QQQ" not in html


def test_corte_en_etapa_3_usa_solo_activos_de_esa_etapa(raiz):
    r, out = informe(raiz, "h-corta3")
    html = out.read_text()
    assert "CORTA en la etapa 3" in html and "DSR bajo en QQQ" in html
    assert "QQQ" in html and "EFA" in html and "<td>SPY</td>" not in html
    assert "al menos 2 trades, pnl &gt; 0, Sharpe con tasa 0 &gt;= 0,10 y caída máxima &lt;= 35%" in html
    assert "<th>Sharpe registrado</th>" in html and "puede pasar V30 con Sharpe registrado negativo" in html
    assert "<th>pnl" not in html.lower() and "123.4" not in html and "Prueba abierta: no" in html


def test_hipotesis_desconocida_falla(raiz):
    r, out = informe(raiz, "nada")
    assert r.exit_code != 0 and "desconocida" in r.output and not out.exists()


@pytest.mark.parametrize("hid", ["cartera-tendencia-momentum", "cartera-transversal",
                                 "cartera-estacionalidad", "cartera-nucleo-curva"])
def test_informe_advierte_overlay_en_cuatro_carteras_conocidas(raiz, hid):
    registrar_estado(hid, "3", "CORTA", "no supera al benchmark")
    archivo = Path(__file__).resolve().parents[1] / "hipotesis" / f"{hid}.yaml"
    r, out = informe(raiz, hid, archivo)
    assert r.exit_code == 0, r.output
    html = out.read_text()
    assert "Advertencias etapa 0 (no bloquean)" in html
    assert "<li>overlay_solo_largo</li>" in html
    assert "desvio_de_la_cita" not in html and "muestra_solapada" not in html
    assert "Veredicto: CORTA en la etapa 3" in html


@pytest.mark.parametrize("fiel", [False, True])
def test_informe_deriva_advertencias_del_yaml_sin_cambiar_veredicto(raiz, fiel):
    import yaml

    propuesta = {"id": "h-avisos", "autor": "fede", "justificacion": "prueba",
                 "timeframe": "1d", "clase": "momentum-12m", "parametros": {"lookback": [252]},
                 "universo": "ETFs líquidos", "activos": ["SPY"],
                 "fidelidad": "desvío: cita largo-corto", "muestra_de_la_cita": "1985-2009"}
    if fiel:
        propuesta.update(fidelidad="igual a la cita", muestra_de_la_cita="1985-2006", activos=["CL"])
    archivo = raiz / "avisos.yaml"
    archivo.write_text(yaml.safe_dump(propuesta))
    registrar_estado("h-avisos", "0", "PASA", "admitida")
    r, out = informe(raiz, "h-avisos", archivo)
    assert r.exit_code == 0, r.output
    html = out.read_text()
    for codigo in ("desvio_de_la_cita", "muestra_solapada", "overlay_solo_largo"):
        assert (f"<li>{codigo}</li>" in html) == (not fiel)
    assert ("Advertencias etapa 0" in html) == (not fiel)
    assert "Veredicto: PASA en la etapa 0" in html


def test_informe_cartera_dos_piernas_contiene_universos_y_fila_cartera_sin_precios(raiz):
    import yaml
    yaml_path = raiz / "cartera.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "id": "h-cartera",
        "autor": "fede",
        "justificacion": "Cartera de 2 piernas.",
        "timeframe": "1d",
        "piernas": [
            {"clase": "momentum-12m", "parametros": {"lookback": [252]},
             "activos": ["SPY", "QQQ"], "universo": "Universo 1: 12 ETFs de acciones"},
            {"clase": "sma-cross", "parametros": {"fast": [1], "slow": [210]},
             "activos": ["TLT", "IEF"], "universo": "Universo 2: ETFs de bonos"}
        ]
    }))
    registrar_estado("h-cartera", "1", "PASA", "ok")
    with (raiz / "intentos.csv").open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_INTENTOS, restval="")
        w.writerow({"hipotesis": "h-cartera", "etapa": "1", "activo": "cartera", "sharpe": "0.45", "trades": "120", "pnl": "5000.0"})
        w.writerow({"hipotesis": "h-cartera", "etapa": "1", "activo": "SPY", "sharpe": "0.20", "trades": "60", "pnl": "2500.0"})
    r, out = informe(raiz, "h-cartera", archivo=yaml_path)
    assert r.exit_code == 0, r.output
    html = out.read_text()
    assert "Universo 1: 12 ETFs de acciones" in html
    assert "Universo 2: ETFs de bonos" in html
    assert "<td>cartera</td>" in html
    assert html.index("<td>cartera</td>") < html.index("<td>SPY</td>")
    assert "<th>pnl" not in html.lower() and "5000.0" not in html and "2500.0" not in html
    assert "Cartera" in html and "Pierna" in html and "Igual riesgo" in html


def test_informe_etapa3_muestra_cuatro_cifras_recargo_y_glosario_nuevo(raiz):
    from quantagent.cli.hipotesis import motivo_etapa3_cartera

    for e in ("1", "2"):
        registrar_estado("h-etapa3-completa", e, "PASA", "ok")
    fila_e3 = {"dsr": 0.80, "dsr_crudo": 0.55, "anios_positivos": "3 de 4", "benchmark": 0.12,
               "benchmark_mismo_peso": 0.34, "n_efectivo": 5}
    registrar_estado("h-etapa3-completa", "3", "PASA", motivo_etapa3_cartera("cartera pasa V30 y DSR", fila_e3))
    with (raiz / "intentos.csv").open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CAMPOS_INTENTOS, restval="").writerow({"hipotesis": "h-etapa3-completa", "etapa": "3", "activo": "cartera", "sharpe": "0.45", "trades": "100"})
    r, out = informe(raiz, "h-etapa3-completa")
    assert r.exit_code == 0, r.output
    html = out.read_text()
    for h, v in (("N crudo", "6"), ("N efectivo", "5"), ("DSR efectivo", "0.80"), ("DSR crudo", "0.55"),
                 ("Recargo paper", "75% (6 sem)"), ("Años positivos", "3 de 4"),
                 ("Benchmark 1/N (corte)", "0.12"),
                 ("Benchmark con pesos de cartera (diagnóstico)", "0.34")):
        assert f"<th>{h}</th>" in html and f"<td>{v}</td>" in html
    assert "pasa con 0,95" not in html and "fija el recargo de paper trading" in html


def test_informe_etapa3_motivo_sin_claves_muestra_guiones(raiz):
    for e, v, m in (("1", "PASA", "ok"), ("2", "PASA", "ok"), ("3", "CORTA", "DSR bajo en QQQ")):
        registrar_estado("h-sin-claves", e, v, m)
    r, out = informe(raiz, "h-sin-claves")
    assert r.exit_code == 0 and "<h2>Criterio de etapa 3</h2>" in out.read_text() and out.read_text().count("<td>-</td>") >= 7


def test_informe_cli_muestra_sharpes_en_exceso_y_sin_datos_indica_ausencia(raiz, monkeypatch):
    from quantagent.fabrica import informe as modulo

    registrar_estado("h-exceso", "1", "PASA", "ok")
    yaml_path = raiz / "h.yaml"
    yaml_path.write_text("id: h-exceso\nautor: f\njustificacion: j\ntimeframe: 1d\nclase: rsi\nparametros: {rsi_period: [7], oversold_threshold: [30]}\nactivos: [SPY]\nuniverso: prueba\n")
    calls = []

    def diagnosticar(h, carpeta, snapshot):
        calls.append((carpeta, snapshot))
        return {"disponible": True, "sharpe_cartera": 0.31, "sharpe_benchmark": 0.12}

    monkeypatch.setattr(modulo, "diagnostico_exceso", diagnosticar, raising=False)
    r, out = informe(raiz, "h-exceso", yaml_path)
    assert r.exit_code == 0, r.output
    assert "no disponible" in out.read_text() and calls == []
    r = CliRunner().invoke(backtest_group, ["hipotesis", "informe", "h-exceso", "--out", str(out),
                                             "--archivo", str(yaml_path), "--carpeta-equity", str(raiz),
                                             "--snapshot", "snap"])
    assert r.exit_code == 0, r.output
    assert calls == [(str(raiz), "snap")] and "<td>0.31</td><td>0.12</td>" in out.read_text()


def test_diagnostico_exceso_solo_lee_curvas_y_fechas_guardadas(raiz, monkeypatch):
    import pandas as pd
    from quantagent.fabrica import informe as modulo
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    fechas = pd.date_range("2020-01-01", periods=4)
    carpeta = raiz / "equity"
    carpeta.mkdir()
    with (carpeta / "sma-SPY.csv").open("w") as f:
        f.write("date,equity\n" + "".join(f"{d.date()},{100+i*2}\n" for i, d in enumerate(fechas)))
    h = Hipotesis("h", "f", "j", piernas=(Pierna("sma", {}, ("SPY",), "u"),))
    frame = pd.DataFrame({"close": [100, 101, 102, 103], "SPY": [100, 101, 102, 103]}, index=fechas)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda *a, **k: frame)
    def forbidden(*a, **k):
        raise AssertionError("motor llamado")
    monkeypatch.setattr("quantagent.cli.backtest.run_backtest", forbidden)
    monkeypatch.setattr("quantagent.fabrica.validacion._curva_y_v30_cartera", forbidden)
    monkeypatch.setattr("quantagent.fabrica.validacion.registrar_intento", forbidden)
    monkeypatch.setattr("quantagent.cli.sweep.registrar_intento", forbidden)
    original = __import__("quantagent.fabrica.criterio", fromlist=["curva_benchmark"]).curva_benchmark
    fechas_benchmark = []
    def benchmark(cierres, *a, **k):
        fechas_benchmark.extend(cierres.index)
        return original(cierres, *a, **k)
    monkeypatch.setattr("quantagent.fabrica.criterio.curva_benchmark", benchmark)
    from quantagent.data.macro import _escribir
    from quantagent.fabrica.cartera import curva_cartera, pesos
    from quantagent.fabrica.criterio import curva_benchmark, sharpes_en_exceso
    macro = pd.DataFrame({"fecha": [fechas[0]], "realtime_start": [fechas[0]],
                          "realtime_end": [pd.Timestamp("2099-01-01")], "valor": [3.0]})
    _escribir(raiz / "macro-2026-10", "macro-2026-10", {"DGS3MO": macro})
    (raiz / "hipotesis-candado.csv").write_text("h,prueba,2020-01-01\n")
    entradas = ["intentos.csv", "hipotesis.csv", "hipotesis-candado.csv"]
    antes = {n: (raiz / n).read_bytes() for n in entradas}
    dsr_antes = modulo.datos_informe("h-corta3")["dsr"]
    eq = pd.Series([100, 102, 104, 106], index=fechas)
    precios = frame[["SPY"]]
    curva = curva_cartera(precios, {0: ["SPY"]}, {(0, "SPY"): eq.pct_change().dropna()})
    ponderaciones = pesos(precios, {0: ["SPY"]}).T.groupby(level=1).sum().T
    bench = curva_benchmark(precios, pesos=ponderaciones).reindex(curva.index)
    esperado = sharpes_en_exceso(curva, bench, base_dir=raiz)
    fechas_benchmark.clear()
    resultado = modulo.diagnostico_exceso(h, carpeta, "snapshot", raiz)
    assert (resultado["sharpe_cartera"], resultado["sharpe_benchmark"]) == pytest.approx(esperado)
    assert fechas_benchmark == list(fechas)
    assert {n: (raiz / n).read_bytes() for n in antes} == antes
    assert modulo.datos_informe("h-corta3")["dsr"] == dsr_antes
    assert "curva faltante: SPY" in modulo.diagnostico_exceso(h, carpeta / "nada", "snapshot", raiz)["motivo"]
    monkeypatch.setattr("quantagent.fabrica.criterio.serie_punto_en_el_tiempo",
                        lambda *a, **k: pd.Series(float("nan"), index=a[2]))
    sin_tasa = modulo.diagnostico_exceso(h, carpeta, "snapshot", raiz)
    assert sin_tasa["disponible"] is False and "DGS3MO sin cobertura" in sin_tasa["motivo"]
    yaml_path = raiz / "h-corta3.yaml"
    yaml_path.write_text("id: h-corta3\nautor: f\njustificacion: j\ntimeframe: 1d\nclase: sma-cross\nparametros: {fast: [2], slow: [10]}\nactivos: [SPY]\nuniverso: prueba\n")
    out = raiz / "informe.html"
    cli = CliRunner().invoke(backtest_group, ["hipotesis", "informe", "h-corta3", "--out", str(out),
                                               "--archivo", str(yaml_path), "--carpeta-equity", str(carpeta),
                                               "--snapshot", "snapshot"])
    assert cli.exit_code == 0, cli.output
    assert "Sharpe en exceso de DGS3MO" in out.read_text()
    assert {n: (raiz / n).read_bytes() for n in antes} == antes
    assert modulo.datos_informe("h-corta3")["dsr"] == dsr_antes


