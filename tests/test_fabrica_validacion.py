import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.data.snapshot import write_snapshot
from quantagent.fabrica import validacion
from quantagent.fabrica.hipotesis import cargar_hipotesis


@pytest.mark.parametrize("cantidad,esperado", [(4, True), (3, False)])
def test_alarma_mitad_del_filtro(cantidad, esperado):
    resultados = [(str(i), 0.10, 30) for i in range(8)]
    filas = [{"activo": str(i), "sharpe_tasa_0": "0" if i < cantidad else "0.01"}
             for i in reversed(range(8))]
    assert validacion.alarma_filtro_motor(resultados, filas) == (
        esperado, [str(i) for i in range(cantidad)])


@pytest.mark.parametrize("vacio", ["", None])
def test_alarma_vacio_y_filtro_bajo(vacio):
    resultados = [("SPY", 0.10, 30), ("TLT", 0.2, 30), ("GLD", 0.3, 30),
                  ("QQQ", 0.09, 30)]
    filas = [{"activo": a, "sharpe_tasa_0": s}
             for a, s in [("SPY", vacio), ("TLT", "-0.1"), ("GLD", "0.1"), ("QQQ", "-1")]]
    assert validacion.alarma_filtro_motor(resultados, filas) == (True, ["SPY", "TLT"])
    filas[1]["sharpe_tasa_0"] = "0.1"
    assert validacion.alarma_filtro_motor(resultados, filas) == (False, ["SPY"])


@pytest.mark.parametrize("sharpes", [[], [0.09, None, float("nan")]])
def test_alarma_sin_filtro_aprobado(sharpes):
    resultados = [(str(i), s, 30) for i, s in enumerate(sharpes)]
    filas = [{"activo": str(i), "sharpe_tasa_0": "-1"} for i in range(len(sharpes))]
    assert validacion.alarma_filtro_motor(resultados, filas) == (False, [])


@pytest.mark.parametrize("abrir_prueba,etapa,desde", [(False, "2", "2007-01-01"), (True, "3", "2019-01-01")])
def test_motor_dos_activos(tmp_path, monkeypatch, abrir_prueba, etapa, desde):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range(desde, periods=150)
    close = 100 + 10 * np.sin(np.arange(150) / 5)
    velas = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "adj_close": close, "volume": 1000.0}, index=fechas)
    write_snapshot("sintetico", {"SPY": velas, "TLT": velas}, base_dir=tmp_path)
    h = cargar_hipotesis(Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml")
    params = {"SPY": {"rsi_period": 7}, "TLT": {"rsi_period": 21}}
    real, recibidos = validacion.subprocess.run, {}

    def observar(cmd, **kwargs):
        activo = cmd[cmd.index("--symbols") + 1]
        recibidos[activo] = json.loads(Path(cmd[cmd.index("--params") + 1]).read_text())
        assert ("--abrir-prueba" in cmd) == abrir_prueba
        return real(cmd, **kwargs)

    monkeypatch.setattr(validacion.subprocess, "run", observar)
    carpeta = tmp_path / "salida"
    filas = validacion.correr_motor(h, "sintetico", params, desde, fechas[-1].date(), carpeta, etapa, abrir_prueba)
    assert recibidos == {a: {h.clase: p} for a, p in params.items()}
    assert [f["activo"] for f in filas] == ["SPY", "TLT"]
    assert filas[0]["pnl"] != filas[1]["pnl"]
    for activo in params:
        for sub in ("trades", "equity"):
            with (carpeta / sub / f"rsi-{activo}.csv").open() as f:
                assert len(list(csv.DictReader(f))) > 0
    with (tmp_path / "intentos.csv").open() as f:
        intentos = list(csv.DictReader(f))
    assert len(intentos) == 2
    for intento, fila in zip(intentos, filas):
        assert intento["hipotesis"] == h.id and intento["etapa"] == etapa
        assert json.loads(intento["params"]) == params[fila["activo"]]
        assert intento["pnl"] == fila["pnl"]
        assert intento["sharpe"] == fila["sharpe_tasa_0"]
        assert intento["sharpe_base"] == "tasa_0"
    with pytest.raises(RuntimeError, match="AUSENTE"):
        validacion.correr_motor(h, "sintetico", {"AUSENTE": {}}, desde, fechas[-1].date(), carpeta, etapa, abrir_prueba)
    with (tmp_path / "intentos.csv").open() as f:
        assert list(csv.DictReader(f)) == intentos


@pytest.mark.parametrize("caso", ["pasa", "sin validar", "ninguno pasa", "otro directorio"])
def test_etapa2(tmp_path, monkeypatch, caso):
    pytest.importorskip("backtesting")
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2007-01-01", periods=150)
    close = 100 + 10 * np.sin(np.arange(150) / 5)
    velas = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "adj_close": close, "volume": 1000.0}, index=fechas)
    write_snapshot("sintetico", {"SPY": velas, "TLT": velas}, base_dir=tmp_path)
    h = cargar_hipotesis(Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml")
    params = {"SPY": {"rsi_period": 7}, "TLT": {"rsi_period": 21}}
    if caso == "ninguno pasa":
        params = {a: {"oversold_threshold": 0.0, "overbought_threshold": 100.0} for a in params}
    real = validacion.subprocess.run

    def sin_validar(cmd, **kwargs):
        resultado = real(cmd, **kwargs)
        if "--tabla" in cmd and cmd[cmd.index("--tabla") + 1].endswith("TLT.csv"):
            tabla = Path(cmd[cmd.index("--tabla") + 1])
            with tabla.open(newline="") as f:
                reader = csv.DictReader(f)
                campos, filas = reader.fieldnames, list(reader)
            filas[0]["control"] = "sin validar: diferencia forzada"
            with tabla.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=campos)
                writer.writeheader()
                writer.writerows(filas)
        return resultado

    if caso == "sin validar":
        monkeypatch.setattr(validacion.subprocess, "run", sin_validar)
    if caso == "otro directorio":
        monkeypatch.chdir(tmp_path)
    veredicto, filas = validacion.validar_etapa2(h, "sintetico", params, tmp_path / "salida")
    assert veredicto == ("PASA" if caso in ("pasa", "otro directorio") else "CORTA")
    assert [f["activo"] for f in filas] == ["SPY", "TLT"]
    assert filas[0]["control"].startswith("coinciden")
    with (tmp_path / "salida" / "celdas.csv").open() as f:
        assert [(c["pierna_clase"], c["activo"]) for c in csv.DictReader(f)] == [("rsi", "SPY"), ("rsi", "TLT")]
    if caso == "ninguno pasa":
        assert all(f["v30"] == "NO PASA" and f["motivos"] for f in filas)
    else:
        assert any(f["v30"] == "PASA" for f in filas)
    if caso == "sin validar":
        assert "TLT: sin validar: diferencia forzada" in filas[1]["motivos"]
    else:
        assert filas[1]["control"].startswith("coinciden")


@pytest.mark.parametrize("dsr,pnl,esperado", [(0.95, 1, "PASA"), (0.94, 1, "CORTA"),
                                          (0.99, -1, "CORTA"), (None, 1, "CORTA")])
def test_etapa3_regla(tmp_path, monkeypatch, dsr, pnl, esperado):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    h = cargar_hipotesis(Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml")
    (tmp_path / "equity").mkdir()
    for activo in ("SPY", "TLT"):
        (tmp_path / "equity" / f"rsi-{activo}.csv").write_text("date,equity\n2019-01-01,100\n")
    filas = [{"activo": a, "trades": "2", "pnl": str(p), "sharpe_tasa_0": "0.10",
              "max_drawdown": "0.35"} for a, p in (("SPY", -1), ("TLT", pnl))]
    monkeypatch.setattr(validacion, "correr_motor", lambda *args, **kwargs: filas)
    monkeypatch.setattr(validacion, "dsr_contra_intentos", lambda *args: dsr)
    veredicto, resultado = validacion.validar_etapa3(h, "sintetico", {}, tmp_path)
    assert veredicto == esperado
    assert resultado[0]["v30"] == "NO PASA" and resultado[0]["motivos"]
    assert resultado[1]["v30"] == ("PASA" if pnl > 0 else "NO PASA")
    assert resultado[1]["dsr"] == dsr
    assert bool(resultado[1]["motivos"]) == (esperado == "CORTA")


def test_etapa3_real(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2018-12-01", "2023-01-31")
    close = 100 + 10 * np.sin(np.arange(len(fechas)) / 5)
    velas = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "adj_close": close, "volume": 1000.0}, index=fechas)
    write_snapshot("sintetico", {"SPY": velas, "TLT": velas}, base_dir=tmp_path)
    h = cargar_hipotesis(Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml")
    carpeta = tmp_path / "salida"
    veredicto, filas = validacion.validar_etapa3(
        h, "sintetico", {"SPY": {"rsi_period": 7}, "TLT": {"rsi_period": 21}}, carpeta)
    for fila in filas:
        with (carpeta / "equity" / f"rsi-{fila['activo']}.csv").open() as f:
            curva = list(csv.DictReader(f))
        assert all("2019-01-01" <= r["date"][:10] <= "2022-12-31" for r in curva)
        equity = [float(r["equity"]) for r in curva]
        assert fila["dsr"] == validacion.dsr_contra_intentos(equity, tmp_path / "intentos.csv")
        pasa, _ = validacion.evaluar(*(float(fila[c]) for c in
                                     ("trades", "pnl", "sharpe_tasa_0", "max_drawdown")))
        assert fila["v30"] == ("PASA" if pasa else "NO PASA")
    assert filas[0]["dsr"] != filas[1]["dsr"]
    assert veredicto == ("PASA" if any(not f["motivos"] for f in filas) else "CORTA")
    with (tmp_path / "intentos.csv").open() as f:
        intentos = list(csv.DictReader(f))
    assert [(r["activo"], r["hipotesis"], r["etapa"]) for r in intentos] == [
        (a, h.id, "3") for a in ("SPY", "TLT")]


def test_etapa2_cartera_compare_falso(tmp_path, monkeypatch):
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2010-01-01", periods=100)
    eq_dir = tmp_path / "salida" / "equity"
    eq_dir.mkdir(parents=True)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(100, 0.001))}).to_csv(eq_dir / "rsi-SPY.csv", index=False)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(100, -0.0002))}).to_csv(eq_dir / "sma-TLT.csv", index=False)

    h = Hipotesis("h-c", "autor", "just", "1d", piernas=(Pierna("rsi", {}, ("SPY",), "U"), Pierna("sma", {}, ("TLT",), "U")))
    monkeypatch.setattr(validacion, "correr_motor", lambda h_obj, *a, **k: [
        {"activo": "SPY", "estrategia": "rsi", "trades": "4", "pnl": "1000", "sharpe_tasa_0": "0.5", "max_drawdown": "0.05"},
    ] if len(h_obj.piernas) == 1 else [
        {"activo": "SPY", "estrategia": "rsi", "trades": "4", "pnl": "1000", "sharpe_tasa_0": "0.5", "max_drawdown": "0.05"},
        {"activo": "TLT", "estrategia": "sma", "trades": "2", "pnl": "-200", "sharpe_tasa_0": "-0.2", "max_drawdown": "0.08"},
    ])
    no_coincide = False

    def fake_subp(cmd, **kwargs):
        if "--tabla" in cmd:
            t = Path(cmd[cmd.index("--tabla") + 1])
            ctrl = "sin validar: dif" if "TLT" in str(t) and no_coincide else "coinciden 5 de 5"
            t.write_text(f"activo,control\nX,{ctrl}\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validacion.subprocess, "run", fake_subp)

    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas)}))
    veredicto, filas = validacion.validar_etapa2(h, "sintetico", {("rsi", "SPY"): {}, ("sma", "TLT"): {}}, tmp_path / "salida")
    assert veredicto == "PASA" and filas[1]["v30"] == "NO PASA"

    no_coincide = True
    veredicto, filas = validacion.validar_etapa2(h, "sintetico", {("rsi", "SPY"): {}, ("sma", "TLT"): {}}, tmp_path / "salida")
    assert veredicto == "CORTA" and "TLT: sin validar: dif" in filas[1]["motivos"]

    no_coincide = False
    h1 = Hipotesis("h-1p", "autor", "just", "1d", clase="rsi", parametros={}, activos=("SPY",), universo="U")
    veredicto1, filas1 = validacion.validar_etapa2(h1, "sintetico", {"SPY": {}}, tmp_path / "salida")
    assert veredicto1 == "PASA" and filas1[0]["v30"] == "PASA"


def test_etapa2_cartera_snapshot_ilegible_falla(tmp_path):
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    h = Hipotesis("h-c", "autor", "just", "1d", piernas=(Pierna("rsi", {}, ("SPY",), "U"), Pierna("sma", {}, ("TLT",), "U")))
    with pytest.raises(Exception):
        validacion.evaluar_etapa2_cartera(h, [], tmp_path, snapshot="ilegible")
    assert validacion.evaluar_etapa2_cartera(h, [], tmp_path, snapshot=None) == (False, ["curva de cartera vacía"])


@pytest.mark.parametrize("filtro,motor,esperado", [
    (0.15, -0.05, True),
    (0.15, 0.0, True),
    (0.15, 0.01, False),
    (0.09, -0.05, False),
    (None, -0.05, False),
    (float("nan"), -0.05, False),
    (0.20, None, True),
    (0.20, "", True),
])
def test_alarma_cartera_dispara_y_no_dispara(filtro, motor, esperado):
    assert validacion.alarma_cartera(filtro, motor) is esperado


def test_etapa2_cartera_2_piernas_3_activos_agrega_una_fila_intentos(tmp_path, monkeypatch):
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2010-01-01", periods=100)
    eq_dir = tmp_path / "salida" / "equity"
    eq_dir.mkdir(parents=True)
    p1_activos = ("SPY", "TLT", "GLD")
    p2_activos = ("EEM", "IWM", "DIA")
    for a in p1_activos:
        pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(100, 0.001))}).to_csv(eq_dir / f"rsi-{a}.csv", index=False)
    for a in p2_activos:
        pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(100, 0.0005))}).to_csv(eq_dir / f"sma-{a}.csv", index=False)

    h = Hipotesis(
        "h-c2x3", "autor", "just", "1d",
        piernas=(
            Pierna("rsi", {"rsi_period": [7]}, p1_activos, "U"),
            Pierna("sma", {"fast": [20]}, p2_activos, "U"),
        ),
    )
    filas_falsas = [
        {"activo": a, "estrategia": "rsi", "trades": "5", "pnl": "100", "sharpe_tasa_0": "0.4", "max_drawdown": "0.02"} for a in p1_activos
    ] + [
        {"activo": a, "estrategia": "sma", "trades": "5", "pnl": "100", "sharpe_tasa_0": "0.4", "max_drawdown": "0.02"} for a in p2_activos
    ]
    monkeypatch.setattr(validacion, "correr_motor", lambda *a, **k: filas_falsas)

    def fake_subp(cmd, **kwargs):
        if "--tabla" in cmd:
            t = Path(cmd[cmd.index("--tabla") + 1])
            t.write_text("activo,control\nX,coinciden 5 de 5\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validacion.subprocess, "run", fake_subp)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas)}))

    veredicto, filas = validacion.validar_etapa2(h, "sintetico", {(p.clase, a): {} for p in h.piernas for a in p.activos}, tmp_path / "salida")
    assert veredicto == "PASA"
    assert len(filas) == 6
    with (tmp_path / "intentos.csv").open() as f:
        intentos = list(csv.DictReader(f))
    assert len(intentos) == 1
    intento = intentos[0]
    assert intento["activo"] == "cartera"
    assert intento["etapa"] == "2"
    assert intento["sharpe_base"] == "tasa_0"
    assert intento["trades"] == "30"
    assert intento["estrategia"] == "rsi+sma"


def test_etapa2_cartera_escribe_celdas_por_pierna_y_activo(tmp_path, monkeypatch):
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2010-01-01", periods=100)
    eq_dir = tmp_path / "salida" / "equity"
    eq_dir.mkdir(parents=True)
    h = Hipotesis("h-celdas", "autor", "just", "1d", piernas=(
        Pierna("rsi", {"rsi_period": [7]}, ("SPY", "TLT"), "U"), Pierna("sma", {"fast": [20]}, ("SPY", "TLT"), "U")))
    filas_falsas = [{"activo": a, "estrategia": p.clase, "trades": str(n), "pnl": "100", "sharpe_tasa_0": "0.4",
                     "max_drawdown": "0.02", "dsr": "0.9"} for n, p in enumerate(h.piernas, 5) for a in p.activos]
    for f in filas_falsas:
        pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(100, 0.001))}).to_csv(
            eq_dir / f"{f['estrategia']}-{f['activo']}.csv", index=False)
    monkeypatch.setattr(validacion, "correr_motor", lambda *a, **k: filas_falsas)

    def fake_subp(cmd, **kwargs):
        Path(cmd[cmd.index("--tabla") + 1]).write_text("activo,control\nX,coinciden 5 de 5\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validacion.subprocess, "run", fake_subp)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas)}))
    validacion.validar_etapa2(h, "sintetico", {(p.clase, a): {} for p in h.piernas for a in p.activos}, tmp_path / "salida")
    with (tmp_path / "salida" / "celdas.csv").open() as f:
        lector = csv.DictReader(f)
        celdas = list(lector)
    assert lector.fieldnames == validacion.COLUMNAS_CELDAS
    assert [(c["pierna_clase"], c["activo"], c["trades"]) for c in celdas] == [
        ("rsi", "SPY", "5"), ("rsi", "TLT", "5"), ("sma", "SPY", "6"), ("sma", "TLT", "6")]
    assert {(c["hipotesis"], c["desde"], c["hasta"], c["pnl"], c["max_drawdown"]) for c in celdas} == {
        ("h-celdas", "2007-01-01", "2018-12-31", "100", "0.02")}


def test_etapa2_cartera_dos_piernas_mismo_activo_parametros_distintos(tmp_path, monkeypatch):
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2010-01-01", periods=50)
    p1 = Pierna("rsi", {"rsi_period": [7]}, ("SPY",), "U")
    p2 = Pierna("sma-cross", {"fast": [20]}, ("SPY",), "U")
    h = Hipotesis("h-mismo-activo", "autor", "just", "1d", piernas=(p1, p2))
    params = {
        ("rsi", "SPY"): {"rsi_period": 14},
        ("sma-cross", "SPY"): {"fast": 50, "slow": 200},
    }

    recibidos_motor = {}
    recibidos_control = {}

    def fake_subp(cmd, **kwargs):
        if "--strategies" in cmd:
            strat = cmd[cmd.index("--strategies") + 1]
            activo = cmd[cmd.index("--symbols") + 1]
            recibidos_motor[(strat, activo)] = json.loads(Path(cmd[cmd.index("--params") + 1]).read_text())
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"activo,estrategia,trades,pnl,sharpe_tasa_0,max_drawdown\n{activo},{strat},5,100,0.5,0.02\n")
            eq = out.parent / "equity" / f"{strat}-{activo}.csv"
            eq.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(50, 0.001))}).to_csv(eq, index=False)
        elif "--tabla" in cmd:
            t = Path(cmd[cmd.index("--tabla") + 1])
            strat, activo = t.stem.rsplit("-", 1)
            recibidos_control[(strat, activo)] = json.loads(Path(cmd[cmd.index("--params") + 1]).read_text())
            t.write_text("activo,control\nX,coinciden 5 de 5\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validacion.subprocess, "run", fake_subp)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas)}))

    veredicto, filas = validacion.validar_etapa2(h, "sintetico", params, tmp_path / "salida")
    assert veredicto == "PASA"
    assert len(filas) == 2
    assert recibidos_motor[("rsi", "SPY")] == {"rsi": {"rsi_period": 14}}
    assert recibidos_motor[("sma-cross", "SPY")] == {"sma-cross": {"fast": 50, "slow": 200}}
    assert recibidos_control[("rsi", "SPY")] == {"rsi": {"rsi_period": 14}}
    assert recibidos_control[("sma-cross", "SPY")] == {"sma-cross": {"fast": 50, "slow": 200}}


def test_etapa3_cartera_criterio_pasa_con_dsr_bajo(tmp_path, monkeypatch):
    """Cartera con DSR bajo pero Sharpe > 0, > benchmark, >=3 años positivos y DD bajo PASA."""
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.bdate_range("2019-01-01", "2022-12-31")
    fechas_snap = pd.bdate_range("2007-01-01", "2022-12-31")
    (tmp_path / "intentos.csv").write_text("activo,hipotesis,etapa,sharpe\ncartera,h-c,2,0.5\n")
    eq_dir = tmp_path / "salida" / "equity"
    eq_dir.mkdir(parents=True)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(len(fechas), 0.001))}).to_csv(eq_dir / "rsi-SPY.csv", index=False)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(len(fechas), 0.0005))}).to_csv(eq_dir / "sma-TLT.csv", index=False)

    h = Hipotesis("h-c", "autor", "just", "1d", piernas=(Pierna("rsi", {}, ("SPY",), "U"), Pierna("sma", {}, ("TLT",), "U")))
    monkeypatch.setattr(validacion, "correr_motor", lambda *a, **k: [
        {"activo": "SPY", "estrategia": "rsi", "trades": "4", "pnl": "1000", "sharpe_tasa_0": "0.5", "max_drawdown": "0.05"},
        {"activo": "TLT", "estrategia": "sma", "trades": "2", "pnl": "200", "sharpe_tasa_0": "0.2", "max_drawdown": "0.08"},
    ])
    monkeypatch.setattr(validacion, "dsr_contra_intentos", lambda *a, **k: 0.10)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas_snap)}))

    veredicto, filas = validacion.validar_etapa3(h, "sintetico", {("rsi", "SPY"): {}, ("sma", "TLT"): {}}, tmp_path / "salida")
    assert veredicto == "PASA"
    assert filas[0]["v30"] == "PASA"
    assert filas[0]["motivos"] == []
    assert filas[0]["dsr"] == 0.10
    assert filas[0]["recargo"] == (50, 12)


def test_etapa3_cartera_solo_2_anios_positivos_corta(tmp_path, monkeypatch):
    """La misma cartera con solo 2 de 4 años positivos CORTA con el motivo escrito."""
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.bdate_range("2019-01-01", "2022-12-31")
    fechas_snap = pd.bdate_range("2007-01-01", "2022-12-31")
    (tmp_path / "intentos.csv").write_text("activo,hipotesis,etapa,sharpe\ncartera,h-c,2,0.5\n")
    retornos = np.where(fechas.year >= 2021, -0.0005, 0.001)
    eq_dir = tmp_path / "salida" / "equity"
    eq_dir.mkdir(parents=True)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + retornos)}).to_csv(eq_dir / "rsi-SPY.csv", index=False)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + retornos)}).to_csv(eq_dir / "sma-TLT.csv", index=False)

    h = Hipotesis("h-c", "autor", "just", "1d", piernas=(Pierna("rsi", {}, ("SPY",), "U"), Pierna("sma", {}, ("TLT",), "U")))
    monkeypatch.setattr(validacion, "correr_motor", lambda *a, **k: [
        {"activo": "SPY", "estrategia": "rsi", "trades": "4", "pnl": "1000", "sharpe_tasa_0": "0.5", "max_drawdown": "0.05"},
        {"activo": "TLT", "estrategia": "sma", "trades": "2", "pnl": "200", "sharpe_tasa_0": "0.2", "max_drawdown": "0.08"},
    ])
    monkeypatch.setattr(validacion, "dsr_contra_intentos", lambda *a, **k: 0.90)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas_snap)}))

    veredicto, filas = validacion.validar_etapa3(h, "sintetico", {("rsi", "SPY"): {}, ("sma", "TLT"): {}}, tmp_path / "salida")
    assert veredicto == "CORTA"
    assert filas[0]["v30"] == "NO PASA"
    assert any("2 de 4" in m or "(< 3)" in m for m in filas[0]["motivos"])


def test_etapa3_cartera_menor_al_benchmark_corta_con_dsr_alto(tmp_path, monkeypatch):
    """Sharpe de cartera menor al del benchmark corta aunque el DSR sea alto."""
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.bdate_range("2019-01-01", "2022-12-31")
    fechas_snap = pd.bdate_range("2007-01-01", "2022-12-31")
    (tmp_path / "intentos.csv").write_text("activo,hipotesis,etapa,sharpe\ncartera,h-c,2,0.5\n")
    eq_dir = tmp_path / "salida" / "equity"
    eq_dir.mkdir(parents=True)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(len(fechas), 0.0001))}).to_csv(eq_dir / "rsi-SPY.csv", index=False)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(len(fechas), 0.0001))}).to_csv(eq_dir / "sma-TLT.csv", index=False)

    h = Hipotesis("h-c", "autor", "just", "1d", piernas=(Pierna("rsi", {}, ("SPY",), "U"), Pierna("sma", {}, ("TLT",), "U")))
    monkeypatch.setattr(validacion, "correr_motor", lambda *a, **k: [
        {"activo": "SPY", "estrategia": "rsi", "trades": "4", "pnl": "100", "sharpe_tasa_0": "0.1", "max_drawdown": "0.01"},
        {"activo": "TLT", "estrategia": "sma", "trades": "2", "pnl": "100", "sharpe_tasa_0": "0.1", "max_drawdown": "0.01"},
    ])
    monkeypatch.setattr(validacion, "dsr_contra_intentos", lambda *a, **k: 0.99)
    ret_snap = pd.Series(0.0, index=fechas_snap)
    ret_snap.loc["2019-01-01":"2022-12-31"] = 0.002
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame(
        {"close": pd.Series(100.0 * np.cumprod(1 + ret_snap.values), index=fechas_snap)}
    ))

    veredicto, filas = validacion.validar_etapa3(h, "sintetico", {("rsi", "SPY"): {}, ("sma", "TLT"): {}}, tmp_path / "salida")
    assert veredicto == "CORTA"
    assert filas[0]["v30"] == "NO PASA"
    assert any("benchmark" in m for m in filas[0]["motivos"])


def test_etapa3_cartera_dsr_cero_cumple_lo_demas_pasa(tmp_path, monkeypatch):
    """Falla si se vuelve a poner el corte por DSR (cartera con DSR 0,00 pasa si cumple lo demás)."""
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.bdate_range("2019-01-01", "2022-12-31")
    fechas_snap = pd.bdate_range("2007-01-01", "2022-12-31")
    (tmp_path / "intentos.csv").write_text("activo,hipotesis,etapa,sharpe\ncartera,h-c,2,0.5\n")
    eq_dir = tmp_path / "salida" / "equity"
    eq_dir.mkdir(parents=True)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(len(fechas), 0.001))}).to_csv(eq_dir / "rsi-SPY.csv", index=False)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(len(fechas), 0.001))}).to_csv(eq_dir / "sma-TLT.csv", index=False)

    h = Hipotesis("h-c", "autor", "just", "1d", piernas=(Pierna("rsi", {}, ("SPY",), "U"), Pierna("sma", {}, ("TLT",), "U")))
    monkeypatch.setattr(validacion, "correr_motor", lambda *a, **k: [
        {"activo": "SPY", "estrategia": "rsi", "trades": "4", "pnl": "1000", "sharpe_tasa_0": "0.5", "max_drawdown": "0.05"},
        {"activo": "TLT", "estrategia": "sma", "trades": "2", "pnl": "200", "sharpe_tasa_0": "0.2", "max_drawdown": "0.08"},
    ])
    monkeypatch.setattr(validacion, "dsr_contra_intentos", lambda *a, **k: 0.0)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas_snap)}))

    veredicto, filas = validacion.validar_etapa3(h, "sintetico", {("rsi", "SPY"): {}, ("sma", "TLT"): {}}, tmp_path / "salida")
    assert veredicto == "PASA"
    assert filas[0]["v30"] == "PASA"
    assert filas[0]["motivos"] == []
    assert filas[0]["dsr"] == 0.0


def test_etapa3_cartera_2_piernas_3_activos_agrega_una_fila_intentos(tmp_path, monkeypatch):
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.bdate_range("2019-01-01", "2022-12-31")
    fechas_snap = pd.bdate_range("2007-01-01", "2022-12-31")
    (tmp_path / "intentos.csv").write_text("activo,hipotesis,etapa,sharpe\ncartera,h-c2x3,2,0.5\n")
    eq_dir = tmp_path / "salida" / "equity"
    eq_dir.mkdir(parents=True)
    p1_activos = ("SPY", "TLT", "GLD")
    p2_activos = ("EEM", "IWM", "DIA")
    for a in p1_activos:
        pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(len(fechas), 0.001))}).to_csv(eq_dir / f"rsi-{a}.csv", index=False)
    for a in p2_activos:
        pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(len(fechas), 0.0005))}).to_csv(eq_dir / f"sma-{a}.csv", index=False)

    h = Hipotesis(
        "h-c2x3", "autor", "just", "1d",
        piernas=(
            Pierna("rsi", {"rsi_period": [7]}, p1_activos, "U"),
            Pierna("sma", {"fast": [20]}, p2_activos, "U"),
        ),
    )
    filas_falsas = [
        {"activo": a, "estrategia": "rsi", "trades": "5", "pnl": "100", "sharpe_tasa_0": "0.4", "max_drawdown": "0.02"} for a in p1_activos
    ] + [
        {"activo": a, "estrategia": "sma", "trades": "5", "pnl": "100", "sharpe_tasa_0": "0.4", "max_drawdown": "0.02"} for a in p2_activos
    ]
    monkeypatch.setattr(validacion, "correr_motor", lambda *a, **k: filas_falsas)
    monkeypatch.setattr(validacion, "dsr_contra_intentos", lambda *a: 0.96)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas_snap)}))

    veredicto, filas = validacion.validar_etapa3(h, "sintetico", {(p.clase, a): {} for p in h.piernas for a in p.activos}, tmp_path / "salida")
    assert veredicto == "PASA"
    assert len(filas) == 6
    with (tmp_path / "intentos.csv").open() as f:
        intentos = list(csv.DictReader(f))
    assert len(intentos) == 2
    intento = intentos[1]
    assert intento["activo"] == "cartera"
    assert intento["etapa"] == "3"
    assert intento["sharpe_base"] == "tasa_0"
    assert intento["trades"] == "30"
    assert intento["desde"] == "2019-01-01"
    assert intento["hasta"] == "2022-12-31"


def test_etapa3_cartera_sin_etapa2_o_benchmark_corta(tmp_path, monkeypatch):
    """Falla si se saltea el chequeo de etapa 2 (falta intento en intentos.csv o falta benchmark etapa 2)."""
    from quantagent.fabrica.hipotesis import Hipotesis, Pierna

    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.bdate_range("2019-01-01", "2022-12-31")
    eq_dir = tmp_path / "salida" / "equity"
    eq_dir.mkdir(parents=True)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(len(fechas), 0.001))}).to_csv(eq_dir / "rsi-SPY.csv", index=False)
    pd.DataFrame({"date": fechas, "equity": 100_000 * np.cumprod(1 + np.full(len(fechas), 0.0005))}).to_csv(eq_dir / "sma-TLT.csv", index=False)

    h = Hipotesis("h-c", "autor", "just", "1d", piernas=(Pierna("rsi", {}, ("SPY",), "U"), Pierna("sma", {}, ("TLT",), "U")))
    monkeypatch.setattr(validacion, "correr_motor", lambda *a, **k: [
        {"activo": "SPY", "estrategia": "rsi", "trades": "4", "pnl": "1000", "sharpe_tasa_0": "0.5", "max_drawdown": "0.05"},
        {"activo": "TLT", "estrategia": "sma", "trades": "2", "pnl": "200", "sharpe_tasa_0": "0.2", "max_drawdown": "0.08"},
    ])
    monkeypatch.setattr(validacion, "dsr_contra_intentos", lambda *a, **k: 0.90)

    # Caso 1: snapshot sin 2007-2018 (sb2 is None) y sin intentos.csv (s2 is None)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas)}))
    veredicto, filas = validacion.validar_etapa3(h, "sintetico", {("rsi", "SPY"): {}, ("sma", "TLT"): {}}, tmp_path / "salida")
    assert veredicto == "CORTA"
    assert filas[0]["v30"] == "NO PASA"
    assert "sin Sharpe de etapa 2" in filas[0]["motivos"]
    assert "sin benchmark de etapa 2" in filas[0]["motivos"]

    # Caso 2: con benchmark pero sin fila de etapa 2 en intentos.csv
    fechas_snap = pd.bdate_range("2007-01-01", "2022-12-31")
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas_snap)}))
    veredicto, filas = validacion.validar_etapa3(h, "sintetico", {("rsi", "SPY"): {}, ("sma", "TLT"): {}}, tmp_path / "salida")
    assert veredicto == "CORTA"
    assert "sin Sharpe de etapa 2" in filas[0]["motivos"]
    assert "sin benchmark de etapa 2" not in filas[0]["motivos"]

    # Caso 3: con fila de etapa 2 pero sin benchmark
    (tmp_path / "intentos.csv").write_text("activo,hipotesis,etapa,sharpe\ncartera,h-c,2,0.5\n")
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas)}))
    veredicto, filas = validacion.validar_etapa3(h, "sintetico", {("rsi", "SPY"): {}, ("sma", "TLT"): {}}, tmp_path / "salida")
    assert veredicto == "CORTA"
    assert "sin Sharpe de etapa 2" not in filas[0]["motivos"]
    assert "sin benchmark de etapa 2" in filas[0]["motivos"]





