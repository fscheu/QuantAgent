import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.data.snapshot import write_snapshot
from quantagent.fabrica import validacion
from quantagent.fabrica.hipotesis import cargar_hipotesis


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
