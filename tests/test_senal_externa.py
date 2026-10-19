"""Tests para QuantAgent-d4cq.63 (P05.4): SenalExterna y celdas transversales en validación."""
import subprocess
from pathlib import Path
import pandas as pd
import pytest

from quantagent.fabrica import validacion
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.hipotesis import Hipotesis, Pierna
from quantagent.strategy.registry import build_strategy, cli_aliases
from quantagent.strategy.senal_externa_strategy import SenalExterna


def test_senal_externa_entry_exit(tmp_path):
    csv_file = tmp_path / "senal.csv"
    dates = pd.date_range("2020-01-01", periods=4)
    pd.DataFrame({"timestamp": dates, "posicion": [0, 1, 1, 0]}).to_csv(csv_file, index=False)
    strat = SenalExterna(ruta=str(csv_file))
    assert strat.generate_signal([{"timestamp": dates[0]}], "SPY", "1d", 100.0) is None
    sig1 = strat.generate_signal([{"timestamp": dates[1]}], "SPY", "1d", 100.0)
    assert sig1 is not None and sig1.decision == "LONG"
    df2 = pd.DataFrame({"timestamp": dates[:3], "close": [100.0, 101.0, 102.0]})
    df3 = pd.DataFrame({"timestamp": dates[:4], "close": [100.0, 101.0, 102.0, 101.0]})
    assert strat.should_exit(None, 102.0, df2) == (False, None)
    assert strat.should_exit(None, 101.0, df3) == (True, "SENAL_EXTERNA_EXIT")
    assert "senal-externa" in cli_aliases()
    assert isinstance(build_strategy("SenalExterna", ruta=str(csv_file)), SenalExterna)


def test_admitir_rechaza_senal_externa():
    p = Pierna("senal-externa", {}, ("SPY",), "U")
    h = Hipotesis("h-bad", "autor", "x" * 50, "1d", piernas=(p,))
    velas = pd.DataFrame({"timestamp": pd.date_range("2010-01-01", periods=10), "close": 100.0})
    ok, motivos = admitir(h, velas)
    assert not ok and any("clase interna" in m for m in motivos)


def test_etapa2_celda_difiere_corta(tmp_path, monkeypatch):
    fechas = pd.date_range("2010-01-01", periods=30)
    p = Pierna("momentum-transversal", {}, ("SPY", "TLT"), "U")
    h = Hipotesis("h-t", "autor", "x" * 50, "1d", piernas=(p,))

    def fake_subp(cmd, **kwargs):
        if "--strategies" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            sym = cmd[cmd.index("--symbols") + 1]
            out.write_text(f"activo,estrategia,trades,pnl,sharpe_tasa_0,max_drawdown\n{sym},senal-externa,5,100,0.5,0.02\n")
            eq = out.parent / "equity" / f"senal-externa-{sym}.csv"
            eq.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": fechas, "equity": 100_000 * (1 + 0.001)}).to_csv(eq, index=False)
        elif "--tabla" in cmd:
            t = Path(cmd[cmd.index("--tabla") + 1])
            sym = "SPY" if "SPY" in t.name else "TLT"
            ctrl = "coinciden 5 de 5" if sym == "SPY" else "sin validar: distinta cantidad de trades (motor=5, port=4)"
            t.write_text(f"activo,control\n{sym},{ctrl}\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validacion.subprocess, "run", fake_subp)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas)}))
    monkeypatch.setattr(validacion, "evaluar_etapa2_cartera", lambda *a, **k: (True, []))

    params = {a: {} for a in ("SPY", "TLT")}
    veredicto, filas = validacion.validar_etapa2(h, "sintetico", params, tmp_path / "salida")
    assert veredicto == "CORTA"
    assert any("sin validar" in m for f in filas for m in f["motivos"])


def test_etapa2_activo_sin_trades_no_es_fallo(tmp_path, monkeypatch):
    fechas = pd.date_range("2010-01-01", periods=30)
    p = Pierna("momentum-transversal", {}, ("SPY", "TLT"), "U")
    h = Hipotesis("h-t2", "autor", "x" * 50, "1d", piernas=(p,))

    def fake_subp(cmd, **kwargs):
        if "--strategies" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            sym = cmd[cmd.index("--symbols") + 1]
            trades = 5 if sym == "SPY" else 0
            out.write_text(f"activo,estrategia,trades,pnl,sharpe_tasa_0,max_drawdown\n{sym},senal-externa,{trades},100,0.5,0.02\n")
            eq = out.parent / "equity" / f"senal-externa-{sym}.csv"
            eq.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": fechas, "equity": 100_000 * (1 + 0.001)}).to_csv(eq, index=False)
        elif "--tabla" in cmd:
            t = Path(cmd[cmd.index("--tabla") + 1])
            t.write_text("activo,control\nSPY,coinciden 5 de 5\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(validacion.subprocess, "run", fake_subp)
    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", lambda snap, sym: pd.DataFrame({"close": pd.Series(100.0, index=fechas)}))
    monkeypatch.setattr(validacion, "evaluar_etapa2_cartera", lambda *a, **k: (True, []))

    params = {a: {} for a in p.activos}
    veredicto, filas = validacion.validar_etapa2(h, "sintetico", params, tmp_path / "salida")
    assert veredicto == "PASA"
    tlt = next(f for f in filas if f["activo"] == "TLT")
    assert tlt["control"] == "coinciden 0 de 0"
    assert tlt["v30"] == "PASA" and tlt["motivos"] == []


def test_ranking_etapa3_no_usa_cierres_posteriores_al_corte(tmp_path, monkeypatch):
    fechas = pd.date_range("2018-01-01", "2024-01-01", freq="B")
    p = Pierna("momentum-transversal", {}, ("SPY", "TLT", "GLD"), "U")
    h = Hipotesis("h-t3", "autor", "x" * 50, "1d", piernas=(p,))
    corte = "2022-12-31"

    recibido_hasta = {}
    def fake_read(snap, sym):
        return pd.DataFrame({"close": pd.Series(100.0, index=fechas)})

    def fake_subp(cmd, **kwargs):
        if "--strategies" in cmd:
            sym = cmd[cmd.index("--symbols") + 1]
            recibido_hasta[sym] = cmd[cmd.index("--to") + 1]
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"activo,estrategia,trades,pnl,sharpe_tasa_0,max_drawdown\n{sym},senal-externa,5,100,0.5,0.02\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", fake_read)
    monkeypatch.setattr(validacion.subprocess, "run", fake_subp)

    carpeta = tmp_path / "salida"
    params = {a: {} for a in p.activos}
    validacion.correr_motor(h, "sintetico", params, "2019-01-01", corte, carpeta, "3", abrir_prueba=True)
    assert all(h == corte for h in recibido_hasta.values())
    for a in p.activos:
        df_senal = pd.read_csv(carpeta / f"senal-momentum-transversal-{a}.csv")
        assert pd.to_datetime(df_senal["timestamp"]).max() <= pd.to_datetime(corte)
    senales_antes = {a: (carpeta / f"senal-momentum-transversal-{a}.csv").read_text() for a in p.activos}

    def fake_read_mutado(snap, sym):
        s = pd.Series(100.0, index=fechas)
        s.loc["2023-06-01":] = 999999.0
        return pd.DataFrame({"close": s})

    monkeypatch.setattr("quantagent.data.snapshot.read_snapshot", fake_read_mutado)
    carpeta2 = tmp_path / "salida2"
    validacion.correr_motor(h, "sintetico", params, "2019-01-01", corte, carpeta2, "3", abrir_prueba=True)
    senales_despues = {a: (carpeta2 / f"senal-momentum-transversal-{a}.csv").read_text() for a in p.activos}
    assert senales_antes == senales_despues
