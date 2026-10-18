import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica import admision
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.critico import cargar_existentes, criticar
from quantagent.fabrica.hipotesis import cargar_hipotesis

H = cargar_hipotesis(Path(__file__).parent / "fixtures" / "hipotesis" / "rsi-valida.yaml")


@pytest.fixture
def velas():
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, 120))
    idx = pd.bdate_range("2010-01-04", periods=120)
    return pd.DataFrame(
        {"timestamp": idx, "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
        index=idx,
    )


def test_valida_con_velas_sinteticas(velas):
    assert admitir(H, velas) == (True, [])


def test_siete_parametros_libres(velas, monkeypatch):
    monkeypatch.setattr(admision, "_senales", lambda h, c, v: pd.DataFrame({"entry": 0}, index=v.index))
    grilla = {f"p{i}": [1, 2] for i in range(7)}
    monkeypatch.setattr(admision.Pierna, "combinaciones", lambda self: [{}])
    ok, motivos = admitir(replace(H, piernas=(replace(H.piernas[0], parametros=grilla),)), velas)
    assert not ok and len(motivos) == 1 and "7" in motivos[0] and "6" in motivos[0]


def test_dos_piernas_cuatro_parametros_libres_corta(velas, monkeypatch):
    monkeypatch.setattr(admision, "_senales", lambda h, c, v: pd.DataFrame({"entry": 0}, index=v.index))
    monkeypatch.setattr(admision.Pierna, "combinaciones", lambda self: [{}])
    p1 = replace(H.piernas[0], parametros={f"p{i}": [1, 2] for i in range(4)})
    p2 = replace(H.piernas[0], parametros={f"q{i}": [1, 2] for i in range(4)})
    h2 = replace(H, piernas=(p1, p2))
    ok, motivos = admitir(h2, velas)
    assert not ok
    assert any("8 parámetros libres" in m for m in motivos)


def test_clase_no_registrada(velas):
    p_invalida = replace(H.piernas[0], clase="no-existe")
    h = replace(H, piernas=(p_invalida,))
    ok, motivos = admitir(h, velas)
    assert not ok
    assert any("clase no registrada" in m for m in motivos)


def test_justificacion_corta_y_activo_ajeno_juntos(velas):
    ok, motivos = admitir(replace(H, justificacion="corta", piernas=(replace(H.piernas[0], activos=("AAPL",)),)), velas)
    assert not ok and len(motivos) == 2
    assert "justificación" in motivos[0] and "AAPL" in motivos[1]


def test_estrategia_que_mira_el_futuro(velas, monkeypatch):
    monkeypatch.setattr(
        admision, "_senales", lambda h, c, v: pd.DataFrame({"entry": (v["close"].shift(-1) > v["close"]).astype(int)})
    )
    ok, motivos = admitir(H, velas)
    assert not ok and all("truncamiento" in m for m in motivos)


def test_universo_coincide_con_el_manifiesto_amplio():
    manifiesto = Path(__file__).parent / "fixtures" / "snapshots" / "etf-1d-2026-10-amplio.json"
    assert admision.UNIVERSO_AMPLIO == frozenset(json.loads(manifiesto.read_text())["symbols"])


def test_simbolo_del_universo_ampliado_se_admite_y_uno_ajeno_sigue_rechazado(velas):
    # XLV está entre los 29 del manifiesto pero no estaba en los 12 de M2; AAPL no está en ninguno.
    ok, motivos = admitir(replace(H, piernas=(replace(H.piernas[0], activos=("SPY", "XLV")),)), velas)
    assert (ok, motivos) == (True, [])
    ok, motivos = admitir(replace(H, piernas=(replace(H.piernas[0], activos=("SPY", "XLV", "AAPL")),)), velas)
    assert not ok and len(motivos) == 1 and "AAPL" in motivos[0] and "XLV" not in motivos[0]


def _pierna(clase, activos, parametros):
    return replace(H.piernas[0], clase=clase, activos=tuple(activos), parametros=parametros)


RSI = {"rsi_period": [14]}
SMA = {"fast": [10], "slow": [50]}


def test_critico_cartera_con_piernas_en_otro_orden_es_duplicada():
    a, b = _pierna("rsi", ["SPY"], RSI), _pierna("sma-cross", ["QQQ"], SMA)
    existente = replace(H, id="cartera-a", piernas=(a, b))
    nueva = replace(H, id="cartera-b", piernas=(b, a))
    assert criticar(nueva, [existente]) == ("duplicada", "duplicada de cartera-a", "cartera-a")


def test_critico_otro_activo_es_variante_con_la_familia_de_la_mas_antigua():
    vieja = replace(H, id="vieja", piernas=(_pierna("rsi", ["SPY"], RSI),))
    posterior = replace(H, id="posterior", piernas=(_pierna("rsi", ["SPY", "QQQ"], RSI),))
    nueva = replace(H, id="nueva", piernas=(_pierna("rsi", ["QQQ"], RSI),))
    veredicto, _, familia = criticar(nueva, [vieja, posterior])
    assert (veredicto, familia) == ("variante", "vieja")
    assert criticar(replace(nueva, piernas=(_pierna("rsi", ["QQQ"], {"rsi_period": [7]}),)), [vieja])[0] == "variante"


def test_admitir_rechaza_duplicada_y_acepta_variante(velas):
    existente = replace(H, id="ya-existe")
    ok, motivos = admitir(replace(H, id="copia"), velas, [existente])
    assert not ok and motivos == ["duplicada de ya-existe"]
    variante = replace(H, id="variante", piernas=(replace(H.piernas[0], activos=("QQQ",)),))
    assert admitir(variante, velas, [existente]) == (True, [])


def test_las_hipotesis_actuales_pasan_la_etapa_0_unas_contra_otras(velas):
    existentes = cargar_existentes()
    assert len(existentes) >= 5
    for h in existentes:
        assert criticar(h, existentes)[0] != "duplicada", h.id


def test_universo_se_congela_en_la_primera_admision(velas, tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    assert admitir(H, velas, []) == (True, [])
    assert (tmp_path / "hipotesis-universos.csv").read_text().count("\n") == 1
    # misma hipótesis y universo, otra grilla: sigue admitida y no agrega filas
    h_grilla = replace(H, piernas=(replace(H.piernas[0], parametros={"rsi_period": [7, 14], "oversold_threshold": [30.0]}),))
    assert admitir(h_grilla, velas, []) == (True, [])
    assert (tmp_path / "hipotesis-universos.csv").read_text().count("\n") == 1
    ok, motivos = admitir(replace(H, piernas=(replace(H.piernas[0], activos=("SPY", "QQQ")),)), velas, [])
    assert not ok and len(motivos) == 1 and motivos[0].startswith("universo distinto del declarado el ")
    assert motivos[0].endswith(" (D2)")


def test_hipotesis_rechazada_no_congela_universo(velas, tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    ok, _ = admitir(replace(H, justificacion="corta"), velas, [])
    assert not ok and not (tmp_path / "hipotesis-universos.csv").exists()
