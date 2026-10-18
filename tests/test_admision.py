import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica import admision
from quantagent.fabrica.admision import admitir
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
