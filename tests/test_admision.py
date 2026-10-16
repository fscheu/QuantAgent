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
    monkeypatch.setattr(type(H), "combinaciones", lambda self: [{}])
    ok, motivos = admitir(replace(H, parametros=grilla), velas)
    assert not ok and len(motivos) == 1 and "7" in motivos[0] and "6" in motivos[0]


def test_justificacion_corta_y_activo_ajeno_juntos(velas):
    ok, motivos = admitir(replace(H, justificacion="corta", activos=("AAPL",)), velas)
    assert not ok and len(motivos) == 2
    assert "justificación" in motivos[0] and "AAPL" in motivos[1]


def test_estrategia_que_mira_el_futuro(velas, monkeypatch):
    monkeypatch.setattr(
        admision, "_senales", lambda h, c, v: pd.DataFrame({"entry": (v["close"].shift(-1) > v["close"]).astype(int)})
    )
    ok, motivos = admitir(H, velas)
    assert not ok and all("truncamiento" in m for m in motivos)
