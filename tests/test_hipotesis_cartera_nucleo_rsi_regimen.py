"""QuantAgent-d4cq.38 (M02): la cartera nucleo + RSI con regimen carga con sus 3 piernas y pasa la admision."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica.admision import admitir
from quantagent.fabrica.hipotesis import cargar_hipotesis

RAIZ = Path(__file__).parent
YAML = RAIZ.parent / "hipotesis" / "cartera-nucleo-rsi-regimen.yaml"
NUCLEO = RAIZ.parent / "hipotesis" / "cartera-tendencia-momentum.yaml"
SNAPSHOT = RAIZ / "fixtures" / "snapshots" / "etf-1d-2026-10-amplio.json"


@pytest.fixture(scope="module")
def velas():
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.01, 800)))
    idx = pd.bdate_range("2010-01-04", periods=800)
    return pd.DataFrame(
        {"timestamp": idx, "open": close, "high": close * 1.005, "low": close * 0.995, "close": close,
         "volume": rng.uniform(500, 2000, 800)},
        index=idx,
    )


def test_el_nucleo_es_el_de_la_cartera_tendencia_momentum():
    h, n = cargar_hipotesis(YAML), cargar_hipotesis(NUCLEO)
    assert h.piernas[:2] == n.piernas


def test_la_pierna_rsi_es_de_acciones_dentro_del_snapshot_y_sin_parametros_libres():
    h = cargar_hipotesis(YAML)
    simbolos = set(json.loads(SNAPSHOT.read_text())["symbols"])
    rsi = h.piernas[2]
    assert rsi.clase == "rsi-regimen"
    assert set(rsi.activos) < simbolos and len(rsi.activos) == 17
    assert not {"AGG", "DBC", "FXE", "GLD", "IEF", "LQD", "SHY", "SLV", "TIP", "TLT", "USO", "VNQ"} & set(rsi.activos)
    assert all(len(p.combinaciones()) == 1 for p in h.piernas)
    assert "10.1093/rfs/hhs066" in h.fuente


def test_admision_acepta_la_cartera(velas):
    assert admitir(cargar_hipotesis(YAML), velas) == (True, [])
