"""QuantAgent-d4cq.54: la cartera tendencia + momentum carga con sus 2 piernas y pasa la admisión."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica.admision import _senales, admitir
from quantagent.fabrica.hipotesis import cargar_hipotesis

RAIZ = Path(__file__).parent
YAML = RAIZ.parent / "hipotesis" / "cartera-tendencia-momentum.yaml"
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


def test_estructura_de_la_cartera():
    h = cargar_hipotesis(YAML)
    simbolos = set(json.loads(SNAPSHOT.read_text())["symbols"])
    assert [p.clase for p in h.piernas] == ["momentum-12m", "sma-cross"]
    assert h.piernas[0].parametros == {"lookback": [252]}
    assert h.piernas[1].parametros == {"fast": [1], "slow": [210]}
    for p in h.piernas:
        assert set(p.activos) == simbolos and len(p.activos) == 29
        assert len(p.combinaciones()) == 1
    assert "doi.org" in h.fuente


def test_admision_acepta_la_cartera(velas):
    h = cargar_hipotesis(YAML)
    assert admitir(h, velas) == (True, [])


def test_cada_pierna_genera_senales(velas):
    h = cargar_hipotesis(YAML)
    for p in h.piernas:
        assert (_senales(p, p.combinaciones()[0], velas)["entry"] != 0).any()
