"""Tests de la hipótesis cambio-de-mes (QuantAgent-rdcf, puv2.2)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from quantagent.backtesting.signals import bar_signals
from quantagent.fabrica.admision import _senales, admitir
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica.propuestas import entradas_de_hipotesis

HIPOTESIS = Path(__file__).resolve().parents[1] / "hipotesis"
YAML = HIPOTESIS / "cambio-de-mes.yaml"


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


def test_cambio_de_mes_carga_con_cero_parametros_libres_y_pasa_admision(velas):
    h = cargar_hipotesis(YAML)
    assert h.id == "cambio-de-mes"
    assert h.clase == "turn-of-the-month"
    assert set(h.activos) == {"EEM", "EFA", "EWJ", "EWZ", "FXI", "IWM", "QQQ", "SPY"}
    libres = sum(sum(len(v) > 1 for v in p.parametros.values()) for p in h.piernas)
    assert libres == 0
    assert admitir(h, velas) == (True, [])
    assert (_senales(h, h.combinaciones()[0], velas)["entry"] == 1).any()


def test_hipotesis_dir_carga_completa_para_proponer_y_tanda():
    archivos = sorted(HIPOTESIS.glob("*.yaml"))
    assert YAML in archivos
    cargadas = [cargar_hipotesis(f) for f in archivos]
    assert any(h.id == "cambio-de-mes" for h in cargadas)
    h_cdm = next(h for h in cargadas if h.id == "cambio-de-mes")
    entradas = entradas_de_hipotesis(h_cdm)
    assert len(entradas) == 1 and entradas[0]["clase"] == "turn-of-the-month"
