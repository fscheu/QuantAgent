"""QuantAgent-d4cq.33: las hipótesis de L11 cargan y pasan la admisión de etapa 0."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica.admision import _senales, admitir
from quantagent.fabrica.hipotesis import cargar_hipotesis

HIPOTESIS = Path(__file__).parent.parent / "hipotesis"


@pytest.fixture(scope="module")
def velas():
    # Sintéticas con deriva positiva y volumen variable: ~3 años, para que los cortes de la prueba
    # de truncamiento (1/2 y 3/4) caigan después de las 303 velas que pide el máximo de 52 semanas.
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.01, 800)))
    idx = pd.bdate_range("2010-01-04", periods=800)
    return pd.DataFrame(
        {"timestamp": idx, "open": close, "high": close * 1.005, "low": close * 0.995, "close": close,
         "volume": rng.uniform(500, 2000, 800)},
        index=idx,
    )


@pytest.mark.parametrize("archivo", ["maximo-52-semanas.yaml", "momentum-12m.yaml"])
def test_carga_y_admision(archivo, velas):
    h = cargar_hipotesis(HIPOTESIS / archivo)
    assert admitir(h, velas) == (True, [])
    # La admisión no prueba nada si la estrategia nunca entra: exige al menos una señal.
    assert (_senales(h, h.combinaciones()[0], velas)["entry"] != 0).any()
