"""QuantAgent-d4cq.66: las carteras de las familias de primas cargan, citan el catálogo y pasan la admisión."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quantagent.fabrica.admision import admitir
from quantagent.fabrica.hipotesis import cargar_hipotesis

RAIZ = Path(__file__).parent.parent
CARTERAS = {"cartera-transversal": ["momentum-transversal", "baja-volatilidad"], "cartera-estacionalidad": ["halloween"]}
PRIMAS = {p["id"]: p for p in yaml.safe_load((RAIZ / "catalogo" / "primas.yaml").read_text())["primas"]}
# Prima del catálogo que respalda a cada clase.
PRIMA_DE = {"momentum-transversal": "momentum-transversal", "baja-volatilidad": "baja-volatilidad-transversal",
            "halloween": "estacionalidad-halloween"}


def _cargar(nombre):
    return cargar_hipotesis(RAIZ / "hipotesis" / f"{nombre}.yaml")


@pytest.mark.parametrize("nombre", CARTERAS)
def test_piernas_universo_y_cero_parametros_libres(nombre):
    h = _cargar(nombre)
    assert [p.clase for p in h.piernas] == CARTERAS[nombre]
    for p in h.piernas:
        assert p.parametros == {} and p.combinaciones() == [{}]
        prima = PRIMAS[PRIMA_DE[p.clase]]
        assert sorted(p.activos) == sorted(prima["universo"]["activos"])
        assert len(p.universo) > 40  # dice el criterio de definición, no solo una lista


@pytest.mark.parametrize("nombre", CARTERAS)
def test_la_cita_de_cada_fuente_coincide_con_el_catalogo(nombre):
    h = _cargar(nombre)
    citados = set(re.findall(r"doi\.org/(\S+)", h.fuente))
    esperados = {d for p in h.piernas for d in PRIMAS[PRIMA_DE[p.clase]]["fuente"]}
    assert citados == esperados


@pytest.mark.parametrize("nombre", CARTERAS)
def test_admision_acepta_la_cartera(nombre):
    h = _cargar(nombre)
    activos = sorted({a for p in h.piernas for a in p.activos})
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2010-01-04", periods=900)
    cierres = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (900, len(activos))), axis=0),
                           index=idx, columns=activos)
    c = cierres[activos[0]]
    velas = pd.DataFrame({"timestamp": idx, "open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1.0},
                         index=idx)
    assert admitir(h, velas, [], cierres) == (True, [])
