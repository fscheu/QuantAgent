"""QuantAgent-d4cq.74 (M06): la cartera nucleo + pendiente de curva carga con sus 3 piernas y pasa la admision."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.data.macro import crear_snapshot_macro
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.hipotesis import cargar_hipotesis

RAIZ = Path(__file__).parent
YAML = RAIZ.parent / "hipotesis" / "cartera-nucleo-curva.yaml"
NUCLEO = RAIZ.parent / "hipotesis" / "cartera-tendencia-momentum.yaml"
SNAPSHOT = RAIZ / "fixtures" / "snapshots" / "etf-1d-2026-10-amplio.json"
FIN = pd.Timestamp("9999-12-31")


class ClienteSintetico:
    """T10Y2Y que cambia de signo dentro de las velas de la fixture, para que la pierna opere."""

    def observaciones(self, serie, desde, hasta, todas_las_versiones=True):
        filas = [("2010-01-01", 0.8, "2010-01-01", FIN), ("2011-06-01", -0.3, "2011-06-01", FIN),
                 ("2012-06-01", 0.6, "2012-06-01", FIN)]
        return pd.DataFrame(
            [(pd.Timestamp(f), v, pd.Timestamp(a), pd.Timestamp(b)) for f, v, a, b in filas],
            columns=["fecha", "valor", "realtime_start", "realtime_end"],
        )


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


def test_la_pierna_de_curva_es_de_bonos_medios_y_largos_dentro_del_snapshot_y_sin_parametros_libres():
    h = cargar_hipotesis(YAML)
    simbolos = set(json.loads(SNAPSHOT.read_text())["symbols"])
    curva = h.piernas[2]
    assert curva.clase == "curva-pendiente"
    assert list(curva.activos) == ["AGG", "IEF", "LQD", "TIP", "TLT"] and set(curva.activos) < simbolos
    assert curva.parametros == {"snapshot": ["macro-2026-10"], "serie": ["T10Y2Y"]}
    assert "SHY queda afuera" in curva.universo
    assert all(len(p.combinaciones()) == 1 for p in h.piernas)
    assert "10.1257/0002828053828581" in h.fuente


def test_admision_acepta_la_cartera(velas, tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    crear_snapshot_macro("macro-2026-10", ["T10Y2Y"], ClienteSintetico(), base_dir=tmp_path)
    assert admitir(cargar_hipotesis(YAML), velas) == (True, [])
