"""Tests de CurvaPendienteStrategy (QuantAgent-d4cq.72, M04): snapshot macro sintético, sin red ni clave FRED."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.backtesting.signals import bar_signals, check_no_lookahead_macro
from quantagent.data.macro import crear_snapshot_macro
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.strategy.curva_pendiente_strategy import CurvaPendienteStrategy
from quantagent.strategy.registry import cli_aliases

FIN = pd.Timestamp("9999-12-31")
H = cargar_hipotesis(Path(__file__).parent / "fixtures" / "hipotesis" / "rsi-valida.yaml")

# Pendiente: +0.5 desde el 03-02, negativa (-0.2) desde el 03-16, y el 03-23 se revisa lo del 03-16 a +0.3.
FILAS = [
    ("2020-03-02", 0.5, "2020-03-02", FIN),
    ("2020-03-16", -0.2, "2020-03-16", "2020-03-22"),
    ("2020-03-16", 0.3, "2020-03-23", FIN),
]


class ClienteSintetico:
    def observaciones(self, serie, desde, hasta, todas_las_versiones=True):
        return pd.DataFrame(
            [(pd.Timestamp(f), v, pd.Timestamp(a), pd.Timestamp(b)) for f, v, a, b in FILAS],
            columns=["fecha", "valor", "realtime_start", "realtime_end"],
        )


@pytest.fixture
def snap(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    crear_snapshot_macro("m", ["T10Y2Y"], ClienteSintetico(), base_dir=tmp_path)
    return tmp_path


def _velas(inicio="2020-02-24", n=40):
    idx = pd.bdate_range(inicio, periods=n)
    close = 100 + np.arange(n, dtype=float)
    return pd.DataFrame({"timestamp": idx, "open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)


def test_pendiente_positiva_entra_largo_y_negativa_sale(snap):
    velas = _velas()
    sen = bar_signals(CurvaPendienteStrategy("m"), velas, "TLT", "1d")
    dia = lambda s: pd.Timestamp(s)  # noqa: E731
    assert sen.loc[dia("2020-02-28"), "entry"] == 0  # todavía no hay pendiente conocida
    assert sen.loc[dia("2020-03-02"), "entry"] == 1 and sen.loc[dia("2020-03-13"), "entry"] == 1
    assert not sen.loc[dia("2020-03-13"), "exit"]
    assert sen.loc[dia("2020-03-16"), "entry"] == 0 and sen.loc[dia("2020-03-16"), "exit"]
    assert sen.loc[dia("2020-03-20"), "exit"]


def test_una_revision_posterior_no_cambia_la_senal_pasada(snap):
    sen = bar_signals(CurvaPendienteStrategy("m"), _velas(), "TLT", "1d")
    # El 03-16 se revisa a +0.3 recién el 03-23: ese lunes ya cuenta la revisión, pero los días 16 a 20 no.
    assert (sen.loc["2020-03-16":"2020-03-20", "entry"] == 0).all()
    assert sen.loc["2020-03-23", "entry"] == 1


def test_pasa_la_prueba_de_vintage_de_m03(snap):
    est = CurvaPendienteStrategy("m")
    cortes = [pd.Timestamp("2020-03-16"), pd.Timestamp("2020-03-20"), pd.Timestamp("2020-03-23")]
    assert check_no_lookahead_macro(lambda c: bar_signals(est, c, "TLT", "1d"), _velas(), cortes, "m") == []


def test_admitir_acepta_la_pierna_y_exige_snapshot(snap):
    velas = _velas()
    pierna = replace(H.piernas[0], clase="curva-pendiente", parametros={"snapshot": ["m"], "serie": ["T10Y2Y"]})
    assert admitir(replace(H, piernas=(pierna,)), velas) == (True, [])
    sin_snapshot = replace(pierna, parametros={"serie": ["T10Y2Y"]})
    ok, motivos = admitir(replace(H, piernas=(sin_snapshot,)), velas)
    assert not ok and "snapshot" in motivos[0]


def test_registrada_con_alias_y_requiere_macro():
    assert cli_aliases()["curva-pendiente"] == "CurvaPendienteStrategy"
    est = CurvaPendienteStrategy("m")
    assert est.requiere_macro and est.serie == "T10Y2Y"
