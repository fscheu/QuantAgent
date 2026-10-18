"""QuantAgent-d4cq.62: etapa 1 y admisión con piernas transversales."""
import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica import transversal
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.cartera import combinar, pesos
from quantagent.fabrica.filtro import filtrar_pierna, veredicto_etapa1
from quantagent.fabrica.hipotesis import Hipotesis, Pierna

COSTOS = {"commission_pct": 0.0, "slippage_pct": 0.0}
ACTIVOS = ("SPY", "QQQ", "IWM", "EFA", "EEM", "XLE")


def _velas(n=365 * 9, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    out = {}
    for i, a in enumerate(ACTIVOS):
        close = 100 * np.cumprod(1 + rng.normal(0.0002 * (i - 2), 0.01, n))
        out[a] = pd.DataFrame({"timestamp": idx, "open": close, "high": close * 1.01, "low": close * 0.99,
                               "close": close, "volume": 1e6}, index=idx)
    return out


def _hipotesis(clase="momentum-transversal"):
    pierna = Pierna(clase=clase, parametros={}, activos=ACTIVOS, universo="ETFs de acciones")
    return Hipotesis(id="t", autor="a", justificacion="x" * 50, piernas=(pierna,)), pierna


@pytest.fixture(scope="module")
def resultado():
    h, p = _hipotesis()
    paneles = _velas()
    return h, p, paneles, filtrar_pierna(h, p, paneles, costos=COSTOS, capital=100_000.0)


def test_los_retornos_fuera_de_muestra_los_acepta_combinar_junto_a_una_pierna_por_activo(resultado):
    _, _, paneles, r = resultado
    ret = r["retornos"].copy()
    assert list(ret.columns) == [(0, a) for a in ACTIVOS]
    assert ret.index.min().year == 2014 and ret.index.is_unique  # 4 años de ajuste: la primera prueba arranca en 2014
    cierres = pd.DataFrame({a: v["close"] for a, v in paneles.items()})
    ret[(1, "SPY")] = cierres["SPY"].pct_change().reindex(ret.index).fillna(0.0)
    w = pesos(cierres, {0: list(ACTIVOS), 1: ["SPY"]}).reindex(ret.index)
    cartera = combinar(ret, w.dropna())
    assert cartera.notna().all() and cartera.abs().max() > 0


def test_los_trades_son_la_suma_de_entradas_de_todos_los_activos_y_cuentan_para_la_cartera(resultado):
    _, p, paneles, r = resultado
    cierres = pd.DataFrame({a: v["close"] for a, v in paneles.items()})
    panel = transversal.señales_transversales(cierres, transversal.METRICAS[p.clase])
    entradas = ((panel == 1) & (panel.shift(1, fill_value=0) == 0)).loc["2014-01-01":]
    assert r["trades"] > 0 and abs(r["trades"] - int(entradas.sum().sum())) <= len(ACTIVOS) * 2  # bordes de ventana
    assert r["trades"] >= 30
    assert veredicto_etapa1(r["sharpe_tasa_0"], r["trades"])[1] != "menos de 30 trades en la cartera"


def test_el_resultado_no_depende_de_velas_posteriores(resultado):
    h, p, paneles, r = resultado
    recorte = {a: v.loc[:"2017-12-31"] for a, v in paneles.items()}
    r2 = filtrar_pierna(h, p, recorte, costos=COSTOS, capital=100_000.0)
    comun = r2["retornos"].loc[:"2017-12-31"]
    pd.testing.assert_frame_equal(comun, r["retornos"].loc[comun.index[0]:comun.index[-1]], check_freq=False)


def _cierres_admision(n=900):
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, 6)), axis=0), index=idx, columns=list(ACTIVOS))


def _velas_admision(cierres):
    c = cierres["SPY"]
    return pd.DataFrame({"timestamp": cierres.index, "open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=cierres.index)


def test_admitir_acepta_una_clase_transversal(monkeypatch):
    h, _ = _hipotesis()
    cierres = _cierres_admision()
    assert admitir(h, _velas_admision(cierres), [], cierres) == (True, [])


def test_admitir_rechaza_senal_externa_y_una_metrica_que_mira_el_futuro(monkeypatch):
    cierres = _cierres_admision()
    velas = _velas_admision(cierres)
    h, _ = _hipotesis("senal-externa")
    ok, motivos = admitir(h, velas, [], cierres)
    assert not ok and any("senal-externa" in m for m in motivos)

    monkeypatch.setitem(transversal.METRICAS, "trampa", lambda c: c.shift(-5) / c - 1)
    h, _ = _hipotesis("trampa")
    ok, motivos = admitir(h, velas, [], cierres)
    assert not ok and any("truncamiento" in m for m in motivos)
