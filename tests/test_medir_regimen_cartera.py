import numpy as np
import pandas as pd
from scripts import medir_regimen_cartera as script


def test_contrastar_azar_diferencia_y_p_valor():
    idx = pd.bdate_range("2015-01-01", periods=100)
    ret = pd.Series(0.01, index=idx)
    reg = pd.Series(["alcista_alta_vol"] * 50 + ["bajista_alta_vol"] * 50, index=idx)
    obs, azar, p_val = script.contrastar_azar(ret, reg, n_rep=20, seed=1)
    assert np.isclose(obs, 0.0) and 0.0 <= p_val <= 1.0


def test_medir_produce_tabla_y_conclusion(monkeypatch, capsys):
    idx = pd.bdate_range("2015-01-01", periods=200)
    ret_c = pd.Series(np.linspace(0.001, 0.005, 200), index=idx)
    reg = pd.Series(["alcista_baja_vol", "bajista_baja_vol"] * 100, index=idx)

    monkeypatch.setattr(script, "calcular_retornos_cartera", lambda h, s: (ret_c, [pd.Series(0.002, index=idx), pd.Series(-0.001, index=idx)]))
    monkeypatch.setattr(script, "read_snapshot", lambda s, a: pd.DataFrame({"close": pd.Series(np.arange(200), index=idx)}))
    monkeypatch.setattr(script, "clasificar_regimen", lambda spy: reg)

    res = script.medir(n_rep=10)
    assert len(res["tabla"]) == 2 and res["decision"] == "se archiva"
    salida = capsys.readouterr().out
    assert "Régimen" in salida and "Cartera" in salida and "Conclusión: se archiva" in salida
