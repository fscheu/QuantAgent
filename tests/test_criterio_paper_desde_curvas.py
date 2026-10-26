import numpy as np
import pandas as pd
import pytest

from quantagent.data.snapshot import write_snapshot
from quantagent.fabrica.candado import abierto
from quantagent.fabrica.hipotesis import Hipotesis, Pierna
from scripts.criterio_paper_desde_curvas import calcular_criterio_paper, main, reconstruir_curva_cartera


def _velas(fechas, ret):
    c = 100.0 * np.cumprod(1.0 + ret)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "adj_close": c, "volume": 1000.0}, index=fechas)


def _preparar_fixtures(tmp_path):
    f2, f3 = pd.bdate_range("2017-01-02", "2018-12-31"), pd.bdate_range("2019-01-02", "2022-12-30")
    fechas, osc2, osc3 = f2.append(f3), 1e-4 * np.sin(np.arange(len(f2))), 1e-4 * np.sin(np.arange(len(f3)))
    ret_car3 = np.where(f3.year == 2021, -0.0003, 0.0015) + osc3
    ret_bajo = np.concatenate([np.full(len(f2), 0.0001) + osc2, np.where(f3.year == 2021, -0.0005, 0.0002) + osc3])
    ret_alto3 = np.concatenate([np.full(len(f2), 0.0001) + osc2, np.full(len(f3), 0.004) + osc3])
    write_snapshot("snap_bajo", {"SPY": _velas(fechas, ret_bajo), "TLT": _velas(fechas, ret_bajo)}, base_dir=tmp_path)
    write_snapshot("snap_alto", {"SPY": _velas(fechas, ret_alto3), "TLT": _velas(fechas, ret_alto3)}, base_dir=tmp_path)
    yaml_path = tmp_path / "cartera-sint.yaml"
    yaml_path.write_text("id: cartera-sint\nautor: fede\njustificacion: test\ntimeframe: 1d\npiernas:\n  - {clase: rsi, parametros: {rsi_period: [14]}, activos: [SPY], universo: U}\n  - {clase: sma-cross, parametros: {fast: [50]}, activos: [TLT], universo: U}\n", encoding="utf-8")
    e2, e3 = tmp_path / "etapa2" / "cartera-sint" / "equity", tmp_path / "etapa3" / "cartera-sint" / "equity"
    e2.mkdir(parents=True)
    e3.mkdir(parents=True)
    for nombre in ("rsi-SPY.csv", "sma-cross-TLT.csv"):
        pd.DataFrame({"date": f2, "equity": 100_000 * np.cumprod(1 + 0.001 + osc2)}).to_csv(e2 / nombre, index=False)
        pd.DataFrame({"date": f3, "equity": 100_000 * np.cumprod(1 + ret_car3)}).to_csv(e3 / nombre, index=False)
    return yaml_path, e2, e3


def test_criterio_paper_desde_curvas_pasa_corta_y_sin_escrituras(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    yaml_path, _, e3 = _preparar_fixtures(tmp_path)
    registros = [tmp_path / n for n in ("intentos.csv", "hipotesis.csv", "hipotesis-candado.csv")]
    res = calcular_criterio_paper(yaml_path, tmp_path / "etapa2", tmp_path / "etapa3", "snap_bajo")
    assert res["veredicto"] == "PASA" and res["pasa"] is True and res["motivos"] == []
    assert res["anios_positivos"] == (3, 4) and 0 < res["max_drawdown"] <= 0.35
    assert res["sharpe_etapa2"] > res["benchmark_etapa2"] and res["sharpe_etapa3"] > res["benchmark_etapa3"]
    assert not abierto("cartera-sint", "prueba") and all(not r.exists() for r in registros)
    for r in registros:
        r.write_text("intacto\n", encoding="utf-8")
    res_corta = calcular_criterio_paper(yaml_path, tmp_path / "etapa2", tmp_path / "etapa3", "snap_alto")
    assert res_corta["veredicto"] == "CORTA" and any("Sharpe de cartera" in m and "<= benchmark" in m for m in res_corta["motivos"])
    assert not abierto("cartera-sint", "prueba") and all(r.read_text(encoding="utf-8") == "intacto\n" for r in registros)
    (e3 / "sma-cross-TLT.csv").unlink()
    with pytest.raises(FileNotFoundError, match="sma-cross-TLT"):
        calcular_criterio_paper(yaml_path, tmp_path / "etapa2", tmp_path / "etapa3", "snap_bajo")


def test_criterio_paper_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    yaml_path, _, e3 = _preparar_fixtures(tmp_path)
    int_csv = tmp_path / "intentos.csv"
    int_csv.write_text("hipotesis,etapa,estrategia,params,activo,desde,hasta,sharpe,sharpe_base\nh1,2,rsi,{},SPY,2007-01-01,2018-12-31,0.8,tasa_0\nh1,2,sma,{},TLT,2007-01-01,2018-12-31,0.5,tasa_0\n", encoding="utf-8")
    txt_int = int_csv.read_text(encoding="utf-8")
    assert main([str(yaml_path), str(tmp_path / "etapa2"), str(tmp_path / "etapa3"), "snap_bajo"]) == 0
    out, _ = capsys.readouterr()
    for esperada in ("cartera-sint", "Sharpe etapa 3", "Benchmark etapa 3", "3 de 4", "DSR con N efectivo (2)", "DSR con N crudo (2)", "Recargo de paper", "Veredicto: PASA"):
        assert esperada in out
    assert int_csv.read_text(encoding="utf-8") == txt_int and not (tmp_path / "hipotesis.csv").exists() and not abierto("cartera-sint", "prueba")
    assert main([str(yaml_path), str(tmp_path / "etapa2"), str(tmp_path / "etapa3"), "snap_alto"]) == 0
    out_corta, _ = capsys.readouterr()
    assert "Veredicto: CORTA" in out_corta and "Motivos:" in out_corta and "<= benchmark" in out_corta
    (e3 / "sma-cross-TLT.csv").unlink()
    assert main([str(yaml_path), str(tmp_path / "etapa2"), str(tmp_path / "etapa3"), "snap_bajo"]) == 1
    _, err = capsys.readouterr()
    assert "Curva faltante: sma-cross-TLT" in err
    assert int_csv.read_text(encoding="utf-8") == txt_int and not (tmp_path / "hipotesis.csv").exists()


@pytest.mark.parametrize("etapa", ["etapa2", "etapa3"])
@pytest.mark.parametrize("layout", ["raiz", "corrida", "equity", "plano"])
def test_reconstruccion_transversal_t4_por_layout(tmp_path, etapa, layout):
    h = Hipotesis("cartera-sint", "fede", "test", piernas=(Pierna("momentum-transversal", {}, ("SPY", "TLT"), "U"),))
    inicio = "2007-01-02" if etapa == "etapa2" else "2019-01-02"
    fechas = pd.bdate_range(inicio, periods=3)
    cierres = pd.DataFrame({"SPY": [100, 110, 121], "TLT": [100, 90, 81]}, index=fechas)
    root = tmp_path / etapa
    corrida = root / h.id if layout in ("raiz", "plano") else root
    equity = corrida if layout == "plano" else corrida / "equity"
    equity.mkdir(parents=True)
    entrada = equity if layout == "equity" else root if layout in ("raiz", "plano") else corrida
    for activo, valores in (("SPY", [100, 110, 121]), ("TLT", [100, 90, 81])):
        pd.DataFrame({"date": fechas, "equity": valores}).to_csv(equity / f"momentum-transversal-{activo}.csv", index=False)
    pd.DataFrame({"timestamp": fechas, "posicion": [1, 1, 1]}).to_csv(
        corrida / "senal-momentum-transversal-SPY.csv", index=False)
    pd.DataFrame({"timestamp": fechas, "posicion": [0, 0, 0]}).to_csv(
        corrida / "senal-momentum-transversal-TLT.csv", index=False)

    curva = reconstruir_curva_cartera(h, entrada, cierres)

    pd.testing.assert_series_equal(curva, pd.Series([110_000.0, 121_000.0], index=fechas[1:], name=None), check_freq=False)
    (corrida / "senal-momentum-transversal-TLT.csv").unlink()
    with pytest.raises(FileNotFoundError, match="senal-momentum-transversal-TLT.csv"):
        reconstruir_curva_cartera(h, entrada, cierres)
