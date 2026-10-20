import numpy as np
import pandas as pd
import pytest

from quantagent.data.snapshot import write_snapshot
from quantagent.fabrica.candado import abierto
from scripts.criterio_paper_desde_curvas import calcular_criterio_paper


def _velas(fechas, ret):
    c = 100.0 * np.cumprod(1.0 + ret)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "adj_close": c, "volume": 1000.0}, index=fechas)


def test_criterio_paper_desde_curvas_pasa_corta_y_sin_escrituras(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
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
