"""Tests del screener semanal por sector y tabla candidatos_semanales (M5-32)."""

from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.fabrica.screener import (
    ETF_SECTOR,
    candidatos_semanales,
    candidatos_semanales_panel,
)


def _datos_sinteticos(tickers, etf, rets=None, fund_vals=None, fecha="2020-06-01"):
    fechas = pd.date_range("2020-01-01", periods=150, freq="D")
    precios = {etf: pd.Series(100.0, index=fechas)}
    rets = rets or {}
    for t in tickers:
        r63, r126 = rets.get(t, (0.0, 0.0))
        s = pd.Series(100.0, index=fechas)
        s.iloc[-1], s.iloc[-64] = 100.0 * (1.0 + r63), 100.0
        s.iloc[-127] = 100.0 * (1.0 + r63) / (1.0 + r126)
        precios[t] = s
    paneles = {
        m: pd.DataFrame(
            {t: (fund_vals or {}).get(t, {}).get(m, 0.1) for t in tickers},
            index=pd.DatetimeIndex([pd.Timestamp(fecha)]), dtype=float,
        )
        for m in ["roic_4t", "margen_op_yoy", "roe_4t", "margen_pretax_yoy", "aceleracion_ingresos", "crecimiento_eps"]
    }
    return precios, paneles


def test_etf_directo_etapa3():
    df = candidatos_semanales("2020-06-01", ["SPY", "EEM", "EWZ", "DESCONOCIDO"])
    assert list(df["etf"]) == ["SPY", "EEM", "EWZ"]
    assert (df["sector"] == "pais").all() and (df["simbolo"] == df["etf"]).all()
    assert df["score"].isna().all() and df["rank"].isna().all()


def test_zscore_recortado(monkeypatch):
    fecha, etf, tickers = "2020-06-01", "XLK", ["T1", "T2", "T3", "T4", "T5"]
    monkeypatch.setattr("quantagent.data.sp500.miembros", lambda c, f: tickers)
    monkeypatch.setattr("quantagent.data.sp500.sector", lambda c, t, f: ETF_SECTOR[etf])
    fund = {"T1": {"roic_4t": 100000.0}, "T5": {"roic_4t": -100000.0}}
    precios, paneles = _datos_sinteticos(tickers, etf, fund_vals=fund, fecha=fecha)
    res = candidatos_semanales(fecha, [etf], precios=precios, paneles_fundamentales=paneles)
    assert all(-3.0 <= row["calidad"] <= 3.0 and -3.0 <= row["score"] <= 3.0 for _, row in res.iterrows())


def test_exclusion_faltantes_y_cuartil_inferior_fr(monkeypatch):
    fecha, etf = "2020-06-01", "XLK"
    tickers = ["T_OK1", "T_OK2", "T_OK3", "T_1FALTA", "T_2FALTAS", "T_BAJO_FR"]
    monkeypatch.setattr("quantagent.data.sp500.miembros", lambda c, f: tickers)
    monkeypatch.setattr("quantagent.data.sp500.sector", lambda c, t, f: ETF_SECTOR[etf])
    rets = {"T_OK1": (0.2, 0.3), "T_OK2": (0.15, 0.25), "T_OK3": (0.1, 0.2), "T_1FALTA": (0.05, 0.1), "T_BAJO_FR": (-0.8, -0.9)}
    precios, paneles = _datos_sinteticos(tickers, etf, rets=rets, fund_vals={"T_1FALTA": {"crecimiento_eps": np.nan}}, fecha=fecha)
    del precios["T_2FALTAS"]
    res = candidatos_semanales(fecha, [etf], precios=precios, paneles_fundamentales=paneles, top=5)
    simbolos = set(res["simbolo"])
    assert "T_2FALTAS" not in simbolos and "T_BAJO_FR" not in simbolos and "T_1FALTA" in simbolos


def test_financials_usa_roe_y_pretax(monkeypatch):
    fecha, etf, tickers = "2020-06-01", "XLF", ["BANK_A", "BANK_B"]
    monkeypatch.setattr("quantagent.data.sp500.miembros", lambda c, f: tickers)
    monkeypatch.setattr("quantagent.data.sp500.sector", lambda c, t, f: ETF_SECTOR[etf])
    fund = {
        "BANK_A": {"roe_4t": 0.30, "margen_pretax_yoy": 0.10, "roic_4t": 0.01, "margen_op_yoy": 0.01},
        "BANK_B": {"roe_4t": 0.05, "margen_pretax_yoy": 0.01, "roic_4t": 0.50, "margen_op_yoy": 0.20},
    }
    precios, paneles = _datos_sinteticos(tickers, etf, fund_vals=fund, fecha=fecha)
    res = candidatos_semanales(fecha, [etf], precios=precios, paneles_fundamentales=paneles)
    assert res.iloc[0]["simbolo"] == "BANK_A" and res.iloc[0]["score"] > res.iloc[1]["score"]


def test_top3_y_desempate(monkeypatch):
    fecha, etf, tickers = "2020-06-01", "XLK", ["SYM_Z", "SYM_A", "SYM_M", "SYM_LOW"]
    monkeypatch.setattr("quantagent.data.sp500.miembros", lambda c, f: tickers)
    monkeypatch.setattr("quantagent.data.sp500.sector", lambda c, t, f: ETF_SECTOR[etf])
    fund = {"SYM_Z": {"roic_4t": 0.5}, "SYM_A": {"roic_4t": 0.5}, "SYM_M": {"roic_4t": 0.3}, "SYM_LOW": {"roic_4t": -0.2}}
    precios, paneles = _datos_sinteticos(tickers, etf, fund_vals=fund, fecha=fecha)
    res = candidatos_semanales(fecha, [etf], precios=precios, paneles_fundamentales=paneles, top=3)
    assert list(res["simbolo"]) == ["SYM_A", "SYM_Z", "SYM_M"]
    assert list(res["rank"]) == [1, 2, 3]


def test_candidatos_semanales_panel(monkeypatch):
    tickers, fechas = ["T1", "T2"], ["2020-06-05", "2020-06-12"]
    monkeypatch.setattr("quantagent.data.sp500.miembros", lambda c, f: tickers)
    monkeypatch.setattr("quantagent.data.sp500.sector", lambda c, t, f: "Information Technology")
    precios, _ = _datos_sinteticos(tickers, "XLK")
    idx = pd.DatetimeIndex([pd.Timestamp(f) for f in fechas])
    paneles = {m: pd.DataFrame(0.1, index=idx, columns=tickers) for m in [
        "roic_4t", "margen_op_yoy", "roe_4t", "margen_pretax_yoy", "aceleracion_ingresos", "crecimiento_eps",
    ]}
    res = candidatos_semanales_panel(fechas, ["XLK"], precios=precios, paneles_fundamentales=paneles)
    assert len(res) == 4 and set(res["fecha"]) == {pd.Timestamp(f) for f in fechas}


def test_planilla_aceptacion_coincide_top3():
    ruta_csv = Path("docs/06_implementation/QuantAgent-fnpr.4.3-IM-screener-2019-06-28.csv")
    assert ruta_csv.exists()
    df = pd.read_csv(ruta_csv)
    assert len(df) == 136
    assert df[(df["etf"] == "XLK") & (df["rank"] <= 3)].sort_values("rank")["simbolo"].tolist() == ["ORCL", "GPN", "MSFT"]
    assert df[(df["etf"] == "XLF") & (df["rank"] <= 3)].sort_values("rank")["simbolo"].tolist() == ["CINF", "BBT", "SPGI"]
