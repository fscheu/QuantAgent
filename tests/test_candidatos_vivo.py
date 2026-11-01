"""Adaptador vivo sobre snapshots temporales y precios sintéticos (M5-55b)."""

import json

import numpy as np
import pandas as pd
import pytest

from quantagent.data.fundamentales import metricas_fundamentales
from quantagent.data.snapshot import write_snapshot
from quantagent.data.sp500 import CURRENT_CSV, HISTORICAL_CSV, crear_snapshot_sp500
from quantagent.fabrica.candidatos_vivo import candidatos_en_vivo
from quantagent.fabrica.screener import COLS_CANDIDATOS, candidatos_semanales

FUND = "fundamentales-test"
ACCIONES = ["AAA", "BBB", "CCC", "DDD"]


class Proveedor:
    def __init__(self, precios):
        self.precios = precios
        self.pedidos = []

    def get_ohlc(self, simbolo, timeframe, inicio, fin):
        self.pedidos.append((simbolo, timeframe, inicio, fin))
        s = self.precios[simbolo]
        s = s[(s.index >= inicio) & (s.index < fin)]
        return pd.DataFrame({"timestamp": s.index, "close": s.to_numpy()})


@pytest.fixture
def datos(tmp_path):
    csv = tmp_path / "csv"
    csv.mkdir()
    sectores = [(t, "Information Technology") for t in ACCIONES] + [("BANK", "Financials")]
    pd.DataFrame([{"date": "2026-08-18", "tickers": ",".join(t for t, _ in sectores)}]).to_csv(csv / HISTORICAL_CSV, index=False)
    pd.DataFrame([{"Symbol": t, "GICS Sector": s} for t, s in sectores]).to_csv(csv / CURRENT_CSV, index=False)
    crear_snapshot_sp500("sp500-2026-10", base_dir=tmp_path, csv_dir=csv)
    frames = {}
    for k, t in enumerate(ACCIONES):
        filas = []
        for i, fin in enumerate(pd.date_range("2024-09-30", periods=8, freq="QE")):
            for concepto, valor in {"ingresos": 100 + i * (k + 1), "OperatingIncomeLoss": 20 + i * (k + 1),
                                    "EarningsPerShareDiluted": 2 + i, "StockholdersEquity": 200}.items():
                filas.append({"ticker": t, "concepto": concepto, "val": valor,
                              "start": str((fin - pd.offsets.QuarterEnd() + pd.Timedelta(days=1)).date()),
                              "end": str(fin.date()), "filed": str((fin + pd.Timedelta(days=30)).date()), "accn": str(i)})
        frames[t] = pd.DataFrame(filas, index=pd.to_datetime([r["end"] for r in filas]))
    write_snapshot(FUND, frames, source="sec-edgar", base_dir=tmp_path)
    fechas = pd.bdate_range("2025-11-03", "2026-11-03")
    precios = {t: pd.Series(100 * np.exp(np.arange(len(fechas)) * k / 10000), index=fechas)
               for k, t in enumerate(["XLK", *ACCIONES, "SPY"])}
    return tmp_path, precios


@pytest.mark.parametrize("historia", [None, 100])
def test_equivalencia_con_screener_y_universo(datos, historia):
    base, precios = datos
    if historia:
        precios["DDD"] = precios["DDD"].loc[:"2026-10-30"].iloc[-historia:]
    proveedor = Proveedor(precios)
    fecha = pd.Timestamp("2026-10-30")
    paneles = metricas_fundamentales(FUND, ACCIONES, pd.DatetimeIndex([fecha]), base_dir=base)
    esperado = candidatos_semanales(fecha, ["XLK", "SPY"], precios=precios,
                                   paneles_fundamentales=paneles, composicion=str(base / "sp500-2026-10"))
    res, descartes = candidatos_en_vivo("2026-10-31T12:00:00Z", iter(["XLK", "SPY"]), proveedor, FUND, base_dir=base)
    pd.testing.assert_frame_equal(res, esperado)
    assert set(res["simbolo"]) >= {"CCC", "DDD", "SPY"}
    assert descartes == []
    assert {p[0] for p in proveedor.pedidos} == {"XLK", "SPY", *ACCIONES}
    assert all(p[1:] == ("1d", pd.Timestamp("2025-10-30"), pd.Timestamp("2026-10-31")) for p in proveedor.pedidos)


@pytest.mark.parametrize("historia", [0, 63, -30])
def test_descarta_accion_sin_precio_util(datos, historia):
    base, precios = datos
    s = precios["DDD"].loc[:"2026-10-30"]
    precios["DDD"] = s.iloc[:0] if historia == 0 else s.iloc[-historia:] if historia > 0 else s.iloc[:historia]
    res, descartes = candidatos_en_vivo("2026-10-31T12:00:00Z", ["XLK"], Proveedor(precios), FUND, base_dir=base)
    assert "DDD" not in set(res["simbolo"])
    assert len(res) > 0
    assert len(descartes) == 1 and descartes[0]["simbolo"] == "DDD"
    assert "sin precio" in descartes[0]["motivo"]


def test_accion_fuera_del_snapshot(datos):
    base, precios = datos
    ruta = base / FUND / "manifest.json"
    manifest = json.loads(ruta.read_text())
    del manifest["symbols"]["DDD"]
    ruta.write_text(json.dumps(manifest))
    res, descartes = candidatos_en_vivo("2026-10-31T12:00:00Z", ["XLK"], Proveedor(precios), FUND, base_dir=base)
    assert "DDD" not in set(res["simbolo"]) and len(res) > 0
    assert descartes == [{"simbolo": "DDD", "motivo": f"sin fundamentales en snapshot {FUND}"}]


@pytest.mark.parametrize("snapshot", [None, "inexistente", "vacio"])
def test_sin_fundamentales_no_consulta_precios(datos, snapshot):
    base, precios = datos
    if snapshot == "vacio":
        carpeta = base / snapshot
        carpeta.mkdir()
        (carpeta / "manifest.json").write_text(json.dumps({"symbols": {}}))
    proveedor = Proveedor(precios)
    res, descartes = candidatos_en_vivo("2026-10-31T12:00:00Z", ["XLK", "SPY"], proveedor, snapshot, base_dir=base)
    assert res.empty and list(res.columns) == COLS_CANDIDATOS
    assert len(descartes) == 1 and descartes[0]["simbolo"] is None
    assert "fundamentales" in descartes[0]["motivo"]
    assert proveedor.pedidos == []


@pytest.mark.parametrize("as_of, sesion", [("2026-10-30T19:59:00Z", "2026-10-29"),
                                          ("2026-10-30T20:01:00Z", "2026-10-30")])
def test_reloj_inyectado_antes_y_despues_del_cierre(datos, as_of, sesion):
    base, precios = datos
    proveedor = Proveedor(precios)
    res, _ = candidatos_en_vivo(as_of, ["XLK"], proveedor, FUND, base_dir=base)
    assert set(res["fecha"]) == {pd.Timestamp(sesion)}
    assert all(p[3] == pd.Timestamp(sesion) + pd.Timedelta(days=1) for p in proveedor.pedidos)
