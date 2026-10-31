import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/m5"))
import etapa3  # noqa: E402


def test_sesgo_semanal_peso_y_fundamentales_sin_red():
    fechas = pd.to_datetime(["2019-12-23", "2019-12-30", "2020-01-02"])
    decisiones = dict(zip(fechas, fechas - pd.Timedelta(days=3)))
    pesos = pd.DataFrame(0., index=fechas, columns=etapa3.h3.scr.ETF_SECTOR)
    pesos.loc[fechas[0], ["XLK", "XLF"]] = [.6, .4]
    pesos.loc[fechas[2], "XLK"] = .9
    tech, fin = "Information Technology", "Financials"
    filas = [(decisiones[fechas[0]], tech, ["rs3", "rs6"]) for _ in range(4)]
    filas += [(decisiones[fechas[0]], fin, ["rs3", "roic_4t", "crecimiento_eps"]),
              (decisiones[fechas[0]], fin, ["rs6"]),  # Un solo NaN no descarta.
              (decisiones[fechas[1]], tech, ["rs3", "rs6"]),  # Efectivo: sin exposición.
              (decisiones[fechas[2]], tech, ["rs6", "crecimiento_eps"])]
    detalle = pd.DataFrame(filas, columns=["fecha", "sector", "faltan"])
    res = etapa3.contar_sesgo(detalle, pesos, decisiones)
    tech_res = res[res.sector == tech]
    assert tech_res.descartes.tolist() == [4, 0, 1]
    assert tech_res.puestos_max.tolist() == [3, 0, 1]
    assert tech_res.peso_max.tolist() == pytest.approx([.6, 0, .9])
    assert res[res.sector == fin].peso_max.sum() == 0
    assert res.loc[res.peso_max > 0, "fecha"].nunique() == 2
    assert res.peso_max.sum() / res.fecha.nunique() == pytest.approx(.5)
    assert res.groupby("anio").peso_max.sum().to_dict() == {2019: .6, 2020: .9}
