from datetime import date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quantagent.data.snapshot import write_snapshot
from quantagent.database import Base
from quantagent.digest_m5 import build_digest_m5, consultar_datos_m5
from quantagent.digest_m5_semanal import comparacion_sombra, resumen_semanal
from quantagent.models import EquitySnapshot, Order, OrderSide, OrderStatus, OrderType, Position, TradeIntentRecord


def test_candidatos_descartes_y_falla_global(tmp_path):
    pd.DataFrame([dict(fecha="2026-10-23", habilitados="XLK,XLE"),
                  dict(fecha="2026-10-30", habilitados="XLF")]).to_csv(tmp_path / "m5-regimen-semanal.csv", index=False)
    pd.DataFrame([dict(fecha="2026-10-23", etf="XLK", simbolo="AAPL", rank=2),
                  dict(fecha="2026-10-23", etf="XLK", simbolo="MSFT", rank=1)]).to_csv(
                      tmp_path / "m5-candidatos-semanales.csv", index=False)
    pd.DataFrame([dict(fecha="2026-10-23", simbolo="XOM", motivo="sin precio"),
                  dict(fecha="2026-10-23", simbolo="CVX", motivo="sin fundamentales"),
                  dict(fecha="2026-10-30", simbolo="", motivo="screener: falla global")]).to_csv(
                      tmp_path / "m5-candidatos-descartes.csv", index=False)
    texto = "\n".join(resumen_semanal(date(2026, 10, 29), None, 3000, carpeta=tmp_path))
    assert "Information Technology (XLK): MSFT (rank 1), AAPL (rank 2)" in texto
    assert "Energy (XLE): solo-ETF; sin precio; sin fundamentales" in texto
    assert "Descartadas por precio: 1; XOM: sin precio" in texto
    assert "Descartadas por fundamentales: 1; CVX: sin fundamentales" in texto
    assert "2026-10 (28 días" in texto
    assert "XLF" not in texto
    texto = "\n".join(resumen_semanal(date(2026, 10, 30), None, 3000, carpeta=tmp_path))
    assert "Financials (XLF): solo-ETF; screener: falla global" in texto
    assert "AAPL" not in texto
    write_snapshot("fundamentales-sp500-2026-10-28",
                   {"AAPL": pd.DataFrame({"val": [1]}, index=pd.to_datetime(["2026-10-28"]))},
                   base_dir=tmp_path)
    texto = "\n".join(resumen_semanal(date(2026, 10, 30), None, 3000, carpeta=tmp_path))
    assert "Fundamentales vigentes: 2026-10-28 (2 días" in texto


@pytest.mark.parametrize("faltante", [False, True])
def test_retornos_sombra_semanales_acumulados_y_sin_precio(tmp_path, faltante):
    class Precios:
        def get_ohlc(self, symbol, timeframe, start, end):
            precios = [100, 110, 121] if symbol != "AAPL" else [100, 120, 108]
            df = pd.DataFrame(dict(timestamp=pd.to_datetime(["2026-10-16", "2026-10-23", "2026-10-30"]),
                                   close=precios))
            return df.iloc[:-1] if faltante and symbol == "AAPL" else df

    sombra = pd.DataFrame([dict(fecha=f, activo=s, peso_h1=h1, peso_h3=h3)
                           for f in ["2026-10-16", "2026-10-23"]
                           for s, h1, h3 in [("XLK", 1, 0.5), ("AAPL", 0, 0.5)]] +
                          [dict(fecha="2026-10-30", activo="FUTURO", peso_h1=1, peso_h3=1)])
    engine = create_engine(f"sqlite:///{tmp_path / 'm5.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([EquitySnapshot(snapshot_date=date(2026, 10, d), taken_at=datetime(2026, 10, d),
                                        equity=e, cash=e, invested=0) for d, e in [(16, 3000), (23, 3300)]])
        session.commit()
        texto = "\n".join(comparacion_sombra(date(2026, 10, 30), sombra, Precios(), session, 3630))
    assert "H1: +10.00% / +21.00%" in texto
    assert "1/N: +10.00% / +21.00%" in texto
    assert ("H3: sin dato / sin dato" if faltante else "H3: +0.00% / +15.00%") in texto
    assert "Pipeline: +10.00% / +21.00%" in texto


def test_acciones_fraccionarias_y_sin_actividad(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'm5.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        orden = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
                      quantity=0.123456, status=OrderStatus.FILLED)
        session.add(orden)
        session.flush()
        session.add_all([Position(symbol="AAPL", quantity=0.123456, average_entry_price=100,
                                  current_price=100, unrealized_pnl=0, unrealized_pnl_pct=0, side=OrderSide.BUY),
                         TradeIntentRecord(bucket_id="m5", strategy_id="test", symbol="AAPL", status="aprobado",
                                           order_id=orden.id, created_at=datetime(2026, 10, 29),
                                           intent={"entry_price": 100, "stop_loss": 90})])
        session.commit()
        texto = build_digest_m5(date(2026, 10, 29), consultar_datos_m5(session, date(2026, 10, 29)))
        assert "AAPL 0.1235 @ 100.00, stop 90.00" in texto
        assert "AAPL: 0.1235" in texto
        texto = build_digest_m5(date(2026, 10, 30), consultar_datos_m5(session, date(2026, 10, 30)))
        assert "sin actividad" in texto.lower()
        assert "AAPL: 0.1235" in texto
