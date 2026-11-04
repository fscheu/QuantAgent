import pandas as pd
import pytest
from datetime import date, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quantagent.database import Base
from quantagent.models import MarketBar

from apps.api.market_motor import SECTORS, UNIVERSE, estado_motor, filas_sector


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__])
    with Session(engine) as session:
        yield session


def test_motor_habilitados_bloqueados_y_score(tmp_path, db):
    symbols = list(SECTORS)
    row = {"fecha": "2026-10-30", "estado": "risk-on", "habilitados": ",".join(symbols[:3]),
           "bloqueados": symbols[3]}
    row.update({f"score_{s}": 1 - i for i, s in enumerate(UNIVERSE)})
    pd.DataFrame([row]).to_csv(tmp_path / "m5-regimen-semanal.csv", index=False)
    assert estado_motor(tmp_path)["state"] == "risk_on"
    rows = filas_sector(db, tmp_path)
    assert [r["rank"] for r in rows[:3]] == [1, 2, 3]
    assert rows[0]["enabled"] and rows[3]["blocked"]
    assert rows[3]["blocked_reason"] == "filtro de caída"
    assert all(r["blocked"] and r["blocked_reason"] for r in rows[3:11])


def test_archivos_ausentes_y_formato_invalido(tmp_path, db):
    assert estado_motor(tmp_path) is None
    rows = filas_sector(db, tmp_path)
    assert all(not r["enabled"] and not r["blocked"] and r["momentum_score"] is None for r in rows)
    assert rows[0]["returns"] == {key: None for key in ("d1", "w1", "m1", "m3", "ytd")}
    pd.DataFrame([{"fecha": "2026-10-30"}]).to_csv(tmp_path / "m5-regimen-semanal.csv", index=False)
    with pytest.raises(ValueError, match="faltan columnas"):
        estado_motor(tmp_path)


def test_retornos_sector_en_porcentaje_con_barras_sinteticas(db):
    start = date(2024, 10, 1)
    for i in range(64):
        close = 100 + i
        db.add(MarketBar(symbol="XLB", session_date=start + timedelta(days=i), open=close,
                         high=close, low=close, close=close, volume=1, source="test"))
    db.commit()
    row = next(row for row in filas_sector(db) if row["symbol"] == "XLB")
    assert row["returns"] == pytest.approx({
        "d1": (163 / 162 - 1) * 100,
        "w1": (163 / 158 - 1) * 100,
        "m1": (163 / 143 - 1) * 100,
        "m3": (163 / 101 - 1) * 100,
        "ytd": None,
    })
