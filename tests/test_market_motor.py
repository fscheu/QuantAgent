import pandas as pd
import pytest
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
    pd.DataFrame([{"fecha": "2026-10-30"}]).to_csv(tmp_path / "m5-regimen-semanal.csv", index=False)
    with pytest.raises(ValueError, match="faltan columnas"):
        estado_motor(tmp_path)
