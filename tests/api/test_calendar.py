from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import calendar as calendar_route
from apps.api.calendar_sync import sync_calendar
from apps.api.rebalances import proximos_rebalanceos
from quantagent.database import Base
from quantagent.models import Account, CalendarEvent

HIPOTESIS = Path(__file__).resolve().parents[2] / "hipotesis"


def _cuenta(strategy, id="a", status="active"):
    return SimpleNamespace(id=id, name=f"Cuenta {id}", strategy=strategy, status=status)


def _cargar(strategy):
    from quantagent.fabrica.hipotesis import cargar_hipotesis

    return cargar_hipotesis(HIPOTESIS / f"{strategy}.yaml")


def test_rebalance_mensual_saltea_feriado():
    fechas = proximos_rebalanceos([_cuenta("sma-tendencia")], date(2026, 11, 1), 4,
                                  {date(2026, 11, 2)}, _cargar)
    assert [(cuenta.id, dia) for cuenta, dia in fechas] == [("a", date(2026, 11, 3))]


def test_rebalance_semanal_usa_lunes_o_martes_si_es_feriado():
    cuenta = _cuenta("m5-h1")
    lunes = proximos_rebalanceos([cuenta], date(2026, 11, 2), 3, set(), _cargar)
    feriado = proximos_rebalanceos([cuenta], date(2026, 11, 2), 3, {date(2026, 11, 2)}, _cargar)
    assert [dia for _, dia in lunes] == [date(2026, 11, 2)]
    assert [dia for _, dia in feriado] == [date(2026, 11, 3)]


def test_rebalance_fin_de_mes_son_cinco_sesiones():
    fechas = proximos_rebalanceos([_cuenta("rebalanceo-fin-de-mes")], date(2026, 11, 24), 7, set(), _cargar)
    assert [dia for _, dia in fechas] == [date(2026, 11, 24), date(2026, 11, 25), date(2026, 11, 26),
                                           date(2026, 11, 27), date(2026, 11, 30)]


def test_rebalance_omite_yaml_ausente_y_respeta_ventana():
    fechas = proximos_rebalanceos([_cuenta("missing"), _cuenta("sma-tendencia", id="b")],
                                  date(2026, 11, 1), 2, set(), _cargar)
    assert [(cuenta.id, dia) for cuenta, dia in fechas] == [("b", date(2026, 11, 2))]


class AlpacaDay:
    def __init__(self, day, close):
        self.date = date.fromisoformat(day)
        self.close = time.fromisoformat(close)


def test_sync_upserts_holiday_and_early_close_idempotently(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'calendar.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[CalendarEvent.__table__])
    days = [AlpacaDay("2026-11-25", "16:00"), AlpacaDay("2026-11-27", "13:00"),
            AlpacaDay("2026-11-30", "16:00")]
    with Session(engine) as db:
        assert sync_calendar(db, lambda: days) == 2
        assert sync_calendar(db, lambda: days) == 2
        events = db.query(CalendarEvent).order_by(CalendarEvent.date, CalendarEvent.kind).all()
        assert [(e.date.isoformat(), e.kind, e.source) for e in events] == [
            ("2026-11-26", "holiday", "alpaca"), ("2026-11-27", "early_close", "alpaca")]


def test_calendar_endpoint_filters_window_and_formats_dates(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'calendar.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[CalendarEvent.__table__, Account.__table__])
    with Session(engine) as db:
        db.add_all([
            CalendarEvent(date=date(2026, 11, 26), kind="holiday", title="Thanksgiving", source="alpaca"),
            CalendarEvent(date=date(2026, 12, 1), kind="early_close", title="Early close", source="alpaca"),
            CalendarEvent(date=date(2027, 1, 15), kind="holiday", title="Future holiday", source="alpaca"),
        ])
        db.commit()

    with Session(engine) as db:
        result = calendar_route(days=30, db=db)
    assert result[0].date == "2026-11-26"
    assert result[0].kind == "holiday"
    assert len(result) == 1
