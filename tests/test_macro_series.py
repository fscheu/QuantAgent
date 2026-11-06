from datetime import date
from decimal import Decimal

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from apps.api.market_macro import recession_periods, series_macro
from quantagent.data.macro_series import update_macro_series
from quantagent.database import Base
from quantagent.models import MacroObservation


class FredFake:
    def __init__(self, values):
        self.values = values

    def observaciones(self, serie, desde, hasta, todas_las_versiones=False):
        return self.values[serie]


def frame(days, values):
    return pd.DataFrame({"fecha": pd.to_datetime(days), "valor": values})


def database():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[MacroObservation.__table__])
    return engine


def test_ingestion_discards_missing_and_preserves_first_seen_date():
    engine = database()
    fred = FredFake({"T10Y2Y": frame(["2024-01-01", "2024-01-02"], [1.25, "."])})
    with Session(engine) as session:
        assert update_macro_series(session, fred, ("T10Y2Y",), start=date(2024, 1, 1)) == {"T10Y2Y": 1}
        original = session.get(MacroObservation, ("T10Y2Y", date(2024, 1, 1))).published_at
        fred.values["T10Y2Y"] = frame(["2024-01-01"], [2.0])
        assert update_macro_series(session, fred, ("T10Y2Y",), start=date(2024, 1, 1)) == {"T10Y2Y": 0}
        row = session.get(MacroObservation, ("T10Y2Y", date(2024, 1, 1)))
        assert row.value == Decimal("2.00000000")
        assert row.published_at == original
        assert len(session.scalars(select(MacroObservation)).all()) == 1


def test_macro_change_one_month_and_stale_business_day_boundaries():
    engine = database()
    with Session(engine) as session:
        update_macro_series(session, FredFake({"T10Y2Y": frame(
            ["2024-01-01", "2024-01-31"], [1.0, 1.4]
        )}), ("T10Y2Y",), start=date(2024, 1, 1))
        assert series_macro(session, today=date(2024, 2, 5))[0]["change_1m"] == 0.4
        assert series_macro(session, today=date(2024, 2, 5))[0]["stale"] is False
        assert series_macro(session, today=date(2024, 2, 6))[0]["stale"] is False
        assert series_macro(session, today=date(2024, 2, 7))[0]["stale"] is True
        assert series_macro(session, today=date(2024, 2, 6), expected_release=date(2024, 2, 2))[0]["stale"] is False
        assert series_macro(session, today=date(2024, 2, 7), expected_release=date(2024, 2, 2))[0]["stale"] is True


def test_stale_weekend_and_monday_count_only_business_days():
    engine = database()
    with Session(engine) as session:
        update_macro_series(session, FredFake({"T10Y2Y": frame(["2024-02-02"], [1.0])}),
                            ("T10Y2Y",), start=date(2024, 2, 2))
        assert series_macro(session, today=date(2024, 2, 5))[0]["stale"] is False
        assert series_macro(session, today=date(2024, 2, 8))[0]["stale"] is False
        assert series_macro(session, today=date(2024, 2, 9))[0]["stale"] is True


def test_recession_runs_including_open_run_and_empty_series():
    engine = database()
    with Session(engine) as session:
        assert recession_periods(session) == []
        update_macro_series(session, FredFake({"USREC": frame(
            ["2001-01-01", "2001-02-01", "2001-03-01", "2001-04-01", "2001-05-01"],
            [0, 1, 1, 0, 1],
        )}), ("USREC",), start=date(2001, 1, 1))
        assert recession_periods(session) == [
            {"start": "2001-02-01", "end": "2001-03-31"},
            {"start": "2001-05-01", "end": "2001-05-01"},
        ]
