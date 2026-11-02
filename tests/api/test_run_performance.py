from datetime import date, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import portfolio_performance, run_performance
from quantagent.database import Base
from quantagent.models import Account, EquitySnapshot


def test_run_performance_is_the_assigned_account_report(tmp_path):
    master_engine = create_engine(f"sqlite:///{tmp_path / 'master.db'}")
    account_engine = create_engine(f"sqlite:///{tmp_path / 'account.db'}")
    Base.metadata.create_all(master_engine)
    Base.metadata.create_all(account_engine)
    with Session(master_engine) as db:
        db.add(Account(id="paper-alpha", name="Cuenta Paper", env="paper", kind="simulated",
                       database_url=f"sqlite:///{tmp_path / 'account.db'}", strategy="Estrategia Alpha",
                       status="active"))
        db.commit()
    with Session(account_engine) as db:
        start = date(2026, 1, 1)
        dates = [start + timedelta(days=day) for day in range(25)]
        db.add_all(EquitySnapshot(snapshot_date=day, taken_at=datetime.combine(day, datetime.min.time()),
                                  equity=10000 + index * 25, cash=10000, invested=0)
                   for index, day in enumerate(dates))
        db.commit()

    try:
        with Session(master_engine) as db:
            portfolio_report = portfolio_performance("paper-alpha", "all", db)
            run_report = run_performance("paper-alpha", "all", db)
        assert run_report["subject"]["type"] == "run"
        assert run_report["subject"]["name"] == "Estrategia Alpha"
        assert run_report["subject"]["backtest_id"] is None
        assert run_report["metrics"] == portfolio_report["metrics"]
        assert run_report["series"] == portfolio_report["series"]
    finally:
        master_engine.dispose()
        account_engine.dispose()
