from click.testing import CliRunner
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.orm import Session

from quantagent.cli.reconcile import trading_group
from quantagent.database import Base
from quantagent.models import EquitySnapshot


class Broker:
    def __init__(self, fail=False):
        self.fail = fail

    def get_account(self):
        if self.fail:
            raise RuntimeError("offline")
        return {"equity": "120", "cash": "30"}


def test_snapshot_upserts_and_broker_failure_does_not_write(tmp_path, monkeypatch):
    db = tmp_path / "portfolio.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine, tables=[])
    engine.dispose()
    broker = Broker()
    monkeypatch.setattr("quantagent.cli.reconcile.crear_broker_alpaca_paper", lambda: broker)
    runner = CliRunner()
    args = ["snapshot", "--db", str(db), "--portfolio", "paper-m4"]
    assert runner.invoke(trading_group, args).exit_code == 0
    broker.get_account = lambda: {"equity": "130", "cash": "35"}
    assert runner.invoke(trading_group, args).exit_code == 0
    with Session(create_engine(f"sqlite:///{db}")) as session:
        assert session.scalar(select(func.count()).select_from(EquitySnapshot)) == 1
        row = session.scalar(select(EquitySnapshot))
        assert (row.equity, row.cash, row.invested) == (130, 35, 95)
    broker.get_account = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    result = runner.invoke(trading_group, args)
    assert result.exit_code != 0
    assert "Alpaca account query failed" in result.output
    failed_db = tmp_path / "failed.db"
    result = runner.invoke(trading_group, ["snapshot", "--db", str(failed_db), "--portfolio", "paper-m4"])
    assert result.exit_code != 0
    failed_engine = create_engine(f"sqlite:///{failed_db}")
    assert not inspect(failed_engine).has_table("equity_snapshots")
    failed_engine.dispose()
