from datetime import date, datetime, timezone
import json

from click.testing import CliRunner
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from apps.api.accounts import add_account
from apps.api.alarm_config import AlarmConfig
from apps.api.alarms import evaluate_account_alarms
from apps.api.app import account_summary, home_status
from apps.api.cli import api_group
from quantagent.database import Base
from quantagent.models import Account, CalendarEvent, EquitySnapshot, Event, Order, OrderSide, Position


def _account_values(db_url):
    return dict(id="acct", name="Cuenta", env="paper", kind="simulated", database_url=db_url,
                strategy="demo")


def _source(path, *, positions=True, snapshots=True):
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    tables = [Position.__table__, Order.__table__]
    if snapshots:
        tables.append(EquitySnapshot.__table__)
    Base.metadata.create_all(engine, tables=tables)
    with Session(engine) as db:
        if positions:
            db.add(Position(symbol="A", quantity=10, average_entry_price=10, current_price=10,
                unrealized_pnl=0, unrealized_pnl_pct=0, side=OrderSide.BUY,
                opened_at=datetime(2026, 10, 30), updated_at=datetime(2026, 10, 30)))
        if snapshots:
            db.add_all([EquitySnapshot(snapshot_date=day, taken_at=taken, equity=value, cash=0, invested=value)
                for day, taken, value in [(date(2026, 10, 29), datetime(2026, 10, 29), 100),
                                          (date(2026, 10, 30), datetime(2026, 10, 30), 90)]])
        db.commit()
    engine.dispose()
    return url


def _master(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'master.db'}")
    Base.metadata.create_all(engine, tables=[Account.__table__, CalendarEvent.__table__, Event.__table__])
    return engine


def test_drawdown_and_stale_and_shadow_evaluation(tmp_path):
    source_url = _source(tmp_path / "account.db")
    engine = _master(tmp_path)
    with Session(engine) as master:
        account = add_account(master, **_account_values(source_url))
        assert account_summary("acct", master).alarms[0].kind == "drawdown_limit"
        alarms = evaluate_account_alarms(account, master, datetime(2026, 10, 30, 20, tzinfo=timezone.utc))
        assert {alarm.kind for alarm in alarms} == {"drawdown_limit"}
        assert alarms[0].severity == "critical"
        master.add(Event(type="drawdown_limit", severity="critical", occurred_at=datetime(2026, 10, 30, 19, tzinfo=timezone.utc),
            account_id="acct", entity_id=None, payload={"drawdown_pct": 10, "limit_pct": 8}))
        master.commit()
        assert evaluate_account_alarms(account, master, datetime(2026, 10, 30, 20, tzinfo=timezone.utc))[0].since.startswith("2026-10-30T19")
        source = create_engine(source_url)
        with Session(source) as db:
            db.query(EquitySnapshot).delete()
            db.add_all([EquitySnapshot(snapshot_date=date(2026, 10, day), taken_at=datetime(2026, 10, 30, 19), equity=100, cash=0, invested=100) for day in (29, 30)])
            db.commit()
        source.dispose()
        shadow_file = (tmp_path / "account.sombra.json")
        shadow_file.write_text(json.dumps({"2026-10-30": {"acct": {"pesos": {"A": 0.7}}}}))
        alarms = evaluate_account_alarms(account, master, datetime(2026, 10, 30, 20, tzinfo=timezone.utc))
        assert [a.kind for a in alarms] == ["shadow_divergence"]
        shadow_file.unlink()
        assert evaluate_account_alarms(account, master,
            datetime(2026, 10, 30, 20, tzinfo=timezone.utc)) == []
    engine.dispose()


def test_market_stale_threshold_and_holiday(tmp_path):
    engine = _master(tmp_path)
    url = _source(tmp_path / "stale.db", snapshots=False)
    with Session(engine) as master:
        account = add_account(master, **(_account_values(url) | {"id": "stale"}))
        # 15:00 ET on a weekday: 45 minutes exceeds the open threshold.
        now = datetime(2026, 10, 30, 19, 45, tzinfo=timezone.utc)
        assert "data_stale" in [a.kind for a in evaluate_account_alarms(account, master, now, AlarmConfig())]
        # At 18:00 ET, the closed threshold is 24 hours.
        assert "data_stale" not in [a.kind for a in evaluate_account_alarms(account, master,
            datetime(2026, 10, 30, 22, tzinfo=timezone.utc), AlarmConfig())]
        master.add(CalendarEvent(date=date(2026, 10, 30), kind="holiday", title="Feriado", source="manual"))
        master.commit()
        assert "data_stale" not in [a.kind for a in evaluate_account_alarms(account, master, now, AlarmConfig())]
    engine.dispose()


def test_home_status_and_evaluate_command_idempotency(tmp_path, monkeypatch):
    engine = _master(tmp_path)
    url = _source(tmp_path / "critical.db")
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{tmp_path / 'master.db'}")
    with Session(engine) as master:
        account = add_account(master, **_account_values(url))
        empty_url = _source(tmp_path / "empty.db", positions=False, snapshots=False)
        add_account(master, **(_account_values(empty_url) | {"id": "empty"}))
        status = home_status(master)
        assert status.level == "critical" and "alarma crítica" in status.headline
        assert status.link == "/cuenta/acct"
    runner = CliRunner()
    assert runner.invoke(api_group, ["evaluate-alarms"]).exit_code == 0
    assert runner.invoke(api_group, ["evaluate-alarms"]).output.strip() == "0 nuevas, 0 resueltas"
    with Session(engine) as master:
        first = master.scalar(select(Event).where(Event.type == "drawdown_limit"))
        assert first is not None and first.entity_id is None
    source = create_engine(url)
    with Session(source) as db:
        db.query(EquitySnapshot).delete()
        db.commit()
        db.add_all([EquitySnapshot(snapshot_date=date(2026, 10, day), taken_at=datetime(2026, 10, 30), equity=100, cash=0, invested=100) for day in (29, 30)])
        db.commit()
    source.dispose()
    assert runner.invoke(api_group, ["evaluate-alarms"]).output.strip() == "0 nuevas, 1 resueltas"
    with Session(engine) as master:
        resolved = master.scalar(select(Event).where(Event.type == "alarm_resolved"))
        assert resolved.payload["alarm_type"] == "drawdown_limit"
        assert home_status(master).level == "warning"
    engine.dispose()
