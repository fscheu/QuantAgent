from datetime import date

import pandas as pd
import pandas_market_calendars as mcal
from click.testing import CliRunner
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quantagent.cli.__main__ import cli
from quantagent.data.snapshot import write_snapshot
from quantagent.models import Base, MacroObservation, MarketBar


def test_update_cli_dry_run_and_repeat_are_idempotent(tmp_path, monkeypatch):
    db = tmp_path / "bars.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__])
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{db}")

    class Alpaca:
        def fetch(self, symbols, start, end):
            frame = pd.DataFrame({"open": [1], "high": [2], "low": [1], "close": [2], "volume": [10]},
                                 index=pd.to_datetime(["2024-01-02"]))
            return {"SPY": frame}

    monkeypatch.setattr("quantagent.cli.market_bars.DailyAlpacaClient", Alpaca)
    runner = CliRunner()
    args = ["mercado", "barras", "actualizar", "--simbolos", "SPY"]
    dry = runner.invoke(cli, args + ["--dry-run"])
    assert dry.exit_code == 0, dry.output
    with Session(engine) as session:
        assert session.scalars(select(MarketBar)).all() == []
    first = runner.invoke(cli, args + ["--desde", "2024-01-02"])
    second = runner.invoke(cli, args + ["--desde", "2024-01-02"])
    assert first.exit_code == second.exit_code == 0, first.output + second.output
    with Session(engine) as session:
        rows = session.scalars(select(MarketBar)).all()
        assert len(rows) == 1
        assert rows[0].session_date == date(2024, 1, 2)


def test_update_cli_reports_symbol_missing_from_alpaca(tmp_path, monkeypatch):
    db = tmp_path / "bars.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__])
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{db}")

    class Alpaca:
        def fetch(self, symbols, start, end):
            return {}

    monkeypatch.setattr("quantagent.cli.market_bars.DailyAlpacaClient", Alpaca)
    result = CliRunner().invoke(cli, ["mercado", "barras", "actualizar", "--simbolos", "SPY,QQQ"])
    assert result.exit_code == 0, result.output
    assert "SPY: 0 barras" in result.output
    assert "QQQ: 0 barras" in result.output


def test_fail_if_stale_exits_nonzero_but_prints_summary(tmp_path, monkeypatch):
    db = tmp_path / "bars.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__])
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{db}")

    class Alpaca:
        def fetch(self, symbols, start, end):
            previous = mcal.get_calendar("NYSE").valid_days(start_date=end - pd.Timedelta(days=7), end_date=end)[-2].date()
            frame = pd.DataFrame({"open": [1], "high": [2], "low": [1], "close": [2], "volume": [10]},
                                 index=pd.to_datetime([previous]))
            return {"SPY": frame}

    monkeypatch.setattr("quantagent.cli.market_bars.DailyAlpacaClient", Alpaca)
    result = CliRunner().invoke(cli, ["mercado", "barras", "actualizar", "--simbolos", "SPY", "--fallar-si-stale"])
    assert result.exit_code != 0
    assert "SPY: 1 barras" in result.output


def test_fail_if_stale_succeeds_with_current_session(tmp_path, monkeypatch):
    db = tmp_path / "bars.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__])
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{db}")

    class Alpaca:
        def fetch(self, symbols, start, end):
            frame = pd.DataFrame({"open": [1], "high": [2], "low": [1], "close": [2], "volume": [10]},
                                 index=pd.to_datetime([end]))
            return {"SPY": frame}

    monkeypatch.setattr("quantagent.cli.market_bars.DailyAlpacaClient", Alpaca)
    result = CliRunner().invoke(cli, ["mercado", "barras", "actualizar", "--simbolos", "SPY", "--fallar-si-stale"])
    assert result.exit_code == 0, result.output
    assert "SPY: 1 barras" in result.output


def test_macro_update_cli_uses_fred_and_dry_run(tmp_path, monkeypatch):
    db = tmp_path / "macro.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine, tables=[MacroObservation.__table__])
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{db}")

    class Fred:
        def observaciones(self, serie, desde, hasta, todas_las_versiones=False):
            return pd.DataFrame({"fecha": pd.to_datetime(["2024-01-01"]), "valor": [1.25]})

    monkeypatch.setattr("quantagent.data.sources.fred.FredClient", Fred)
    runner = CliRunner()
    args = ["mercado", "macro", "actualizar", "--series", "T10Y2Y", "--desde", "2024-01-01"]
    dry = runner.invoke(cli, args + ["--dry-run"])
    assert dry.exit_code == 0, dry.output
    with Session(engine) as session:
        assert session.scalars(select(MacroObservation)).all() == []
    saved = runner.invoke(cli, args)
    assert saved.exit_code == 0, saved.output
    with Session(engine) as session:
        rows = session.scalars(select(MacroObservation)).all()
        assert len(rows) == 1 and rows[0].value == 1.25


def test_update_cli_snapshot_persists_unless_dry_run(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "snapshots"
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(snapshot_dir))
    frame = pd.DataFrame({"open": [1], "high": [2], "low": [1], "close": [2],
                          "volume": [10], "adj_close": [2]},
                         index=pd.to_datetime(["2024-01-02"]))
    write_snapshot("small", {"SPY": frame}, base_dir=snapshot_dir)

    db = tmp_path / "bars.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__])
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setattr("quantagent.cli.market_bars.DailyAlpacaClient", lambda: None)
    args = ["mercado", "barras", "actualizar", "--snapshot", "small", "--simbolos", "SPY"]
    runner = CliRunner()

    dry = runner.invoke(cli, args + ["--dry-run"])
    assert dry.exit_code == 0, dry.output
    with Session(engine) as session:
        assert session.scalars(select(MarketBar)).all() == []

    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    with Session(engine) as session:
        rows = session.scalars(select(MarketBar)).all()
        assert len(rows) == 1
        assert rows[0].session_date == date(2024, 1, 2)
