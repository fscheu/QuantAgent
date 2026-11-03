import csv
import math
import statistics

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from apps.api.app import app
from quantagent.cli.backtest import backtest_group
from quantagent.database import Base
from quantagent.models import BacktestRun


@pytest.fixture
def destination(tmp_path):
    url = f"sqlite:///{tmp_path / 'import.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    yield url, engine
    engine.dispose()


def test_import_all_curves_dry_run_idempotency_and_api(tmp_path, monkeypatch, destination):
    url, engine = destination
    root = tmp_path / "local-data"
    groups = ["control", "research/etapa2/cartera-nucleo-curva", "research/etapa3/cartera-nucleo-curva",
              "m5-arquetipo-a", "reserva/rebalanceo-fin-de-mes"]
    for group in groups:
        filename = "AGG.csv" if group.startswith("reserva/") else "curva-pendiente-AGG.csv"
        curve = root / group / "equity" / filename
        curve.parent.mkdir(parents=True)
        curve.write_text("date,equity\n2024-01-01,100000\n2024-01-02,110000\n2024-01-03,99000\n2024-01-04,108900\n")
    trades = root / "m5-arquetipo-a/trades/curva-pendiente-AGG.csv"
    trades.parent.mkdir()
    trades.write_text("symbol,exit_time,pnl\nAGG,2024-01-02,10000\nAGG,2024-01-03,-11000\nAGG,2024-01-04,9900\n")
    # Non-equity CSVs must never become runs.
    (root / "celdas.csv").write_text("not,a,curve\n")
    curves = tmp_path / "output"
    args = ["import", str(root), "--database-url", url, "--equity-dir", str(curves)]
    runner = CliRunner()
    dry = runner.invoke(backtest_group, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "Importaría: 5; existentes: 0" in dry.output
    assert not curves.exists()
    with Session(engine) as db:
        assert db.scalars(select(BacktestRun)).all() == []
    result = runner.invoke(backtest_group, args)
    assert result.exit_code == 0, result.output
    with Session(engine) as db:
        rows = db.scalars(select(BacktestRun)).all()
        assert len(rows) == 5
        assert len({row.name for row in rows}) == 5
        for row in rows:
            assert row.assets == ["AGG"] and row.timeframe == "1d"
            assert row.start_date.isoformat() == "2024-01-01T00:00:00"
            assert row.end_date.isoformat() == "2024-01-04T00:00:00"
            assert row.total_pnl == 8900
            assert row.max_drawdown == pytest.approx(0.1)
            periods = 365.25
            expected = (statistics.mean([0.1, -0.1, 0.1]) - 0.02 / periods) / statistics.stdev([0.1, -0.1, 0.1]) * math.sqrt(periods)
            assert row.sharpe_ratio == pytest.approx(expected)
            assert row.profit_factor is None
            assert row.config_snapshot["source"] == row.data_source
            assert row.config_snapshot["strategy"] == ("rebalanceo-fin-de-mes" if row.data_source.startswith("reserva/") else "curva-pendiente")
            if row.config_snapshot["group"] == "m5-arquetipo-a":
                assert row.total_trades == 3 and row.win_rate == pytest.approx(2 / 3)
                assert row.config_snapshot["hypothesis_name"] == "arquetipo-a"
            else:
                assert row.total_trades is None and row.win_rate is None
        stage3 = next(row for row in rows if "etapa3" in row.name)
        assert stage3.config_snapshot["stage"] == "etapa3"
        assert stage3.config_snapshot["hypothesis_name"] == "cartera-nucleo-curva"
        ids = {row.id for row in rows}
    repeated = runner.invoke(backtest_group, args)
    assert repeated.exit_code == 0, repeated.output
    assert "Importadas: 0; existentes: 5" in repeated.output
    with Session(engine) as db:
        assert set(db.scalars(select(BacktestRun.id))) == ids
    assert len(list(curves.glob("*.csv"))) == 5
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", url)
    monkeypatch.setenv("QUANTAGENT_BACKTEST_EQUITY_DIR", str(curves))
    monkeypatch.delenv("QUANTAGENT_WEB_SNAPSHOT", raising=False)
    with TestClient(app) as client:
        listed = client.get("/api/backtests")
        assert listed.status_code == 200
        assert len(listed.json()) == 5
        for run in listed.json():
            assert run["total_return"] == pytest.approx(8.9)
            report = client.get(f"/api/performance/backtest/{run['id']}")
            assert report.status_code == 200, report.text
            assert [point["value"] for point in report.json()["series"]["equity"]] == [100000, 110000, 99000, 108900]
            with (curves / f"{run['id']}.csv").open() as source:
                assert csv.DictReader(source).fieldnames == ["timestamp", "equity"]


@pytest.mark.parametrize("data", ["date,equity\n", "date,equity\n2024-01-01,nan\n",
                                      "date,equity\n2024-01-02,100\n2024-01-01,101\n"])
def test_invalid_curve_does_not_create_run(tmp_path, destination, data):
    url, engine = destination
    equity = tmp_path / "input/equity"
    equity.mkdir(parents=True)
    (equity / "rsi-SPY.csv").write_text(data)
    result = CliRunner().invoke(backtest_group, ["import", str(equity.parent), "--database-url", url,
                                               "--equity-dir", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "equity/rsi-SPY.csv" in result.output
    with Session(engine) as db:
        assert db.scalars(select(BacktestRun)).all() == []
    assert not (tmp_path / "out").exists()


def test_defaults_flat_curve_and_unrelated_trades(tmp_path, monkeypatch, destination):
    url, engine = destination
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUANTAGENT_BACKTEST_EQUITY_DIR", str(tmp_path / "curves"))
    equity = tmp_path / "local-data/group/equity"
    equity.mkdir(parents=True)
    (equity / "rsi-SPY.csv").write_text("date,equity\n2024-01-01,100000\n2024-01-02,100000\n")
    trades = equity.parent / "trades"
    trades.mkdir()
    (trades / "rsi-SPY.csv").write_text("symbol,exit_time,pnl\nAGG,2024-01-02,100\n")
    result = CliRunner().invoke(backtest_group, ["import", "--database-url", url])
    assert result.exit_code == 0, result.output
    with Session(engine) as db:
        run = db.scalars(select(BacktestRun)).one()
        assert run.sharpe_ratio is None
        assert run.total_pnl == 0 and run.max_drawdown == 0
        assert run.total_trades is None and run.win_rate is None
        assert (tmp_path / "curves" / f"{run.id}.csv").is_file()
