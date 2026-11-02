import csv
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import app, get_db
from quantagent.database import Base
from quantagent.models import BacktestRun


def test_backtest_performance_aligns_two_saved_curves_and_lists_unattached_run(tmp_path, monkeypatch):
    database = tmp_path / "backtests.db"
    engine = create_engine(f"sqlite:///{database}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        first = BacktestRun(name="Run A", timeframe="1d", assets=["SPY"],
                            start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 4),
                            config_snapshot={})
        second = BacktestRun(name="Run B", timeframe="1d", assets=["SPY"],
                             start_date=datetime(2024, 1, 2), end_date=datetime(2024, 1, 5),
                             config_snapshot={})
        db.add_all([first, second])
        db.commit()
        first_id, second_id = first.id, second.id

    curves = tmp_path / "curves"
    curves.mkdir()
    monkeypatch.setenv("QUANTAGENT_BACKTEST_EQUITY_DIR", str(curves))
    monkeypatch.delenv("QUANTAGENT_WEB_SNAPSHOT", raising=False)
    for run_id, dates, values in (
        (first_id, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], [100, 101, 102, 103]),
        (second_id, ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], [200, 202, 204, 206]),
    ):
        with (curves / f"{run_id}.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["timestamp", "equity", "cash", "positions_value", "drawdown_pct"])
            writer.writerows((date, value, value, 0, 0) for date, value in zip(dates, values))

    def override_db():
        with Session(engine) as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        report = client.get(f"/api/performance/backtest/{first_id}?range=all&align_with=backtest:{second_id}")
        assert report.status_code == 200
        data = report.json()
        assert data["period"]["start"] == "2024-01-02"
        assert data["period"]["end"] == "2024-01-04"
        assert [point["date"] for point in data["series"]["equity"]] == ["2024-01-02", "2024-01-03", "2024-01-04"]

        listed = client.get("/api/backtests?q=Run A")
        assert listed.status_code == 200
        assert listed.json()[0]["hypothesis_name"] is None
        assert listed.json()[0]["id"] == first_id
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()
