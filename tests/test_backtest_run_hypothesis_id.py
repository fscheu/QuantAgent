from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.models import BacktestRun, Base


def test_backtest_run_persists_optional_hypothesis_id():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    start = datetime(2024, 1, 1)

    runs = [
        BacktestRun(
            timeframe="1d",
            assets=["SPY"],
            start_date=start,
            end_date=datetime(2024, 2, 1),
            config_snapshot={},
            hypothesis_id=hypothesis_id,
        )
        for hypothesis_id in (None, "hypothesis-123")
    ]
    session.add_all(runs)
    session.commit()

    persisted = session.query(BacktestRun).order_by(BacktestRun.id).all()
    assert [run.hypothesis_id for run in persisted] == [None, "hypothesis-123"]

    session.close()
    engine.dispose()
