from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.backtesting.backtest import Backtest
from quantagent.models import Base, MarketData
from quantagent.strategy.fifty_two_week_high_strategy import FiftyTwoWeekHighStrategy
from quantagent.strategy.rsi_strategy import RSIMeanReversionStrategy
from quantagent.strategy.triple_screen_strategy import TripleScreenStrategy

END_DATE = datetime(2026, 6, 30)
START_INTRADAY = END_DATE - timedelta(days=365)
START_DAILY = END_DATE - timedelta(days=540)


def _make_session_factory() -> sessionmaker:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _seed_hourly_market_data(session, symbol: str, start_date: datetime, end_date: datetime) -> None:
    from decimal import Decimal

    current = start_date - timedelta(days=10)
    close = 100.0
    rows = []
    while current <= end_date:
        rows.append(
            MarketData(
                symbol=symbol,
                timeframe="1h",
                timestamp=current,
                open=Decimal(str(close - 0.5)),
                high=Decimal(str(close + 1.0)),
                low=Decimal(str(close - 1.0)),
                close=Decimal(str(close)),
                volume=Decimal("1000"),
            )
        )
        close += 0.05
        current += timedelta(hours=1)
    session.bulk_save_objects(rows)
    session.commit()


def _seed_four_hour_market_data(session, symbol: str, start_date: datetime, end_date: datetime) -> None:
    from decimal import Decimal

    current = start_date - timedelta(days=30)
    close = 200.0
    rows = []
    while current <= end_date:
        rows.append(
            MarketData(
                symbol=symbol,
                timeframe="4h",
                timestamp=current,
                open=Decimal(str(close - 1.0)),
                high=Decimal(str(close + 2.0)),
                low=Decimal(str(close - 2.0)),
                close=Decimal(str(close)),
                volume=Decimal("1500"),
            )
        )
        close += 0.2
        current += timedelta(hours=4)
    session.bulk_save_objects(rows)
    session.commit()


def _seed_daily_spx_breakout_data(session, symbol: str, start_date: datetime, end_date: datetime) -> None:
    from decimal import Decimal

    current = start_date - timedelta(days=120)
    rows = []
    idx = 0
    while current <= end_date:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        if idx < 500:
            close = 100 + (idx * 0.05)
            volume = 1000
            high = close + 1
        else:
            close = 130 + ((idx - 500) * 0.4)
            volume = 4000
            high = close + 3

        rows.append(
            MarketData(
                symbol=symbol,
                timeframe="1d",
                timestamp=current,
                open=Decimal(str(close - 0.5)),
                high=Decimal(str(high)),
                low=Decimal(str(close - 1.5)),
                close=Decimal(str(close)),
                volume=Decimal(str(volume)),
            )
        )
        idx += 1
        current += timedelta(days=1)
    session.bulk_save_objects(rows)
    session.commit()


RUNS = [
    {
        "name": "RSI Mean Reversion",
        "strategy": RSIMeanReversionStrategy(),
        "assets": ["BTC"],
        "timeframe": "1h",
        "start_date": START_INTRADAY,
        "end_date": END_DATE,
        "run_name": "qa-rsi-btc-1h-365d",
    },
    {
        "name": "Triple Screen",
        "strategy": TripleScreenStrategy(),
        "assets": ["BTC"],
        "timeframe": "4h",
        "start_date": START_INTRADAY,
        "end_date": END_DATE,
        "run_name": "qa-triple-screen-btc-4h-365d",
    },
    {
        "name": "52-Week High Momentum",
        "strategy": FiftyTwoWeekHighStrategy(),
        "assets": ["SPX"],
        "timeframe": "1d",
        "start_date": START_DAILY,
        "end_date": END_DATE,
        "run_name": "qa-52w-spx-1d-540d",
    },
]

CONFIG = {
    "base_position_pct": 0.05,
    "max_daily_loss_pct": 0.05,
    "max_position_pct": 0.10,
    "slippage_pct": 0.01,
    "market_hours_filter": True,
}


def main() -> None:
    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "config": {
            "start_intraday": START_INTRADAY.isoformat(),
            "start_daily": START_DAILY.isoformat(),
            "end_date": END_DATE.isoformat(),
        },
        "runs": [],
    }

    SessionLocal = _make_session_factory()

    for spec in RUNS:
        session = SessionLocal()
        try:
            if spec["timeframe"] == "1h":
                _seed_hourly_market_data(session, spec["assets"][0], spec["start_date"], spec["end_date"])
            elif spec["timeframe"] == "4h":
                _seed_four_hour_market_data(session, spec["assets"][0], spec["start_date"], spec["end_date"])
            elif spec["timeframe"] == "1d":
                _seed_daily_spx_breakout_data(session, spec["assets"][0], spec["start_date"], spec["end_date"])

            backtest = Backtest(
                start_date=spec["start_date"],
                end_date=spec["end_date"],
                assets=spec["assets"],
                timeframe=spec["timeframe"],
                initial_capital=100000.0,
                config=CONFIG,
                db_session=session,
                strategy=spec["strategy"],
            )
            metrics = backtest.run(name=spec["run_name"])
            output["runs"].append(
                {
                    "name": spec["name"],
                    "assets": spec["assets"],
                    "timeframe": spec["timeframe"],
                    "start_date": spec["start_date"].isoformat(),
                    "end_date": spec["end_date"].isoformat(),
                    "run_name": spec["run_name"],
                    "total_candles_processed": backtest.total_candles_processed,
                    "metrics": asdict(metrics),
                }
            )
        finally:
            session.close()

    out_path = Path("/home/azureuser/repos/projects/QuantAgent/tmp/three_strategy_backtests.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(out_path)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
