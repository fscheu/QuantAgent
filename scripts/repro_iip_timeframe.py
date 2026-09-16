"""Repro script for QuantAgent-iip: 4H backtest counter stuck at 0 / trades
identical to the 1H run.

Seeds deterministic OHLCV data for BTC on both the 1h and 4h timeframes,
then runs the RSI strategy backtest for 1h followed by 4h *in the same DB
session* -- the exact sequence the original bug report was filed against,
where the second (4h) run allegedly leaked state from the first (1h) run.

Usage:
    python scripts/repro_iip_timeframe.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Guards against a stray, stale `quantagent` install shadowing the venv's
# editable one when this file is run directly (`python scripts/foo.py` puts
# the script's own directory on sys.path[0], not the repo root -- see
# QuantAgent-bqe for the underlying environment issue).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from quantagent.backtesting.backtest import Backtest
from quantagent.models import Base, MarketData
from quantagent.strategy.rsi_strategy import RSIMeanReversionStrategy

SYMBOL = "BTC"
END_DATE = datetime(2026, 6, 30)
START_DATE = END_DATE - timedelta(days=90)

CONFIG = {
    "base_position_pct": 0.05,
    "max_daily_loss_pct": 0.05,
    "max_position_pct": 0.10,
    "slippage_pct": 0.01,
    "market_hours_filter": False,
}


def _make_session_factory() -> sessionmaker:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _seed_market_data(session, timeframe: str, delta: timedelta, seed_start: float) -> int:
    """Deterministic sine-ish oscillation so RSI actually swings overbought/oversold."""
    import math

    current = START_DATE - timedelta(days=10)
    rows = []
    idx = 0
    while current <= END_DATE:
        close = seed_start + 15.0 * math.sin(idx / 6.0) + (idx * 0.01)
        rows.append(
            MarketData(
                symbol=SYMBOL,
                timeframe=timeframe,
                timestamp=current,
                open=Decimal(str(round(close - 0.5, 4))),
                high=Decimal(str(round(close + 1.0, 4))),
                low=Decimal(str(round(close - 1.0, 4))),
                close=Decimal(str(round(close, 4))),
                volume=Decimal("1000"),
            )
        )
        idx += 1
        current += delta
    session.bulk_save_objects(rows)
    session.commit()
    return len(rows)


def _run_backtest(session, timeframe: str, run_name: str):
    backtest = Backtest(
        start_date=START_DATE,
        end_date=END_DATE,
        assets=[SYMBOL],
        timeframe=timeframe,
        initial_capital=100000.0,
        config=CONFIG,
        db_session=session,
        strategy=RSIMeanReversionStrategy(),
    )
    metrics = backtest.run(name=run_name)
    return backtest.total_candles_processed, metrics


def main() -> None:
    SessionLocal = _make_session_factory()
    session = SessionLocal()

    try:
        seeded_1h = _seed_market_data(session, "1h", timedelta(hours=1), seed_start=100.0)
        seeded_4h = _seed_market_data(session, "4h", timedelta(hours=4), seed_start=100.0)
        print(f"Seeded {seeded_1h} candles (1h) and {seeded_4h} candles (4h) for {SYMBOL}")
        print()

        # Run 1H first, then 4H in the SAME session -- reproduces the bug
        # sequence reported in QuantAgent-iip.
        candles_1h, metrics_1h = _run_backtest(session, "1h", "repro-iip-1h")
        candles_4h, metrics_4h = _run_backtest(session, "4h", "repro-iip-4h")
    finally:
        session.close()

    print("=" * 60)
    print(f"{'':12}{'1h':>15}{'4h':>15}")
    print(f"{'evaluations':12}{candles_1h:>15}{candles_4h:>15}")
    print(f"{'trades':12}{metrics_1h.total_trades:>15}{metrics_4h.total_trades:>15}")
    print(f"{'pnl':12}{metrics_1h.total_pnl:>15.2f}{metrics_4h.total_pnl:>15.2f}")
    print("=" * 60)

    print()
    print("1h metrics:", asdict(metrics_1h))
    print("4h metrics:", asdict(metrics_4h))


if __name__ == "__main__":
    main()
