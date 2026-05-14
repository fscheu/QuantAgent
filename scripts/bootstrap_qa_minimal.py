#!/usr/bin/env python3
"""Bootstrap a deterministic minimal QA dataset for Streamlit and qa-validator.

Usage:
    python scripts/bootstrap_qa_minimal.py --reset
    python scripts/bootstrap_qa_minimal.py --reset --db-url postgresql://user:pass@host:5432/db
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Allow running from repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantagent.models import (  # noqa: E402
    ActivePosition,
    Environment,
    ExitPolicy,
    Fill,
    Log,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    SchedulerHeartbeat,
    Signal,
    Trade,
    TradeSignal,
)

RESET_TABLES = (
    "active_positions",
    "scheduler_heartbeats",
    "fills",
    "trades",
    "orders",
    "positions",
    "signals",
    "logs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a deterministic minimal QA dataset."
    )
    parser.add_argument(
        "--db-url", default=None, help="Database URL (overrides DATABASE_URL env var)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate the QA bootstrap tables before inserting rows",
    )
    return parser.parse_args()


def resolve_db_url(db_url_arg: str | None) -> str:
    if db_url_arg:
        return db_url_arg

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(root, ".env"))
    load_dotenv(os.path.join(root, ".env.local"), override=True)

    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        print("ERROR: DATABASE_URL not set. Pass --db-url or export DATABASE_URL.")
        sys.exit(1)
    return url


def reset_tables(session) -> None:
    tables_sql = ", ".join(RESET_TABLES)
    session.execute(text(f"TRUNCATE {tables_sql} RESTART IDENTITY CASCADE"))
    session.commit()


def _make_signal(symbol: str, signal: TradeSignal, generated_at: datetime, summary: str) -> Signal:
    return Signal(
        symbol=symbol,
        signal=signal,
        confidence=0.82,
        timeframe="1h",
        rsi=54.2,
        macd=1.8,
        stochastic=61.5,
        roc=2.3,
        williams_r=-38.4,
        pattern="qa_bootstrap",
        trend="uptrend",
        analysis_summary=summary,
        generated_at=generated_at,
        environment=Environment.PAPER,
        thread_id="qa-bootstrap-thread",
        checkpoint_id="qa-bootstrap-checkpoint",
        state_snapshot={"source": "bootstrap_qa_minimal"},
        model_provider="bootstrap",
        model_name="deterministic-fixture",
        temperature=0.0,
        agent_version="qa-bootstrap-v1",
        graph_version="qa-bootstrap-v1",
    )


def _make_order(
    symbol: str,
    side: OrderSide,
    created_at: datetime,
    trigger_signal_id: int,
    quantity: Decimal,
    price: Decimal,
    status: OrderStatus = OrderStatus.FILLED,
) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        price=price,
        status=status,
        created_at=created_at,
        updated_at=created_at,
        filled_at=created_at + timedelta(seconds=30),
        filled_quantity=quantity,
        average_fill_price=price,
        comment="Inserted by bootstrap_qa_minimal",
        environment=Environment.PAPER,
        trigger_signal_id=trigger_signal_id,
    )


def _make_fill(order_id: int, quantity: Decimal, price: Decimal, filled_at: datetime) -> Fill:
    return Fill(
        order_id=order_id,
        quantity=quantity,
        price=price,
        commission=Decimal("1.25"),
        filled_at=filled_at,
    )


def _make_trade(
    symbol: str,
    order_id: int,
    quantity: Decimal,
    entry_price: Decimal,
    side: OrderSide,
    opened_at: datetime,
    *,
    exit_price: Decimal | None,
    pnl: Decimal | None,
    pnl_pct: float | None,
    closed_at: datetime | None,
    entry_signal: str,
    exit_signal: str | None,
    notes: str,
) -> Trade:
    return Trade(
        symbol=symbol,
        order_id=order_id,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        side=side,
        pnl=pnl,
        pnl_pct=pnl_pct,
        commission=Decimal("2.50"),
        entry_signal=entry_signal,
        exit_signal=exit_signal,
        timeframe="1h",
        opened_at=opened_at,
        closed_at=closed_at,
        notes=notes,
        environment=Environment.PAPER,
    )


def seed_minimal_dataset(session) -> dict[str, int]:
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)

    closed_signal = _make_signal(
        symbol="AAPL",
        signal=TradeSignal.LONG,
        generated_at=now - timedelta(minutes=40),
        summary="Closed validation trade for deterministic QA PnL coverage.",
    )
    session.add(closed_signal)
    session.flush()

    closed_order = _make_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        created_at=now - timedelta(minutes=39),
        trigger_signal_id=closed_signal.id,
        quantity=Decimal("10"),
        price=Decimal("187.50"),
    )
    session.add(closed_order)
    session.flush()
    closed_signal.order_id = closed_order.id

    closed_fill = _make_fill(
        order_id=closed_order.id,
        quantity=Decimal("10"),
        price=Decimal("187.50"),
        filled_at=now - timedelta(minutes=38, seconds=30),
    )
    session.add(closed_fill)

    closed_trade = _make_trade(
        symbol="AAPL",
        order_id=closed_order.id,
        quantity=Decimal("10"),
        entry_price=Decimal("187.50"),
        side=OrderSide.BUY,
        opened_at=now - timedelta(minutes=38),
        exit_price=Decimal("190.20"),
        pnl=Decimal("27.00"),
        pnl_pct=1.44,
        closed_at=now - timedelta(minutes=12),
        entry_signal="long",
        exit_signal="take_profit",
        notes="Closed bootstrap trade for dashboard daily PnL.",
    )
    session.add(closed_trade)
    session.flush()

    open_signal = _make_signal(
        symbol="BTC-USD",
        signal=TradeSignal.LONG,
        generated_at=now - timedelta(minutes=9),
        summary="Open validation trade for deterministic QA runtime coverage.",
    )
    session.add(open_signal)
    session.flush()

    open_order = _make_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        created_at=now - timedelta(minutes=8),
        trigger_signal_id=open_signal.id,
        quantity=Decimal("0.015"),
        price=Decimal("64000.00"),
    )
    session.add(open_order)
    session.flush()
    open_signal.order_id = open_order.id

    open_fill = _make_fill(
        order_id=open_order.id,
        quantity=Decimal("0.015"),
        price=Decimal("64000.00"),
        filled_at=now - timedelta(minutes=7, seconds=30),
    )
    session.add(open_fill)

    open_trade = _make_trade(
        symbol="BTC-USD",
        order_id=open_order.id,
        quantity=Decimal("0.015"),
        entry_price=Decimal("64000.00"),
        side=OrderSide.BUY,
        opened_at=now - timedelta(minutes=7),
        exit_price=None,
        pnl=None,
        pnl_pct=None,
        closed_at=None,
        entry_signal="long",
        exit_signal=None,
        notes="Open bootstrap trade backing active QA position.",
    )
    session.add(open_trade)
    session.flush()

    open_position = Position(
        symbol="BTC-USD",
        quantity=Decimal("0.015"),
        average_entry_price=Decimal("64000.00"),
        current_price=Decimal("64650.00"),
        unrealized_pnl=Decimal("9.75"),
        unrealized_pnl_pct=1.02,
        side=OrderSide.BUY,
        opened_at=now - timedelta(minutes=7),
        updated_at=now - timedelta(minutes=1),
    )
    session.add(open_position)

    active_position = ActivePosition(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        entry_price=Decimal("64000.00"),
        stop_loss=Decimal("62800.00"),
        take_profit=Decimal("66000.00"),
        quantity=Decimal("0.015"),
        decision_timestamp=now - timedelta(minutes=7),
        candles_since_entry=2,
        exit_policy=ExitPolicy.SL_TP_ONLY,
        max_hold_candles=12,
        prediction_horizon=3,
        candles_direction=["up", "up", "flat"],
        trailing_stop_pct=None,
        highest_price_seen=Decimal("64650.00"),
        lowest_price_seen=Decimal("63920.00"),
        trade_id=open_trade.id,
        signal_id=open_signal.id,
        is_active=True,
        closed_at=None,
        close_reason=None,
        accuracy=0.84,
        environment=Environment.PAPER,
    )
    session.add(active_position)

    heartbeat_old = SchedulerHeartbeat(
        timestamp=now - timedelta(minutes=25),
        completed_at=now - timedelta(minutes=24, seconds=48),
        status="completed",
        environment=Environment.PAPER,
        assets=["AAPL", "BTC-USD"],
        stats={"processed": 2, "total": 2, "errors": 0, "duration_seconds": 12.4},
        last_trade_id=closed_trade.id,
        error_message=None,
    )
    heartbeat_current = SchedulerHeartbeat(
        timestamp=now - timedelta(minutes=3),
        completed_at=now - timedelta(minutes=2, seconds=49),
        status="completed",
        environment=Environment.PAPER,
        assets=["AAPL", "BTC-USD"],
        stats={"processed": 2, "total": 2, "errors": 0, "duration_seconds": 11.2},
        last_trade_id=open_trade.id,
        error_message=None,
    )
    session.add_all([heartbeat_old, heartbeat_current])

    session.add_all(
        [
            Log(
                timestamp=now - timedelta(minutes=6),
                level="INFO",
                module="quantagent.qa.bootstrap",
                message="Inserted deterministic QA bootstrap dataset.",
                environment="paper",
                symbol="BTC-USD",
                event_type="qa_bootstrap",
                extra_data={"script": "bootstrap_qa_minimal", "dataset": "minimal"},
                thread_id="qa-bootstrap-thread",
                checkpoint_id="qa-bootstrap-checkpoint",
            ),
            Log(
                timestamp=now - timedelta(minutes=4),
                level="INFO",
                module="quantagent.scheduler",
                message="Paper scheduler cycle completed successfully.",
                environment="paper",
                symbol=None,
                event_type="scheduler_cycle_completed",
                extra_data={"processed": 2, "errors": 0},
                thread_id="qa-bootstrap-thread",
                checkpoint_id="qa-bootstrap-checkpoint",
            ),
            Log(
                timestamp=now - timedelta(minutes=2),
                level="WARNING",
                module="quantagent.position_monitor",
                message="Trailing stop not armed for BTC-USD bootstrap fixture.",
                environment="paper",
                symbol="BTC-USD",
                event_type="position_monitor_warning",
                extra_data={"reason": "bootstrap_fixture", "severity": "low"},
                thread_id="qa-bootstrap-thread",
                checkpoint_id="qa-bootstrap-checkpoint",
            ),
        ]
    )

    session.commit()

    return {
        "signals": 2,
        "orders": 2,
        "fills": 2,
        "trades": 2,
        "positions": 1,
        "active_positions": 1,
        "scheduler_heartbeats": 2,
        "logs": 3,
    }


def main() -> None:
    args = parse_args()
    db_url = resolve_db_url(args.db_url)
    masked = db_url.split("@")[-1] if "@" in db_url else db_url
    print(f"Connecting to: ...@{masked}")

    engine = create_engine(db_url, pool_pre_ping=True, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        if args.reset:
            print("[1/2] Resetting QA bootstrap tables...")
            reset_tables(session)
            print("  Reset complete.")

        print("[2/2] Seeding deterministic QA bootstrap dataset...")
        inserted = seed_minimal_dataset(session)
        for table_name, count in inserted.items():
            print(f"  {table_name}: {count}")
        print("Done.")
    except Exception:
        session.rollback()
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
