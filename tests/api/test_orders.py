from datetime import date, datetime

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.accounts import add_account
from apps.api.app import account_orders
from quantagent.database import Base
from quantagent.models import Account, Environment, Order, OrderSide, OrderStatus, OrderType, TradeIntentRecord


def account_values(url):
    return dict(id="paper-orders", name="Paper Orders", env="paper", kind="alpaca", database_url=url,
                alpaca_account_ref="alpaca-1", bucket_id="orders", strategy="orders")


@pytest.fixture
def setup(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'accounts.db'}")
    Base.metadata.create_all(engine, tables=[Account.__table__])
    db = Session(engine)
    url = f"sqlite:///{tmp_path / 'account.db'}"
    source_engine = create_engine(url)
    Base.metadata.create_all(source_engine)
    add_account(db, **account_values(url))
    yield db, source_engine, url
    db.close()
    source_engine.dispose()
    engine.dispose()


def add_order(source, created, status=OrderStatus.PENDING, **kwargs):
    order = Order(symbol="ABC", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=2,
                  status=status, created_at=created, updated_at=created, environment=Environment.PAPER,
                  **kwargs)
    source.add(order)
    source.flush()
    return order


def test_orders_use_new_york_calendar_day_across_dst(setup):
    db, engine, _ = setup
    with Session(engine) as source:
        add_order(source, datetime(2026, 11, 2, 4, 59))  # Nov 1, 23:59 EST
        add_order(source, datetime(2026, 11, 2, 5))      # Nov 2, 00:00 EST
        source.commit()
    assert [o.symbol for o in account_orders("paper-orders", date(2026, 11, 1).isoformat(), db)] == ["ABC"]
    assert [o.created_at for o in account_orders("paper-orders", date(2026, 11, 2).isoformat(), db)] == ["2026-11-02T05:00:00Z"]


def test_orders_translate_reasons_and_leave_filled_empty(setup):
    db, engine, _ = setup
    with Session(engine) as source:
        filled = add_order(source, datetime(2026, 10, 30, 16), OrderStatus.FILLED)
        rejected = add_order(source, datetime(2026, 10, 30, 17), OrderStatus.REJECTED)
        source.add(TradeIntentRecord(bucket_id="b", strategy_id="s", symbol="ABC", intent={}, status="rejected",
                                     reject_reason="market_closed", order_id=rejected.id))
        source.commit()
    rows = account_orders("paper-orders", date(2026, 10, 30).isoformat(), db)
    assert [row.reject_reason for row in rows] == [None, "Market is closed"]
    assert rows[0].reject_detail is rows[0].reject_reason_raw is None


def test_unknown_rejection_code_and_empty_day(setup):
    db, engine, _ = setup
    with Session(engine) as source:
        rejected = add_order(source, datetime(2026, 10, 30, 16), OrderStatus.REJECTED)
        source.add(TradeIntentRecord(bucket_id="b", strategy_id="s", symbol="ABC", intent={}, status="rejected",
                                     reject_reason="new_code", order_id=rejected.id))
        source.commit()
    assert account_orders("paper-orders", date(2026, 10, 30).isoformat(), db)[0].reject_reason == "Rejected (reason not recognised)"
    assert account_orders("paper-orders", date(2026, 10, 31).isoformat(), db) == []


def test_legacy_database_without_trade_intents(setup):
    db, _, _ = setup
    legacy = create_engine(f"sqlite:///{db.bind.url.database.rsplit('/', 1)[0]}/legacy.db")
    Order.__table__.create(legacy)
    with Session(legacy) as source:
        add_order(source, datetime(2026, 10, 30, 16), OrderStatus.REJECTED)
        source.commit()
    db.get(Account, "paper-orders").database_url = str(legacy.url)
    db.commit()
    assert account_orders("paper-orders", date(2026, 10, 30).isoformat(), db)[0].reject_reason is None
    legacy.dispose()


def test_bad_date_returns_422(setup):
    db, _, _ = setup
    with pytest.raises(HTTPException) as error:
        account_orders("paper-orders", "not-a-date", db)
    assert error.value.status_code == 422
