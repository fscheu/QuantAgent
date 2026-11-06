import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from apps.api.accounts import account_session_factory, add_account
from apps.api.app import accounts
from apps.api.app import account_equity, account_positions, account_summary
from quantagent.database import Base
from quantagent.models import Account, ActivePosition, Environment, EquitySnapshot, ExitPolicy, Fill, MarketData, Order, OrderSide, OrderType, OrderStatus, Position


def account_values(**overrides):
    values = dict(id="paper-nucleo-curva", name="Paper Núcleo Curva", env="paper", kind="alpaca",
                  database_url="", alpaca_account_ref="alpaca-1", bucket_id="nucleo-curva", strategy="nucleo-curva")
    return values | overrides


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'accounts.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[Account.__table__])
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_account_add_validations(db):
    values = account_values(database_url=str(db.bind.url.database))
    row = add_account(db, **values)
    assert row.id == "paper-nucleo-curva"
    assert add_account(db, **values | dict(id="paper-sim", kind="simulated", alpaca_account_ref=None,
                                            bucket_id="paper-sim")).alpaca_account_ref is None
    with pytest.raises(ValueError, match="slug already exists"):
        add_account(db, **values)
    with pytest.raises(ValueError, match="existing readable"):
        add_account(db, **values | dict(id="bad", database_url="/missing/account.db"))
    with pytest.raises(ValueError, match="cannot be simulated"):
        add_account(db, **values | dict(id="bad-live", env="live", kind="simulated",
                                         alpaca_account_ref=None))


def test_account_sessions_are_read_only_and_isolated(tmp_path, db):
    urls = []
    for idx in (1, 2):
        url = f"sqlite:///{tmp_path / f'account-{idx}.db'}"
        engine = create_engine(url)
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE sample (value INTEGER)")
            connection.exec_driver_sql("INSERT INTO sample VALUES (?)", (idx,))
        engine.dispose()
        urls.append(url)
    for idx, url in enumerate(urls, 1):
        account = add_account(db, **account_values(id=f"account-{idx}", database_url=url,
                                                   alpaca_account_ref=None, bucket_id=None))
        session = account_session_factory(account)()
        try:
            assert session.execute(text("SELECT value FROM sample")).scalar() == idx
            with pytest.raises(Exception, match="readonly"):
                session.execute(text("INSERT INTO sample VALUES (9)"))
        finally:
            session.close()


def test_archive_hides_account_from_endpoint(db):
    add_account(db, **account_values(database_url=str(db.bind.url.database)))
    assert [row.model_dump() for row in accounts(db)] == [
        {"id": "paper-nucleo-curva", "name": "Paper Núcleo Curva", "env": "paper"}
    ]
    db.get(Account, "paper-nucleo-curva").status = "archived"
    db.commit()
    assert accounts(db) == []


def test_account_summary_and_positions_read_account_sqlite(tmp_path, db):
    account_url = f"sqlite:///{tmp_path / 'paper.db'}"
    source_engine = create_engine(account_url)
    Base.metadata.create_all(source_engine, tables=[table for table in Base.metadata.sorted_tables
                                                    if table is not EquitySnapshot.__table__])
    with Session(source_engine) as source:
        source.add_all([
            Position(symbol="LONG", quantity=2, average_entry_price=10, current_price=12,
                     unrealized_pnl=4, unrealized_pnl_pct=.2, side=OrderSide.BUY,
                     opened_at=__import__('datetime').datetime(2026, 10, 30),
                     updated_at=__import__('datetime').datetime(2026, 10, 30, 15)),
            Position(symbol="SHORT", quantity=-3, average_entry_price=20, current_price=18,
                     unrealized_pnl=6, unrealized_pnl_pct=.1, side=OrderSide.SELL,
                     opened_at=__import__('datetime').datetime(2026, 10, 30),
                     updated_at=__import__('datetime').datetime(2026, 10, 30, 15)),
        ])
        for symbol, side, qty, entry, price in [("LONG", OrderSide.BUY, 2, 10, 12),
                                                ("SHORT", OrderSide.SELL, 3, 20, 18)]:
            order = Order(symbol=symbol, side=side, quantity=qty, order_type=OrderType.MARKET,
                          status=OrderStatus.FILLED, environment=Environment.PAPER)
            source.add(order)
            source.flush()
            source.add(Fill(order_id=order.id, quantity=qty, price=entry))
            source.add(MarketData(symbol=symbol, timeframe="1d", timestamp=__import__('datetime').datetime(2026, 10, 30),
                                  open=price, high=price, low=price, close=price, volume=100))
        source.add_all([
            ActivePosition(symbol="LONG", side=OrderSide.BUY, entry_price=10, stop_loss=9,
                take_profit=15, quantity=2, decision_timestamp=__import__('datetime').datetime(2026, 10, 30),
                exit_policy=ExitPolicy.SL_TP_ONLY, environment=Environment.PAPER, is_active=True),
            ActivePosition(symbol="SHORT", side=OrderSide.SELL, entry_price=20, stop_loss=21,
                take_profit=15, quantity=3, decision_timestamp=__import__('datetime').datetime(2026, 10, 30),
                exit_policy=ExitPolicy.SL_TP_ONLY, environment=Environment.PAPER, is_active=True),
        ])
        source.commit()
    add_account(db, **account_values(database_url=account_url))
    summary = account_summary("paper-nucleo-curva", db).model_dump()
    assert summary["status"] == "active"
    assert (summary["long_value"], summary["short_value"], summary["day_change"]) == (24, 54, None)
    assert summary["equity"] is None and summary["cash"] is None and summary["spark_30d"] is None
    positions = [p.model_dump() for p in account_positions("paper-nucleo-curva", db)]
    assert [(p["side"], p["qty"], p["market_value"], p["stop_price"], p["stop_distance"])
            for p in positions] == [("long", 2, 24, 9, 3), ("short", 3, 54, 21, 3)]
    source_engine.dispose()


@pytest.mark.parametrize("activity,snapshots,reason,cash", [
    ("none", None, "not_opened", None),
    ("none", [], "not_opened", None),
    ("none", [(3000, 3000)], "not_opened", 3000),
    ("none", [(3000, 3000), (0, 0)], "not_funded", 0),
    ("none", [(0, 0), (3000, 3000)], "not_opened", 3000),
    ("none", [(3000, 0)], "not_opened", 0),
    ("none", [(0, 3000)], "not_opened", 3000),
    ("order", [], "unknown", None),
    ("fill", [(3000, 3000)], "unknown", 3000),
    ("order", [(0, 0)], "not_funded", 0),
])
def test_account_without_positions_has_no_data(tmp_path, db, activity, snapshots, reason, cash):
    from datetime import date, datetime, timedelta

    account_url = f"sqlite:///{tmp_path / 'empty.db'}"
    source_engine = create_engine(account_url)
    Base.metadata.create_all(source_engine, tables=[table for table in Base.metadata.sorted_tables
                                                    if snapshots is not None or table is not EquitySnapshot.__table__])
    with Session(source_engine) as source:
        for index, (equity, available) in enumerate(snapshots or []):
            source.add(EquitySnapshot(snapshot_date=date(2026, 10, 29) + timedelta(days=index),
                                      taken_at=datetime(2026, 10, 30), equity=equity,
                                      cash=available, invested=equity - available))
        if activity == "order":
            source.add(Order(symbol="SPY", side=OrderSide.BUY, quantity=1,
                             order_type=OrderType.MARKET, status=OrderStatus.REJECTED,
                             environment=Environment.PAPER))
        elif activity == "fill":
            # Incomplete local history still proves there was activity.
            source.add(Fill(order_id=99, quantity=1, price=100))
        source.commit()
    add_account(db, **account_values(database_url=account_url))
    response = account_summary("paper-nucleo-curva", db).model_dump()
    assert response["status"] == "no_data"
    assert response["no_data_reason"] == reason
    assert response["cash"] == cash
    assert "equity" not in response and "long_value" not in response
    source_engine.dispose()


def test_account_positions_does_not_trust_legacy_positions_without_fills(tmp_path, db):
    account_url = f"sqlite:///{tmp_path / 'legacy.db'}"
    source_engine = create_engine(account_url)
    Position.__table__.create(source_engine)
    with source_engine.begin() as connection:
        connection.exec_driver_sql("""
            CREATE TABLE active_positions (
                symbol VARCHAR(20), side VARCHAR(4), stop_loss FLOAT,
                environment VARCHAR(10), is_active BOOLEAN
            )
        """)
    with Session(source_engine) as source:
        source.add(Position(symbol="LEGACY", quantity=-1, average_entry_price=20, current_price=18,
                            unrealized_pnl=2, unrealized_pnl_pct=.1, side=OrderSide.SELL,
                            opened_at=__import__('datetime').datetime(2026, 10, 30),
                            updated_at=__import__('datetime').datetime(2026, 10, 30, 15)))
        source.commit()
    add_account(db, **account_values(database_url=account_url))

    positions = account_positions("paper-nucleo-curva", db)

    assert positions == []
    source_engine.dispose()


def test_equity_history_summary_and_ranges(tmp_path, db):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from fastapi import HTTPException

    hoy = datetime.now(ZoneInfo("America/New_York")).date()
    account_url = f"sqlite:///{tmp_path / 'history.db'}"
    engine = create_engine(account_url)
    Position.__table__.create(engine)
    Order.__table__.create(engine)
    EquitySnapshot.__table__.create(engine)
    with Session(engine) as source:
        source.add(Position(symbol="A", quantity=1, average_entry_price=10, current_price=10,
            unrealized_pnl=0, unrealized_pnl_pct=0, side=OrderSide.BUY,
            opened_at=datetime(2026, 10, 1), updated_at=datetime(2026, 10, 30)))
        dates = [hoy - timedelta(days=offset) for offset in (10, 8, 1, 0)]
        source.add_all([EquitySnapshot(snapshot_date=day, taken_at=datetime(2026, 10, 30),
            equity=value, cash=20, invested=value - 20) for day, value in zip(dates, [100, 110, 120, 130])])
        source.commit()
    add_account(db, **account_values(database_url=account_url))
    summary = account_summary("paper-nucleo-curva", db)
    assert (summary.equity, summary.cash, summary.day_change, summary.day_change_pct) == (130, 20, 10, 10 / 120)
    assert (summary.week_change, summary.week_change_pct, summary.spark_30d) == (20, 20 / 110, [100, 110, 120, 130])
    assert [point.account for point in account_equity("paper-nucleo-curva", "1W", db)] == [120, 130]
    assert len(account_equity("paper-nucleo-curva", "1M", db)) == 4
    assert len(account_equity("paper-nucleo-curva", "3M", db)) == 4
    assert len(account_equity("paper-nucleo-curva", "YTD", db)) == 4
    assert len(account_equity("paper-nucleo-curva", "ALL", db)) == 4
    with pytest.raises(HTTPException) as missing:
        account_equity("missing", "1M", db)
    assert missing.value.status_code == 404
    engine.dispose()
