import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from apps.api.accounts import account_session_factory, add_account
from apps.api.app import accounts
from apps.api.app import account_positions, account_summary
from quantagent.database import Base
from quantagent.models import Account, ActivePosition, Environment, ExitPolicy, OrderSide, Position


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
    Base.metadata.create_all(source_engine)
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
    positions = [p.model_dump() for p in account_positions("paper-nucleo-curva", db)]
    assert [(p["side"], p["qty"], p["market_value"], p["stop_price"], p["stop_distance"])
            for p in positions] == [("long", 2, 24, 9, 3), ("short", 3, 54, 21, 3)]
    source_engine.dispose()


def test_account_without_positions_has_no_data(tmp_path, db):
    account_url = f"sqlite:///{tmp_path / 'empty.db'}"
    source_engine = create_engine(account_url)
    Base.metadata.create_all(source_engine)
    add_account(db, **account_values(database_url=account_url))
    response = account_summary("paper-nucleo-curva", db).model_dump()
    assert response["status"] == "no_data"
    assert "equity" not in response and "long_value" not in response
    source_engine.dispose()


def test_account_positions_supports_legacy_active_positions_schema(tmp_path, db):
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

    assert len(positions) == 1
    assert (positions[0].side, positions[0].stop_price, positions[0].stop_distance) == ("short", None, None)
    source_engine.dispose()
