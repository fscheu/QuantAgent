import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from apps.api.accounts import account_session_factory, add_account
from apps.api.app import accounts
from quantagent.database import Base
from quantagent.models import Account


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
