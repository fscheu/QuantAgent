import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.accounts import add_account
from apps.api.app import accounts
from quantagent.database import Base
from quantagent.models import Account


def account_values(**overrides):
    values = dict(id="paper-nucleo-curva", name="Paper Núcleo Curva", env="paper", kind="alpaca",
                  alpaca_account_ref="alpaca-1", bucket_id="nucleo-curva", strategy="nucleo-curva")
    return values | overrides


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'accounts.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[Account.__table__])
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_account_add_validations(db):
    row = add_account(db, **account_values())
    assert row.id == "paper-nucleo-curva"
    assert add_account(db, **account_values(id="paper-sim", kind="simulated", alpaca_account_ref=None,
                                            bucket_id="paper-sim")).alpaca_account_ref is None
    with pytest.raises(ValueError, match="slug already exists"):
        add_account(db, **account_values())
    with pytest.raises(ValueError, match="required for alpaca"):
        add_account(db, **account_values(id="bad", alpaca_account_ref=None))
    with pytest.raises(ValueError, match="cannot be simulated"):
        add_account(db, **account_values(id="bad-live", env="live", kind="simulated",
                                         alpaca_account_ref=None))


def test_archive_hides_account_from_endpoint(db):
    add_account(db, **account_values())
    assert [row.model_dump() for row in accounts(db)] == [
        {"id": "paper-nucleo-curva", "name": "Paper Núcleo Curva", "env": "paper"}
    ]
    db.get(Account, "paper-nucleo-curva").status = "archived"
    db.commit()
    assert accounts(db) == []
