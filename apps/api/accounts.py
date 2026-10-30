"""Write operations for configured trading accounts."""

import os
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from quantagent.models import Account


def write_session():
    url = os.environ["QUANTAGENT_DATABASE_URL"]
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    return Session(engine)


def add_account(db: Session, *, id: str, name: str, env: str, kind: str,
                database_url: str, alpaca_account_ref: str | None = None,
                bucket_id: str | None = None, strategy: str) -> Account:
    if env not in {"paper", "live"}:
        raise ValueError("env must be paper or live")
    if kind not in {"alpaca", "simulated"}:
        raise ValueError("kind must be alpaca or simulated")
    if not database_url:
        raise ValueError("database_url is required")
    url = database_url.removeprefix("sqlite:///") if database_url.startswith("sqlite:///") else database_url
    if database_url.startswith("sqlite://") and not database_url.startswith("sqlite:///"):
        url = None
    if url is None or not Path(url).is_file() or not os.access(url, os.R_OK):
        raise ValueError("database_url must point to an existing readable SQLite database")
    if env == "live" and kind == "simulated":
        raise ValueError("live accounts cannot be simulated")
    if db.get(Account, id) is not None:
        raise ValueError(f"account slug already exists: {id}")
    account = Account(id=id, name=name, env=env, kind=kind,
                      alpaca_account_ref=alpaca_account_ref, database_url=database_url, bucket_id=bucket_id,
                      strategy=strategy, status="active")
    db.add(account)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ValueError("account slug already exists") from exc
    db.refresh(account)
    return account


def list_accounts(db: Session, *, include_archived: bool = False):
    query = select(Account)
    if not include_archived:
        query = query.where(Account.status == "active")
    return db.scalars(query.order_by(Account.id)).all()


def account_session_factory(account: Account):
    from .app import session_factory

    return session_factory(account.database_url)


def archive_account(db: Session, id: str) -> Account:
    account = db.get(Account, id)
    if account is None:
        raise ValueError(f"account not found: {id}")
    account.status = "archived"
    db.commit()
    return account
