"""Write operations for configured trading accounts."""

import os

from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from quantagent.models import Account


def write_session():
    url = os.environ["QUANTAGENT_DATABASE_URL"]
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    return Session(engine)


def add_account(db: Session, *, id: str, name: str, env: str, kind: str,
                alpaca_account_ref: str | None, bucket_id: str, strategy: str) -> Account:
    if env not in {"paper", "live"}:
        raise ValueError("env must be paper or live")
    if kind not in {"alpaca", "simulated"}:
        raise ValueError("kind must be alpaca or simulated")
    if (kind == "alpaca" and not alpaca_account_ref) or (kind == "simulated" and alpaca_account_ref is not None):
        raise ValueError("alpaca_account_ref is required for alpaca and forbidden for simulated")
    if env == "live" and kind == "simulated":
        raise ValueError("live accounts cannot be simulated")
    if db.get(Account, id) is not None:
        raise ValueError(f"account slug already exists: {id}")
    account = Account(id=id, name=name, env=env, kind=kind,
                      alpaca_account_ref=alpaca_account_ref, bucket_id=bucket_id,
                      strategy=strategy, status="active")
    db.add(account)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ValueError("account slug or bucket_id already exists") from exc
    db.refresh(account)
    return account


def list_accounts(db: Session, *, include_archived: bool = False):
    query = select(Account)
    if not include_archived:
        query = query.where(Account.status == "active")
    return db.scalars(query.order_by(Account.id)).all()


def archive_account(db: Session, id: str) -> Account:
    account = db.get(Account, id)
    if account is None:
        raise ValueError(f"account not found: {id}")
    account.status = "archived"
    db.commit()
    return account
