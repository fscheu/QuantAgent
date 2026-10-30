"""FastAPI application and read-only database dependency."""

import os

from fastapi import FastAPI
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

app = FastAPI(title="QuantAgent API")


def session_factory(url=None):
    url = url or os.environ["QUANTAGENT_DATABASE_URL"]
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def readonly_sqlite(connection, _):
            connection.execute("PRAGMA query_only=ON")
    else:
        @event.listens_for(engine, "begin")
        def readonly_transaction(connection):
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
    return sessionmaker(bind=engine, class_=Session, autoflush=False)


def get_db():
    db = session_factory()()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}
