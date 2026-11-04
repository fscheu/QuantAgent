"""Acceso de solo lectura al registro de hipótesis."""

import os

from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

_engines = {}


def _session_factory():
    url = os.environ.get("QUANTAGENT_REGISTRO_URL")
    if not url:
        raise HTTPException(
            status_code=503,
            detail="QUANTAGENT_REGISTRO_URL no está configurada; el registro no está disponible.",
        )
    if url not in _engines:
        engine = create_engine(
            url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {}
        )
        if url.startswith("sqlite"):
            @event.listens_for(engine, "connect")
            def readonly_sqlite(connection, _):
                connection.execute("PRAGMA query_only=ON")
        else:
            @event.listens_for(engine, "begin")
            def readonly_transaction(connection):
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        _engines[url] = sessionmaker(bind=engine, class_=Session, autoflush=False)
    return _engines[url]


def registro_session():
    """Dependencia FastAPI que entrega una sesión de lectura del registro."""
    try:
        db = _session_factory()()
        db.connection()
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503, detail="No se pudo conectar con la base del registro."
        ) from exc
    try:
        yield db
    finally:
        db.close()
