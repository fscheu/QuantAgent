"""Acceso de solo lectura al registro de hipótesis."""

import os
from datetime import timezone
from types import SimpleNamespace

from fastapi import HTTPException
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from quantagent.models import Cycle

_engines = {}


def eventos_de_ciclos(session_registro):
    """Derive feed events from closed registry cycles without writing to events."""
    from quantagent.fabrica.registro_db import leer_recibo
    from quantagent.fabrica.registro_vista import _LIMITE_REVISION

    cycles = session_registro.scalars(select(Cycle).where(Cycle.closed_at.is_not(None))
                                      .order_by(Cycle.closed_at, Cycle.cycle)).all()
    result = []
    streak = 0
    for cycle in cycles:
        closed_at = cycle.closed_at
        if closed_at.tzinfo is None:
            closed_at = closed_at.replace(tzinfo=timezone.utc)
        receipt = leer_recibo(cycle.receipt or {}) or {}
        promoted = receipt.get("promovidas", 0)
        streak = 0 if promoted else streak + 1
        dropped = sum(item.get("estado_final") in ("cortada", "detenida")
                      for item in (receipt.get("por_hipotesis") or {}).values())
        base = {"cycle": cycle.cycle, "entered": len(receipt.get("hipotesis") or []),
                "dropped": dropped, "promoted": promoted}
        result.append(SimpleNamespace(id=f"cycle_closed:{cycle.cycle}", type="cycle_closed",
            severity="info", occurred_at=closed_at, account_id=None, entity_id=cycle.cycle,
            payload=base))
        if streak >= _LIMITE_REVISION:
            result.append(SimpleNamespace(id=f"review_rule_reached:{cycle.cycle}",
                type="review_rule_reached", severity="info", occurred_at=closed_at,
                account_id=None, entity_id=cycle.cycle,
                payload={"cycle": cycle.cycle, "streak": streak}))
    return result


def eventos_de_ciclos_disponibles():
    """Return derived events when the optional registry database is reachable."""
    try:
        db = _session_factory()()
        try:
            return eventos_de_ciclos(db)
        finally:
            db.close()
    except (HTTPException, SQLAlchemyError):
        return []


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
