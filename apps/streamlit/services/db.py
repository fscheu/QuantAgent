from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import streamlit as st


@dataclass
class DbHandle:
    ok: bool
    error: Optional[str]
    SessionLocal: Any = None
    models: Any = None

    def get_latest_heartbeat(self, environment: str) -> Optional[dict]:
        """
        Get the latest scheduler heartbeat for an environment.

        Args:
            environment: Environment name (paper/prod/backtest)

        Returns:
            Dict with heartbeat data or None
        """
        if not self.ok:
            return None

        try:
            from quantagent.models import Environment, SchedulerHeartbeat

            with self.SessionLocal() as session:
                hb = (
                    session.query(SchedulerHeartbeat)
                    .filter_by(environment=Environment(environment))
                    .order_by(SchedulerHeartbeat.timestamp.desc())
                    .first()
                )

                if not hb:
                    return None

                return {
                    "id": hb.id,
                    "timestamp": hb.timestamp,
                    "completed_at": hb.completed_at,
                    "status": hb.status,
                    "environment": hb.environment.value,
                    "assets": hb.assets,
                    "stats": hb.stats,
                    "last_trade_id": hb.last_trade_id,
                    "error_message": hb.error_message,
                }
        except Exception:
            # Table may not exist yet (pre-migration)
            return None

    def get_recent_heartbeats(
        self, environment: str, limit: int = 10
    ) -> list[dict]:
        """
        Get recent scheduler heartbeats for an environment.

        Args:
            environment: Environment name (paper/prod/backtest)
            limit: Maximum number of heartbeats to return

        Returns:
            List of heartbeat dicts (most recent first)
        """
        if not self.ok:
            return []

        try:
            from quantagent.models import Environment, SchedulerHeartbeat

            with self.SessionLocal() as session:
                heartbeats = (
                    session.query(SchedulerHeartbeat)
                    .filter_by(environment=Environment(environment))
                    .order_by(SchedulerHeartbeat.timestamp.desc())
                    .limit(limit)
                    .all()
                )

                return [
                    {
                        "id": hb.id,
                        "timestamp": hb.timestamp,
                        "completed_at": hb.completed_at,
                        "status": hb.status,
                        "environment": hb.environment.value,
                        "assets": hb.assets,
                        "stats": hb.stats,
                        "last_trade_id": hb.last_trade_id,
                        "error_message": hb.error_message,
                    }
                    for hb in heartbeats
                ]
        except Exception:
            # Table may not exist yet (pre-migration)
            return []


    def get_paper_llm_metrics(self, environment: str, hours_back: int = 24) -> dict:
        """
        Return aggregated LLM telemetry metrics for a given environment.

        Args:
            environment: Environment string (e.g. 'paper', 'backtest').
            hours_back: Time window in hours (default 24).

        Returns:
            Aggregate dict or empty dict on failure / no DB.
        """
        if not self.ok:
            return {}
        try:
            from quantagent.llm_telemetry import get_environment_metrics

            with self.SessionLocal() as session:
                return get_environment_metrics(session, environment, hours_back)
        except Exception:
            return {}


@st.cache_resource(show_spinner=False)
def get_db_handle() -> DbHandle:
    """Try to import DB session and models. Return a handle or an error.

    Handles missing DATABASE_URL gracefully to keep UI usable without DB.
    """
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return DbHandle(
                False, "DATABASE_URL is not set. See docs/MIGRATIONS.md to configure."
            )

        import quantagent.models as models  # type: ignore
        from quantagent.database import SessionLocal  # type: ignore

        # Try opening a session to validate connectivity
        try:
            s = SessionLocal()
            s.close()
        except Exception as e:  # pragma: no cover
            return DbHandle(False, f"DB connection error: {e}")

        return DbHandle(True, None, SessionLocal=SessionLocal, models=models)
    except Exception as e:  # pragma: no cover
        return DbHandle(False, f"DB import error: {e}")
