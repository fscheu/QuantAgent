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


@st.cache_resource(show_spinner=False)
def get_db_handle() -> DbHandle:
    """Try to import DB session and models. Return a handle or an error.

    Handles missing DATABASE_URL gracefully to keep UI usable without DB.
    """
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return DbHandle(False, "DATABASE_URL is not set. See docs/MIGRATIONS.md to configure.")

        from quantagent.database import SessionLocal  # type: ignore
        import quantagent.models as models  # type: ignore

        # Try opening a session to validate connectivity
        try:
            s = SessionLocal()
            s.close()
        except Exception as e:  # pragma: no cover
            return DbHandle(False, f"DB connection error: {e}")

        return DbHandle(True, None, SessionLocal=SessionLocal, models=models)
    except Exception as e:  # pragma: no cover
        return DbHandle(False, f"DB import error: {e}")

