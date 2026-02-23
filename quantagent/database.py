"""SQLAlchemy database configuration and engine setup."""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from quantagent import settings

_DEFAULT_TEST_DB_URL = "sqlite+pysqlite:///:memory:"
_TEST_FALLBACK_ENV_FLAGS = {"1", "true", "yes", "on"}


def _flag_enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in _TEST_FALLBACK_ENV_FLAGS

# Base class for all models (safe to import without DB configured)
Base = declarative_base()

# Lazy-initialized module-level singletons
_engine = None
_SessionLocal = None


def _should_use_test_fallback() -> bool:
    """Return True when running under pytest or an explicit fallback flag."""
    if _flag_enabled("QUANTAGENT_DISABLE_SQLITE_FALLBACK"):
        return False
    if "pytest" in sys.modules:
        return True
    return _flag_enabled("QUANTAGENT_ALLOW_SQLITE_FALLBACK")


def _resolve_database_url() -> str:
    """Resolve DATABASE_URL with graceful fallback for unit tests."""
    try:
        return settings.require("DATABASE_URL")
    except ValueError as exc:
        if _should_use_test_fallback():
            return os.getenv("QUANTAGENT_TEST_DATABASE_URL", _DEFAULT_TEST_DB_URL)
        raise exc


def _get_engine():
    """Create or return the cached engine. Validates DATABASE_URL on first call only."""
    global _engine
    if _engine is None:
        url = _resolve_database_url()
        if url.startswith("sqlite"):
            _engine = create_engine(
                url, connect_args={"check_same_thread": False}, poolclass=StaticPool
            )
        else:
            _engine = create_engine(url, pool_pre_ping=True, echo=False)
    return _engine


def _get_session_factory():
    """Create or return the cached session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return _SessionLocal


# Keep backward-compatible names as properties via module-level getters.
# Code that accesses `database.engine` or `database.SessionLocal` directly
# will now trigger lazy init (and validation) on first access.

class _LazyEngine:
    """Descriptor-like proxy so `database.engine` still works."""
    def __repr__(self):
        return repr(_get_engine())
    def __getattr__(self, name):
        return getattr(_get_engine(), name)

class _LazySessionLocal:
    """Descriptor-like proxy so `database.SessionLocal()` still works."""
    def __call__(self, *args, **kwargs):
        return _get_session_factory()(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(_get_session_factory(), name)


engine = _LazyEngine()
SessionLocal = _LazySessionLocal()


def get_db() -> Session:
    """Get database session for dependency injection."""
    db = _get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database by creating all tables."""
    Base.metadata.create_all(bind=_get_engine())


def drop_all_tables() -> None:
    """Drop all tables (use with caution!)."""
    Base.metadata.drop_all(bind=_get_engine())
