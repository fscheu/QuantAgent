"""SQLAlchemy database configuration and engine setup."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from quantagent import settings

# Base class for all models (safe to import without DB configured)
Base = declarative_base()

# Lazy-initialized module-level singletons
_engine = None
_SessionLocal = None


def _get_engine():
    """Create or return the cached engine. Validates DATABASE_URL on first call only."""
    global _engine
    if _engine is None:
        url = settings.require("DATABASE_URL")
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
