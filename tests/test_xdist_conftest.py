"""Tests for pytest-xdist worker database configuration in tests/conftest.py."""

import os
import pytest
from tests.conftest import _configure_xdist_worker_database


def test_configure_xdist_worker_database_rejects_non_sqlite(monkeypatch):
    """When running under pytest-xdist with a non-SQLite database, fail early with UsageError."""
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")

    with pytest.raises(pytest.UsageError, match="parallel execution is only supported with SQLite"):
        _configure_xdist_worker_database()


def test_configure_xdist_worker_database_allows_sqlite(monkeypatch, tmp_path):
    """When running under pytest-xdist with SQLite, derive worker-specific database."""
    db_file = tmp_path / "test.db"
    db_file.touch()
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")

    _configure_xdist_worker_database()
    assert os.environ["DATABASE_URL"] == f"sqlite:///{tmp_path / 'test_gw1.db'}"


def test_configure_xdist_worker_database_noop_when_not_xdist(monkeypatch):
    """When not running under pytest-xdist, leave DATABASE_URL untouched."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")

    _configure_xdist_worker_database()
    assert os.environ["DATABASE_URL"] == "postgresql://user:pass@localhost/db"
