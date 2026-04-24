"""
Tests for SQLite/JSONB type incompatibility fix (QuantAgent-sfc).

Validates that the Log model (and any other models using JSON types)
are compatible with SQLite in-memory databases used in the test suite.

Issue context:
- Previously: Log model used PostgreSQL-specific JSONB type
- Problem: SQLAlchemy couldn't compile JSONB for SQLite engine
- Fix: Replaced JSONB with standard JSON type (compatible with both engines)

Acceptance criteria:
- pytest tests/ --collect-only completes with 0 errors
- At least 450 tests pass when run with SQLite test fixtures
- Log model can be created in SQLite without compilation errors
"""

import json
from datetime import datetime

import pytest
from sqlalchemy import Column, Integer, String, Text, create_engine, inspect
from sqlalchemy.orm import sessionmaker

from quantagent.database import Base
from quantagent.models import Log


# ============================================================================
# AC1: Log Model SQLite Compatibility
# ============================================================================


class TestLogModelSQLiteCompatibility:
    """Validate Log model works with SQLite engine (no JSONB)."""

    def test_log_model_creates_in_sqlite(self):
        """Verify Log model table can be created in SQLite without errors."""
        # Create in-memory SQLite engine (like conftest fixtures)
        engine = create_engine("sqlite:///:memory:")
        
        # Attempt to create the logs table
        try:
            Base.metadata.create_all(engine, tables=[Log.__table__])
        except Exception as e:
            pytest.fail(f"Failed to create Log table in SQLite: {e}")
        
        # Verify table exists
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "logs" in tables, "logs table not created"

    def test_log_model_insert_with_extra_data(self):
        """Verify Log records with extra_data can be inserted in SQLite."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine, tables=[Log.__table__])
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create log with JSON extra_data
        test_log = Log(
            level="INFO",
            module="test.module",
            message="Test message with extra data",
            environment="test",
            symbol="BTC",
            event_type="test_event",
            extra_data={"key1": "value1", "nested": {"key2": 123}},
        )
        
        try:
            session.add(test_log)
            session.commit()
        except Exception as e:
            pytest.fail(f"Failed to insert Log with extra_data in SQLite: {e}")
        
        # Verify retrieval
        retrieved = session.query(Log).first()
        assert retrieved is not None
        assert retrieved.extra_data == {"key1": "value1", "nested": {"key2": 123}}
        
        session.close()

    def test_log_model_query_json_field(self):
        """Verify JSON field can be queried in SQLite."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine, tables=[Log.__table__])
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Insert multiple logs with different extra_data
        logs = [
            Log(
                level="INFO",
                module="test",
                message=f"Log {i}",
                extra_data={"index": i, "type": "test"}
            )
            for i in range(5)
        ]
        session.add_all(logs)
        session.commit()
        
        # Query all logs (JSON field should not cause errors)
        try:
            results = session.query(Log).all()
            assert len(results) == 5
        except Exception as e:
            pytest.fail(f"Failed to query Log records with JSON field: {e}")
        
        session.close()

    def test_log_model_null_extra_data(self):
        """Verify Log records with NULL extra_data work in SQLite."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine, tables=[Log.__table__])
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create log without extra_data
        test_log = Log(
            level="WARNING",
            module="test.null",
            message="Log without extra data",
            extra_data=None,
        )
        
        session.add(test_log)
        session.commit()
        
        retrieved = session.query(Log).first()
        assert retrieved.extra_data is None
        
        session.close()


# ============================================================================
# AC2: No JSONB Import Remains
# ============================================================================


class TestNoJSONBInCodebase:
    """Validate JSONB type has been removed from models."""

    def test_models_does_not_import_jsonb(self):
        """Verify quantagent/models.py does not import JSONB."""
        import quantagent.models as models_module
        
        # Check module doesn't have JSONB attribute
        assert not hasattr(models_module, "JSONB"), \
            "JSONB should not be imported in models module"

    def test_log_model_uses_json_not_jsonb(self):
        """Verify Log.extra_data column uses JSON, not JSONB."""
        from sqlalchemy import JSON
        from sqlalchemy.dialects.postgresql import JSONB
        
        # Get the column type
        extra_data_col = Log.__table__.columns["extra_data"]
        col_type = extra_data_col.type
        
        # Should be JSON type
        assert isinstance(col_type, JSON), \
            f"extra_data should use JSON type, got {type(col_type)}"
        
        # Should NOT be JSONB
        assert not isinstance(col_type, JSONB), \
            "extra_data should not use JSONB type"


# ============================================================================
# AC3: SQLite Test Collection
# ============================================================================


class TestSQLiteTestCollection:
    """Validate test collection works without JSONB compilation errors."""

    def test_pytest_collect_only_succeeds(self, tmp_path):
        """Verify pytest --collect-only completes without JSONB errors."""
        import subprocess
        
        # Run pytest --collect-only in the tests directory
        result = subprocess.run(
            ["pytest", "tests/", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Should exit 0 (or 5 for no tests collected in subset)
        assert result.returncode in [0, 5], \
            f"pytest collection failed with code {result.returncode}: {result.stderr}"
        
        # Should not contain JSONB compilation errors
        error_markers = [
            "Compiler can't render element of type JSONB",
            "CompileError",
            "type JSONB",
        ]
        
        for marker in error_markers:
            assert marker not in result.stderr, \
                f"pytest collection output contains JSONB error: {marker}"
            assert marker not in result.stdout, \
                f"pytest collection output contains JSONB error: {marker}"


# ============================================================================
# AC4: Multi-Engine Compatibility
# ============================================================================


class TestMultiEngineCompatibility:
    """Validate Log model works with both SQLite and PostgreSQL dialects."""

    def test_log_model_compiles_for_sqlite(self):
        """Verify Log table DDL can be compiled for SQLite."""
        from sqlalchemy.schema import CreateTable
        
        engine = create_engine("sqlite:///:memory:")
        
        try:
            ddl = CreateTable(Log.__table__).compile(engine)
            ddl_str = str(ddl)
        except Exception as e:
            pytest.fail(f"Failed to compile Log table DDL for SQLite: {e}")
        
        # Verify table definition contains expected columns
        assert "extra_data" in ddl_str.lower()
        assert "level" in ddl_str.lower()
        assert "message" in ddl_str.lower()

    def test_log_model_compiles_for_postgresql(self):
        """Verify Log table DDL can be compiled for PostgreSQL."""
        from sqlalchemy.schema import CreateTable
        
        # Use PostgreSQL dialect (don't need actual connection)
        engine = create_engine("postgresql://user:pass@localhost/db")
        
        try:
            ddl = CreateTable(Log.__table__).compile(engine)
            ddl_str = str(ddl)
        except Exception as e:
            pytest.fail(f"Failed to compile Log table DDL for PostgreSQL: {e}")
        
        # Verify table definition contains expected columns
        assert "extra_data" in ddl_str.lower()
        assert "json" in ddl_str.lower()  # Should use JSON type


# ============================================================================
# Integration Test
# ============================================================================


class TestLogModelIntegration:
    """Integration test mimicking real test suite usage."""

    def test_fixture_pattern_with_log_model(self):
        """
        Validate Log model works in typical test fixture pattern.
        
        This mimics how other test files create in-memory SQLite databases
        and use models (e.g., test_portfolio_manager.py, test_position_monitor.py).
        """
        # Setup (typical fixture pattern)
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Usage: create and query Log records
        test_logs = [
            Log(
                level="INFO",
                module="test.integration",
                message=f"Integration test log {i}",
                environment="test",
                symbol="BTC",
                event_type="backtest_tick",
                extra_data={"tick": i, "price": 50000 + i * 10},
                thread_id="main",
                checkpoint_id=f"cp-{i}",
            )
            for i in range(10)
        ]
        
        session.add_all(test_logs)
        session.commit()
        
        # Query
        results = session.query(Log).filter(Log.level == "INFO").all()
        assert len(results) == 10
        
        # Verify JSON field access
        for log in results:
            assert "tick" in log.extra_data
            assert "price" in log.extra_data
        
        session.close()
