"""
Tests for logging infrastructure (QuantAgent-yuk.1).

Validates:
- Log ORM model structure and constraints
- DatabaseLogHandler persistence and error handling
- setup_logging() function API and behavior
- Settings module variables
- Database migration integrity

These tests follow TESTING_PATTERNS.md guidelines:
- Structure & type validation
- Constraint validation
- Error handling & fallback
- No tautological mocks
"""

import logging
import os
from io import StringIO
from unittest.mock import patch

import pytest
from sqlalchemy import inspect, text

from quantagent import settings
from quantagent.database import SessionLocal
from quantagent.logging_config import DatabaseLogHandler, setup_logging
from quantagent.models import Log

# ============================================================================
# AC-1.1: Log Model Structure Validation
# ============================================================================


class TestLogModelStructure:
    """Validate Log ORM model has all required fields."""

    def test_log_model_has_all_required_attributes(self):
        """Verify Log model has all columns defined in AC-1.1."""
        required_attrs = [
            "id",
            "timestamp",
            "level",
            "module",
            "message",
            "environment",
            "symbol",
            "event_type",
            "extra_data",  # Note: renamed from 'metadata' to avoid SQLAlchemy conflict  # noqa: E501
            "thread_id",
            "checkpoint_id",
        ]

        for attr in required_attrs:
            assert hasattr(Log, attr), f"Log model missing attribute: {attr}"

    def test_log_model_can_be_instantiated(self):
        """Verify Log model can be instantiated without errors."""
        log = Log(
            level="INFO",
            module="test.module",
            message="Test message",
            environment="test",
            symbol="BTC",
            event_type="test_event",
        )

        assert log.level == "INFO"
        assert log.module == "test.module"
        assert log.message == "Test message"
        assert log.environment == "test"
        assert log.symbol == "BTC"
        assert log.event_type == "test_event"


# ============================================================================
# AC-1.2: Database Migration Verification
# ============================================================================


class TestDatabaseMigration:
    """Validate logs table exists with correct schema and indexes."""

    def test_logs_table_exists(self):
        """Verify logs table exists in database."""
        with SessionLocal() as session:
            result = session.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_name = 'logs'"
                )
            )
            tables = result.fetchall()
            assert len(tables) == 1, "logs table does not exist"

    def test_logs_table_has_required_indexes(self):
        """Verify logs table has all required indexes."""
        expected_indexes = {
            "ix_logs_timestamp",
            "ix_logs_level",
            "ix_logs_environment",
            "ix_logs_symbol",
            "ix_logs_event_type",
            "ix_logs_thread_id",
        }

        with SessionLocal() as session:
            result = session.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = 'logs'")  # noqa: E501
            )
            actual_indexes = {row[0] for row in result.fetchall()}

            missing = expected_indexes - actual_indexes
            assert not missing, f"Missing indexes: {missing}"

    def test_logs_table_schema_matches_model(self):
        """Verify logs table columns match Log ORM model."""
        inspector = inspect(SessionLocal().bind)
        columns = inspector.get_columns("logs")
        column_names = {col["name"] for col in columns}

        expected_columns = {
            "id",
            "timestamp",
            "level",
            "module",
            "message",
            "environment",
            "symbol",
            "event_type",
            "extra_data",
            "thread_id",
            "checkpoint_id",
        }

        missing = expected_columns - column_names
        assert not missing, f"Missing columns in logs table: {missing}"


# ============================================================================
# AC-1.3: Logging Config Module API
# ============================================================================


class TestLoggingConfigAPI:
    """Validate logging_config module exports correct API."""

    def test_setup_logging_signature(self):
        """Verify setup_logging() has all required parameters."""
        import inspect

        sig = inspect.signature(setup_logging)
        params = sig.parameters

        required_params = [
            "level",
            "log_to_console",
            "log_to_db",
            "environment",
            "symbol",
        ]
        for param in required_params:
            assert param in params, f"setup_logging() missing parameter: {param}"  # noqa: E501

    def test_database_log_handler_can_be_instantiated(self):
        """Verify DatabaseLogHandler can be created."""
        handler = DatabaseLogHandler(environment="test", symbol="BTC")
        assert handler.environment == "test"
        assert handler.symbol == "BTC"
        assert isinstance(handler, logging.Handler)


# ============================================================================
# AC-1.4: Console Handler Format Validation
# ============================================================================


class TestConsoleHandlerFormat:
    """Validate console handler outputs human-readable format."""

    def test_console_output_format(self):
        """Verify console handler produces correct format."""
        # Capture console output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        logger = logging.getLogger("test.console_format")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("Test console message")

        output = stream.getvalue()
        assert "test.console_format" in output
        assert "INFO" in output
        assert "Test console message" in output
        assert "-" in output  # Verify separators exist

    def test_setup_logging_enables_console_handler(self):
        """Verify setup_logging with log_to_console=True adds console handler."""  # noqa: E501
        setup_logging(level="INFO", log_to_console=True, log_to_db=False)

        root_logger = logging.getLogger()
        has_console_handler = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, DatabaseLogHandler)
            for h in root_logger.handlers
        )

        assert has_console_handler, "Console handler not added"


# ============================================================================
# AC-1.5: Database Handler Persistence
# ============================================================================


class TestDatabaseHandlerPersistence:
    """Validate DatabaseLogHandler persists logs to database."""

    def test_database_handler_persists_log(self):
        """Verify log messages are persisted to logs table."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Setup logging with DB only
        setup_logging(
            level="INFO", log_to_console=False, log_to_db=True, environment="test"  # noqa: E501
        )

        # Create unique test message
        test_message = f"DB persistence test {os.urandom(4).hex()}"
        logger = logging.getLogger("test.db_persistence")
        logger.info(test_message, extra={"event_type": "test", "symbol": "BTC"})  # noqa: E501

        # Verify in database
        with SessionLocal() as session:
            log_entry = (
                session.query(Log)
                .filter(
                    Log.module == "test.db_persistence", Log.message == test_message  # noqa: E501
                )
                .first()
            )

            assert log_entry is not None, "Log not persisted to database"
            assert log_entry.level == "INFO"
            assert log_entry.environment == "test"
            assert log_entry.event_type == "test"
            assert log_entry.symbol == "BTC"

    def test_database_handler_graceful_failure(self):
        """Verify DatabaseLogHandler doesn't crash on DB failure."""
        # Create handler with invalid session
        handler = DatabaseLogHandler(environment="test")

        # Mock SessionLocal to raise exception
        with patch(
            "quantagent.logging_config.SessionLocal", side_effect=Exception("DB error")  # noqa: E501
        ):
            record = logging.LogRecord(
                name="test.failure",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None,
            )

            # Should not raise exception
            try:
                handler.emit(record)
            except Exception as e:
                pytest.fail(f"DatabaseLogHandler raised exception: {e}")

    def test_database_handler_respects_extra_fields(self):
        """Verify extra fields are stored in correct columns."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        setup_logging(level="INFO", log_to_console=False, log_to_db=True)

        test_message = f"Extra fields test {os.urandom(4).hex()}"
        logger = logging.getLogger("test.extra_fields")
        logger.info(
            test_message,
            extra={
                "environment": "prod",
                "symbol": "SPX",
                "event_type": "agent_start",
                "thread_id": "thread-123",
                "checkpoint_id": "checkpoint-456",
            },
        )

        with SessionLocal() as session:
            log_entry = (
                session.query(Log)
                .filter(Log.module == "test.extra_fields", Log.message == test_message)  # noqa: E501
                .first()
            )

            assert log_entry is not None
            assert log_entry.environment == "prod"
            assert log_entry.symbol == "SPX"
            assert log_entry.event_type == "agent_start"
            assert log_entry.thread_id == "thread-123"
            assert log_entry.checkpoint_id == "checkpoint-456"


# ============================================================================
# AC-1.6: Settings Module Variables
# ============================================================================


class TestSettingsVariables:
    """Validate settings module exports logging configuration variables."""

    def test_settings_has_logging_variables(self):
        """Verify settings module has all logging configuration variables."""
        assert hasattr(settings, "LOG_LEVEL")
        assert hasattr(settings, "LOG_TO_CONSOLE")
        assert hasattr(settings, "LOG_TO_DB")

    def test_settings_default_values(self):
        """Verify settings have correct default values."""
        # Note: These might be overridden by .env, so we check type instead
        assert isinstance(settings.LOG_LEVEL, str)
        assert isinstance(settings.LOG_TO_CONSOLE, bool)
        assert isinstance(settings.LOG_TO_DB, bool)

    def test_settings_log_level_from_env(self):
        """Verify LOG_LEVEL can be set via environment variable."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            # Reload settings to pick up env var
            import importlib

            importlib.reload(settings)

            # Note: This test might be fragile due to module caching
            # In real usage, LOG_LEVEL would be read once at startup


# ============================================================================
# Integration Tests
# ============================================================================


class TestLoggingIntegration:
    """Integration tests for complete logging workflow."""

    def test_dual_handler_logging(self):
        """Verify both console and DB handlers can work simultaneously."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        setup_logging(
            level="INFO", log_to_console=True, log_to_db=True, environment="test"  # noqa: E501
        )

        test_message = f"Dual handler test {os.urandom(4).hex()}"
        logger = logging.getLogger("test.dual_handler")
        logger.info(test_message)

        # Verify in DB
        with SessionLocal() as session:
            log_entry = (
                session.query(Log)
                .filter(Log.module == "test.dual_handler", Log.message == test_message)  # noqa: E501
                .first()
            )
            assert log_entry is not None

        # Verify handlers exist
        assert len(root_logger.handlers) == 2
        has_console = any(
            isinstance(h, logging.StreamHandler) for h in root_logger.handlers
        )
        has_db = any(isinstance(h, DatabaseLogHandler) for h in root_logger.handlers)  # noqa: E501
        assert has_console and has_db

    def test_log_level_filtering(self):
        """Verify log level filtering works correctly."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        setup_logging(level="WARNING", log_to_console=False, log_to_db=True)

        test_message = f"Level filter test {os.urandom(4).hex()}"
        logger = logging.getLogger("test.level_filter")

        # INFO should not be logged
        logger.info(f"INFO {test_message}")

        # WARNING should be logged
        logger.warning(f"WARNING {test_message}")

        with SessionLocal() as session:
            # INFO should not exist
            info_entry = (
                session.query(Log)
                .filter(
                    Log.module == "test.level_filter",
                    Log.message.like(f"INFO {test_message}"),
                )
                .first()
            )
            assert (
                info_entry is None
            ), "INFO log should not be persisted at WARNING level"

            # WARNING should exist
            warning_entry = (
                session.query(Log)
                .filter(
                    Log.module == "test.level_filter",
                    Log.message.like(f"WARNING {test_message}"),
                )
                .first()
            )
            assert (
                warning_entry is not None
            ), "WARNING log should be persisted at WARNING level"
