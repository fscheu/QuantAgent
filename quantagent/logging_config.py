"""Centralized logging configuration with database persistence."""

import logging
from datetime import datetime
from typing import Optional

from . import settings
from .database import SessionLocal


class DatabaseLogHandler(logging.Handler):
    """Custom logging handler that persists logs to PostgreSQL."""

    def __init__(
        self, environment: Optional[str] = None, symbol: Optional[str] = None
    ):
        super().__init__()
        self.environment = environment
        self.symbol = symbol

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the database."""
        try:
            from .models import Log

            session = SessionLocal()
            try:
                extra = getattr(record, "__dict__", {})
                log_entry = Log(
                    timestamp=datetime.utcnow(),
                    level=record.levelname,
                    module=record.name,
                    message=self.format(record),
                    environment=extra.get("environment", self.environment),
                    symbol=extra.get("symbol", self.symbol),
                    event_type=extra.get("event_type"),
                    extra_data=extra.get("extra_data"),
                    thread_id=extra.get("thread_id"),
                    checkpoint_id=extra.get("checkpoint_id"),
                )
                session.add(log_entry)
                session.commit()
            finally:
                session.close()
        except Exception:
            self.handleError(record)


def setup_logging(
    level: Optional[str] = None,
    log_to_console: Optional[bool] = None,
    log_to_db: Optional[bool] = None,
    environment: Optional[str] = None,
    symbol: Optional[str] = None,
) -> None:
    """
    Configure logging with dual handlers (console + database).

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to settings.LOG_LEVEL.
        log_to_console: Enable console output.
                        Defaults to settings.LOG_TO_CONSOLE.
        log_to_db: Enable database persistence.
                   Defaults to settings.LOG_TO_DB.
        environment: Environment tag (backtest, paper, prod). Optional.
        symbol: Symbol tag for filtering. Optional.
    """
    level = level or settings.LOG_LEVEL
    if log_to_console is None:
        log_to_console = settings.LOG_TO_CONSOLE
    if log_to_db is None:
        log_to_db = settings.LOG_TO_DB

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()

    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    if log_to_db:
        db_handler = DatabaseLogHandler(environment=environment, symbol=symbol)
        db_handler.setLevel(logging.INFO)
        root_logger.addHandler(db_handler)
