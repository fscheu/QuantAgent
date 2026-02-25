"""Shared helpers for QuantAgent CLI commands."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, Optional, Sequence

import click
from sqlalchemy.orm import Session

from quantagent import database, settings

_current_url = None


def _ensure_database_url() -> None:
    global _current_url
    env_url = os.getenv('DATABASE_URL', '').strip()
    if env_url and env_url != settings.DATABASE_URL:
        settings.DATABASE_URL = env_url
    if _current_url != settings.DATABASE_URL:
        _current_url = settings.DATABASE_URL
        # Reset cached engine/session whenever the target URL changes
        if hasattr(database, '_engine'):
            database._engine = None
        if hasattr(database, '_SessionLocal'):
            database._SessionLocal = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope for database operations."""

    _ensure_database_url()
    try:
        session = database.SessionLocal()
        session.expire_on_commit = False
    except Exception as exc:  # pragma: no cover - defensive guard around DB init
        raise click.ClickException(f"Database connection failed: {exc}") from exc

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def load_json_payload(config_text: Optional[str]) -> Dict[str, Any]:
    """Load JSON payload from --config option or stdin."""

    raw_payload = config_text
    if raw_payload is None:
        stdin_data = click.get_text_stream("stdin").read().strip()
        if not stdin_data:
            raise click.ClickException("Provide JSON via --config or pipe it through stdin.")
        raw_payload = stdin_data

    try:
        data = json.loads(raw_payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - exercised via tests
        raise click.ClickException(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise click.ClickException("JSON payload must be an object.")
    return data


def ensure_required_fields(payload: Dict[str, Any], required: Sequence[str]) -> None:
    """Validate that required fields are present in payload."""

    missing = [field for field in required if field not in payload]
    if missing:
        missing_list = ", ".join(missing)
        raise click.ClickException(f"Missing required field(s): {missing_list}")


def ensure_json_object(value: Any, field_name: str) -> Dict[str, Any]:
    """Ensure a payload entry is a JSON object and return it."""

    if not isinstance(value, dict):
        raise click.ClickException(f"Field '{field_name}' must be a JSON object.")
    return value


def format_timestamp(value: Optional[datetime]) -> str:
    """Format timestamps consistently for CLI output."""

    if value is None:
        return "-"
    return value.replace(microsecond=0).isoformat(sep=" ")


def serialize_profile(profile: Any) -> Dict[str, Any]:
    """Convert a StrategyConfig instance to a serializable dict."""

    return {
        "id": profile.id,
        "name": profile.name,
        "kind": profile.kind,
        "json_config": profile.json_config,
        "version": profile.version,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


def echo_json(data: Any) -> None:
    """Render data as JSON without trailing noise."""

    click.echo(json.dumps(data, indent=2, default=str))
