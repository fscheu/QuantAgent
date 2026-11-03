import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect

from quantagent.cli.sweep import CAMPOS_INTENTOS
from quantagent.models import HypothesisAttempt


ROOT = Path(__file__).resolve().parents[1]


def _migrate(database_url: str, target: str) -> None:
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    subprocess.run(
        [sys.executable, "-m", "alembic", *target.split()],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_hypothesis_migration_upgrade_and_downgrade_on_empty_sqlite(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'registro.db'}"
    _migrate(database_url, "upgrade head")
    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert {"hypotheses", "hypothesis_events", "hypothesis_attempts", "cycles"}.issubset(
        inspector.get_table_names()
    )
    _migrate(database_url, "downgrade 9d7c4a1b2e60")
    assert not {
        "hypotheses", "hypothesis_events", "hypothesis_attempts", "cycles"
    }.intersection(inspect(engine).get_table_names())
    engine.dispose()


def test_every_csv_attempt_field_has_a_column_or_extra_destination():
    columns = set(HypothesisAttempt.__table__.columns.keys())
    mapped_csv_fields = {
        "fecha_utc": "ts",
        "estrategia": "strategy",
        "activo": "asset",
        "snapshot": "snapshot",
        "params": "params",
        "trades": "trades",
        "pnl": "pnl",
        "sharpe": "sharpe",
        "hipotesis": "hypothesis_id",
        "etapa": "stage",
        "estado": "status",
    }
    extra_fields = {"desde", "hasta", "costos", "tamano", "sharpe_base"}
    assert set(CAMPOS_INTENTOS) == set(mapped_csv_fields) | extra_fields
    for field in CAMPOS_INTENTOS:
        assert field in mapped_csv_fields or field in extra_fields
    assert set(mapped_csv_fields.values()).issubset(columns)
    assert extra_fields and "extra" in columns


def test_importing_models_does_not_change_accounts_or_backtest_runs(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'schema.db'}"
    _migrate(database_url, "upgrade head")
    engine = create_engine(database_url)
    before = {
        table: tuple((column["name"], str(column["type"]), column["nullable"]) for column in inspect(engine).get_columns(table))
        for table in ("accounts", "backtest_runs")
    }
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from quantagent.database import Base; import quantagent.models; "
            "assert {'accounts', 'backtest_runs'} <= set(Base.metadata.tables)",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    after = {
        table: tuple((column["name"], str(column["type"]), column["nullable"]) for column in inspect(engine).get_columns(table))
        for table in ("accounts", "backtest_runs")
    }
    assert after == before
    engine.dispose()
