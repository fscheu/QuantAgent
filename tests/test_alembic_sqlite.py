import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect


def test_alembic_upgrade_head_on_empty_sqlite(tmp_path):
    database = tmp_path / "empty.db"
    url = f"sqlite:///{database}"
    env = os.environ | {"DATABASE_URL": url, "NUMBA_CACHE_DIR": "/tmp/numba-c"}
    python = sys.executable
    subprocess.run([python, "-m", "alembic", "upgrade", "head"], check=True, env=env)
    current = subprocess.run([python, "-m", "alembic", "current"], check=True, env=env, capture_output=True, text=True)
    assert "(head)" in current.stdout
    with create_engine(url).connect() as connection:
        schema = inspect(connection)
        assert schema.has_table("accounts")
        assert schema.has_table("equity_snapshots")
        assert "broker_stop_order_id" in {column["name"] for column in schema.get_columns("active_positions")}
