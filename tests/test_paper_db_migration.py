from click.testing import CliRunner
from sqlalchemy import create_engine, inspect, text

from apps.api.cli import api_group
from quantagent.database import Base
from quantagent import models  # noqa: F401


def test_paper_db_migration_dry_run_and_apply_preserve_data(tmp_path):
    database = tmp_path / "paper.db"
    url = f"sqlite:///{database}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO accounts (id, name, env, kind, database_url, strategy, status, created_at) VALUES ('paper', 'Paper', 'paper', 'alpaca', 'sqlite:///source.db', 'test', 'active', CURRENT_TIMESTAMP)"))
        connection.execute(text("DROP TABLE trade_intents"))
        connection.execute(text("ALTER TABLE active_positions DROP COLUMN broker_stop_order_id"))

    runner = CliRunner()
    result = runner.invoke(api_group, ["migrate-paper-db", "--database-url", url])
    assert result.exit_code == 0, result.output
    schema = inspect(engine)
    assert not schema.has_table("trade_intents")
    assert "broker_stop_order_id" not in {column["name"] for column in schema.get_columns("active_positions")}
    assert not schema.has_table("alembic_version")

    result = runner.invoke(api_group, ["migrate-paper-db", "--database-url", url, "--apply"])
    assert result.exit_code == 0, result.output
    schema = inspect(engine)
    assert schema.has_table("trade_intents")
    assert "broker_stop_order_id" in {column["name"] for column in schema.get_columns("active_positions")}
    assert schema.has_table("alembic_version")
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT name FROM accounts WHERE id='paper'")) == "Paper"

    result = runner.invoke(api_group, ["migrate-paper-db", "--database-url", url, "--apply"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "schema is complete"
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT name FROM accounts WHERE id='paper'")) == "Paper"
