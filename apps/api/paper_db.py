"""Repair a paper database created before Alembic migrations were adopted."""

import os
from pathlib import Path

import click
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


def migrate_paper_db(database_url, apply=False):
    from quantagent import models  # noqa: F401
    from quantagent.database import Base

    engine = create_engine(database_url)
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    missing_tables = sorted(set(Base.metadata.tables) - existing_tables)
    missing_columns = {}
    blocked_columns = []
    for table_name in sorted(set(Base.metadata.tables) & existing_tables):
        present = {column["name"] for column in inspector.get_columns(table_name)}
        missing = [column for column in Base.metadata.tables[table_name].columns if column.name not in present]
        if missing:
            missing_columns[table_name] = missing
            for column in missing:
                python_default = column.default.arg if column.default is not None else None
                if not column.nullable and column.server_default is None and not isinstance(python_default, (str, int, float, bool)):
                    blocked_columns.append(f"{table_name}.{column.name}")

    for table_name in missing_tables:
        click.echo(f"missing table: {table_name}")
    for table_name, columns in missing_columns.items():
        for column in columns:
            click.echo(f"missing column: {table_name}.{column.name}")
    if not missing_tables and not missing_columns:
        click.echo("schema is complete")
    if not apply:
        click.echo("dry-run; pass --apply to repair and stamp head")
        return
    if blocked_columns:
        raise click.ClickException("cannot add required columns without a default: " + ", ".join(blocked_columns))

    with engine.begin() as connection:
        for table_name in missing_tables:
            Base.metadata.tables[table_name].create(connection, checkfirst=True)
        for table_name, columns in missing_columns.items():
            for column in columns:
                default = column.server_default.arg if column.server_default is not None else (column.default.arg if column.default is not None else None)
                default_sql = ""
                if default is not None:
                    if isinstance(default, str):
                        default_sql = " DEFAULT '" + default.replace("'", "''") + "'"
                    else:
                        default_sql = f" DEFAULT {default}"
                nullable_sql = " NOT NULL" if not column.nullable else ""
                type_sql = column.type.compile(dialect=connection.dialect)
                connection.execute(text(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {type_sql}{default_sql}{nullable_sql}'
                ))

    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)
    from alembic.script import ScriptDirectory

    head = ScriptDirectory.from_config(config).get_current_head()
    needs_stamp = not inspect(engine).has_table("alembic_version")
    if not needs_stamp:
        with engine.connect() as connection:
            needs_stamp = connection.scalar(text("SELECT version_num FROM alembic_version")) != head
    if needs_stamp:
        # alembic/env.py prefers DATABASE_URL over sqlalchemy.url; point it at this database.
        previous = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = database_url
        try:
            command.stamp(config, "head")
        finally:
            if previous is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = previous
        click.echo("paper database repaired and stamped at head")
    elif missing_tables or missing_columns:
        click.echo("paper database repaired")
