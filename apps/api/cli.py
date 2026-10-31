"""CLI commands for serving the HTTP API."""

import json
from pathlib import Path

import click
import uvicorn
from sqlalchemy import select


@click.group(help="Run the QuantAgent HTTP API.")
def api_group():
    pass


@api_group.command(help="Serve the read-only QuantAgent API.", epilog="Example: quantagent api serve --port 8000")
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface address to bind.")
@click.option("--port", default=8000, show_default=True, type=int, help="TCP port to bind.")
def serve(host, port):
    uvicorn.run("apps.api.app:app", host=host, port=port)


@api_group.command(help="Export or check the live OpenAPI contract.",
                  epilog="Example: quantagent api schema --check")
@click.option("--output", type=click.Path(path_type=Path), default=Path("docs/api/openapi.json"),
              show_default=True, help="JSON contract output path.")
@click.option("--check", "check_only", is_flag=True,
              help="Compare with the output file without writing; default: export.")
def schema(output, check_only):
    from .app import app

    serialized = json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"
    if check_only:
        if not output.is_file() or output.read_text() != serialized:
            raise click.exceptions.Exit(1)
        click.echo(f"OpenAPI contract matches {output}.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized)
    click.echo(f"Wrote OpenAPI contract to {output}.")


@api_group.command("sync-calendar", help="Synchronize Alpaca market holidays and early closes.", epilog="Example: quantagent api sync-calendar")
def sync_calendar_command():
    from .calendar_sync import sync_from_alpaca

    click.echo(f"Synchronized {sync_from_alpaca()} calendar events.")


@api_group.command("migrate-paper-db", help="Repair a legacy paper database and stamp it at the current Alembic head.",
                   epilog="Example: quantagent api migrate-paper-db --database-url sqlite:////srv/quantagent/paper.db --apply")
@click.option("--database-url", required=True, help="SQLAlchemy database URL to inspect and repair.")
@click.option("--apply", is_flag=True, help="Apply the schema repair; default: dry-run only.")
def migrate_paper_db_command(database_url, apply):
    from .paper_db import migrate_paper_db

    migrate_paper_db(database_url, apply=apply)


@api_group.group("accounts", help="Manage configured trading accounts.")
def accounts_group():
    pass


@accounts_group.command("add", help="Register an account and its strategy.",
                        epilog="Example: quantagent api accounts add --id paper-nucleo-curva --name 'Paper Nucleo Curva' --env paper --kind alpaca --database-url sqlite:////srv/quantagent/paper.db --strategy nucleo-curva")
@click.option("--id", "account_id", required=True, help="Unique account slug.")
@click.option("--name", required=True, help="Display name for the account.")
@click.option("--env", "environment", required=True, type=click.Choice(["paper", "live"]), help="Trading environment: paper or live.")
@click.option("--kind", required=True, type=click.Choice(["alpaca", "simulated"]), help="Account source: Alpaca or simulated.")
@click.option("--alpaca-account-ref", help="Informational Alpaca account reference.")
@click.option("--database-url", required=True, help="Readable SQLite database URL for this account.")
@click.option("--bucket-id", help="Informational trade-intent bucket identifier.")
@click.option("--strategy", required=True, help="Single strategy assigned to this account.")
def account_add(account_id, name, environment, kind, alpaca_account_ref, database_url, bucket_id, strategy):
    from .accounts import add_account, write_session

    with write_session() as db:
        try:
            account = add_account(db, id=account_id, name=name, env=environment, kind=kind,
                                  alpaca_account_ref=alpaca_account_ref, database_url=database_url, bucket_id=bucket_id,
                                  strategy=strategy)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    click.echo(account.id)


@accounts_group.command("list", help="List active accounts.", epilog="Example: quantagent api accounts list")
def account_list():
    from .accounts import list_accounts, write_session

    with write_session() as db:
        rows = list_accounts(db)
    for row in rows:
        click.echo(f"{row.id}\t{row.name}\t{row.env}\t{row.strategy}")


@accounts_group.command("show", help="Show one account by slug.", epilog="Example: quantagent api accounts show paper-nucleo-curva")
@click.argument("account_id")
def account_show(account_id):
    from .accounts import write_session
    from quantagent.models import Account

    with write_session() as db:
        row = db.get(Account, account_id)
    if row is None:
        raise click.ClickException(f"account not found: {account_id}")
    for field in ("id", "name", "env", "kind", "alpaca_account_ref", "bucket_id", "strategy", "status", "created_at"):
        click.echo(f"{field}: {getattr(row, field)}")


@accounts_group.command("archive", help="Archive an account so it is hidden from the API.", epilog="Example: quantagent api accounts archive paper-nucleo-curva")
@click.argument("account_id")
def account_archive(account_id):
    from .accounts import archive_account, write_session

    with write_session() as db:
        try:
            archive_account(db, account_id)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    click.echo(f"Archived {account_id}.")
