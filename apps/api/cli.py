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


@api_group.command("sync-fred-calendar", help="Synchronize FRED releases and manual calendar events.",
                   epilog="Example: quantagent api sync-fred-calendar")
def sync_fred_calendar_command():
    from .calendar_fred import sync_fred_calendar

    click.echo(f"Synchronized {sync_fred_calendar()} calendar events.")


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
    from quantagent.models import Account

    from .accounts import write_session

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


@accounts_group.group("flows", help="Register and reconcile manual deposits and withdrawals.")
def account_flows_group():
    pass


def _account_flow_session(account_id):
    from quantagent.models import Account

    from .accounts import account_session_factory, write_session
    master = write_session()
    account = master.get(Account, account_id)
    if account is None or account.status != "active":
        master.close()
        raise click.ClickException(f"active account not found: {account_id}")
    return master, account_session_factory(account)()


@account_flows_group.command("register", help="Register a manual account deposit or withdrawal.",
    epilog="Example: quantagent api accounts flows register paper-a --date 2026-11-01 --amount 1000 --type deposit")
@click.argument("account_id")
@click.option("--date", "flow_date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]), help="Flow date in YYYY-MM-DD format.")
@click.option("--amount", required=True, type=float, help="Positive amount in the account currency.")
@click.option("--type", "kind", required=True, type=click.Choice(["deposit", "withdrawal"]), help="Flow type; withdrawal is stored with a negative sign.")
def account_flow_register(account_id, flow_date, amount, kind):
    from quantagent.models import CashFlow
    if amount <= 0:
        raise click.BadParameter("must be positive", param_hint="--amount")
    master, session = _account_flow_session(account_id)
    try:
        session.add(CashFlow(flow_date=flow_date.date(), amount=amount if kind == "deposit" else -amount,
                             kind=kind, reconciled=False))
        session.commit()
    finally:
        session.close()
        master.close()
    click.echo("Flow registered.")


@account_flows_group.command("list", help="List manual flows for an account.", epilog="Example: quantagent api accounts flows list paper-a")
@click.argument("account_id")
def account_flow_list(account_id):
    from sqlalchemy import select

    from quantagent.models import CashFlow
    master, session = _account_flow_session(account_id)
    try:
        for row in session.scalars(select(CashFlow).order_by(CashFlow.flow_date, CashFlow.id)):
            click.echo(f"{row.id}\t{row.flow_date}\t{row.amount:.2f}\t{row.kind}\t{'reconciled' if row.reconciled else 'unreconciled'}")
    finally:
        session.close()
        master.close()


@account_flows_group.command("reconcile", help="Mark a registered flow as reconciled.", epilog="Example: quantagent api accounts flows reconcile paper-a 12")
@click.argument("account_id")
@click.argument("flow_id", type=int)
def account_flow_reconcile(account_id, flow_id):
    from quantagent.models import CashFlow
    master, session = _account_flow_session(account_id)
    try:
        row = session.get(CashFlow, flow_id)
        if row is None:
            raise click.ClickException(f"flow not found: {flow_id}")
        row.reconciled = True
        session.commit()
    finally:
        session.close()
        master.close()
    click.echo(f"Flow {flow_id} reconciled.")


@api_group.command("evaluate-alarms", help="Evaluate active account alarms and append state changes to events.",
                   epilog="Example: quantagent api evaluate-alarms")
def evaluate_alarms_command():
    from datetime import datetime, timezone

    from quantagent.models import Account, Event

    from .accounts import write_session
    from .alarms import evaluate_account_alarms
    from .events import record_event

    now = datetime.now(timezone.utc)
    created = resolved_count = 0
    with write_session() as db:
        accounts = db.scalars(select(Account).where(Account.status == "active").order_by(Account.id)).all()
        for account in accounts:
            evaluated = evaluate_account_alarms(account, db, now)
            rows = db.scalars(select(Event).where(Event.account_id == account.id,
                Event.type.in_(["drawdown_limit", "shadow_divergence", "data_stale", "alarm_resolved"]))
                .order_by(Event.occurred_at, Event.id)).all()
            opened = {}
            for event in rows:
                if event.type == "alarm_resolved":
                    opened.pop((event.payload or {}).get("alarm_type"), None)
                else:
                    opened[event.type] = event
            current = {alarm.kind: alarm for alarm in evaluated}
            for kind, alarm in current.items():
                if kind not in opened:
                    record_event(db, kind, alarm.severity, now, account.id, None, alarm.payload)
                    created += 1
            for kind, event in opened.items():
                if kind not in current:
                    duration = max(0, int((now - event.occurred_at.replace(tzinfo=timezone.utc)).total_seconds()))
                    title = {"drawdown_limit": "Drawdown sobre el límite", "shadow_divergence": "Sombra desalineada",
                             "data_stale": "Sin datos recientes"}[kind]
                    record_event(db, "alarm_resolved", "neutral", now, account.id, None,
                        {"title": title, "duration_seconds": duration, "alarm_type": kind})
                    resolved_count += 1
        db.commit()
    click.echo(f"{created} nuevas, {resolved_count} resueltas")
