"""Link imported backtest runs to known hypotheses."""

from pathlib import Path

import click
import yaml
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quantagent.fabrica.investigador import HIPOTESIS_DIR
from quantagent.models import BacktestRun, HypothesisAttempt, HypothesisEvent


def known_hypothesis_ids(session, hipotesis_dir=HIPOTESIS_DIR):
    ids = set()
    directory = Path(hipotesis_dir)
    if directory.exists():
        for path in sorted((*directory.glob("*.yaml"), *directory.glob("*.yml"))):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError):
                continue
            if isinstance(data, dict) and data.get("id"):
                ids.add(data["id"])
    ids.update(value for value in session.scalars(select(HypothesisEvent.hypothesis_id).distinct()) if value)
    ids.update(value for value in session.scalars(select(HypothesisAttempt.hypothesis_id).distinct()) if value)
    return ids


def resolve_hypothesis_id(hypothesis_name, known_ids):
    return hypothesis_name if hypothesis_name in known_ids else None


@click.command(
    "vincular", help="Vincula corridas sin hipótesis a IDs conocidos.",
    epilog="Example: quantagent backtest vincular --database-url sqlite:////tmp/backtests.db",
)
@click.option("--database-url", required=True, help="URL SQLAlchemy de la base existente.")
@click.option("--hipotesis-dir", type=click.Path(file_okay=False, path_type=Path), default=HIPOTESIS_DIR,
              show_default=True, help=f"Directorio de hipótesis YAML; default: {HIPOTESIS_DIR}.")
@click.option("--dry-run", is_flag=True, help="Cuenta los cambios sin escribir en la base (default: desactivado).")
def vincular_command(database_url, hipotesis_dir, dry_run):
    engine = create_engine(database_url)
    linked = unchanged = 0
    try:
        with Session(engine) as session:
            known_ids = known_hypothesis_ids(session, hipotesis_dir)
            runs = session.scalars(select(BacktestRun).where(BacktestRun.hypothesis_id.is_(None))).all()
            for run in runs:
                hypothesis_id = resolve_hypothesis_id((run.config_snapshot or {}).get("hypothesis_name"), known_ids)
                if hypothesis_id:
                    linked += 1
                    if not dry_run:
                        run.hypothesis_id = hypothesis_id
                else:
                    unchanged += 1
            if not dry_run:
                session.commit()
    finally:
        engine.dispose()
    click.echo(f"{'Vincularía' if dry_run else 'Vinculadas'}: {linked}; sin cambios: {unchanged}")
