"""add backtest_run_id to signals and replay_source_run_id to backtest_runs

Revision ID: d1e2f3a4b5c6
Revises: 84e8444d93b9, a3b4c5d6e7f8
Create Date: 2026-05-11 07:43:38.000000

QuantAgent-375: Scopes replay signal lookup to the selected source run.
- signals.backtest_run_id: FK to backtest_runs.id, identifies which run produced each signal.
- backtest_runs.replay_source_run_id: self-referential FK, records provenance for replay runs.

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, Sequence[str], None] = ("84e8444d93b9", "a3b4c5d6e7f8")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "backtest_runs",
        sa.Column("replay_source_run_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_backtest_runs_replay_source_run_id",
        "backtest_runs",
        "backtest_runs",
        ["replay_source_run_id"],
        ["id"],
    )
    op.create_index(
        "idx_backtest_runs_replay_source_run_id",
        "backtest_runs",
        ["replay_source_run_id"],
    )

    op.add_column(
        "signals",
        sa.Column("backtest_run_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_signals_backtest_run_id",
        "signals",
        "backtest_runs",
        ["backtest_run_id"],
        ["id"],
    )
    op.create_index(
        "idx_signals_backtest_run_id",
        "signals",
        ["backtest_run_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_signals_backtest_run_id", table_name="signals")
    op.drop_constraint("fk_signals_backtest_run_id", "signals", type_="foreignkey")
    op.drop_column("signals", "backtest_run_id")

    op.drop_index(
        "idx_backtest_runs_replay_source_run_id", table_name="backtest_runs"
    )
    op.drop_constraint(
        "fk_backtest_runs_replay_source_run_id", "backtest_runs", type_="foreignkey"
    )
    op.drop_column("backtest_runs", "replay_source_run_id")
