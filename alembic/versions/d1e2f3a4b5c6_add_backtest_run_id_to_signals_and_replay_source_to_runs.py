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
    with op.batch_alter_table("backtest_runs") as batch_op:
        batch_op.add_column(sa.Column("replay_source_run_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key("fk_backtest_runs_replay_source_run_id", "backtest_runs", ["replay_source_run_id"], ["id"])
    op.create_index(
        "idx_backtest_runs_replay_source_run_id",
        "backtest_runs",
        ["replay_source_run_id"],
    )

    with op.batch_alter_table("signals") as batch_op:
        batch_op.add_column(sa.Column("backtest_run_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key("fk_signals_backtest_run_id", "backtest_runs", ["backtest_run_id"], ["id"])
    op.create_index(
        "idx_signals_backtest_run_id",
        "signals",
        ["backtest_run_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_signals_backtest_run_id", table_name="signals")
    with op.batch_alter_table("signals") as batch_op:
        batch_op.drop_constraint("fk_signals_backtest_run_id", type_="foreignkey")
        batch_op.drop_column("backtest_run_id")

    op.drop_index(
        "idx_backtest_runs_replay_source_run_id", table_name="backtest_runs"
    )
    with op.batch_alter_table("backtest_runs") as batch_op:
        batch_op.drop_constraint("fk_backtest_runs_replay_source_run_id", type_="foreignkey")
        batch_op.drop_column("replay_source_run_id")
