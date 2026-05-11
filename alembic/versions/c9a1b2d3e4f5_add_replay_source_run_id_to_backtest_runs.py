"""add replay_source_run_id to backtest_runs

Revision ID: c9a1b2d3e4f5
Revises: 044fd7cf77d5
Create Date: 2026-05-10 07:37:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "c9a1b2d3e4f5"
down_revision: Union[str, Sequence[str], None] = "044fd7cf77d5"
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


def downgrade() -> None:
    op.drop_index("idx_backtest_runs_replay_source_run_id", table_name="backtest_runs")
    op.drop_constraint(
        "fk_backtest_runs_replay_source_run_id", "backtest_runs", type_="foreignkey"
    )
    op.drop_column("backtest_runs", "replay_source_run_id")
