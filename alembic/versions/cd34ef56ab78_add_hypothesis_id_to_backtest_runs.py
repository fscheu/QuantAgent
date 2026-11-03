"""Add hypothesis ID to backtest runs."""

from alembic import op
import sqlalchemy as sa


revision = "cd34ef56ab78"
down_revision = "ab12cd34ef56"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("backtest_runs") as batch_op:
        batch_op.add_column(sa.Column("hypothesis_id", sa.String(length=200), nullable=True))
    op.create_index("ix_backtest_runs_hypothesis_id", "backtest_runs", ["hypothesis_id"])


def downgrade():
    op.drop_index("ix_backtest_runs_hypothesis_id", table_name="backtest_runs")
    with op.batch_alter_table("backtest_runs") as batch_op:
        batch_op.drop_column("hypothesis_id")
