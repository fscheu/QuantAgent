"""add backtest_run_id to trades

Revision ID: 25efa4fd0a2b
Revises: d1e2f3a4b5c6
Create Date: 2026-09-16 00:15:10.506239

QuantAgent-plan30-D07: Scopes trades to the BacktestRun that produced them,
same pattern as signals.backtest_run_id (d1e2f3a4b5c6) and
active_positions.backtest_run_id (84e8444d93b9). NULL means the trade was
not produced by a backtest run (live/paper).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '25efa4fd0a2b'
down_revision: Union[str, Sequence[str], None] = 'd1e2f3a4b5c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "trades",
        sa.Column("backtest_run_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_trades_backtest_run_id",
        "trades",
        "backtest_runs",
        ["backtest_run_id"],
        ["id"],
    )
    op.create_index(
        "idx_trades_backtest_run_id",
        "trades",
        ["backtest_run_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_trades_backtest_run_id", table_name="trades")
    op.drop_constraint("fk_trades_backtest_run_id", "trades", type_="foreignkey")
    op.drop_column("trades", "backtest_run_id")
