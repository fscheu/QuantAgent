"""Add scheduler_heartbeats table for dashboard monitoring

Revision ID: a3b4c5d6e7f8
Revises: 1543e9a20a69
Create Date: 2026-04-27 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a3b4c5d6e7f8"
down_revision: Union[str, Sequence[str], None] = "1543e9a20a69"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    environment_enum = postgresql.ENUM(
        "BACKTEST", "PAPER", "PROD", name="environment", create_type=False
    )

    # Create scheduler_heartbeats table
    op.create_table(
        "scheduler_heartbeats",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column(
            "environment",
            environment_enum,
            nullable=False,
        ),
        sa.Column("assets", sa.JSON(), nullable=True),
        sa.Column("stats", sa.JSON(), nullable=True),
        sa.Column("last_trade_id", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["last_trade_id"], ["trades.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index(
        op.f("ix_scheduler_heartbeats_timestamp"),
        "scheduler_heartbeats",
        ["timestamp"],
        unique=False,
    )
    op.create_index(
        op.f("ix_scheduler_heartbeats_environment"),
        "scheduler_heartbeats",
        ["environment"],
        unique=False,
    )
    op.create_index(
        "idx_heartbeat_env_ts",
        "scheduler_heartbeats",
        ["environment", "timestamp"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_heartbeat_env_ts", table_name="scheduler_heartbeats")
    op.drop_index(
        op.f("ix_scheduler_heartbeats_environment"),
        table_name="scheduler_heartbeats",
    )
    op.drop_index(
        op.f("ix_scheduler_heartbeats_timestamp"), table_name="scheduler_heartbeats"
    )
    op.drop_table("scheduler_heartbeats")
