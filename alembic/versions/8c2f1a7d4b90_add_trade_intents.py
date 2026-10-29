"""Add persisted trade intents."""

import sqlalchemy as sa

from alembic import op

revision = "8c2f1a7d4b90"
down_revision = "7d2e4a91c6b3"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "trade_intents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("bucket_id", sa.String(length=100), nullable=False),
        sa.Column("strategy_id", sa.String(length=100), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("intent", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("reject_reason", sa.String(length=100), nullable=True),
        sa.Column("order_id", sa.Integer(), sa.ForeignKey("orders.id"), nullable=True),
        sa.Column("reasoning_metadata", sa.JSON(), nullable=False),
    )
    op.create_index("ix_trade_intents_bucket_id", "trade_intents", ["bucket_id"])
    op.create_index("ix_trade_intents_strategy_id", "trade_intents", ["strategy_id"])
    op.create_index("ix_trade_intents_symbol", "trade_intents", ["symbol"])
    op.create_index("ix_trade_intents_status", "trade_intents", ["status"])


def downgrade():
    op.drop_index("ix_trade_intents_status", table_name="trade_intents")
    op.drop_index("ix_trade_intents_symbol", table_name="trade_intents")
    op.drop_index("ix_trade_intents_strategy_id", table_name="trade_intents")
    op.drop_index("ix_trade_intents_bucket_id", table_name="trade_intents")
    op.drop_table("trade_intents")
