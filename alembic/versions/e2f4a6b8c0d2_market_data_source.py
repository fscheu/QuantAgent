"""Separate market data by source; retain the oldest row of each duplicate key."""

import sqlalchemy as sa

from alembic import op

revision = "e2f4a6b8c0d2"
down_revision = "25efa4fd0a2b"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""DELETE FROM market_data WHERE id NOT IN (
        SELECT MIN(id) FROM market_data GROUP BY symbol, timeframe, timestamp
    )""")
    op.add_column("market_data", sa.Column(
        "source", sa.String(100), nullable=False, server_default="legacy"
    ))
    op.create_index("uq_market_data_source", "market_data",
                    ["symbol", "timeframe", "timestamp", "source"], unique=True)


def downgrade():
    op.drop_index("uq_market_data_source", table_name="market_data")
    with op.batch_alter_table("market_data") as batch_op:
        batch_op.drop_column("source")
