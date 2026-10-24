"""Persist Alpaca order identifiers."""

import sqlalchemy as sa

from alembic import op

revision = "7d2e4a91c6b3"
down_revision = "4c2a1d8e7f90"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("orders") as batch_op:
        batch_op.add_column(sa.Column("alpaca_order_id", sa.String(100), nullable=True))


def downgrade():
    with op.batch_alter_table("orders") as batch_op:
        batch_op.drop_column("alpaca_order_id")
