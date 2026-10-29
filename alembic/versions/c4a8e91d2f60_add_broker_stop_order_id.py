"""Add broker stop order reference to active positions."""

from alembic import op
import sqlalchemy as sa

revision = "c4a8e91d2f60"
down_revision = "8c2f1a7d4b90"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("active_positions", sa.Column("broker_stop_order_id", sa.String(length=100), nullable=True))


def downgrade() -> None:
    op.drop_column("active_positions", "broker_stop_order_id")
