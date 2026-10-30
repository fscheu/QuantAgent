"""Add daily equity snapshots."""

from alembic import op
import sqlalchemy as sa

revision = "f31a7c9d2e10"
down_revision = "e45226"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "equity_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("snapshot_date", sa.Date(), nullable=False, unique=True),
        sa.Column("taken_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("invested", sa.Float(), nullable=False),
    )


def downgrade():
    op.drop_table("equity_snapshots")
