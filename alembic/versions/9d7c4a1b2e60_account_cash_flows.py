"""Add manually registered account cash flows."""
import sqlalchemy as sa

from alembic import op

revision = "9d7c4a1b2e60"
down_revision = "f31a7c9d2e10"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("cash_flows",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("flow_date", sa.Date(), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("reconciled", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade():
    op.drop_table("cash_flows")
