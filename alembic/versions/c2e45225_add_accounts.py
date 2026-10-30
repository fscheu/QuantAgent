"""Add configured trading accounts."""

from alembic import op
import sqlalchemy as sa

revision = "c2e45225"
down_revision = "b71c9e2a4f60"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "accounts",
        sa.Column("id", sa.String(length=100), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("env", sa.String(length=10), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("alpaca_account_ref", sa.String(length=100), nullable=True),
        sa.Column("bucket_id", sa.String(length=100), nullable=False, unique=True),
        sa.Column("strategy", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("env IN ('paper', 'live')"),
        sa.CheckConstraint("kind IN ('alpaca', 'simulated')"),
        sa.CheckConstraint("status IN ('active', 'archived')"),
        sa.CheckConstraint("(kind = 'alpaca' AND alpaca_account_ref IS NOT NULL) OR "
                           "(kind = 'simulated' AND alpaca_account_ref IS NULL)"),
        sa.CheckConstraint("NOT (env = 'live' AND kind = 'simulated')"),
    )


def downgrade():
    op.drop_table("accounts")
