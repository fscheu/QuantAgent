"""Configure account-specific read-only databases."""

from alembic import op
import sqlalchemy as sa

revision = "e45226"
down_revision = "c2e45225"
branch_labels = None
depends_on = None


def upgrade():
    old = sa.Table(
        "accounts", sa.MetaData(),
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("env", sa.String(10), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("alpaca_account_ref", sa.String(100)),
        sa.Column("bucket_id", sa.String(100)),
        sa.Column("strategy", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("env IN ('paper', 'live')"),
        sa.CheckConstraint("kind IN ('alpaca', 'simulated')"),
        sa.CheckConstraint("status IN ('active', 'archived')"),
        sa.CheckConstraint("NOT (env = 'live' AND kind = 'simulated')"),
    )
    with op.batch_alter_table("accounts", copy_from=old) as batch:
        batch.add_column(sa.Column("database_url", sa.String(500), nullable=False, server_default=""))
        batch.alter_column("bucket_id", existing_type=sa.String(100), nullable=True)


def downgrade():
    with op.batch_alter_table("accounts") as batch:
        batch.drop_column("database_url")
