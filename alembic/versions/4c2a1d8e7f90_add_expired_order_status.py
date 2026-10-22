"""Allow expired orders in PostgreSQL."""

from alembic import op

revision = "4c2a1d8e7f90"
down_revision = "e2f4a6b8c0d2"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "DO $$ BEGIN "
            "IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'orderstatus') THEN "
            "ALTER TYPE orderstatus ADD VALUE IF NOT EXISTS 'EXPIRED'; "
            "END IF; END $$"
        )


def downgrade():
    pass
