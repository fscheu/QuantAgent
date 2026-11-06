"""Create macro observations."""

import sqlalchemy as sa

from alembic import op

revision = "6e45241_macro_observations"
down_revision = "5e45236_market_bars"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "macro_observations",
        sa.Column("series_id", sa.String(length=20), primary_key=True),
        sa.Column("observation_date", sa.Date(), primary_key=True),
        sa.Column("value", sa.Numeric(18, 8), nullable=False),
        sa.Column("published_at", sa.Date(), nullable=False),
    )


def downgrade():
    op.drop_table("macro_observations")
