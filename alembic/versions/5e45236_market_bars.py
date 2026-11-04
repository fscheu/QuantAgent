"""Create daily market bars."""

from alembic import op
import sqlalchemy as sa

revision = "5e45236_market_bars"
down_revision = "cd34ef56ab78"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "market_bars",
        sa.Column("symbol", sa.String(length=20), primary_key=True),
        sa.Column("session_date", sa.Date(), primary_key=True),
        sa.Column("open", sa.Numeric(18, 8), nullable=False),
        sa.Column("high", sa.Numeric(18, 8), nullable=False),
        sa.Column("low", sa.Numeric(18, 8), nullable=False),
        sa.Column("close", sa.Numeric(18, 8), nullable=False),
        sa.Column("volume", sa.Numeric(20, 8), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
    )


def downgrade():
    op.drop_table("market_bars")
