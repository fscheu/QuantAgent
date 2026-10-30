"""Add Alpaca calendar events."""

from alembic import op
import sqlalchemy as sa

revision = "aa91bc07de33"
down_revision = ("c4a8e91d2f60", "044fd7cf77d5")
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "calendar_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False, server_default="alpaca"),
        sa.UniqueConstraint("date", "kind", "source", name="uq_calendar_date_kind_source"),
    )


def downgrade():
    op.drop_table("calendar_events")
