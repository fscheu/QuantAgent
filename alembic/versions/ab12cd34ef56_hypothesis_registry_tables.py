"""Add relational storage for the hypothesis registry."""

from alembic import op
import sqlalchemy as sa


revision = "ab12cd34ef56"
down_revision = "9d7c4a1b2e60"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "hypotheses",
        sa.Column("id", sa.String(length=100), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "hypothesis_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hypothesis_id", sa.String(length=100), nullable=True),
        sa.Column("event", sa.String(length=100), nullable=False),
        sa.Column("stage", sa.String(length=50), nullable=True),
        sa.Column("verdict", sa.String(length=50), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("cycle", sa.String(length=7), nullable=True),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_hypothesis_events_hypothesis_id", "hypothesis_events", ["hypothesis_id"])
    op.create_index("ix_hypothesis_events_cycle", "hypothesis_events", ["cycle"])
    op.create_table(
        "hypothesis_attempts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hypothesis_id", sa.String(length=100), nullable=True),
        sa.Column("snapshot", sa.String(length=300), nullable=True),
        sa.Column("strategy", sa.String(length=100), nullable=True),
        sa.Column("asset", sa.String(length=100), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("sharpe", sa.Float(), nullable=True),
        sa.Column("trades", sa.Integer(), nullable=True),
        sa.Column("pnl", sa.Float(), nullable=True),
        sa.Column("stage", sa.String(length=50), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "status IS NULL OR status IN ('valida', 'invalida_bug')",
            name="ck_hypothesis_attempts_status",
        ),
    )
    op.create_index("ix_hypothesis_attempts_hypothesis_id", "hypothesis_attempts", ["hypothesis_id"])
    op.create_table(
        "cycles",
        sa.Column("cycle", sa.String(length=7), primary_key=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hypothesis_ids", sa.JSON(), nullable=False),
        sa.Column("receipt", sa.JSON(), nullable=True),
    )


def downgrade():
    op.drop_table("cycles")
    op.drop_index("ix_hypothesis_attempts_hypothesis_id", table_name="hypothesis_attempts")
    op.drop_table("hypothesis_attempts")
    op.drop_index("ix_hypothesis_events_cycle", table_name="hypothesis_events")
    op.drop_index("ix_hypothesis_events_hypothesis_id", table_name="hypothesis_events")
    op.drop_table("hypothesis_events")
    op.drop_table("hypotheses")
