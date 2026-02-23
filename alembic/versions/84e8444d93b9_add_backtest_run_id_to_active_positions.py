"""add_backtest_run_id_to_active_positions

Revision ID: 84e8444d93b9
Revises: 044fd7cf77d5
Create Date: 2026-02-18 18:25:47.647309

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '84e8444d93b9'
down_revision: Union[str, Sequence[str], None] = '044fd7cf77d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


INDEX_ISOLATION = 'idx_active_position_isolation'
FK_NAME = 'fk_active_positions_backtest_run'


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'active_positions',
        sa.Column('backtest_run_id', sa.Integer(), nullable=True),
    )
    op.create_index(
        op.f('ix_active_positions_backtest_run_id'),
        'active_positions',
        ['backtest_run_id'],
        unique=False,
    )
    op.create_foreign_key(
        FK_NAME,
        'active_positions',
        'backtest_runs',
        ['backtest_run_id'],
        ['id'],
        ondelete='RESTRICT',
    )

    # Replace the previous two-column index with one that scopes by run and environment
    op.drop_index('idx_symbol_is_active', table_name='active_positions')
    op.create_index(
        INDEX_ISOLATION,
        'active_positions',
        ['symbol', 'is_active', 'backtest_run_id', 'environment'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(INDEX_ISOLATION, table_name='active_positions')
    op.drop_index(op.f('ix_active_positions_backtest_run_id'), table_name='active_positions')
    op.drop_constraint(FK_NAME, 'active_positions', type_='foreignkey')
    op.drop_column('active_positions', 'backtest_run_id')
    op.create_index(
        'idx_symbol_is_active',
        'active_positions',
        ['symbol', 'is_active'],
        unique=False,
    )
