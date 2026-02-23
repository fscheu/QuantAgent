"""merge logging and active_positions branches

Revision ID: 044fd7cf77d5
Revises: 6f9b34635f4b, f7d3bad02cae
Create Date: 2026-01-11 18:21:36.954967

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = '044fd7cf77d5'
down_revision: Union[str, Sequence[str], None] = ('6f9b34635f4b', 'f7d3bad02cae')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
