"""auto_migration

Revision ID: 3eeae1dd709f
Revises: bf303ebd0e94
Create Date: 2026-07-27 22:09:52.028973

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3eeae1dd709f'
down_revision: Union[str, Sequence[str], None] = 'bf303ebd0e94'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    with op.batch_alter_table("articles", schema=None) as batch_op:
        batch_op.alter_column(
            "entity",
            existing_type=sa.String(),   # whatever the OLD type actually was
            type_=sa.String(),           # the NEW type you want
            existing_nullable=True,
        )


def downgrade():
    with op.batch_alter_table("articles", schema=None) as batch_op:
        batch_op.alter_column(
            "entity",
            existing_type=sa.String(),
            type_=sa.String(),
            existing_nullable=True,
        )