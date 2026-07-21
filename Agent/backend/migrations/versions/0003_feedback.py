"""feedback table

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-21
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=True),
        sa.Column("rating", sa.String(length=4), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_id", "turn_index", name="uq_feedback_session_turn"),
    )
    op.create_index("ix_feedback_session_id", "feedback", ["session_id"])
    op.create_index("ix_feedback_user_id", "feedback", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_feedback_user_id", table_name="feedback")
    op.drop_index("ix_feedback_session_id", table_name="feedback")
    op.drop_table("feedback")
