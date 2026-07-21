"""credit accounts + usage ledger

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-21
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "credit_accounts",
        sa.Column("user_id", sa.String(length=32),
                  sa.ForeignKey("users.user_id"), primary_key=True),
        sa.Column("week_key", sa.String(length=10), nullable=False),
        sa.Column("weekly_left", sa.Integer(), nullable=False),
        sa.Column("month_key", sa.String(length=7), nullable=False),
        sa.Column("month_cost_usd", sa.Float(), nullable=False),
    )
    op.create_table(
        "usage_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(length=32), nullable=True),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("llm_calls", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("estimated", sa.Boolean(), nullable=False),
        sa.Column("error_type", sa.String(length=24), nullable=True),
    )
    op.create_index("ix_usage_log_user_id", "usage_log", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_usage_log_user_id", table_name="usage_log")
    op.drop_table("usage_log")
    op.drop_table("credit_accounts")
