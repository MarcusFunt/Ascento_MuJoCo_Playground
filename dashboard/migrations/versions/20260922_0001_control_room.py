"""Create dashboard semantic index."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260922_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dashboard_runs",
        sa.Column("id", sa.String(length=128), primary_key=True),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("artifact_name", sa.Text(), nullable=False),
        sa.Column("task", sa.String(length=128), nullable=True),
        sa.Column("stage", sa.String(length=128), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=True),
        sa.Column("total_iterations", sa.Integer(), nullable=True),
        sa.Column("percent_complete", sa.Float(), nullable=True),
        sa.Column("reward", sa.Float(), nullable=True),
        sa.Column("episode_length", sa.Float(), nullable=True),
        sa.Column("kl", sa.Float(), nullable=True),
        sa.Column("entropy", sa.Float(), nullable=True),
        sa.Column("repository_status", sa.String(length=32), nullable=True),
        sa.Column("run_commit", sa.String(length=64), nullable=True),
        sa.Column("modified_at", sa.Float(), nullable=True),
        sa.Column("curriculum", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_dashboard_runs_task", "dashboard_runs", ["task"])
    op.create_index("ix_dashboard_runs_state", "dashboard_runs", ["state"])
    op.create_table(
        "dashboard_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(length=128), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_dashboard_events_run_id", "dashboard_events", ["run_id"])
    op.create_index("ix_dashboard_events_event_type", "dashboard_events", ["event_type"])
    op.create_index("ix_dashboard_events_created_at", "dashboard_events", ["created_at"])
    op.create_table(
        "dashboard_checkpoints",
        sa.Column("id", sa.String(length=256), primary_key=True),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=True),
        sa.Column("stable", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_dashboard_checkpoints_run_id", "dashboard_checkpoints", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_dashboard_checkpoints_run_id", table_name="dashboard_checkpoints")
    op.drop_table("dashboard_checkpoints")
    op.drop_index("ix_dashboard_events_created_at", table_name="dashboard_events")
    op.drop_index("ix_dashboard_events_event_type", table_name="dashboard_events")
    op.drop_index("ix_dashboard_events_run_id", table_name="dashboard_events")
    op.drop_table("dashboard_events")
    op.drop_index("ix_dashboard_runs_state", table_name="dashboard_runs")
    op.drop_index("ix_dashboard_runs_task", table_name="dashboard_runs")
    op.drop_table("dashboard_runs")
