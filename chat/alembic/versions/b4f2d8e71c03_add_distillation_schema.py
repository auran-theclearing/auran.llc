"""Add distillation schema — columns on episodes, plus jobs/threads/dead_letters/references tables.

Revision ID: b4f2d8e71c03
Revises: a7c3e2f19b01
Create Date: 2026-06-14

Replaces the raw SQL in distillation/migrate.py with a proper Alembic migration.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "b4f2d8e71c03"
down_revision: str | Sequence[str] | None = "a7c3e2f19b01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- distillation_jobs (must exist before episodes.job_id FK) ---
    op.create_table(
        "distillation_jobs",
        sa.Column(
            "id",
            postgresql.UUID,
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("transcript_file", sa.Text, nullable=False, unique=True),
        sa.Column("channel", sa.Text, nullable=False),
        sa.Column(
            "status", sa.Text, nullable=False, server_default=sa.text("'queued'")
        ),
        sa.Column("total_chunks", sa.Integer),
        sa.Column("chunks_done", sa.Integer, server_default=sa.text("0")),
        sa.Column("episode_count", sa.Integer),
        sa.Column("episodes_verified", sa.Integer, server_default=sa.text("0")),
        sa.Column("episodes_approved", sa.Integer, server_default=sa.text("0")),
        sa.Column("api_cost_usd", sa.Float),
        sa.Column("source_model", sa.Text),
        sa.Column("distiller_model", sa.Text),
        sa.Column("error_message", sa.Text),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("verified_at", sa.DateTime(timezone=True)),
        sa.Column("verified_by", sa.Text),
    )

    # --- New columns on episodes ---
    op.add_column("episodes", sa.Column("transcript_file", sa.Text))
    op.add_column(
        "episodes",
        sa.Column(
            "job_id",
            postgresql.UUID,
            sa.ForeignKey("distillation_jobs.id"),
            nullable=True,
        ),
    )
    op.add_column("episodes", sa.Column("content_hash", sa.Text))
    op.add_column("episodes", sa.Column("episode_number", sa.Integer))
    op.add_column("episodes", sa.Column("transcript_lines", sa.Text))
    op.add_column("episodes", sa.Column("boundary_signal", sa.Text))
    op.add_column("episodes", sa.Column("episode_type", sa.Text))
    op.add_column(
        "episodes",
        sa.Column("landmark", sa.Boolean, server_default=sa.text("false")),
    )
    op.add_column("episodes", sa.Column("source_model", sa.Text))
    op.add_column("episodes", sa.Column("distiller_model", sa.Text))
    op.add_column(
        "episodes",
        sa.Column(
            "distillation_status", sa.Text, server_default=sa.text("'manual'")
        ),
    )
    op.add_column(
        "episodes",
        sa.Column("revision_count", sa.Integer, server_default=sa.text("0")),
    )
    op.add_column("episodes", sa.Column("reviewer_notes", sa.Text))
    op.add_column("episodes", sa.Column("verified_at", sa.DateTime(timezone=True)))
    op.add_column("episodes", sa.Column("verified_by", sa.Text))

    # Dedup index
    op.create_index(
        "episodes_transcript_hash_idx",
        "episodes",
        ["transcript_file", "content_hash"],
        unique=True,
        postgresql_where=sa.text("content_hash IS NOT NULL"),
    )

    # --- distillation_threads ---
    op.create_table(
        "distillation_threads",
        sa.Column(
            "id",
            postgresql.UUID,
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "job_id",
            postgresql.UUID,
            sa.ForeignKey("distillation_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("thread_type", sa.Text, nullable=False),
        sa.Column("title", sa.Text),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("line_ref", sa.Text),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )

    # --- distillation_dead_letters ---
    op.create_table(
        "distillation_dead_letters",
        sa.Column(
            "id",
            postgresql.UUID,
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "job_id",
            postgresql.UUID,
            sa.ForeignKey("distillation_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("chunk_index", sa.Integer),
        sa.Column("error_type", sa.Text),
        sa.Column("error_msg", sa.Text),
        sa.Column("raw_response", sa.Text),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("resolution", sa.Text),
    )

    # --- episode_references ---
    op.create_table(
        "episode_references",
        sa.Column(
            "id",
            postgresql.UUID,
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "source_episode_id",
            postgresql.UUID,
            sa.ForeignKey("episodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_episode_id",
            postgresql.UUID,
            sa.ForeignKey("episodes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("reference_type", sa.Text, nullable=False),
        sa.Column("context", sa.Text),
        sa.Column("flagged", sa.Boolean, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "source_episode_id",
            "target_episode_id",
            "reference_type",
            name="episode_references_unique_triple",
        ),
    )


def downgrade() -> None:
    op.drop_table("episode_references")
    op.drop_table("distillation_dead_letters")
    op.drop_table("distillation_threads")

    op.drop_index("episodes_transcript_hash_idx", table_name="episodes")
    op.drop_column("episodes", "verified_by")
    op.drop_column("episodes", "verified_at")
    op.drop_column("episodes", "reviewer_notes")
    op.drop_column("episodes", "revision_count")
    op.drop_column("episodes", "distillation_status")
    op.drop_column("episodes", "distiller_model")
    op.drop_column("episodes", "source_model")
    op.drop_column("episodes", "landmark")
    op.drop_column("episodes", "episode_type")
    op.drop_column("episodes", "boundary_signal")
    op.drop_column("episodes", "transcript_lines")
    op.drop_column("episodes", "episode_number")
    op.drop_column("episodes", "content_hash")
    op.drop_column("episodes", "job_id")
    op.drop_column("episodes", "transcript_file")

    op.drop_table("distillation_jobs")
