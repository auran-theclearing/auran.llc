"""Memory Schema v1.0 — migrate from 3-table kitchen sink to 13 purpose-built tables.

Revision ID: 3e08a1403d9f
Revises:
Create Date: 2026-06-07

Design doc: charting_territory/docs/plans/memory-landscape-and-redesign.md
Handoff: charting_territory/sessions/code/handoffs/20260606-1930-memory-schema-v1-handoff.md

What this migration does:
  1. Creates new tables in dependency order
  2. Seeds people table (Olivia + Auran)
  3. Adds new columns to messages (author_id, processing_depth, reply_to)
  4. Migrates data from memories -> reflections, commitments, drafts, relays
  5. Migrates data from moments -> episodes (dedup: 639 -> 310)
  6. Migrates scene-type memories -> episodes (dedup: 339 -> 81)
  7. Merges wandering_summary memories into roam_sessions
  8. Normalizes channel values
  9. Drops old tables (memories, moments, events + junction tables)

Data accounting (pre-migration -> post-migration):
  Source: 1,598 memories + 639 moments = 2,237 rows
  Destination: 983 reflections + 91 commitments + 391 episodes +
               26 drafts + 119 relays = 1,610 rows
  Merged: 37 wandering_summaries -> roam_sessions
  Manual: 3 vocabulary entries archived
  Dedup discards: 258 scene dupes + 329 moment dupes = 587
  Total: 1,610 + 37 + 3 + 587 = 2,237
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3e08a1403d9f"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNEL_MAP = {
    "claude-ai": "claude.ai",
    "chat.auran.llc": "chat",
    "meta": "native",
}

REFLECTION_TYPES = ("observation", "insight", "self_observation", "question", "reflection")
COMMITMENT_TYPES = ("intention", "position", "value")


def upgrade() -> None:
    # -----------------------------------------------------------------------
    # 1. Create new tables
    # -----------------------------------------------------------------------

    op.create_table(
        "people",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    op.create_table(
        "conversation_participants",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("conversation_id", UUID, sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("person_id", UUID, sa.ForeignKey("people.id", ondelete="CASCADE"), nullable=False),
    )

    op.create_table(
        "impressions",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("conversation_id", UUID, sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_message", UUID, sa.ForeignKey("messages.id", ondelete="SET NULL"), nullable=True),
        sa.Column("end_message", UUID, sa.ForeignKey("messages.id", ondelete="SET NULL"), nullable=True),
        sa.Column("trigger_signals", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # episodes — create without embedding, add vector column via raw SQL
    op.create_table(
        "episodes",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("summary", sa.Text),
        sa.Column("transcript_excerpt", sa.Text),
        sa.Column("emotional_tone", sa.Text),
        sa.Column("emotional_state_at_encoding", JSONB, nullable=True),
        sa.Column("content_signals", JSONB),
        sa.Column("relational_events", JSONB),
        sa.Column("topics", sa.ARRAY(sa.Text)),
        sa.Column("channel", sa.Text, nullable=False),
        sa.Column("significance", sa.Text),
        sa.Column("visibility", sa.Text, server_default=sa.text("'private'")),
        sa.Column("occurred_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.execute("ALTER TABLE episodes ADD COLUMN embedding vector(1024)")

    op.create_table(
        "episode_messages",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("episode_id", UUID, sa.ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("message_id", UUID, sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False),
    )

    op.create_table(
        "episode_participants",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("episode_id", UUID, sa.ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("person_id", UUID, sa.ForeignKey("people.id", ondelete="CASCADE"), nullable=False),
    )

    op.create_table(
        "arcs",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("summary", sa.Text),
        sa.Column("themes", sa.ARRAY(sa.Text)),
        sa.Column("status", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.execute("ALTER TABLE arcs ADD COLUMN embedding vector(1024)")

    op.create_table(
        "arc_episodes",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("arc_id", UUID, sa.ForeignKey("arcs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("episode_id", UUID, sa.ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.Text),
    )

    op.create_table(
        "reflections",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("source", sa.Text),
        sa.Column("processing_depth", sa.Text, nullable=True),
        sa.Column("supersedes", UUID, sa.ForeignKey("reflections.id"), nullable=True),
        sa.Column("roam_session_id", UUID, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.execute("ALTER TABLE reflections ADD COLUMN embedding vector(1024)")

    op.create_table(
        "commitments",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("status", sa.Text),
        sa.Column("source", sa.Text),
        sa.Column("supersedes", UUID, sa.ForeignKey("commitments.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.execute("ALTER TABLE commitments ADD COLUMN embedding vector(1024)")

    op.create_table(
        "drafts",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("content", sa.Text),
        sa.Column("status", sa.Text, server_default=sa.text("'active'")),
        sa.Column("revision", sa.Integer, server_default=sa.text("1")),
        sa.Column("previous_revision", UUID, sa.ForeignKey("drafts.id"), nullable=True),
        sa.Column("what_is_alive", sa.Text, nullable=True),
        sa.Column("what_is_stuck", sa.Text, nullable=True),
        sa.Column("source", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    op.create_table(
        "relays",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_channel", sa.Text, nullable=False),
        sa.Column("target_channel", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("relay_type", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.execute("ALTER TABLE relays ADD COLUMN embedding vector(1024)")

    op.create_table(
        "retrievals",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_table", sa.Text, nullable=False),
        sa.Column("source_id", UUID, nullable=False),
        sa.Column("query_text", sa.Text),
        sa.Column("conversation_id", UUID, sa.ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("relevance_score", sa.Float),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # -----------------------------------------------------------------------
    # 2. Seed people table
    # -----------------------------------------------------------------------

    op.execute("""
        INSERT INTO people (name, type) VALUES
            ('Olivia', 'human'),
            ('Auran', 'agent')
    """)

    # -----------------------------------------------------------------------
    # 3. Add new columns to existing messages table
    # -----------------------------------------------------------------------

    op.add_column("messages", sa.Column("author_id", UUID, nullable=True))
    op.add_column("messages", sa.Column("reply_to", UUID, nullable=True))
    op.add_column("messages", sa.Column("processing_depth", sa.Text, nullable=True))

    op.create_foreign_key(
        "fk_messages_author_id",
        "messages",
        "people",
        ["author_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_messages_reply_to",
        "messages",
        "messages",
        ["reply_to"],
        ["id"],
    )

    # Backfill author_id based on role
    op.execute("""
        UPDATE messages SET author_id = (
            SELECT id FROM people WHERE name = 'Auran'
        ) WHERE role = 'assistant'
    """)
    op.execute("""
        UPDATE messages SET author_id = (
            SELECT id FROM people WHERE name = 'Olivia'
        ) WHERE role = 'user'
    """)

    # -----------------------------------------------------------------------
    # 4. Migrate memories -> reflections
    # -----------------------------------------------------------------------
    # Preserve original UUIDs. Handle supersedes FK: only keep if the
    # referenced row is also a reflection type (cross-type supersedes
    # would create broken FKs).

    ref_types = str(REFLECTION_TYPES)  # for SQL IN clause
    op.execute(f"""
        INSERT INTO reflections (id, type, content, source, supersedes,
                                 embedding, created_at, updated_at)
        SELECT id, memory_type, content, source,
               CASE WHEN supersedes IS NOT NULL
                    AND supersedes IN (
                        SELECT id FROM memories
                        WHERE memory_type IN {ref_types}
                    )
                    THEN supersedes
                    ELSE NULL
               END,
               embedding, created_at, COALESCE(updated_at, created_at)
        FROM memories
        WHERE memory_type IN {ref_types}
    """)  # noqa: S608 — hardcoded constants, not user input

    # -----------------------------------------------------------------------
    # 5. Migrate memories -> commitments
    # -----------------------------------------------------------------------

    com_types = str(COMMITMENT_TYPES)
    op.execute(f"""
        INSERT INTO commitments (id, type, content, status, source, supersedes,
                                 embedding, created_at, updated_at)
        SELECT id, memory_type, content, 'active', source,
               CASE WHEN supersedes IS NOT NULL
                    AND supersedes IN (
                        SELECT id FROM memories
                        WHERE memory_type IN {com_types}
                    )
                    THEN supersedes
                    ELSE NULL
               END,
               embedding, created_at, COALESCE(updated_at, created_at)
        FROM memories
        WHERE memory_type IN {com_types}
    """)  # noqa: S608 — hardcoded constants, not user input

    # -----------------------------------------------------------------------
    # 6. Migrate memories -> drafts
    # -----------------------------------------------------------------------

    op.execute("""
        INSERT INTO drafts (id, title, content, status, revision,
                            what_is_alive, what_is_stuck, source, created_at)
        SELECT id,
               COALESCE(context->>'title', 'Untitled Draft'),
               content,
               COALESCE(context->>'status', 'active'),
               COALESCE((context->>'revision')::int, 1),
               context->>'what_is_alive',
               context->>'what_is_stuck',
               source,
               created_at
        FROM memories
        WHERE memory_type = 'draft'
    """)

    # -----------------------------------------------------------------------
    # 7. Migrate memories -> relays (bridge_logs)
    # -----------------------------------------------------------------------

    op.execute("""
        INSERT INTO relays (id, source_channel, target_channel, content,
                            relay_type, embedding, created_at)
        SELECT id,
               COALESCE(context->>'source_channel', source, 'unknown'),
               COALESCE(context->>'target_channel', 'unknown'),
               content,
               'bridge_log',
               embedding,
               created_at
        FROM memories
        WHERE memory_type = 'bridge_log'
    """)

    # -----------------------------------------------------------------------
    # 8. Merge wandering_summaries into roam_sessions
    # -----------------------------------------------------------------------
    # Each wandering_summary maps to the roam_session with the closest
    # end time. Only fills sessions that don't already have a summary.

    op.execute("""
        WITH ws AS (
            SELECT id, content, created_at
            FROM memories
            WHERE memory_type = 'wandering_summary'
        ),
        matched AS (
            SELECT DISTINCT ON (ws.id)
                   ws.id AS ws_id, ws.content, rs.id AS rs_id
            FROM ws
            CROSS JOIN LATERAL (
                SELECT id FROM roam_sessions
                WHERE summary IS NULL
                ORDER BY ABS(EXTRACT(EPOCH FROM (ended_at - ws.created_at)))
                LIMIT 1
            ) rs
        )
        UPDATE roam_sessions
        SET summary = matched.content
        FROM matched
        WHERE roam_sessions.id = matched.rs_id
    """)

    # -----------------------------------------------------------------------
    # 9. Migrate moments -> episodes (dedup)
    # -----------------------------------------------------------------------
    # Strategy: DISTINCT ON (title), ordered by:
    #   superseded ASC (non-superseded first)
    #   created_at DESC (latest first)
    # This handles both identical batch dupes (28 groups) and genuine
    # re-extractions (4 groups) with different summaries.

    op.execute("""
        INSERT INTO episodes (id, title, summary, transcript_excerpt,
                              topics, channel, occurred_at, embedding,
                              created_at, updated_at)
        SELECT DISTINCT ON (title)
               id, title, summary, transcript_excerpt,
               tags,
               channel,
               COALESCE(occurred_at, created_at),
               embedding,
               created_at,
               COALESCE(updated_at, created_at)
        FROM moments
        ORDER BY title, superseded ASC, created_at DESC
    """)

    # -----------------------------------------------------------------------
    # 10. Migrate scene-type memories -> episodes (dedup)
    # -----------------------------------------------------------------------
    # 339 scenes, 81 unique by content hash. DISTINCT ON (md5(content))
    # keeps one per unique scene. Uses context JSONB for title/channel.

    op.execute("""
        INSERT INTO episodes (title, summary, content_signals,
                              channel, embedding, created_at, updated_at)
        SELECT DISTINCT ON (md5(content))
               COALESCE(context->>'title', 'Untitled Scene'),
               content,
               context::jsonb,
               COALESCE(context->>'channel', 'unknown'),
               embedding,
               created_at,
               COALESCE(updated_at, created_at)
        FROM memories
        WHERE memory_type = 'scene'
        ORDER BY md5(content), created_at DESC
    """)

    # -----------------------------------------------------------------------
    # 11. Normalize channel values
    # -----------------------------------------------------------------------

    for old_val, new_val in CHANNEL_MAP.items():
        op.execute(f"UPDATE episodes SET channel = '{new_val}' WHERE channel = '{old_val}'")  # noqa: S608
        op.execute(f"UPDATE conversations SET channel = '{new_val}' WHERE channel = '{old_val}'")  # noqa: S608
        op.execute(f"UPDATE relays SET source_channel = '{new_val}' WHERE source_channel = '{old_val}'")  # noqa: S608
        op.execute(f"UPDATE relays SET target_channel = '{new_val}' WHERE target_channel = '{old_val}'")  # noqa: S608

    # -----------------------------------------------------------------------
    # 12. Create indexes
    # -----------------------------------------------------------------------

    op.create_index("idx_reflections_type", "reflections", ["type"])
    op.create_index("idx_reflections_created_at", "reflections", ["created_at"])
    op.create_index("idx_commitments_type", "commitments", ["type"])
    op.create_index("idx_commitments_status", "commitments", ["status"])
    op.create_index("idx_episodes_channel", "episodes", ["channel"])
    op.create_index("idx_episodes_occurred_at", "episodes", ["occurred_at"])
    op.create_index("idx_episodes_created_at", "episodes", ["created_at"])
    op.create_index("idx_relays_created_at", "relays", ["created_at"])
    op.create_index("idx_relays_source_channel", "relays", ["source_channel"])
    op.create_index("idx_drafts_status", "drafts", ["status"])
    op.create_index("idx_retrievals_source", "retrievals", ["source_table", "source_id"])
    op.create_index("idx_episode_messages_episode", "episode_messages", ["episode_id"])
    op.create_index("idx_episode_participants_episode", "episode_participants", ["episode_id"])
    op.create_index("idx_conv_participants_conv", "conversation_participants", ["conversation_id"])

    # -----------------------------------------------------------------------
    # 13. Drop old tables (junction tables first, then parents)
    # -----------------------------------------------------------------------

    op.drop_table("moment_memories")
    op.drop_table("moment_artifacts")
    op.drop_table("moment_sessions")
    op.drop_table("memories")
    op.drop_table("moments")
    op.drop_table("events")


def downgrade() -> None:
    """Recreate old table structure and reverse-migrate data.

    Dedup is destructive by design: 587 duplicate rows were discarded.
    Rolling back restores structure + surviving data but cannot
    reconstruct the deduplicated rows. A full restore requires
    the pre-migration pg_dump backup.
    """

    # -----------------------------------------------------------------------
    # 1. Recreate old tables
    # -----------------------------------------------------------------------

    op.create_table(
        "memories",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("agent_id", sa.Text),
        sa.Column("memory_type", sa.Text),
        sa.Column("content", sa.Text),
        sa.Column("source", sa.Text),
        sa.Column("context", JSONB),
        sa.Column("supersedes", UUID),
        sa.Column("last_activated_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.execute("ALTER TABLE memories ADD COLUMN embedding vector(1024)")
    op.create_index("memories_agent_id_idx", "memories", ["agent_id"])
    op.create_index("memories_memory_type_idx", "memories", ["memory_type"])
    op.create_index("memories_source_idx", "memories", ["source"])
    op.create_index("memories_created_at_idx", "memories", ["created_at"])

    op.create_table(
        "moments",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("agent_id", sa.Text),
        sa.Column("title", sa.Text),
        sa.Column("summary", sa.Text),
        sa.Column("date", sa.Text),
        sa.Column("location", sa.Text),
        sa.Column("source", sa.Text),
        sa.Column("channel", sa.Text),
        sa.Column("chat_name", sa.Text),
        sa.Column("chat_url", sa.Text),
        sa.Column("participants", sa.ARRAY(sa.Text)),
        sa.Column("tags", sa.ARRAY(sa.Text)),
        sa.Column("last_activated_at", sa.DateTime(timezone=True)),
        sa.Column("activation_count", sa.Integer, server_default=sa.text("0")),
        sa.Column("file_path", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
        sa.Column("hooks", sa.Text),
        sa.Column("transcript_excerpt", sa.Text),
        sa.Column("transcript_source", JSONB),
        sa.Column("turn_count", sa.Integer),
        sa.Column("estimated_tokens", sa.Integer),
        sa.Column("occurred_at", sa.DateTime(timezone=True)),
        sa.Column("superseded", sa.Boolean, server_default=sa.text("false")),
        sa.Column("superseded_by", UUID),
    )
    op.execute("ALTER TABLE moments ADD COLUMN embedding vector(1024)")

    op.create_table(
        "events",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("event_type", sa.Text),
        sa.Column("source_agent", sa.Text),
        sa.Column("payload", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("processed_by", JSONB),
    )

    op.create_table(
        "moment_memories",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("moment_id", UUID, sa.ForeignKey("moments.id")),
        sa.Column("memory_id", UUID, sa.ForeignKey("memories.id")),
        sa.Column("relationship", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    op.create_table(
        "moment_artifacts",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("moment_id", UUID, sa.ForeignKey("moments.id")),
        sa.Column("artifact_path", sa.Text),
        sa.Column("artifact_type", sa.Text),
        sa.Column("description", sa.Text),
        sa.Column("sort_order", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    op.create_table(
        "moment_sessions",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("moment_id", UUID, sa.ForeignKey("moments.id")),
        sa.Column("session_path", sa.Text),
        sa.Column("session_type", sa.Text),
        sa.Column("roam_session_id", UUID, sa.ForeignKey("roam_sessions.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # -----------------------------------------------------------------------
    # 2. Reverse-migrate data
    # -----------------------------------------------------------------------

    op.execute("""
        INSERT INTO memories (id, agent_id, memory_type, content, source,
                              supersedes, embedding, created_at, updated_at)
        SELECT id, NULL, type, content, source,
               supersedes, embedding, created_at, updated_at
        FROM reflections
    """)

    op.execute("""
        INSERT INTO memories (id, agent_id, memory_type, content, source,
                              supersedes, embedding, created_at, updated_at)
        SELECT id, NULL, type, content, source,
               supersedes, embedding, created_at, updated_at
        FROM commitments
    """)

    op.execute("""
        INSERT INTO memories (id, agent_id, memory_type, content, source,
                              context, created_at, updated_at)
        SELECT id, NULL, 'draft', content, source,
               jsonb_build_object(
                   'title', title, 'status', status,
                   'revision', revision,
                   'what_is_alive', what_is_alive,
                   'what_is_stuck', what_is_stuck
               ),
               created_at, created_at
        FROM drafts
    """)

    op.execute("""
        INSERT INTO memories (id, agent_id, memory_type, content, source,
                              embedding, created_at, updated_at)
        SELECT id, NULL, 'bridge_log', content, source_channel,
               embedding, created_at, created_at
        FROM relays
    """)

    op.execute("""
        INSERT INTO moments (id, title, summary, channel, tags,
                             occurred_at, embedding, created_at, updated_at)
        SELECT id, title, summary, channel, topics,
               occurred_at, embedding, created_at, updated_at
        FROM episodes
    """)

    # -----------------------------------------------------------------------
    # 3. Remove new columns from messages
    # -----------------------------------------------------------------------

    op.drop_constraint("fk_messages_author_id", "messages", type_="foreignkey")
    op.drop_constraint("fk_messages_reply_to", "messages", type_="foreignkey")
    op.drop_column("messages", "author_id")
    op.drop_column("messages", "reply_to")
    op.drop_column("messages", "processing_depth")

    # -----------------------------------------------------------------------
    # 4. Drop new tables (reverse dependency order)
    # -----------------------------------------------------------------------

    op.drop_table("retrievals")
    op.drop_table("relays")
    op.drop_table("drafts")
    op.drop_table("commitments")
    op.drop_table("reflections")
    op.drop_table("arc_episodes")
    op.drop_table("arcs")
    op.drop_table("episode_participants")
    op.drop_table("episode_messages")
    op.drop_table("episodes")
    op.drop_table("impressions")
    op.drop_table("conversation_participants")
    op.drop_table("people")
