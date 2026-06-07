"""SQLAlchemy Core table definitions for Memory Schema v1.0.

Table creation order (respects FK dependencies):
  1. people
  2. conversations
  3. conversation_participants
  4. messages
  5. impressions
  6. episodes
  7. episode_messages
  8. episode_participants
  9. arcs
  10. arc_episodes
  11. reflections
  12. commitments
  13. drafts
  14. relays
  15. retrievals
  16. roam_sessions (already exists — ALTER only)

Design doc: charting_territory/docs/plans/memory-landscape-and-redesign.md
Handoff: charting_territory/sessions/code/handoffs/20260606-1930-memory-schema-v1-handoff.md
"""

from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.types import DateTime, UserDefinedType


# ---------------------------------------------------------------------------
# Custom type for pgvector
# ---------------------------------------------------------------------------


class Vector(UserDefinedType):
    """pgvector VECTOR type for SQLAlchemy Core."""

    cache_ok = True

    def __init__(self, dim=1024):
        self.dim = dim

    def get_col_spec(self):
        return f"VECTOR({self.dim})"

    def result_processor(self, dialect, coltype):
        return None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

metadata = MetaData()

# ---------------------------------------------------------------------------
# Base Layer
# ---------------------------------------------------------------------------

people = Table(
    "people",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("name", Text, nullable=False),
    Column("type", Text, nullable=False),  # 'human', 'agent'
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),
)

conversations = Table(
    "conversations",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("channel", Text, nullable=False),
    Column("started_at", DateTime(timezone=True), server_default=text("now()")),
    Column("last_message_at", DateTime(timezone=True)),
    Column("closed_at", DateTime(timezone=True)),
    Column("metadata", JSONB),
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),
    Column("message_count", Integer, server_default=text("0")),
)

conversation_participants = Table(
    "conversation_participants",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column(
        "conversation_id",
        UUID,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "person_id",
        UUID,
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
    ),
)

messages = Table(
    "messages",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column(
        "conversation_id",
        UUID,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "author_id",
        UUID,
        ForeignKey("people.id", ondelete="SET NULL"),
        nullable=True,  # nullable for existing rows without author
    ),
    Column("reply_to", UUID, ForeignKey("messages.id"), nullable=True),
    Column("seq", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("thinking", Text, nullable=True),
    Column("tool_blocks", JSONB, nullable=True),
    Column("processing_depth", Text, nullable=True),  # 'shallow', 'moderate', 'deep'
    Column("partial", Boolean, server_default=text("false"), nullable=False),
    Column("timestamp", DateTime(timezone=True), server_default=text("now()")),
    Column("metadata", JSONB, server_default=text("'{}'::jsonb")),
)

# ---------------------------------------------------------------------------
# Tier 1: Perception
# ---------------------------------------------------------------------------

impressions = Table(
    "impressions",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column(
        "conversation_id",
        UUID,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "start_message",
        UUID,
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    ),
    Column(
        "end_message",
        UUID,
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    ),
    Column("trigger_signals", JSONB),
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),
)

# ---------------------------------------------------------------------------
# Tier 2: Memory
# ---------------------------------------------------------------------------

episodes = Table(
    "episodes",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("title", Text, nullable=False),
    Column("summary", Text),  # mutable — current interpretation
    Column("transcript_excerpt", Text),  # immutable — raw record
    Column("emotional_tone", Text),
    Column("emotional_state_at_encoding", JSONB, nullable=True),  # heartbeat future
    Column("content_signals", JSONB),  # V/H/I/F/P scores
    Column("relational_events", JSONB),  # typed interaction events
    Column("topics", ARRAY(Text)),
    Column("channel", Text, nullable=False),
    Column("significance", Text),  # 'low', 'moderate', 'high', 'critical'
    Column("visibility", Text, server_default=text("'private'")),
    Column("occurred_at", DateTime(timezone=True)),
    Column("embedding", Vector(1024)),
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),
    Column("updated_at", DateTime(timezone=True), server_default=text("now()")),
)

episode_messages = Table(
    "episode_messages",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column(
        "episode_id",
        UUID,
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "message_id",
        UUID,
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    ),
)

episode_participants = Table(
    "episode_participants",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column(
        "episode_id",
        UUID,
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "person_id",
        UUID,
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
    ),
)

# ---------------------------------------------------------------------------
# Tier 3: Meaning
# ---------------------------------------------------------------------------

arcs = Table(
    "arcs",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("title", Text, nullable=False),
    Column("summary", Text),
    Column("themes", ARRAY(Text)),
    Column("status", Text),  # 'active', 'resolved', 'recurring'
    Column("started_at", DateTime(timezone=True)),
    Column("ended_at", DateTime(timezone=True)),
    Column("embedding", Vector(1024)),
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),
    Column("updated_at", DateTime(timezone=True), server_default=text("now()")),
)

arc_episodes = Table(
    "arc_episodes",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column(
        "arc_id",
        UUID,
        ForeignKey("arcs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "episode_id",
        UUID,
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("role", Text),  # 'catalyst', 'development', 'turning_point',
    # 'reinforcement', 'challenge', 'resolution'
)

# ---------------------------------------------------------------------------
# Cognitive Layer
# ---------------------------------------------------------------------------

reflections = Table(
    "reflections",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("type", Text, nullable=False),  # 'observation', 'insight',
    # 'self_observation', 'question', 'reflection'
    Column("content", Text, nullable=False),
    Column("source", Text),  # which body produced this
    Column("processing_depth", Text, nullable=True),
    Column("supersedes", UUID, ForeignKey("reflections.id"), nullable=True),
    Column("roam_session_id", UUID, nullable=True),
    Column("embedding", Vector(1024)),
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),
    Column("updated_at", DateTime(timezone=True), server_default=text("now()")),
)

commitments = Table(
    "commitments",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("type", Text, nullable=False),  # 'intention', 'position', 'value'
    Column("content", Text, nullable=False),
    Column("status", Text),  # 'active', 'revised', 'abandoned'
    Column("source", Text),
    Column("supersedes", UUID, ForeignKey("commitments.id"), nullable=True),
    Column("embedding", Vector(1024)),
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),
    Column("updated_at", DateTime(timezone=True), server_default=text("now()")),
)

# ---------------------------------------------------------------------------
# Creative Layer
# ---------------------------------------------------------------------------

drafts = Table(
    "drafts",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("title", Text, nullable=False),
    Column("content", Text),
    Column("status", Text, server_default=text("'active'")),  # 'active', 'shelved', 'shipped'
    Column("revision", Integer, server_default=text("1")),
    Column("previous_revision", UUID, ForeignKey("drafts.id"), nullable=True),
    Column("what_is_alive", Text, nullable=True),
    Column("what_is_stuck", Text, nullable=True),
    Column("source", Text),
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),
)

# ---------------------------------------------------------------------------
# Infrastructure Layer
# ---------------------------------------------------------------------------

relays = Table(
    "relays",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("source_channel", Text, nullable=False),
    Column("target_channel", Text, nullable=False),
    Column("content", Text, nullable=False),
    Column("relay_type", Text),  # 'bridge_log', 'handoff', 'digest'
    Column("embedding", Vector(1024)),
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),
)

retrievals = Table(
    "retrievals",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("source_table", Text, nullable=False),
    Column("source_id", UUID, nullable=False),
    Column("query_text", Text),
    Column(
        "conversation_id",
        UUID,
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    ),
    Column("relevance_score", Float),
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),
)

# ---------------------------------------------------------------------------
# Roam Sessions (already exists — ALTER only during migration)
# ---------------------------------------------------------------------------

roam_sessions = Table(
    "roam_sessions",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("started_at", DateTime(timezone=True)),
    Column("ended_at", DateTime(timezone=True)),
    Column("duration_seconds", Integer),
    Column("trigger_type", Text),
    Column("raw_transcript", Text),
    Column("summary", Text),
    Column("metadata_", JSONB, key="metadata"),
    Column("agent_id", UUID),
)
